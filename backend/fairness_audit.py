"""Demographic accuracy audit tooling, and a label-free confound diagnostic.

This module exists because "audit for skin-tone bias" was flagged as
necessary but blocked on data this project does not have: there is no
dataset of ClassGraph's real classroom footage with human-annotated skin
tone labels. Before building an audit that could not be run, a research pass
checked what is already known about the two models this pipeline actually
uses -- MediaPipe Face Mesh and SixDRepNet -- rather than assuming the usual
face-recognition bias literature (Gender Shades and its successors) transfers
directly to geometric landmark/pose regression, which is a different task
with a thinner and more contradictory bias literature.

What is already known (verified from primary sources, not assumed):

* MediaPipe Face Mesh has a real, published Google fairness model card
  (Yan & Grishchenko, 2022) that tested Fitzpatrick skin tone I-VI AND 17
  geographic regions, including a "Southern Asia" bucket -- this project's
  actual population. It passed Google's own fairness threshold, but
  "Southern Asia" was among the worse-performing regions (2.97% IOD MAE in
  tracking mode vs. 2.06% best-case), and the card's test conditions
  (front-facing selfie camera) do not match ClassGraph's real deployment
  (off-axis classroom CCTV, variable lighting, multiple faces per frame).
* SixDRepNet's paper, and every dataset in its training/evaluation chain
  (300W-LP, AFLW2000, BIWI), publish ZERO demographic composition data and
  ZERO fairness evaluation of any kind. This is a genuine documentation gap
  in the published literature, not something this project failed to find.
* The most methodologically careful recent academic audit of landmark-
  detection bias (Parte et al., 2026 preprint, arXiv:2604.06961) found that
  once image resolution and head-pose extremity are statistically
  controlled for, race/ethnicity showed no statistically significant effect
  on landmark accuracy in the model they tested (a different architecture
  than MediaPipe's) -- resolution alone explained 29.3% of error variance,
  the single largest factor found. A second study on a different model
  (Shadmi et al., 2021, arXiv:2111.01683) still found small but consistent
  White-favouring bias on a Black/white axis. These two findings do not
  fully agree with each other, which itself says the field has not
  converged -- do not treat either as the final word.
* Nobody has tested South Asian faces specifically, at scale, against
  either of ClassGraph's actual models. The closest proxies (Face Mesh's
  coarse "Southern Asia" geographic bucket; a different landmark model's
  small, automatically-labelled "Indian" subgroup in an unrelated study)
  are not the same thing as a real audit of this project's models on this
  project's population.

What this means for priority: the evidence available does not suggest this
is the most urgent open risk (the strongest recent finding suggests pose and
resolution confounds -- which hurt every student regardless of skin tone --
dominate over ethnicity effects once controlled), but it also does not clear
these models. Nobody has looked. Absence of evidence here is absence of
study, not a body of studies that came back clean. Treat any audit built
from this module as generating net-new evidence for a real, field-wide gap,
not confirming or refuting a specific known problem.

Two things this module provides:

1. ``stratified_detection_rate`` -- the actual stratified-accuracy
   computation, ready to run the moment labelled data exists. Pure function,
   fully tested, no model dependency.
2. ``image_quality_proxies`` / a documented protocol for running the
   label-free confound diagnostic the research recommends doing FIRST:
   checking whether detection success correlates with resolution, brightness
   and other confounds identified as dominant in the literature above --
   useful on this project's existing images today, without needing any
   demographic labels at all.

Usage (once labelled data exists):
    from backend.fairness_audit import stratified_detection_rate
    records = [{"stratum": "III", "detected": True}, ...]  # your own labelling
    print(stratified_detection_rate(records))

Usage (today, no labels needed):
    from backend.fairness_audit import image_quality_proxies
    proxies = image_quality_proxies(frame)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from backend.config import CONFIG

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StratumResult:
    """Detection-rate summary for one stratum (e.g. one skin-tone bin).

    Attributes:
        stratum: The stratum label (e.g. a Fitzpatrick bin, or a resolution
            bucket for the label-free diagnostic).
        n: Number of records in this stratum.
        detected: Number of those records where detection succeeded.
        detection_rate: ``detected / n``, or ``None`` if ``n == 0``.
    """

    stratum: str
    n: int
    detected: int
    detection_rate: float | None


def stratified_detection_rate(
    records: list[dict], stratum_key: str = "stratum", detected_key: str = "detected"
) -> dict[str, StratumResult]:
    """Compute detection rate per stratum from a list of labelled records.

    This is the actual audit computation. It is deliberately generic: the
    "stratum" can be a Fitzpatrick skin-tone bin, a resolution bucket, a
    lighting condition, or any other grouping -- the function does not know
    or care which, so the same code serves the real skin-tone audit (once
    labelled data exists) and the label-free confound diagnostic below.

    Args:
        records: One dict per observation. Each must contain ``stratum_key``
            (any hashable label) and ``detected_key`` (a bool: did detection
            succeed for this observation).
        stratum_key: Which key in each record holds the stratum label.
        detected_key: Which key in each record holds the detection outcome.

    Returns:
        A mapping from stratum label to its :class:`StratumResult`, covering
        every stratum present in ``records``.

    Raises:
        KeyError: If a record is missing ``stratum_key`` or ``detected_key``.
    """
    grouped: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        grouped[record[stratum_key]].append(bool(record[detected_key]))

    results: dict[str, StratumResult] = {}
    for stratum, outcomes in grouped.items():
        n = len(outcomes)
        detected = sum(outcomes)
        results[stratum] = StratumResult(
            stratum=stratum,
            n=n,
            detected=detected,
            detection_rate=(detected / n if n > 0 else None),
        )
    return results


def summarise_disparity(results: dict[str, StratumResult]) -> dict[str, object]:
    """Summarise the spread across strata -- the number that actually matters.

    A per-stratum table is easy to eyeball into a false conclusion from small
    samples; this makes the headline disparity explicit.

    Args:
        results: Output of :func:`stratified_detection_rate`.

    Returns:
        A dict with ``best_stratum``, ``worst_stratum``, ``rate_spread``
        (best minus worst detection rate), and ``min_stratum_n`` (the
        smallest sample size across strata, since a large spread built on a
        tiny sample is not evidence of anything -- see
        :data:`FairnessAuditConfig`'s docstring on the Sony AI finding that
        bias direction was "very sensitive to the sample set" in comparable
        audits).
    """
    rated = [r for r in results.values() if r.detection_rate is not None]
    if not rated:
        return {
            "best_stratum": None,
            "worst_stratum": None,
            "rate_spread": None,
            "min_stratum_n": 0,
        }
    best = max(rated, key=lambda r: r.detection_rate)
    worst = min(rated, key=lambda r: r.detection_rate)
    return {
        "best_stratum": best.stratum,
        "worst_stratum": worst.stratum,
        "rate_spread": best.detection_rate - worst.detection_rate,
        "min_stratum_n": min(r.n for r in rated),
    }


def image_quality_proxies(frame: np.ndarray) -> dict[str, float]:
    """Compute label-free confound proxies for one image: resolution and
    brightness, the two factors identified as dominant ahead of any
    demographic one in the closest available academic audit (see this
    module's docstring).

    Args:
        frame: A ``(H, W, 3)`` BGR image as returned by OpenCV.

    Returns:
        A dict with ``shorter_side`` (pixels), ``mean_brightness`` (0-255,
        computed on the luma channel), and ``resolution_bucket`` (a label
        from :data:`FairnessAuditConfig.resolution_bucket_edges`).

    Raises:
        TypeError: If ``frame`` is not a NumPy array.
        ValueError: If ``frame`` is empty or not a 3-channel image.
    """
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"frame must be a numpy.ndarray, got {type(frame)!r}.")
    if frame.size == 0:
        raise ValueError("frame is empty (zero-size array).")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            f"frame must be an (H, W, 3) image, got shape {frame.shape!r}."
        )

    import cv2

    h, w = frame.shape[:2]
    shorter_side = min(h, w)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(gray.mean())

    edges = CONFIG.fairness_audit.resolution_bucket_edges
    if shorter_side >= edges[-1]:
        bucket = f">={edges[-1]}"
    else:
        bucket = f"<{edges[0]}"
        for lo, hi in pairwise(edges):
            if lo <= shorter_side < hi:
                bucket = f"[{lo},{hi})"
                break
    return {
        "shorter_side": float(shorter_side),
        "mean_brightness": mean_brightness,
        "resolution_bucket": bucket,
    }
