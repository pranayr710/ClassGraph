"""Unit tests for :mod:`backend.fairness_audit`.

Pure logic tests (stratified_detection_rate, summarise_disparity) run
everywhere. image_quality_proxies needs cv2 (already a hard project
dependency) but no ML model -- it is a pixel-statistics function, not a
detector.

Required coverage:
    1. stratified_detection_rate groups correctly and handles an empty
       stratum without dividing by zero.
    2. summarise_disparity reports the real spread and the smallest sample
       size, and degrades gracefully with no data.
    3. image_quality_proxies computes sane values and buckets resolution
       correctly at and around the configured edges.
    4. Bad input to image_quality_proxies raises explicit, typed errors.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from backend.fairness_audit import (
    StratumResult,
    image_quality_proxies,
    stratified_detection_rate,
    summarise_disparity,
)

# --------------------------------------------------------------------------- #
# stratified_detection_rate
# --------------------------------------------------------------------------- #


def test_stratified_detection_rate_groups_and_computes_rate() -> None:
    records = [
        {"stratum": "III", "detected": True},
        {"stratum": "III", "detected": True},
        {"stratum": "III", "detected": False},
        {"stratum": "V", "detected": False},
    ]
    results = stratified_detection_rate(records)
    assert set(results) == {"III", "V"}
    assert results["III"] == StratumResult(
        stratum="III", n=3, detected=2, detection_rate=pytest.approx(2 / 3)
    )
    assert results["V"] == StratumResult(
        stratum="V", n=1, detected=0, detection_rate=0.0
    )


def test_stratified_detection_rate_empty_input() -> None:
    assert stratified_detection_rate([]) == {}


def test_stratified_detection_rate_custom_keys() -> None:
    records = [{"skin_tone": "II", "found": True}, {"skin_tone": "II", "found": False}]
    results = stratified_detection_rate(
        records, stratum_key="skin_tone", detected_key="found"
    )
    assert results["II"].detection_rate == pytest.approx(0.5)


def test_stratified_detection_rate_missing_key_raises() -> None:
    with pytest.raises(KeyError):
        stratified_detection_rate([{"stratum": "I"}])  # missing "detected"


# --------------------------------------------------------------------------- #
# summarise_disparity
# --------------------------------------------------------------------------- #


def test_summarise_disparity_reports_spread_and_min_n() -> None:
    records = [
        {"stratum": "I", "detected": True},
        {"stratum": "I", "detected": True},
        {"stratum": "VI", "detected": False},
    ]
    results = stratified_detection_rate(records)
    summary = summarise_disparity(results)
    assert summary["best_stratum"] == "I"
    assert summary["worst_stratum"] == "VI"
    assert summary["rate_spread"] == pytest.approx(1.0)  # 1.0 vs 0.0
    assert summary["min_stratum_n"] == 1  # "VI" has only 1 sample


def test_summarise_disparity_empty_results() -> None:
    summary = summarise_disparity({})
    assert summary == {
        "best_stratum": None,
        "worst_stratum": None,
        "rate_spread": None,
        "min_stratum_n": 0,
    }


def test_summarise_disparity_single_stratum_has_zero_spread() -> None:
    results = stratified_detection_rate([{"stratum": "III", "detected": True}])
    summary = summarise_disparity(results)
    assert summary["rate_spread"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# image_quality_proxies
# --------------------------------------------------------------------------- #


def test_image_quality_proxies_bright_vs_dark() -> None:
    bright = np.full((600, 800, 3), 220, dtype=np.uint8)
    dark = np.full((600, 800, 3), 20, dtype=np.uint8)
    assert (
        image_quality_proxies(bright)["mean_brightness"]
        > image_quality_proxies(dark)["mean_brightness"]
    )


def test_image_quality_proxies_shorter_side_is_min_dimension() -> None:
    frame = np.zeros((600, 1200, 3), dtype=np.uint8)  # h=600, w=1200
    assert image_quality_proxies(frame)["shorter_side"] == 600.0


@pytest.mark.parametrize(
    "shorter_side,expected_bucket",
    [
        (300, "<480"),
        (480, "[480,720)"),
        (700, "[480,720)"),
        (720, "[720,1080)"),
        (1080, ">=1080"),
        (2000, ">=1080"),
    ],
)
def test_image_quality_proxies_resolution_bucket_edges(
    shorter_side: int, expected_bucket: str
) -> None:
    frame = np.zeros((shorter_side, shorter_side + 200, 3), dtype=np.uint8)
    assert image_quality_proxies(frame)["resolution_bucket"] == expected_bucket


def test_image_quality_proxies_rejects_bad_input() -> None:
    with pytest.raises(TypeError):
        image_quality_proxies("not-a-frame")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        image_quality_proxies(np.zeros((0, 0, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        image_quality_proxies(np.zeros((10, 10), dtype=np.uint8))  # missing channel dim
