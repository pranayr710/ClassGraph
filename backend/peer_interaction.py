"""Pairwise "peer-oriented" detection between students — exploratory Stage 3.

Operates on the Stage 1+2 JSONL output (``bbox``, ``posture``, ``track_id``)
after the fact, the same way :mod:`backend.attention` does. Nothing here is
wired into ``schema.json`` or the live capture loop.

Motivation: :mod:`backend.attention` deliberately treats ``gaze_label`` "left"/
"right" as its own ambiguous "oriented_away" bucket rather than counting it as
off-task, because a student turned toward a neighbour might be asking a
genuine question about the lesson, and vision alone cannot tell that apart
from idle chat -- the CSCL/learning-sciences literature's own answer, when it
needed that distinction, was to add a microphone, not a smarter camera
heuristic (Bassiou, Shriberg et al., Interspeech 2016). This module builds
the one piece of that finding vision *can* support: Kendon's F-formation
theory gives a real geometric definition of joint physical orientation
(people arrange their bodies so their individual activity-spaces overlap into
one shared space) that is detectable from position and body orientation
alone, no gaze or audio needed.

What this module DOES: flags a pair of tracked students as "oriented toward
each other" when they are at conversational distance and each one's shoulder
line is close to perpendicular to the line connecting them (see
``_deviation_from_perpendicular`` below for why that geometric test, not
gaze, is used), sustained above a majority fraction of a rolling window so a
single incidental glance does not count.

What this module does NOT claim:

* It does not detect whether an interaction is productive/academic or
  off-task social chat. That distinction lives in the content of what is
  said (Mercer's exploratory-talk research defines it that way explicitly),
  which is out of reach for a vision-only system. This module answers "are
  these two students jointly oriented toward each other," never "should a
  teacher be concerned about this."
* It does not use the SIGNED facing_direction from :mod:`backend.posture`.
  That field's sign is an unconfirmed guess (real-image validation was
  inconclusive -- see its docstring). Instead this module uses each
  shoulder line's UNDIRECTED orientation (the angle of the line, mod 180
  degrees), which has no front/back ambiguity to get wrong: two people
  facing each other, or facing away from each other, both have their
  shoulder lines perpendicular to the line connecting them -- exactly the
  ambiguity a signed vector cannot resolve without a validated sign, but an
  undirected orientation test does not need to resolve at all, because
  either arrangement is a real joint-orientation candidate (Kendon's
  F-formations include both vis-a-vis and side-by-side/L-shaped
  arrangements, which project to different signed directions but the same
  undirected one).
* It is UNVALIDATED against any labelled ground truth, and a real check
  already found a plausible false positive, not just a hypothetical one.
  Run on a real classroom photo (30-frame synthetic clip built by repeating
  it, so ByteTrack could assign ids), the two highest-scoring pairs included
  one (window fraction 1.00) that, rendered and inspected by eye, were two
  students at different, non-adjacent desks with no visible sign of
  interacting -- both independently bent over their own papers. Their
  shoulder lines happened to both be near-perpendicular to the line
  connecting them purely by coincidence of two people in similar
  writing postures being at that particular relative position, not because
  of any real joint orientation. This is exactly the kind of failure mode
  the "UNVALIDATED" warning above means concretely, not just formally: the
  geometric test as it stands can and does fire on unrelated people who
  merely happen to satisfy the angle condition. It also exposed a specific,
  measured calibration gap: the two students were in different, non-adjacent
  rows, yet still passed :func:`_within_conversational_distance` (measured
  gap ~103px against a threshold of ~155px at ``max_gap_to_width_ratio=1.5``)
  -- the default proximity threshold is looser than this classroom's actual
  desk spacing warrants. One example is not enough to responsibly pick a
  tighter number (that would just be a different unvalidated guess), but it
  is enough to flag the default as too generous, not merely "unvalidated in
  the abstract." Do not treat this module's
  output as reliable evidence of interaction without further validation
  against real labelled data -- there is none yet, and every threshold in
  :class:`~backend.config.PeerInteractionConfig` is a documented engineering
  default, not a measured constant.
* It never resolves into a productivity judgement, and never will.

Usage (library):
    from backend.peer_interaction import (
        RollingPeerInteractionTracker, iter_jsonl_pair_signals,
    )
    tracker = RollingPeerInteractionTracker()
    for pair, timestamp_ms, oriented in iter_jsonl_pair_signals("stage1.jsonl"):
        tracker.update(pair, timestamp_ms, oriented)
    print(tracker.active_pairs())

Usage (CLI):
    python -m backend.peer_interaction --jsonl outputs/stage1.jsonl
    python -m backend.peer_interaction --jsonl outputs/stage1.jsonl --pairs
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

from backend.config import CONFIG, PeerInteractionConfig

logger = logging.getLogger(__name__)

Bbox = tuple[float, float, float, float]
Point = tuple[float, float]
PairKey = tuple[int, int]  # always (min(id_a, id_b), max(id_a, id_b))


def _bbox_center(bbox: Bbox) -> Point:
    """Centre point of an ``(x, y, w, h)`` box."""
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def _within_conversational_distance(
    bbox_a: Bbox, bbox_b: Bbox, config: PeerInteractionConfig
) -> bool:
    """Whether two person boxes are close enough to plausibly be interacting.

    Approximates the gap between the two boxes as the distance between their
    centres minus their half-widths along that line -- simple and scale-
    relative (a fixed pixel threshold would not hold across near/far
    students in the same frame), not a precise polygon-edge distance.

    Args:
        bbox_a: First person's box.
        bbox_b: Second person's box.
        config: Peer-interaction settings.

    Returns:
        ``True`` if the approximate gap is within
        :data:`PeerInteractionConfig.max_gap_to_width_ratio` times the
        narrower of the two boxes' widths.
    """
    _, _, wa, _ = bbox_a
    _, _, wb, _ = bbox_b
    ca, cb = _bbox_center(bbox_a), _bbox_center(bbox_b)
    center_dist = math.hypot(ca[0] - cb[0], ca[1] - cb[1])
    gap = max(0.0, center_dist - (wa + wb) / 2.0)
    threshold = min(wa, wb) * config.max_gap_to_width_ratio
    return gap <= threshold


def _line_angle_degrees(p1: Point, p2: Point) -> float | None:
    """Undirected angle of the line through two points, in ``[0, 180)`` degrees.

    Args:
        p1: First point.
        p2: Second point.

    Returns:
        The angle, or ``None`` if the points coincide (no defined line).
    """
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    if dx == 0.0 and dy == 0.0:
        return None
    return math.degrees(math.atan2(dy, dx)) % 180.0


def _angular_distance_mod_180(a: float, b: float) -> float:
    """Smallest difference between two undirected (mod-180) angles, in ``[0, 90]``."""
    diff = abs(a - b) % 180.0
    return min(diff, 180.0 - diff)


def _deviation_from_perpendicular(shoulder_angle: float, bearing_angle: float) -> float:
    """How far a shoulder-line angle is from perpendicular to a bearing.

    Two people jointly oriented toward each other -- whether face-to-face or
    side-by-side (Kendon's vis-a-vis and L-shaped F-formations) -- both have
    their shoulder line running crosswise to the line connecting them, i.e.
    close to perpendicular. This is why the test is on perpendicularity, not
    on the shoulder line matching the bearing directly.

    Args:
        shoulder_angle: Undirected shoulder-line angle in degrees.
        bearing_angle: Undirected angle of the line connecting the two
            people, in degrees.

    Returns:
        ``0.0`` when exactly perpendicular, up to ``90.0`` when parallel.
    """
    return abs(90.0 - _angular_distance_mod_180(shoulder_angle, bearing_angle))


def classify_pair_frame(
    person_a: dict, person_b: dict, config: PeerInteractionConfig | None = None
) -> bool:
    """Whether two people appear jointly oriented toward each other this frame.

    Requires both people's shoulder keypoints (from ``posture``, see
    :mod:`backend.posture`) and conversational proximity. Reads only fields
    already in the schema -- nothing new is inferred beyond combining what
    Stage 1B already outputs.

    Args:
        person_a: One entry from a JSONL record's ``persons`` list.
        person_b: Another entry from the same record.
        config: Peer-interaction settings. Defaults to
            ``CONFIG.peer_interaction``.

    Returns:
        ``True`` if both people are within conversational distance and both
        shoulder lines are within
        :data:`PeerInteractionConfig.orientation_tolerance_degrees` of
        perpendicular to the line connecting them.
    """
    cfg = config if config is not None else CONFIG.peer_interaction

    posture_a, posture_b = person_a["posture"], person_b["posture"]
    if posture_a is None or posture_b is None:
        return False
    l_a, r_a = posture_a["left_shoulder"], posture_a["right_shoulder"]
    l_b, r_b = posture_b["left_shoulder"], posture_b["right_shoulder"]
    if None in (l_a, r_a, l_b, r_b):
        return False

    bbox_a: Bbox = tuple(person_a["bbox"])  # type: ignore[assignment]
    bbox_b: Bbox = tuple(person_b["bbox"])  # type: ignore[assignment]
    if not _within_conversational_distance(bbox_a, bbox_b, cfg):
        return False

    bearing = _line_angle_degrees(_bbox_center(bbox_a), _bbox_center(bbox_b))
    shoulder_a = _line_angle_degrees(tuple(l_a), tuple(r_a))
    shoulder_b = _line_angle_degrees(tuple(l_b), tuple(r_b))
    if bearing is None or shoulder_a is None or shoulder_b is None:
        return False

    dev_a = _deviation_from_perpendicular(shoulder_a, bearing)
    dev_b = _deviation_from_perpendicular(shoulder_b, bearing)
    return (
        dev_a <= cfg.orientation_tolerance_degrees
        and dev_b <= cfg.orientation_tolerance_degrees
    )


def _pair_key(track_id_a: int, track_id_b: int) -> PairKey:
    """Canonical, order-independent key for a pair of track ids."""
    return (
        (track_id_a, track_id_b)
        if track_id_a < track_id_b
        else (track_id_b, track_id_a)
    )


def iter_jsonl_pair_signals(
    path: str | Path, config: PeerInteractionConfig | None = None
) -> Iterator[tuple[PairKey, int, bool]]:
    """Read a Stage 1+2 JSONL file and yield ``(pair_key, timestamp_ms, oriented)``.

    Every unique pair of tracked persons in a frame is checked -- O(n^2) in
    the number of tracked persons per frame, which is negligible at
    classroom scale. Persons with ``track_id is None`` are skipped, same as
    :func:`backend.attention.iter_jsonl_signals`.

    Args:
        path: Path to a JSONL file matching ``schema.json``.
        config: Peer-interaction settings. Defaults to
            ``CONFIG.peer_interaction``.

    Yields:
        One tuple per candidate pair per frame, in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"JSONL file not found: {src}")

    with src.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            timestamp_ms = record["timestamp_ms"]
            tracked = [p for p in record["persons"] if p["track_id"] is not None]
            for person_a, person_b in combinations(tracked, 2):
                oriented = classify_pair_frame(person_a, person_b, config)
                key = _pair_key(person_a["track_id"], person_b["track_id"])
                yield key, timestamp_ms, oriented


@dataclass
class _PairState:
    """Internal per-pair rolling state (module-private)."""

    history: deque = field(default_factory=deque)
    streak_start_ms: int | None = None


class RollingPeerInteractionTracker:
    """Maintains a rolling per-pair window of orientation history.

    One instance covers one video/session. Feed it every ``(pair_key,
    timestamp_ms, oriented)`` in chronological order via :meth:`update`.

    Attributes:
        config: The :class:`PeerInteractionConfig` in effect.
    """

    def __init__(self, config: PeerInteractionConfig | None = None) -> None:
        """Create an empty tracker.

        Args:
            config: Peer-interaction settings. Defaults to
                ``CONFIG.peer_interaction``.
        """
        self.config: PeerInteractionConfig = (
            config if config is not None else CONFIG.peer_interaction
        )
        self._pairs: dict[PairKey, _PairState] = {}

    def update(self, pair: PairKey, timestamp_ms: int, oriented: bool) -> None:
        """Record one frame's orientation result for one pair.

        Args:
            pair: Canonical ``(min_id, max_id)`` pair key.
            timestamp_ms: Frame timestamp in milliseconds. Must be
                non-decreasing per pair (chronological input).
            oriented: This frame's :func:`classify_pair_frame` result.
        """
        state = self._pairs.setdefault(pair, _PairState())
        state.history.append((timestamp_ms, oriented))

        window_start = timestamp_ms - int(self.config.window_seconds * 1000)
        while state.history and state.history[0][0] < window_start:
            state.history.popleft()

        total = len(state.history)
        oriented_count = sum(1 for _, o in state.history if o)
        majority_now = (
            total > 0 and oriented_count / total >= self.config.majority_fraction
        )
        if majority_now:
            if state.streak_start_ms is None:
                state.streak_start_ms = timestamp_ms
        else:
            state.streak_start_ms = None

    def window_fraction(self, pair: PairKey) -> float:
        """Fraction of the current rolling window classified as oriented.

        Args:
            pair: A pair key previously passed to :meth:`update`.

        Returns:
            The fraction in ``[0, 1]``, or ``0.0`` for an unknown pair.
        """
        state = self._pairs.get(pair)
        if state is None or not state.history:
            return 0.0
        total = len(state.history)
        oriented_count = sum(1 for _, o in state.history if o)
        return oriented_count / total

    def is_sustained(self, pair: PairKey) -> bool:
        """Whether this pair has been majority-oriented for at least
        :data:`PeerInteractionConfig.sustained_seconds`, continuously.

        Args:
            pair: A pair key previously passed to :meth:`update`.

        Returns:
            ``True`` only once the current streak has lasted long enough.
        """
        state = self._pairs.get(pair)
        if state is None or state.streak_start_ms is None or not state.history:
            return False
        now_ms = state.history[-1][0]
        return (now_ms - state.streak_start_ms) >= self.config.sustained_seconds * 1000

    def known_pairs(self) -> list[PairKey]:
        """All pair keys seen so far, in first-seen order."""
        return list(self._pairs.keys())

    def active_pairs(self) -> list[PairKey]:
        """Pair keys currently flagged as sustained (see :meth:`is_sustained`)."""
        return [p for p in self.known_pairs() if self.is_sustained(p)]

    def summarise_classroom(self) -> dict[str, object]:
        """Class-level aggregate: how many pairs are currently interacting.

        Deliberately returns a count by default rather than naming pairs --
        mirrors :meth:`backend.attention.RollingAttentionTracker.summarise_classroom`'s
        "never a bare individual verdict, class-level trend by default"
        guardrail. Use :meth:`active_pairs` explicitly for the drill-down.

        Returns:
            A dict with ``pairs_considered`` and ``active_pair_count``.
        """
        pairs = self.known_pairs()
        return {
            "pairs_considered": len(pairs),
            "active_pair_count": len(self.active_pairs()),
        }


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="python -m backend.peer_interaction",
        description=(
            "Summarise a Stage 1+2 JSONL file into pairwise, class-level "
            "peer-orientation signal. Defaults to a count; pass --pairs to "
            "drill into which track_id pairs are currently flagged."
        ),
    )
    parser.add_argument(
        "--jsonl", required=True, type=str, help="Path to a stage1 JSONL file."
    )
    parser.add_argument(
        "--pairs",
        action="store_true",
        help="List the specific track_id pairs currently flagged, instead of just a count.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=CONFIG.log_level,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]`` when ``None``.

    Returns:
        Process exit code (``0`` on success, ``1`` on a handled failure).
    """
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    )

    try:
        tracker = RollingPeerInteractionTracker()
        seen_any = False
        for pair, timestamp_ms, oriented in iter_jsonl_pair_signals(args.jsonl):
            tracker.update(pair, timestamp_ms, oriented)
            seen_any = True
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
        logger.error("Failed to process %s: %s", args.jsonl, exc)
        return 1

    if not seen_any:
        print(
            "No candidate pairs found in this file (fewer than 2 tracked persons per frame?)."
        )
        return 0

    summary = tracker.summarise_classroom()
    print(
        f"Peer-orientation summary — {summary['pairs_considered']} pairs considered, "
        f"{summary['active_pair_count']} currently sustained"
    )
    if args.pairs:
        for pair in tracker.active_pairs():
            print(
                f"  track_id pair {pair}: window fraction {tracker.window_fraction(pair) * 100:.0f}%"
            )

    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    import sys

    sys.exit(main())
