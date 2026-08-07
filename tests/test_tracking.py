"""Unit tests for :mod:`backend.tracking`.

Runs the real ``ultralytics.trackers.BYTETracker`` against synthetic ``Person``
sequences — no weights or GPU are needed for tracking, so this is a single
logic tier, unlike detection/face/headpose which split real-model tests from
fake-model ones.

Every assertion here was confirmed by first running the real tracker and
observing its actual output, not assumed from documentation — ByteTrack's
two-stage confirmation and buffer-expiry behaviour are easy to get wrong by
inspection alone (see backend/tracking.py's module docstring for the
citations this was checked against).

Required coverage:
    1. A continuously-visible person keeps one stable track_id.
    2. track_id follows the physical box, not list position, across frames.
    3. A brand-new person (not on the sequence's frame 1) is unconfirmed on
       their first sighting (None), then gets a real id from the second
       consecutive match onward.
    4. A track survives a gap shorter than track_buffer with the same id, and
       gets a brand-new id (never the old one) once the gap exceeds it.
    5. Empty-frame input is handled without error.
    6. reset() restarts id numbering.
"""

from __future__ import annotations

import pytest

pytest.importorskip("ultralytics")

from backend.config import CONFIG, TrackingConfig  # noqa: E402
from backend.detection import Person  # noqa: E402
from backend.tracking import PersonTracker  # noqa: E402

_BOX = (100, 100, 50, 100)  # (x, y, w, h)


def _moved(
    box: tuple[int, int, int, int], dx: int, dy: int
) -> tuple[int, int, int, int]:
    """Shift a bbox by (dx, dy), keeping its size."""
    x, y, w, h = box
    return (x + dx, y + dy, w, h)


@pytest.fixture
def tracker() -> PersonTracker:
    """A fresh PersonTracker with the project's default tracking config."""
    return PersonTracker(CONFIG.tracking)


def test_continuous_person_keeps_stable_id(tracker: PersonTracker) -> None:
    """One person, present every frame, keeps the same track_id throughout."""
    ids = []
    for i in range(10):
        result = tracker.update([Person(bbox=_moved(_BOX, i, i), confidence=0.9)])
        assert len(result) == 1
        ids.append(result[0])

    assert ids[0] == 1, "a track starting on frame 1 activates immediately"
    assert all(i == 1 for i in ids), f"expected a stable id, got {ids}"


def test_id_follows_the_box_not_list_position(tracker: PersonTracker) -> None:
    """Swapping two persons' order in the input list must not swap their ids."""
    tracker.update([Person(bbox=_BOX, confidence=0.9)])
    far_box = (500, 300, 50, 100)
    tracker.update(
        [Person(bbox=_BOX, confidence=0.9), Person(bbox=far_box, confidence=0.85)]
    )

    # The far box is brand-new this frame (unconfirmed) -> gets a real id only
    # from its second consecutive match. Confirm it, then swap list order.
    result = tracker.update(
        [Person(bbox=_BOX, confidence=0.9), Person(bbox=far_box, confidence=0.85)]
    )
    assert result == [1, 2]

    swapped = tracker.update(
        [Person(bbox=far_box, confidence=0.85), Person(bbox=_BOX, confidence=0.9)]
    )
    assert swapped == [2, 1], f"ids must track physical boxes, got {swapped}"


def test_new_person_unconfirmed_until_second_match(tracker: PersonTracker) -> None:
    """A person appearing after frame 1 is None on first sighting, then gets an id."""
    tracker.update([Person(bbox=_BOX, confidence=0.9)])  # frame 1: id 1

    new_box = (500, 300, 50, 100)
    first_sighting = tracker.update(
        [Person(bbox=_BOX, confidence=0.9), Person(bbox=new_box, confidence=0.85)]
    )
    assert first_sighting[0] == 1
    assert (
        first_sighting[1] is None
    ), "brand-new track must be unconfirmed on frame 1 of its life"

    second_sighting = tracker.update(
        [Person(bbox=_BOX, confidence=0.9), Person(bbox=new_box, confidence=0.85)]
    )
    assert second_sighting == [1, 2], "id should appear on the second consecutive match"


def test_track_survives_gap_within_buffer(tracker: PersonTracker) -> None:
    """A track missing for fewer than track_buffer frames keeps its id."""
    tracker.update([Person(bbox=_BOX, confidence=0.9)])  # id 1, activated
    for _ in range(4):
        tracker.update([Person(bbox=_BOX, confidence=0.9)])

    gap = tracker.config.track_buffer - 5
    for _ in range(gap):
        tracker.update([])  # person out of frame briefly

    result = tracker.update([Person(bbox=_BOX, confidence=0.9)])
    assert result == [
        1
    ], f"expected the original id to survive a short gap, got {result}"


def test_new_id_after_buffer_expires(tracker: PersonTracker) -> None:
    """A gap longer than track_buffer drops the old track; the id is never reused."""
    tracker.update([Person(bbox=_BOX, confidence=0.9)])  # id 1

    for _ in range(tracker.config.track_buffer + 5):
        tracker.update([])

    reappeared = tracker.update([Person(bbox=_BOX, confidence=0.9)])
    assert reappeared == [
        None
    ], "reappearance after buffer expiry restarts confirmation"

    confirmed = tracker.update([Person(bbox=_BOX, confidence=0.9)])
    assert confirmed == [
        2
    ], f"expected a fresh id, never the expired one back, got {confirmed}"


def test_empty_frame_returns_empty_list(tracker: PersonTracker) -> None:
    """No persons in, no track_ids out, and the tracker does not raise.

    Note ByteTrack's internal frame counter advances even on an empty update,
    so a real detection immediately after one is no longer "frame 1 of the
    sequence" and goes through the normal unconfirmed-then-assigned sequence
    rather than activating instantly.
    """
    assert tracker.update([]) == []

    # A subsequent real detection still works: unconfirmed on its first
    # sighting, assigned a real id from its second consecutive match.
    first = tracker.update([Person(bbox=_BOX, confidence=0.9)])
    assert first == [None]
    second = tracker.update([Person(bbox=_BOX, confidence=0.9)])
    assert second == [1]


def test_reset_restarts_id_numbering(tracker: PersonTracker) -> None:
    """reset() clears state so id numbering starts over, as for a new video."""
    tracker.update([Person(bbox=_BOX, confidence=0.9)])
    tracker.update([Person(bbox=_BOX, confidence=0.9)])

    tracker.reset()
    result = tracker.update([Person(bbox=_BOX, confidence=0.9)])
    assert result == [1], "reset() should restart id numbering from 1"


def test_config_knobs_are_applied() -> None:
    """A custom TrackingConfig actually changes tracker behaviour.

    Uses a much shorter track_buffer than the default (5 vs. 30) and confirms
    a gap that would easily survive the default already expires it — proof
    the config value is honoured, not silently ignored in favour of defaults.
    A generous margin (well beyond track_buffer) is used deliberately rather
    than the exact expiry boundary, which is an internal detail of the
    vendored tracker this test should not be coupled to.
    """
    custom = TrackingConfig(track_buffer=5)
    custom_tracker = PersonTracker(custom)
    assert custom_tracker.config.track_buffer == 5

    custom_tracker.update([Person(bbox=_BOX, confidence=0.9)])
    for _ in range(15):  # comfortably past the custom buffer of 5
        custom_tracker.update([])
    reappeared = custom_tracker.update([Person(bbox=_BOX, confidence=0.9)])
    assert reappeared == [
        None
    ], "the custom (shorter) buffer should already have expired"
