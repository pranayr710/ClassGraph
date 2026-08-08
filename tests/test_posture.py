"""Unit tests for :mod:`backend.posture`.

Mirrors :mod:`tests.test_face`'s structure: MediaPipe Pose is wrapped directly
(no injection point), so tests skip cleanly when the Solutions Pose API is
unavailable rather than fake a pass. See ``backend/posture.py``'s module
docstring for why this module returns raw geometry, not a posture label — this
suite tests exactly that surface and nothing more.

Required coverage:
    1. Initialises without error.
    2. On a fixture with a visible body, returns detected keypoints.
    3. len(result) == len(input person_bboxes).
    4. Bad input raises explicit, typed errors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

# Importing backend.posture does NOT import mediapipe (lazy, inside
# PostureAnalyzer), so any pure-logic tests here would run without it.
from backend.config import CONFIG
from backend.posture import PostureAnalyzer, PostureResult

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def _pose_available() -> bool:
    """True if the MediaPipe Solutions Pose API can be constructed."""
    try:
        import mediapipe as mp

        _ = mp.solutions.pose
        return True
    except Exception:  # noqa: BLE001 - any failure => treat as unavailable
        return False


_HAS_POSE = _pose_available()
requires_pose = pytest.mark.skipif(
    not _HAS_POSE, reason="MediaPipe Solutions Pose API unavailable in this build."
)


@pytest.fixture(scope="module")
def analyzer() -> PostureAnalyzer:
    """A shared PostureAnalyzer for the module (graph creation is not free)."""
    if not _HAS_POSE:
        pytest.skip("MediaPipe Solutions Pose API unavailable in this build.")
    pa = PostureAnalyzer()
    yield pa
    pa.close()


def _find_body_fixture() -> Path | None:
    """Return an image likely to show a visible torso, or ``None``.

    Reuses whatever face-detection tests use — ``frontal_face.jpg`` is a
    head-and-shoulders portrait (see its generation notes), which is enough
    for Pose to find a nose and both shoulders.
    """
    if _FIXTURE_DIR.is_dir():
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            for candidate in sorted(_FIXTURE_DIR.glob(pattern)):
                return candidate
    return None


@requires_pose
def test_initializes_without_error() -> None:
    """PostureAnalyzer constructs and closes cleanly."""
    pa = PostureAnalyzer()
    try:
        assert pa.config.model_complexity == CONFIG.posture.model_complexity
    finally:
        pa.close()
    pa.close()  # idempotent second close must not raise


def test_keypoints_detected_on_body_fixture(analyzer: PostureAnalyzer) -> None:
    """A fixture with a visible torso yields detected nose + shoulder points."""
    fixture = _find_body_fixture()
    if fixture is None:
        pytest.skip(
            "No body fixture available (add an image with a visible torso to "
            "tests/fixtures/)."
        )
    frame = cv2.imread(str(fixture))
    assert frame is not None, f"Failed to read fixture: {fixture}"
    h, w = frame.shape[:2]

    results = analyzer.analyze(frame, [(0, 0, w, h)])
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, PostureResult)
    assert result.keypoints_detected, f"No pose detected in {fixture.name}."
    assert result.nose is not None
    nx, ny = result.nose
    assert 0.0 <= nx <= w
    assert 0.0 <= ny <= h
    assert result.shoulder_mid is not None


def test_result_length_matches_input(analyzer: PostureAnalyzer) -> None:
    """Result list is aligned index-wise with the input person_bboxes."""
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    person_bboxes = [(10, 10, 50, 80), (200, 100, 60, 90), (400, 50, 40, 70)]
    results = analyzer.analyze(black, person_bboxes)

    assert len(results) == len(person_bboxes)
    # No body in a black frame => every slot present but empty.
    for result in results:
        assert isinstance(result, PostureResult)
        assert result.keypoints_detected is False
        assert result.nose is None
        assert result.shoulder_mid is None
        assert result.hip_mid is None
        assert result.vertical_lean is None


def test_empty_bboxes_returns_empty_list(analyzer: PostureAnalyzer) -> None:
    """No person bboxes => empty result list (no wasted inference)."""
    black = np.zeros((120, 120, 3), dtype=np.uint8)
    assert analyzer.analyze(black, []) == []


def test_analyze_rejects_bad_input(analyzer: PostureAnalyzer) -> None:
    """Malformed frames and bboxes raise explicit, typed errors."""
    black = np.zeros((60, 60, 3), dtype=np.uint8)
    with pytest.raises(TypeError):
        analyzer.analyze("not-a-frame", [(0, 0, 10, 10)])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        analyzer.analyze(np.zeros((0, 0, 3), dtype=np.uint8), [(0, 0, 10, 10)])
    with pytest.raises(ValueError):
        analyzer.analyze(black, [(0, 0, 10)])  # bbox too short
    with pytest.raises(ValueError):
        analyzer.analyze(black, [(0, 0, 0, 10)])  # non-positive width


def test_degenerate_region_returns_empty(analyzer: PostureAnalyzer) -> None:
    """A person bbox entirely outside the frame yields an empty PostureResult."""
    black = np.zeros((100, 100, 3), dtype=np.uint8)
    # Bbox is fully out of bounds; analyzer must not crash on the empty crop.
    results = analyzer.analyze(black, [(500, 500, 20, 20)])
    assert len(results) == 1
    assert results[0].keypoints_detected is False
