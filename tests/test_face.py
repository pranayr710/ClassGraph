"""Unit tests for :mod:`backend.face`.

The MediaPipe-backed tests require ``mediapipe`` (and, for the face-content
tests, a fixture image). When those are unavailable the affected tests skip
cleanly rather than fake a pass. The pure EAR/geometry math is tested with
synthetic data and runs everywhere.

Required coverage (Person B):
    1. Initialises without error.
    2. On a fixture with a face, returns a FaceResult with 468 landmarks.
    3. EAR is finite and within [0.05, 0.5] on a normal face.
    4. len(result) == len(input person_bboxes).

Per the task, ``person_bboxes`` (normally from detection.py) are mocked here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

# Importing backend.face does NOT import mediapipe (that happens lazily inside
# FaceAnalyzer), so the pure EAR-math tests below run without it.
from backend.config import CONFIG  # noqa: E402 - after importorskip
from backend.face import (  # noqa: E402 - after importorskip
    FaceAnalyzer,
    FaceResult,
    compute_ear,
)

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def _face_mesh_available() -> bool:
    """True if the MediaPipe Solutions Face Mesh API can be constructed."""
    try:
        import mediapipe as mp

        _ = mp.solutions.face_mesh  # noqa: B018 - attribute access is the probe
        return True
    except Exception:  # noqa: BLE001 - any failure => treat as unavailable
        return False


_HAS_FACE_MESH = _face_mesh_available()
requires_face_mesh = pytest.mark.skipif(
    not _HAS_FACE_MESH,
    reason="MediaPipe Solutions Face Mesh API unavailable in this build.",
)


@pytest.fixture(scope="module")
def analyzer() -> FaceAnalyzer:
    """A shared FaceAnalyzer for the module (graph creation is not free)."""
    if not _HAS_FACE_MESH:
        pytest.skip("MediaPipe Solutions Face Mesh API unavailable in this build.")
    fa = FaceAnalyzer()
    yield fa
    fa.close()


def _find_face_fixture() -> Path | None:
    """Return an image likely to contain a clear face, or ``None``.

    Order: any user-supplied image in ``tests/fixtures/``, then Ultralytics'
    bundled ``zidane.jpg`` (two large frontal faces) / ``bus.jpg``. The
    Ultralytics fallback lets these tests run on the target machine with no
    manual fixture, and only skip when nothing is available.
    """
    if _FIXTURE_DIR.is_dir():
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            for candidate in sorted(_FIXTURE_DIR.glob(pattern)):
                return candidate
    try:
        from ultralytics.utils import ASSETS
    except Exception:  # noqa: BLE001 - no fallback available
        return None
    for name in ("zidane.jpg", "bus.jpg"):
        candidate = Path(ASSETS) / name
        if candidate.is_file():
            return candidate
    return None


def _synthetic_eye_landmarks(vertical_gap: float) -> list[tuple[float, float]]:
    """Build 468 dummy landmarks with a controllable left-eye opening.

    The horizontal eye width is fixed at 10 px; ``vertical_gap`` sets the
    distance between upper and lower lids, so the expected EAR is
    ``vertical_gap / 10``.

    Args:
        vertical_gap: Vertical eyelid separation in pixels.

    Returns:
        A list of 468 ``(x, y)`` landmark points.
    """
    pts = [(0.0, 0.0)] * 468
    left = CONFIG.face.left_eye_ear_idx  # (P1, P2, P3, P4, P5, P6)
    top = 5.0 - vertical_gap / 2.0
    bot = 5.0 + vertical_gap / 2.0
    pts[left[0]] = (0.0, 5.0)  # P1 left corner
    pts[left[3]] = (10.0, 5.0)  # P4 right corner -> horizontal = 10
    pts[left[1]] = (3.0, top)  # P2 upper
    pts[left[5]] = (3.0, bot)  # P6 lower
    pts[left[2]] = (7.0, top)  # P3 upper
    pts[left[4]] = (7.0, bot)  # P5 lower
    return pts


@requires_face_mesh
def test_initializes_without_error() -> None:
    """FaceAnalyzer constructs and closes cleanly."""
    fa = FaceAnalyzer()
    try:
        assert fa.config.max_num_faces == CONFIG.face.max_num_faces
    finally:
        fa.close()
    fa.close()  # idempotent second close must not raise


def test_single_face_returns_468_landmarks(analyzer: FaceAnalyzer) -> None:
    """A fixture with a face yields one FaceResult carrying 468 landmarks."""
    fixture = _find_face_fixture()
    if fixture is None:
        pytest.skip(
            "No face fixture available (add an image to tests/fixtures/ or "
            "install ultralytics for its bundled zidane.jpg)."
        )
    frame = cv2.imread(str(fixture))
    assert frame is not None, f"Failed to read fixture: {fixture}"
    h, w = frame.shape[:2]

    # Mock a single person bbox covering the whole frame so a detected face is
    # fully contained and bound to it.
    results = analyzer.analyze(frame, [(0, 0, w, h)])

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, FaceResult)
    assert result.landmarks is not None, f"No face detected in {fixture.name}."
    assert len(result.landmarks) == CONFIG.face.num_landmarks == 468
    for x, y in result.landmarks:
        assert 0.0 <= x <= w
        assert 0.0 <= y <= h
    assert result.face_bbox is not None
    fx, fy, fw, fh = result.face_bbox
    assert fw > 0 and fh > 0


def test_ear_in_valid_range_on_face(analyzer: FaceAnalyzer) -> None:
    """EAR on a normal, open-eyed face is finite and within [0.05, 0.5]."""
    fixture = _find_face_fixture()
    if fixture is None:
        pytest.skip("No face fixture available (see other face tests).")
    frame = cv2.imread(str(fixture))
    assert frame is not None
    h, w = frame.shape[:2]

    results = analyzer.analyze(frame, [(0, 0, w, h)])
    ear = results[0].ear
    assert ear is not None, "Face detected but EAR was None."
    assert np.isfinite(ear)
    assert 0.05 <= ear <= 0.5, f"EAR {ear} outside plausible open-eye range."


def test_result_length_matches_input(analyzer: FaceAnalyzer) -> None:
    """Result list is aligned index-wise with the input person_bboxes."""
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    person_bboxes = [(10, 10, 50, 80), (200, 100, 60, 90), (400, 50, 40, 70)]
    results = analyzer.analyze(black, person_bboxes)

    assert len(results) == len(person_bboxes)
    # No faces in a black frame => every slot present but empty.
    for result in results:
        assert isinstance(result, FaceResult)
        assert result.face_bbox is None
        assert result.landmarks is None
        assert result.ear is None


def test_empty_bboxes_returns_empty_list(analyzer: FaceAnalyzer) -> None:
    """No person bboxes => empty result list (no wasted inference contract)."""
    black = np.zeros((120, 120, 3), dtype=np.uint8)
    assert analyzer.analyze(black, []) == []


def test_compute_ear_open_vs_closed() -> None:
    """EAR math: open eye scores higher than a closed eye, both as expected."""
    left = CONFIG.face.left_eye_ear_idx
    right = CONFIG.face.right_eye_ear_idx

    open_pts = _synthetic_eye_landmarks(vertical_gap=3.0)  # expect ~0.30
    closed_pts = _synthetic_eye_landmarks(vertical_gap=0.5)  # expect ~0.05

    # Only the left eye is populated; right-eye indices stay at origin, so
    # compute_ear falls back to the single valid eye.
    open_ear = compute_ear(open_pts, left, right)
    closed_ear = compute_ear(closed_pts, left, right)

    assert open_ear is not None and closed_ear is not None
    assert open_ear == pytest.approx(0.30, abs=1e-6)
    assert closed_ear == pytest.approx(0.05, abs=1e-6)
    assert open_ear > closed_ear


def test_compute_ear_degenerate_returns_none() -> None:
    """Zero horizontal eye width (all points coincident) yields None."""
    left = CONFIG.face.left_eye_ear_idx
    right = CONFIG.face.right_eye_ear_idx
    coincident = [(1.0, 1.0)] * 468
    assert compute_ear(coincident, left, right) is None


def test_analyze_rejects_bad_input(analyzer: FaceAnalyzer) -> None:
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
