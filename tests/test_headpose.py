"""Unit tests for :mod:`backend.headpose`.

Two tiers, both honest:

* **Logic tier** (runs everywhere): a fake pose model is injected into
  ``HeadPoseEstimator`` so the estimate loop, cropping, index-alignment and
  gaze-label mapping are verified without ``sixdrepnet`` or any weight download.
* **Model tier** (needs ``sixdrepnet`` + weights, and a face fixture): loads the
  real pretrained SixDRepNet and runs it on a frontal face. These skip cleanly
  when the package/fixture is unavailable and run on the target GPU machine.

Required coverage (Person C):
    1. Loads pretrained weights.
    2. On a frontal-face fixture: |yaw| < 15 and gaze_label == "teacher".
    3. Respects input alignment: None in -> None out.
    4. gaze_label is one of the five allowed strings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from backend.config import CONFIG
from backend.headpose import (
    ALLOWED_GAZE_LABELS,
    HeadPoseEstimator,
    HeadPoseResult,
    classify_gaze,
)

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
_HP = CONFIG.headpose


class _FakeSixDRepNet:
    """Stand-in for SixDRepNet returning a fixed ``(pitch, yaw, roll)``.

    Mirrors the package contract: ``predict(bgr_crop) -> (pitch, yaw, roll)``.
    """

    def __init__(self, pitch: float, yaw: float, roll: float) -> None:
        self._pose = (pitch, yaw, roll)
        self.calls = 0

    def predict(self, crop: np.ndarray) -> tuple[float, float, float]:
        """Return the canned pose, ignoring the crop content."""
        assert isinstance(crop, np.ndarray) and crop.size > 0
        self.calls += 1
        return self._pose


def _find_frontal_fixture() -> Path | None:
    """Return a frontal-face fixture image from ``tests/fixtures/``, or None.

    Unlike the face-detection tests, this requires a *frontal* face for the
    strict ``|yaw| < 15`` assertion, so it does NOT fall back to Ultralytics'
    sample images (which are not reliably frontal). Prefers a file named
    ``frontal*`` if present, else any image in the directory.
    """
    if not _FIXTURE_DIR.is_dir():
        return None
    for pattern in ("frontal*.jpg", "frontal*.jpeg", "frontal*.png"):
        for candidate in sorted(_FIXTURE_DIR.glob(pattern)):
            return candidate
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        for candidate in sorted(_FIXTURE_DIR.glob(pattern)):
            return candidate
    return None


# --------------------------------------------------------------------------- #
# Model tier (real SixDRepNet) — skips without the package / a fixture.
# --------------------------------------------------------------------------- #


def test_loads_pretrained_weights() -> None:
    """The real SixDRepNet loads and yields a usable estimator."""
    pytest.importorskip("sixdrepnet")
    estimator = HeadPoseEstimator()
    assert estimator.device in ("cuda", "cpu")
    assert estimator._model is not None


def test_frontal_face_returns_teacher() -> None:
    """A frontal face gives |yaw| < 15 and gaze_label == 'teacher'.

    SixDRepNet expects a face crop, not a whole scene — on a wide image it
    returns garbage (measured: the source webcam frame this fixture was cut
    from gives yaw=19.2, "down" fed whole, vs. yaw=-2.9, "teacher" through the
    real pipeline). This test used to hand the entire fixture image to
    ``estimate()`` on the assumption that "SixDRepNet finds the dominant
    face", which is false and made the test fragile to exactly how tightly
    the fixture happens to be cropped (see the fixture's own commit history:
    only one of several plausible crop margins passed). Getting a real face
    box from FaceAnalyzer first removes that fragility; falling back to the
    whole image keeps this test running on a machine that has sixdrepnet but
    not mediapipe.
    """
    pytest.importorskip("sixdrepnet")
    fixture = _find_frontal_fixture()
    if fixture is None:
        pytest.skip(
            "No frontal-face fixture (drop a frontal face image, ideally named "
            "frontal*.jpg, into tests/fixtures/ to enable this test)."
        )
    frame = cv2.imread(str(fixture))
    assert frame is not None
    h, w = frame.shape[:2]

    face_box: tuple[int, int, int, int] = (0, 0, w, h)
    try:
        from backend.face import FaceAnalyzer

        with FaceAnalyzer() as fa:
            detected = fa.analyze(frame, [(0, 0, w, h)])[0]
        if detected.face_bbox is not None:
            face_box = detected.face_bbox
    except ImportError:
        pass  # mediapipe unavailable here; fall back to the whole image.

    estimator = HeadPoseEstimator()
    results = estimator.estimate(frame, [face_box])
    assert len(results) == 1
    result = results[0]
    assert result is not None, "Expected a pose for the fixture face."
    assert abs(result.yaw) < 15, f"yaw={result.yaw} not frontal (<15)."
    assert result.gaze_label == "teacher"


# --------------------------------------------------------------------------- #
# Logic tier (fake model) — runs everywhere.
# --------------------------------------------------------------------------- #


def test_alignment_none_in_none_out() -> None:
    """Result list is aligned index-wise; None inputs map to None outputs."""
    fake = _FakeSixDRepNet(pitch=0.0, yaw=0.0, roll=0.0)
    estimator = HeadPoseEstimator(model=fake)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    face_bboxes = [None, (10, 10, 100, 100), None, (300, 200, 80, 90)]
    results = estimator.estimate(frame, face_bboxes)

    assert len(results) == len(face_bboxes)
    assert results[0] is None
    assert results[2] is None
    assert isinstance(results[1], HeadPoseResult)
    assert isinstance(results[3], HeadPoseResult)
    # Only the two non-None boxes triggered inference.
    assert fake.calls == 2


def test_pitch_sign_is_flipped_to_down_positive() -> None:
    """estimate() negates SixDRepNet's up-positive pitch to down-positive.

    SixDRepNet is up-positive: its own ``draw_axis`` places the face-direction
    axis at ``y = -cos(yaw) * sin(pitch)``, and image ``y`` grows downward, so a
    positive pitch points the nose up. :class:`HeadPoseResult` documents
    down-positive pitch instead, so the sign must be flipped on the way out.

    Without this, a student bowed over a desk (model pitch about -26) was
    labelled ``"back"`` and ``"down"`` was unreachable in practice.
    """
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    box = (50, 50, 120, 120)

    # Model says "looking down" (negative in its convention).
    looking_down = HeadPoseEstimator(
        model=_FakeSixDRepNet(pitch=-30.0, yaw=0.0, roll=0.0)
    )
    result = looking_down.estimate(frame, [box])[0]
    assert result is not None
    assert result.pitch == 30.0, "down should be positive on the way out"
    assert result.gaze_label == "down"

    # Model says "looking up" (positive in its convention).
    looking_up = HeadPoseEstimator(model=_FakeSixDRepNet(pitch=30.0, yaw=0.0, roll=0.0))
    result = looking_up.estimate(frame, [box])[0]
    assert result is not None
    assert result.pitch == -30.0
    assert result.gaze_label == "back"

    # yaw and roll must be passed through untouched.
    passthrough = HeadPoseEstimator(
        model=_FakeSixDRepNet(pitch=0.0, yaw=12.5, roll=-7.5)
    )
    result = passthrough.estimate(frame, [box])[0]
    assert result is not None
    assert result.yaw == 12.5
    assert result.roll == -7.5


def test_gaze_label_is_allowed() -> None:
    """Every produced gaze_label is one of the five allowed strings."""
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    box = (50, 50, 120, 120)
    # A spread of orientations covering each labelled region. These are raw
    # SixDRepNet values (up-positive pitch), which estimate() negates, so the
    # expected label follows the *negated* pitch.
    poses = [
        (0.0, 0.0),  # teacher
        (0.0, 40.0),  # right
        (0.0, -40.0),  # left
        (-40.0, 0.0),  # -> pitch +40 -> down
        (40.0, 0.0),  # -> pitch -40 -> back
        (30.0, 30.0),  # mixed
        (-22.0, 10.0),  # dead-zone-ish
    ]
    for pitch, yaw in poses:
        fake = _FakeSixDRepNet(pitch=pitch, yaw=yaw, roll=0.0)
        estimator = HeadPoseEstimator(model=fake)
        result = estimator.estimate(frame, [box])[0]
        assert result is not None
        assert result.gaze_label in ALLOWED_GAZE_LABELS


def test_classify_gaze_thresholds() -> None:
    """classify_gaze maps representative angles per the spec precedence."""
    assert classify_gaze(0.0, 0.0, _HP) == "teacher"
    assert classify_gaze(_HP.yaw_side_threshold, 0.0, _HP) == "right"
    assert classify_gaze(-_HP.yaw_side_threshold, 0.0, _HP) == "left"
    assert classify_gaze(0.0, _HP.pitch_down_threshold, _HP) == "down"
    assert classify_gaze(0.0, _HP.pitch_back_threshold, _HP) == "back"
    # yaw takes precedence over pitch when both are past threshold.
    assert classify_gaze(40.0, 40.0, _HP) == "right"
    assert classify_gaze(-40.0, 40.0, _HP) == "left"
    # Every result is a valid label.
    for yaw in (-90, -30, -10, 0, 10, 30, 90):
        for pitch in (-90, -30, -22, 0, 22, 30, 90):
            assert classify_gaze(float(yaw), float(pitch), _HP) in ALLOWED_GAZE_LABELS


def test_classify_gaze_rejects_non_finite() -> None:
    """Non-finite angles raise ValueError rather than mislabelling."""
    with pytest.raises(ValueError):
        classify_gaze(float("nan"), 0.0, _HP)
    with pytest.raises(ValueError):
        classify_gaze(0.0, float("inf"), _HP)


def test_estimate_rejects_bad_input() -> None:
    """Malformed frames and bboxes raise explicit, typed errors."""
    fake = _FakeSixDRepNet(pitch=0.0, yaw=0.0, roll=0.0)
    estimator = HeadPoseEstimator(model=fake)
    frame = np.zeros((60, 60, 3), dtype=np.uint8)

    with pytest.raises(TypeError):
        estimator.estimate("not-a-frame", [None])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        estimator.estimate(np.zeros((0, 0, 3), dtype=np.uint8), [None])
    with pytest.raises(TypeError):
        estimator.estimate(frame, 123)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        estimator.estimate(frame, [(0, 0, 10)])  # bbox too short
    with pytest.raises(ValueError):
        estimator.estimate(frame, [(0, 0, 0, 10)])  # non-positive width


def test_degenerate_crop_returns_none() -> None:
    """A face box fully outside the frame yields None (no crash)."""
    fake = _FakeSixDRepNet(pitch=0.0, yaw=0.0, roll=0.0)
    estimator = HeadPoseEstimator(model=fake)
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    # Box starts well beyond the image bounds -> empty crop -> None.
    results = estimator.estimate(frame, [(1000, 1000, 20, 20)])
    assert results == [None]
    assert fake.calls == 0
