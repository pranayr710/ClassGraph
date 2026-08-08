"""Integration tests for :mod:`backend.integrate`.

The full pipeline is driven with lightweight fakes injected into
``process_video`` for the four heavy modules (the real
Detector/FaceAnalyzer/HeadPoseEstimator/PostureAnalyzer need ML packages +
weights). The fakes return the *real* result dataclasses
(``Person``/``Obj``/``FaceResult``/``HeadPoseResult``/``PostureResult``), so
the wiring, index-alignment, contract assembly, JSONL writing and sample-rate
behaviour are all exercised for real and validated against ``schema.json``.

Person tracking (Stage 2) is deliberately NOT faked here: ``PersonTracker``
needs only ``ultralytics`` + ``lap`` (no weights, no GPU), the same real
dependency ``test_detection.py`` already requires, so ``process_video`` builds
a genuine ``PersonTracker`` and these tests exercise real ByteTrack end to end.

Required coverage:
    1. End-to-end on a 5-second fixture video -> valid JSONL matching schema.
    2. --sample-rate 5 produces ~1/5 the lines of --sample-rate 1.
    3. Stage 2: a continuously-visible person keeps one stable track_id.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
jsonschema = pytest.importorskip("jsonschema")
pytest.importorskip("tqdm")
# PersonTracker (Stage 2) is not faked below, so its real dependency is needed.
pytest.importorskip("ultralytics")

from backend.config import CONFIG  # noqa: E402 - after importorskip
from backend.detection import Obj, Person  # noqa: E402
from backend.face import FaceResult  # noqa: E402
from backend.headpose import HeadPoseResult  # noqa: E402
from backend.integrate import process_video  # noqa: E402
from backend.posture import PostureResult  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCHEMA_PATH = _REPO_ROOT / "schema.json"

_FIXTURE_FPS = 6  # keep the fixture small: 6 fps * 5 s = 30 frames
_FIXTURE_SECONDS = 5
_FIXTURE_W, _FIXTURE_H = 160, 120


class _FakeDetector:
    """Returns one person and one object on every frame."""

    def detect(self, frame: np.ndarray) -> tuple[list[Person], list[Obj]]:
        persons = [Person(bbox=(20, 15, 60, 80), confidence=0.92)]
        objects = [Obj(cls="laptop", bbox=(5, 5, 30, 20), confidence=0.77)]
        return persons, objects


class _FakeFaceAnalyzer:
    """Returns a full 468-landmark FaceResult for each person bbox."""

    def analyze(self, frame: np.ndarray, person_bboxes) -> list[FaceResult]:
        results: list[FaceResult] = []
        for x, y, w, h in person_bboxes:
            landmarks = [(float(x + i % w), float(y + i % h)) for i in range(468)]
            results.append(
                FaceResult(
                    face_bbox=(x + 5, y + 5, max(1, w - 10), max(1, h - 10)),
                    landmarks=landmarks,
                    ear=0.31,
                )
            )
        return results


class _FakeHeadPose:
    """Returns a frontal 'teacher' pose for each non-None face bbox."""

    def estimate(self, frame: np.ndarray, face_bboxes) -> list[HeadPoseResult | None]:
        out: list[HeadPoseResult | None] = []
        for bbox in face_bboxes:
            if bbox is None:
                out.append(None)
            else:
                out.append(
                    HeadPoseResult(yaw=2.0, pitch=-3.0, roll=1.0, gaze_label="teacher")
                )
        return out


class _FakePostureAnalyzer:
    """Returns a fixed PostureResult (keypoints found) for each person bbox."""

    def analyze(self, frame: np.ndarray, person_bboxes) -> list[PostureResult]:
        results: list[PostureResult] = []
        for x, y, w, h in person_bboxes:
            results.append(
                PostureResult(
                    keypoints_detected=True,
                    nose=(float(x + w / 2), float(y + h * 0.1)),
                    shoulder_mid=(float(x + w / 2), float(y + h * 0.3)),
                    hip_mid=(float(x + w / 2), float(y + h * 0.7)),
                    vertical_lean=-0.2,
                )
            )
        return results


def _make_fixture_video(path: Path, n_frames: int) -> int:
    """Write a small synthetic video and return the number of frames written."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(path), fourcc, float(_FIXTURE_FPS), (_FIXTURE_W, _FIXTURE_H)
    )
    assert writer.isOpened(), "VideoWriter failed to open (codec unavailable?)."
    try:
        for i in range(n_frames):
            # Vary pixels a little per frame so it's not a degenerate stream.
            frame = np.full((_FIXTURE_H, _FIXTURE_W, 3), i % 256, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return n_frames


def _fakes() -> dict:
    """Injected-estimator kwargs for process_video.

    ``person_tracker`` is deliberately absent: process_video builds a real
    ``PersonTracker`` (see the module docstring for why that's safe here).
    """
    return {
        "detector": _FakeDetector(),
        "face_analyzer": _FakeFaceAnalyzer(),
        "headpose_estimator": _FakeHeadPose(),
        "posture_analyzer": _FakePostureAnalyzer(),
    }


@pytest.fixture(scope="module")
def schema() -> dict:
    """The frozen Stage 1 JSON schema, loaded once."""
    with _SCHEMA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def test_end_to_end_valid_jsonl(schema: dict, tmp_path: Path) -> None:
    """A 5-second fixture video yields schema-valid JSONL, one line per frame."""
    n_frames = _FIXTURE_FPS * _FIXTURE_SECONDS  # 30
    video = tmp_path / "clip.mp4"
    _make_fixture_video(video, n_frames)

    out = tmp_path / "stage1.jsonl"
    written = process_video(video, out, CONFIG, **_fakes())

    assert written == n_frames
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == written

    validator = jsonschema.Draft202012Validator(schema)
    prev_frame_id = -1
    for line in lines:
        record = json.loads(line)
        validator.validate(record)  # raises on any contract violation

        assert record["frame_id"] > prev_frame_id  # strictly increasing
        prev_frame_id = record["frame_id"]

        assert len(record["persons"]) == 1
        person = record["persons"][0]
        # Stage 2: the fake detector reports the same bbox every frame, so
        # ByteTrack should hold one stable id across the whole clip. Frame 0
        # is ByteTrack's own frame_id 1, which activates a new track
        # immediately (see backend.tracking's module docstring), so the id is
        # present from the very first record here, not just from frame 1 on.
        assert person["track_id"] == 1
        assert len(person["face"]["landmarks"]) == 468
        assert person["head_pose"]["gaze_label"] == "teacher"
        assert person["posture"]["nose"] is not None
        assert person["posture"]["vertical_lean"] == pytest.approx(-0.2)
        assert record["objects"][0]["cls"] == "laptop"


def test_sample_rate_reduces_line_count(tmp_path: Path) -> None:
    """--sample-rate 5 produces about one fifth of the full-rate lines."""
    n_frames = _FIXTURE_FPS * _FIXTURE_SECONDS  # 30
    video = tmp_path / "clip.mp4"
    _make_fixture_video(video, n_frames)

    cfg_full = replace(CONFIG, pipeline=replace(CONFIG.pipeline, sample_rate=1))
    cfg_sampled = replace(CONFIG, pipeline=replace(CONFIG.pipeline, sample_rate=5))

    out_full = tmp_path / "full.jsonl"
    out_sampled = tmp_path / "sampled.jsonl"
    n_full = process_video(video, out_full, cfg_full, **_fakes())
    n_sampled = process_video(video, out_sampled, cfg_sampled, **_fakes())

    assert n_full == n_frames
    # Every 5th frame of N frames -> ceil(N / 5).
    expected = -(-n_frames // 5)
    assert n_sampled == expected
    # "~1/5" sanity band around the exact ceil value.
    assert abs(n_sampled - n_full / 5) <= 1

    # Sampled frame_ids are the multiples of 5 actually present in the source.
    sampled_ids = [
        json.loads(line)["frame_id"]
        for line in out_sampled.read_text(encoding="utf-8").strip().splitlines()
    ]
    assert sampled_ids == list(range(0, n_frames, 5))
