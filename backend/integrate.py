"""Stage 1 integration — wire detection + face + head-pose into JSONL output.

Runs the three perception modules over a video and emits the frozen Stage 1
contract, one JSON object per processed frame:

    Detector       -> persons (bbox, confidence) + objects
    FaceAnalyzer   -> per-person face landmarks + EAR (index-aligned)
    HeadPoseEstimator -> per-face yaw/pitch/roll + gaze label (index-aligned)

The per-person ``track_id`` is always ``null`` in Stage 1 (ByteTrack fills it
in Stage 2). Output validates against ``schema.json``.

Usage (CLI):
    python -m backend.integrate --video in.mp4 --out out.jsonl --sample-rate 5

Usage (API):
    from backend.config import CONFIG
    from backend.integrate import process_video
    n = process_video("in.mp4", "out.jsonl", CONFIG)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Sequence

import numpy as np

from backend.config import CONFIG, Config

if TYPE_CHECKING:  # pragma: no cover - typing only
    from backend.detection import Detector, Obj, Person
    from backend.face import FaceAnalyzer, FaceResult
    from backend.headpose import HeadPoseEstimator, HeadPoseResult

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Structural interfaces (duck-typed) so real modules and test fakes both fit.
# --------------------------------------------------------------------------- #


class DetectorLike(Protocol):
    """Anything exposing ``detect(frame) -> (persons, objects)``."""

    def detect(self, frame: np.ndarray) -> tuple[list["Person"], list["Obj"]]:
        ...


class FaceAnalyzerLike(Protocol):
    """Anything exposing ``analyze(frame, person_bboxes) -> list[FaceResult]``."""

    def analyze(
        self, frame: np.ndarray, person_bboxes: Sequence[Sequence[float]]
    ) -> list["FaceResult"]:
        ...


class HeadPoseLike(Protocol):
    """Anything exposing ``estimate(frame, face_bboxes) -> list[result|None]``."""

    def estimate(
        self, frame: np.ndarray, face_bboxes: Sequence[Sequence[float] | None]
    ) -> list["HeadPoseResult | None"]:
        ...


def _face_to_json(face: "FaceResult | None") -> dict | None:
    """Serialise a FaceResult into the frozen ``face`` object, or ``None``.

    A face is considered present only when it has a bounding box. Landmarks and
    EAR may still be ``None`` within a present face (e.g. degenerate eyes).

    Args:
        face: The per-person face result, or ``None``.

    Returns:
        A dict matching the schema's ``face`` object, or ``None`` when no face
        was matched to this person.
    """
    if face is None or face.face_bbox is None:
        return None
    landmarks = (
        [[float(x), float(y)] for x, y in face.landmarks]
        if face.landmarks is not None
        else None
    )
    return {
        "bbox": [int(v) for v in face.face_bbox],
        "landmarks": landmarks,
        "ear": None if face.ear is None else float(face.ear),
    }


def _headpose_to_json(hp: "HeadPoseResult | None") -> dict | None:
    """Serialise a HeadPoseResult into the frozen ``head_pose`` object, or None.

    Args:
        hp: The per-person head-pose result, or ``None``.

    Returns:
        A dict matching the schema's ``head_pose`` object, or ``None``.
    """
    if hp is None:
        return None
    return {
        "yaw": float(hp.yaw),
        "pitch": float(hp.pitch),
        "roll": float(hp.roll),
        "gaze_label": hp.gaze_label,
    }


def _assemble_frame(
    frame_id: int,
    timestamp_ms: int,
    persons: list["Person"],
    faces: list["FaceResult"],
    headposes: list["HeadPoseResult | None"],
    objects: list["Obj"],
) -> dict:
    """Build one JSONL record in the frozen Stage 1 schema.

    Args:
        frame_id: Zero-indexed source frame number.
        timestamp_ms: Frame presentation time in milliseconds.
        persons: Detected persons for this frame.
        faces: Face results, index-aligned with ``persons``.
        headposes: Head-pose results, index-aligned with ``persons``.
        objects: Detected whitelisted objects.

    Returns:
        A JSON-serialisable dict matching ``schema.json``.

    Raises:
        ValueError: If ``faces``/``headposes`` are not aligned with ``persons``.
    """
    if not (len(persons) == len(faces) == len(headposes)):
        raise ValueError(
            "Misaligned per-person lists: "
            f"persons={len(persons)}, faces={len(faces)}, "
            f"headposes={len(headposes)}."
        )

    person_records = []
    for person, face, hp in zip(persons, faces, headposes):
        person_records.append(
            {
                "track_id": None,  # filled by ByteTrack in Stage 2
                "bbox": [int(v) for v in person.bbox],
                "confidence": float(person.confidence),
                "face": _face_to_json(face),
                "head_pose": _headpose_to_json(hp),
            }
        )

    object_records = [
        {
            "cls": obj.cls,
            "bbox": [int(v) for v in obj.bbox],
            "confidence": float(obj.confidence),
        }
        for obj in objects
    ]

    return {
        "frame_id": int(frame_id),
        "timestamp_ms": int(timestamp_ms),
        "persons": person_records,
        "objects": object_records,
    }


def _build_estimators(
    config: Config,
) -> tuple["Detector", "FaceAnalyzer", "HeadPoseEstimator"]:
    """Construct the three real perception modules from config.

    Imported lazily so the heavy ML dependencies are only required when running
    the real pipeline (tests inject fakes instead).

    Args:
        config: The full pipeline config.

    Returns:
        A ``(detector, face_analyzer, headpose_estimator)`` tuple.

    Raises:
        ImportError: If a required ML package is not installed.
        RuntimeError: If a model fails to load.
    """
    from backend.detection import Detector
    from backend.face import FaceAnalyzer
    from backend.headpose import HeadPoseEstimator

    detector = Detector(config.detection)
    face_analyzer = FaceAnalyzer(config.face)
    headpose_estimator = HeadPoseEstimator(config.headpose)
    return detector, face_analyzer, headpose_estimator


def process_video(
    video_path: str | Path,
    out_jsonl_path: str | Path,
    config: Config = CONFIG,
    *,
    detector: DetectorLike | None = None,
    face_analyzer: FaceAnalyzerLike | None = None,
    headpose_estimator: HeadPoseLike | None = None,
) -> int:
    """Run the full Stage 1 pipeline over a video and write JSONL output.

    Args:
        video_path: Path to the input video file.
        out_jsonl_path: Path to write the JSONL output to. Parent directories
            are created if missing.
        config: The full pipeline config. ``config.pipeline.sample_rate``
            controls frame subsampling; ``config.pipeline.log_every_frames``
            controls how often throughput is logged.
        detector: Optional detector to reuse (constructed from config if None).
        face_analyzer: Optional face analyzer (constructed from config if None).
        headpose_estimator: Optional head-pose estimator (built if None).

    Returns:
        The number of frames processed and written.

    Raises:
        FileNotFoundError: If the input video does not exist.
        RuntimeError: If the video cannot be opened by OpenCV.
        ImportError: If a required ML package is missing and no estimator was
            injected.
    """
    import cv2

    src = Path(video_path)
    if not src.is_file():
        raise FileNotFoundError(f"Input video not found: {src}")

    if detector is None or face_analyzer is None or headpose_estimator is None:
        built_detector, built_face, built_hp = _build_estimators(config)
        detector = detector or built_detector
        face_analyzer = face_analyzer or built_face
        headpose_estimator = headpose_estimator or built_hp

    sample_rate = max(int(config.pipeline.sample_rate), 1)
    log_every = max(int(config.pipeline.log_every_frames), 1)

    capture = cv2.VideoCapture(str(src))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not open video: {src}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        logger.warning("Video reports invalid FPS (%s); timestamps use 0.", fps)
        fps = 0.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    out_path = Path(out_jsonl_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from tqdm import tqdm
    except ImportError as exc:  # pragma: no cover - tqdm is a hard dependency
        raise ImportError("tqdm is required. Install it via requirements.txt.") from exc

    frame_index = 0
    written = 0
    start = time.perf_counter()

    progress = tqdm(
        total=total_frames if total_frames > 0 else None,
        desc="frames",
        unit="frame",
    )
    try:
        with out_path.open("w", encoding="utf-8") as fh:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                progress.update(1)

                if frame_index % sample_rate != 0:
                    frame_index += 1
                    continue

                pos_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
                if pos_ms and pos_ms > 0:
                    timestamp_ms = int(round(pos_ms))
                elif fps > 0:
                    timestamp_ms = int(round(frame_index * 1000.0 / fps))
                else:
                    timestamp_ms = 0

                persons, objects = detector.detect(frame)
                person_bboxes = [p.bbox for p in persons]
                faces = face_analyzer.analyze(frame, person_bboxes)
                face_bboxes = [f.face_bbox for f in faces]
                headposes = headpose_estimator.estimate(frame, face_bboxes)

                record = _assemble_frame(
                    frame_index, timestamp_ms, persons, faces, headposes, objects
                )
                fh.write(json.dumps(record) + "\n")
                written += 1

                if written % log_every == 0:
                    elapsed = time.perf_counter() - start
                    rate = written / elapsed if elapsed > 0 else 0.0
                    logger.info(
                        "Processed %d frames (%.1f FPS, last frame: %d persons, "
                        "%d objects).",
                        written,
                        rate,
                        len(persons),
                        len(objects),
                    )
                frame_index += 1
    finally:
        progress.close()
        capture.release()

    elapsed = time.perf_counter() - start
    rate = written / elapsed if elapsed > 0 else 0.0
    logger.info(
        "Done: %d frames written to %s (%.1f FPS avg).", written, out_path, rate
    )
    return written


def _positive_int(value: str) -> int:
    """Argparse type: parse a strictly-positive integer.

    Args:
        value: The raw CLI string.

    Returns:
        The parsed integer (>= 1).

    Raises:
        argparse.ArgumentTypeError: If ``value`` is not an integer >= 1.
    """
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}.")
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {parsed}.")
    return parsed


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="python -m backend.integrate",
        description=(
            "Run the full ClassGraph Stage 1 pipeline (detection + face + "
            "head pose) over a video and write per-frame JSONL."
        ),
    )
    parser.add_argument("--video", required=True, type=str, help="Input video path.")
    parser.add_argument(
        "--out",
        type=str,
        default=CONFIG.pipeline.default_output,
        help=f"Output JSONL path (default: {CONFIG.pipeline.default_output}).",
    )
    parser.add_argument(
        "--sample-rate",
        type=_positive_int,
        default=CONFIG.pipeline.sample_rate,
        help="Process every Nth frame, integer >= 1 (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default=None,
        help="Override the compute device for detection and head pose.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=CONFIG.log_level,
        help=f"Logging verbosity (default: {CONFIG.log_level}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]`` when ``None``.

    Returns:
        Process exit code (``0`` on success, ``1`` on a handled failure).
        Invalid arguments (e.g. ``--sample-rate 0``) exit ``2`` via argparse.
    """
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    config = replace(
        CONFIG, pipeline=replace(CONFIG.pipeline, sample_rate=args.sample_rate)
    )
    if args.device is not None:
        config = replace(
            config,
            detection=replace(config.detection, device=args.device),
            headpose=replace(config.headpose, device=args.device),
        )

    try:
        written = process_video(args.video, args.out, config)
    except (FileNotFoundError, RuntimeError, ImportError, ValueError) as exc:
        logger.error("Pipeline failed: %s", exc)
        return 1

    logger.info("Wrote %d frames to %s.", written, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    import sys

    sys.exit(main())
