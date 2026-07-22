"""Central configuration for ClassGraph Stage 1 (Perception).

Every tunable knob lives here. No magic numbers in module code. Modify values
here and every module picks them up — this is the contract that lets three
teammates work in parallel without stepping on each other.

Usage:
    from backend.config import CONFIG
    detector = Detector(weights=CONFIG.detection.weights)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = REPO_ROOT / "data"
WEIGHTS_DIR: Path = REPO_ROOT / "weights"
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"


# --------------------------------------------------------------------------- #
# Per-module configs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DetectionConfig:
    """Person A — YOLOv11 detector settings."""

    weights: str = "yolo11m.pt"  # ultralytics auto-downloads if missing
    device: Literal["cuda", "cpu", "auto"] = "auto"

    # Confidence thresholds
    person_conf: float = 0.40
    object_conf: float = 0.35

    # NMS IoU
    iou: float = 0.50

    # COCO class names we care about (person auto-included)
    object_whitelist: tuple[str, ...] = ("cell phone", "laptop", "book")

    # Optional input resize (None = native resolution)
    imgsz: int = 960

    # Batch size when running on video frames
    batch_size: int = 1


@dataclass(frozen=True)
class FaceConfig:
    """Person B — MediaPipe Face Mesh settings."""

    max_num_faces: int = 40  # classroom-scale
    refine_landmarks: bool = True
    min_detection_confidence: float = 0.50
    min_tracking_confidence: float = 0.50

    # EAR (Eye Aspect Ratio) thresholds — used downstream for sleep detection.
    # Here for documentation; the face module only computes the raw value.
    ear_closed_threshold: float = 0.20
    ear_open_typical_range: tuple[float, float] = (0.20, 0.40)

    # MediaPipe eye landmark indices (refined 478-landmark map). These are
    # the 6-point EAR landmarks per eye (P1..P6 in the Soukupova & Cech paper).
    left_eye_ear_idx: tuple[int, ...] = (33, 160, 158, 133, 153, 144)
    right_eye_ear_idx: tuple[int, ...] = (362, 385, 387, 263, 373, 380)

    # Face Mesh runs each frame independently (no cross-frame tracking) since
    # analyze() receives one frame at a time. Set False only for a continuous
    # single-face stream where temporal tracking helps.
    static_image_mode: bool = True

    # Landmarks kept per face. MediaPipe returns 478 with refine_landmarks=True
    # (468 mesh + 10 iris); we keep the canonical 468 to match the frozen
    # schema. Iris points (indices 468-477) are dropped. Refinement still
    # improves the precision of the eye landmarks used for EAR.
    num_landmarks: int = 468

    # A detected face is bound to a person bbox only if at least this fraction
    # of the face's bounding box lies inside that person's bounding box.
    assign_min_containment: float = 0.50


@dataclass(frozen=True)
class HeadPoseConfig:
    """Person C — SixDRepNet head-pose settings."""

    weights: str = "sixdrepnet_300w_lp_alpha1.pth"
    device: Literal["cuda", "cpu", "auto"] = "auto"

    # Gaze bucket thresholds in degrees. Yaw = left/right, Pitch = up/down.
    # Ordered evaluation: "teacher" (frontal) > "down" > "back" > "left"/"right".
    yaw_side_threshold: float = 20.0  # |yaw| >= this -> left or right
    pitch_down_threshold: float = 20.0  # pitch >= this -> looking down
    pitch_back_threshold: float = -25.0  # pitch <= this -> looking backward/up

    # Padding around face bbox before feeding to SixDRepNet (relative to bbox size)
    crop_padding: float = 0.20


@dataclass(frozen=True)
class PipelineConfig:
    """Shared integration (Day 4) settings."""

    # Process every Nth frame (1 = every frame). Higher = faster, less temporal
    # detail. Downstream temporal module needs consistent spacing so keep this
    # fixed for a given output.
    sample_rate: int = 1

    # How often to log throughput (in processed frames).
    log_every_frames: int = 30

    # Where to write JSONL output when using the CLI.
    default_output: str = "outputs/stage1.jsonl"

    # Fail fast if a video can't be opened.
    strict_io: bool = True


@dataclass(frozen=True)
class Config:
    """Top-level config — compose all modules."""

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    headpose: HeadPoseConfig = field(default_factory=HeadPoseConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


# Global instance imported everywhere. Immutable (frozen dataclass) so no
# module can accidentally mutate it at runtime.
CONFIG: Config = Config()
