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

    # Confidence thresholds.
    # person_conf lowered from 0.40: distant back-row students score low, and
    # for engagement statistics a missed student costs more than a stray box.
    person_conf: float = 0.30
    # object_conf left at 0.35 deliberately. Raising it to 0.50 cuts "laptop"
    # detections from 19 to 6 across the sample set, but that is not a pure
    # false-positive win: img04 is a computer lab where the laptops are real,
    # while img01 is an ordinary classroom where they are not. One global
    # threshold cannot separate those cases — tune this only against labelled
    # ground truth.
    object_conf: float = 0.35

    # NMS IoU
    iou: float = 0.50

    # COCO class names we care about (person auto-included)
    object_whitelist: tuple[str, ...] = ("cell phone", "laptop", "book")

    # Inference resolution. YOLO resizes the frame to this before inference, so
    # it directly controls whether distant students survive. Raised from 960:
    # at 960 a back-row student ~60 px tall in a 1920-wide frame shrinks to
    # ~30 px and is lost. Persons detected across 12 classroom images
    # (person_conf=0.30), with per-image latency on an RTX 4050:
    #    960 -> 175 persons,  34 ms
    #   1280 -> 236 persons,  50 ms   <- chosen
    #   1536 -> 271 persons,  72 ms
    #   1920 -> 301 persons,  86 ms
    # Higher keeps helping, but 1280 captures most of the gain at 1.5x cost.
    # Raise it for offline batch runs where latency does not matter.
    imgsz: int = 1280

    # Batch size when running on video frames
    batch_size: int = 1


@dataclass(frozen=True)
class FaceConfig:
    """Person B — MediaPipe Face Mesh settings."""

    max_num_faces: int = 40  # cap per Face Mesh pass (a crop normally has 1)
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

    # Face Mesh runs on a per-person crop, not the whole frame. MediaPipe's
    # face detector downscales its input to a small fixed size, so a face that
    # is small *relative to the frame* is destroyed before detection runs.
    # Measured on real footage with the whole-frame approach:
    #   3840x2160 clip, 1 student  -> 0 faces  (per-person crops: 1/1)
    #   1920x1088 classroom CCTV   -> 0 faces  (per-person crops: 8/20)
    # Padding added around each person box before cropping, as a fraction of
    # the box size. Measured to be actively harmful — padding enlarges the crop,
    # which shrinks the face relative to it and reverses the benefit of
    # cropping at all. Faces found (5 video frames / 20-person CCTV frame):
    #   pad 0.15, full body -> 1/5  and  7/20
    #   pad 0.00, full body -> 5/5  and  8/20   <- default
    # Cropping only the upper part of the person box was also tried and is
    # worse on the classroom frame (top 50% -> 5/20, top 30% -> 1/20), because
    # students bent over desks have their head low in the box. Kept
    # configurable, but raise it only with measurements to justify it.
    person_crop_padding: float = 0.0

    # Overlapping person boxes can both see the same physical face. A candidate
    # whose IoU with an already-assigned face exceeds this is treated as a
    # duplicate and not assigned twice.
    duplicate_face_iou: float = 0.50


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
class PostureConfig:
    """Exploratory — body-pose keypoints as a signal independent of a face.

    Not part of the frozen Stage 1 contract (schema.json has no posture field)
    and not wired into integrate.py's output. See backend/posture.py's module
    docstring for why this exists and what it deliberately does NOT claim.
    """

    model_complexity: int = 1
    static_image_mode: bool = True

    # Recovery of a faceless person's pose keypoints, measured across 167
    # faceless persons in 13 real classroom images:
    #   0.2 -> 111/167 (66%)
    #   0.3 -> 94/167  (56%)   <- chosen: the value actually hand-checked
    #   0.5 (MediaPipe's own default) -> 46/167 (28%)
    #   0.7 -> 18/167  (11%)
    # 0.2 recovers more but was not hand-verified against the real images the
    # way 0.3 was (see the module docstring's montage review) — raise it only
    # after doing the same check.
    min_detection_confidence: float = 0.3

    # MediaPipe Pose's 33-point landmark indices (BlazePose topology) used
    # here. Config-driven per the project's no-magic-numbers rule, though
    # these are fixed by the model, not tunable.
    nose_idx: int = 0
    left_shoulder_idx: int = 11
    right_shoulder_idx: int = 12
    left_hip_idx: int = 23
    right_hip_idx: int = 24

    # A landmark is reported only if its MediaPipe visibility meets this.
    keypoint_min_visibility: float = 0.5


@dataclass(frozen=True)
class AttentionConfig:
    """Exploratory — Stage 3 first slice: windowed, per-student attention signal.

    Consumes Stage 1+2 JSONL output (gaze_label, EAR, posture presence,
    objects, track_id) after the fact; nothing here is wired into
    schema.json or the live capture loop. See backend/attention.py's module
    docstring for the research this operationalises and what it deliberately
    does not claim.

    Every timing default below is an engineering interpolation across several
    adjacent findings in cognitive-science and gaze-based mind-wandering
    research, not a number lifted from a single study that measured this
    exact system. Treat these as tunable starting points to validate against
    real footage, not settled constants.
    """

    # Rolling window for the per-frame category distribution. Gaze-based
    # mind-wandering detectors built for real lecture footage get their best
    # results aggregating over roughly 12 seconds, not single frames (Faber,
    # Bixler & D'Mello). 15s rounds that up with margin for this pipeline's
    # lower frame rate under posture fallback (~8 FPS on an RTX 4050).
    window_seconds: float = 15.0

    # How long a rolling window must stay majority "head_down_with_device"
    # before it is flagged as sustained rather than a normal brief lapse.
    # A single ~2-second break was shown to prevent vigilance decline
    # entirely (Ariga & Lleras 2011); most self-reported classroom lapses
    # last under a minute (Bunce, Flens & Neiles 2010). 90s sits at the
    # midpoint of the literature's 60-120s "this is no longer a blip" range
    # — deliberately not the lower bound, so a single missed glance-back
    # doesn't trip it.
    sustained_seconds: float = 90.0

    # Fraction of the rolling window that must be "head_down_with_device"
    # for the window to count as currently off-task, when accumulating
    # toward sustained_seconds.
    off_task_majority_fraction: float = 0.5

    # Per-student calibration baseline period. A real classroom deployment
    # (Sumer et al. 2021) measured a personal calibration baseline built
    # from the student's own first ~60s of data improving AUC by +0.084 --
    # the one concrete, literature-measured accuracy lever available here.
    calibration_seconds: float = 60.0

    # gaze_label "left"/"right" is deliberately NOT treated as off-task.
    # Kendon's F-formation research gives a real geometric definition of
    # joint peer interaction (reciprocal, sustained body orientation between
    # two tracked people), but detecting it needs pairing across tracked
    # students, which this first slice does not implement -- see the module
    # docstring. Collapsing "turned toward a neighbour" into "distracted"
    # would be exactly the mistake that research warned against, so it is
    # reported as its own "oriented_away" bucket instead: known-ambiguous,
    # not guessed at.
    #
    # gaze_label "down"/"back" is similarly not assumed to be off-task on its
    # own -- gaze aversion while concentrating on a hard problem is a
    # documented, opposite-reading confound (Doherty-Sneddon et al.). It is
    # only treated as a meaningful signal when combined with a nearby
    # "cell phone" detection, which is the one case with a defensible
    # behavioural reading in the existing schema (a phone under a bowed
    # head is a stronger proxy than a bowed head alone, which is equally
    # consistent with reading or writing).
    device_gaze_labels: tuple[str, ...] = ("down", "back")
    device_object_classes: tuple[str, ...] = ("cell phone",)

    # A "cell phone" detection counts as near a person if its box overlaps
    # theirs at all, in image space.
    device_proximity_iou: float = 0.0


@dataclass(frozen=True)
class PeerInteractionConfig:
    """Exploratory -- pairwise "peer-oriented" detection between students.

    Not part of the frozen Stage 1 contract; reads finished Stage 1+2 JSONL
    like backend.attention does. See backend/peer_interaction.py's module
    docstring for the F-formation research this operationalises, and what
    it deliberately does not claim (it detects joint physical orientation
    between two tracked students, never whether their interaction is
    academically productive or off-task -- that distinction is not
    recoverable from vision alone per the CSCL literature this implements a
    decision from).

    Every threshold below is an engineering default, not one measured for
    this exact system -- there is no labelled peer-interaction ground truth
    to calibrate against yet. Validate before trusting the output.
    """

    # Two people count as "at conversational distance" if the shorter of
    # their two bbox widths, scaled by this factor, exceeds the gap between
    # their bboxes. Scale-relative (not a fixed pixel count) so it holds
    # across near/far students in the same frame.
    max_gap_to_width_ratio: float = 1.5

    # How close each person's shoulder-line orientation must be to
    # "perpendicular to the line connecting them" to count as oriented
    # toward each other this frame, in degrees. Wide on purpose: Kendon's
    # F-formations include both vis-a-vis (face-to-face) and L-shaped
    # (cooperative, common in classroom side-by-side seating) arrangements,
    # and the shoulder-line-orientation test used here (see the module
    # docstring for why it is undirected, sidestepping the front/back
    # ambiguity in a single shoulder line) is a coarse proxy for either.
    orientation_tolerance_degrees: float = 35.0

    # Rolling window before judging a pair, and majority fraction of that
    # window required to count the pair as currently oriented toward each
    # other. Same rationale as backend.attention's windowing: Kendon's own
    # turn-taking research shows real conversation has intermittent gaze,
    # so a momentary break must not reset a genuine pairing.
    window_seconds: float = 15.0
    majority_fraction: float = 0.5

    # How long a pair must stay majority-oriented before being reported at
    # all. Deliberately not tied to backend.attention's sustained_seconds:
    # this is reporting a detected joint orientation, not a sustained
    # concern, so it can surface sooner.
    sustained_seconds: float = 20.0


@dataclass(frozen=True)
class FairnessAuditConfig:
    """Exploratory -- tooling for a demographic accuracy audit, and a
    confound diagnostic that needs no labels at all.

    See backend/fairness_audit.py's module docstring for the research this
    is grounded in: a real, primary-source check found MediaPipe Face Mesh
    has a published Google fairness model card (tested across Fitzpatrick
    skin tone AND a "Southern Asia" geographic bucket -- this project's
    actual population), while SixDRepNet has zero published fairness
    evaluation of any kind. Neither has been tested by anyone, anywhere,
    against South Asian faces specifically. This module cannot fill that gap
    without labelled data this project does not yet have -- it makes running
    that audit mechanical once such data exists, and runs the cheaper
    confound diagnostic the research recommends doing first.
    """

    # Fitzpatrick I-VI, matching Google's own MediaPipe Face Mesh fairness
    # card exactly, so any future ClassGraph audit is directly comparable to
    # published numbers rather than using an incompatible scale. Labels
    # should come from trained human annotation, not an automated classifier
    # -- the one academic study found using automated race/skin-tone labels
    # (WFLW's "Indian" subgroup, via a FairFace+CLIP ensemble) is flagged in
    # that same research as a source of label noise, not a shortcut to trust.
    skin_tone_scale: tuple[str, ...] = ("I", "II", "III", "IV", "V", "VI")

    # Resolution buckets (shorter image side, in pixels) for the label-free
    # confound diagnostic. A 2026 academic audit of a different landmark
    # model found image resolution alone explained 29.3% of landmark-error
    # variance -- the single largest factor found, ahead of any demographic
    # one -- which is why this is checked before any skin-tone-labelled
    # audit is attempted.
    resolution_bucket_edges: tuple[int, ...] = (480, 720, 1080)


@dataclass(frozen=True)
class TrackingConfig:
    """Stage 2 — ByteTrack settings, filling the ``track_id`` field Stage 1 always leaves ``null``.

    Wraps ultralytics' own ``BYTETracker``; these six fields are exactly what
    it reads from its ``args`` object (verified against
    ``ultralytics/trackers/byte_tracker.py`` and its default
    ``bytetrack.yaml``), so no local tracking logic is implemented here.
    """

    # First-stage association only matches detections scoring at or above
    # this; second stage recovers weaker ones down to track_low_thresh so an
    # occluded person is not immediately dropped.
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.10

    # A detection with no match starts a new track only if its score is at
    # least this — keeps one-off false-positive detections from spawning IDs.
    new_track_thresh: float = 0.25

    # How many frames a track survives with no matching detection before it is
    # dropped for good (handles brief occlusion / a face turned away).
    track_buffer: int = 30

    # IoU distance threshold for the Hungarian assignment between existing
    # tracks and this frame's detections.
    match_thresh: float = 0.80

    # Blend detection confidence into the assignment cost, not just IoU.
    fuse_score: bool = True


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
    posture: PostureConfig = field(default_factory=PostureConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    peer_interaction: PeerInteractionConfig = field(
        default_factory=PeerInteractionConfig
    )
    fairness_audit: FairnessAuditConfig = field(default_factory=FairnessAuditConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


# Global instance imported everywhere. Immutable (frozen dataclass) so no
# module can accidentally mutate it at runtime.
CONFIG: Config = Config()
