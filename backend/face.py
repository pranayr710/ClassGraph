"""Face landmark + EAR analysis for ClassGraph Stage 1 (Perception).

Wraps MediaPipe Face Mesh to produce, per person, the 468 canonical face-mesh
landmarks (in **image** coordinates, not crop coordinates), a face bounding box,
and an eye-aspect-ratio (EAR) value used downstream for drowsiness/attention.

Design (Person B):

* One Face Mesh inference is run over the **whole frame** with
  ``max_num_faces = config.max_num_faces`` (default 40, classroom scale).
* Each detected face is then bound to the person bounding box that best
  contains it (see :data:`FaceConfig.assign_min_containment`).
* The returned list is **aligned index-wise** with ``person_bboxes``: a person
  with no matching face keeps its slot with all fields ``None``.

This module does not compute head pose (Person C) or track identities
(Stage 2). Landmarks are the canonical 468 mesh points; the 10 iris points that
``refine_landmarks=True`` adds are dropped to match the frozen schema.

Usage:
    from backend.face import FaceAnalyzer
    with FaceAnalyzer() as analyzer:
        results = analyzer.analyze(frame, person_bboxes)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from backend.config import CONFIG, FaceConfig

logger = logging.getLogger(__name__)

# A pixel bounding box: (x, y, w, h), top-left origin, integer pixels.
Bbox = tuple[int, int, int, int]
# A single landmark in image pixel coordinates.
Point = tuple[float, float]


@dataclass(frozen=True)
class FaceResult:
    """Per-person face analysis result.

    All spatial values are in **image** coordinates. Every field is ``None``
    when no face was matched to the corresponding person bounding box; the slot
    is still kept so the result list stays aligned with the input.

    Attributes:
        face_bbox: Face box ``(x, y, w, h)`` in image pixels, or ``None``.
        landmarks: List of ``num_landmarks`` ``(x, y)`` points in image pixels,
            or ``None``.
        ear: Mean eye-aspect-ratio over both eyes, or ``None``.
    """

    face_bbox: Bbox | None
    landmarks: list[Point] | None
    ear: float | None


def _euclidean(a: Point, b: Point) -> float:
    """Return the Euclidean distance between two 2-D points.

    Args:
        a: First point ``(x, y)``.
        b: Second point ``(x, y)``.

    Returns:
        The straight-line distance between ``a`` and ``b``.
    """
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _single_eye_ear(landmarks: Sequence[Point], idx: Sequence[int]) -> float | None:
    """Compute the 6-point EAR for one eye.

    Uses the Soukupova & Cech formula::

        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

    where ``idx`` lists the landmark indices in ``(p1, p2, p3, p4, p5, p6)``
    order (p1/p4 the horizontal eye corners).

    Args:
        landmarks: All face landmarks as ``(x, y)`` points in image pixels.
        idx: The six landmark indices for this eye, in P1..P6 order.

    Returns:
        The EAR for this eye, or ``None`` if an index is out of range or the
        horizontal eye width is zero (degenerate — cannot normalise).

    Raises:
        ValueError: If ``idx`` does not contain exactly six indices.
    """
    if len(idx) != 6:
        raise ValueError(f"Expected 6 eye indices, got {len(idx)}.")
    if any(i < 0 or i >= len(landmarks) for i in idx):
        return None

    p1, p2, p3, p4, p5, p6 = (landmarks[i] for i in idx)
    horizontal = _euclidean(p1, p4)
    if horizontal == 0.0:
        return None
    vertical = _euclidean(p2, p6) + _euclidean(p3, p5)
    return vertical / (2.0 * horizontal)


def compute_ear(
    landmarks: Sequence[Point],
    left_idx: Sequence[int],
    right_idx: Sequence[int],
) -> float | None:
    """Compute the mean eye-aspect-ratio over both eyes.

    Args:
        landmarks: All face landmarks as ``(x, y)`` points in image pixels.
        left_idx: The six left-eye landmark indices in P1..P6 order.
        right_idx: The six right-eye landmark indices in P1..P6 order.

    Returns:
        The average EAR over whichever eyes could be computed, or ``None`` if
        neither eye yields a valid value.

    Raises:
        ValueError: If either index tuple does not contain exactly six indices.
    """
    left = _single_eye_ear(landmarks, left_idx)
    right = _single_eye_ear(landmarks, right_idx)
    valid = [e for e in (left, right) if e is not None]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def _bbox_from_points(points: Sequence[Point], img_w: int, img_h: int) -> Bbox:
    """Compute the tight, image-clamped bounding box of a set of points.

    Args:
        points: The landmark points as ``(x, y)`` image pixels.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        The bounding box ``(x, y, w, h)`` clamped to the image, with ``w`` and
        ``h`` at least 1 pixel.

    Raises:
        ValueError: If ``points`` is empty.
    """
    if not points:
        raise ValueError("Cannot compute a bbox from an empty point set.")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0 = max(0, int(math.floor(min(xs))))
    y0 = max(0, int(math.floor(min(ys))))
    x1 = min(img_w, int(math.ceil(max(xs))))
    y1 = min(img_h, int(math.ceil(max(ys))))
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def _containment(inner: Bbox, outer: Bbox) -> float:
    """Fraction of ``inner``'s area that lies inside ``outer``.

    Args:
        inner: The box whose containment is measured (e.g. a face box).
        outer: The containing box (e.g. a person box).

    Returns:
        A value in ``[0.0, 1.0]``: intersection area divided by ``inner`` area.
    """
    ix0, iy0, iw, ih = inner
    ox0, oy0, ow, oh = outer
    ix1, iy1 = ix0 + iw, iy0 + ih
    ox1, oy1 = ox0 + ow, oy0 + oh

    inter_w = max(0, min(ix1, ox1) - max(ix0, ox0))
    inter_h = max(0, min(iy1, oy1) - max(iy0, oy0))
    inter = inter_w * inter_h

    inner_area = iw * ih
    if inner_area <= 0:
        return 0.0
    return inter / inner_area


def _coerce_bbox(bbox: Sequence[float]) -> Bbox:
    """Validate and convert an input bbox to an integer ``(x, y, w, h)`` tuple.

    Args:
        bbox: A length-4 sequence ``(x, y, w, h)``.

    Returns:
        The bbox as a tuple of ints.

    Raises:
        ValueError: If ``bbox`` is not length 4 or ``w``/``h`` are non-positive.
        TypeError: If any element is not a real number.
    """
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 elements (x, y, w, h), got {len(bbox)}.")
    try:
        x, y, w, h = (int(round(float(v))) for v in bbox)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"bbox elements must be numbers: {bbox!r}") from exc
    if w <= 0 or h <= 0:
        raise ValueError(f"bbox width/height must be positive, got w={w}, h={h}.")
    return (x, y, w, h)


class FaceAnalyzer:
    """MediaPipe Face Mesh wrapper returning per-person landmarks and EAR.

    The underlying Face Mesh graph is created once and reused. It is CPU-bound,
    which is expected and fine (MediaPipe does not use the GPU here). Call
    :meth:`close` when done, or use the analyzer as a context manager.

    Attributes:
        config: The :class:`FaceConfig` in effect for this analyzer.
    """

    def __init__(self, config: FaceConfig | None = None) -> None:
        """Create the Face Mesh graph.

        Args:
            config: Face settings. Defaults to ``CONFIG.face``.

        Raises:
            ImportError: If MediaPipe is not installed.
            RuntimeError: If the Face Mesh graph fails to initialise.
        """
        self.config: FaceConfig = config if config is not None else CONFIG.face

        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "MediaPipe is required for face analysis. "
                "Install it via requirements.txt (`pip install mediapipe`)."
            ) from exc

        # The Face Mesh solution lives in the legacy Solutions API, which ships
        # with the standard mediapipe 0.10 wheel. Some stripped builds expose
        # only the newer Tasks API; fail loudly rather than silently degrade.
        try:
            self._mp_face_mesh = mp.solutions.face_mesh
        except AttributeError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "This MediaPipe build lacks the Solutions Face Mesh API "
                "(mp.solutions.face_mesh). Install the standard wheel pinned in "
                "requirements.txt (mediapipe>=0.10,<0.11)."
            ) from exc
        try:
            self._mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=self.config.static_image_mode,
                max_num_faces=self.config.max_num_faces,
                refine_landmarks=self.config.refine_landmarks,
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
            )
        except Exception as exc:  # noqa: BLE001 - re-raise as RuntimeError
            raise RuntimeError(
                f"Failed to initialise MediaPipe Face Mesh: {exc}"
            ) from exc

        self._closed: bool = False
        logger.info(
            "FaceAnalyzer ready: max_faces=%d refine=%s num_landmarks=%d",
            self.config.max_num_faces,
            self.config.refine_landmarks,
            self.config.num_landmarks,
        )

    def __enter__(self) -> "FaceAnalyzer":
        """Enter the runtime context and return the analyzer."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Exit the runtime context, releasing the Face Mesh graph."""
        self.close()

    def close(self) -> None:
        """Release the underlying MediaPipe Face Mesh resources.

        Idempotent: calling it more than once is safe.
        """
        if not self._closed:
            self._mesh.close()
            self._closed = True

    def _detect_faces(
        self, frame: np.ndarray
    ) -> list[tuple[Bbox, list[Point], float | None]]:
        """Run Face Mesh on a full frame and return detected faces.

        Args:
            frame: A ``(H, W, 3)`` BGR image.

        Returns:
            A list of ``(face_bbox, landmarks, ear)`` for every detected face,
            in the order MediaPipe reports them. Empty if no faces are found.
        """
        import cv2

        img_h, img_w = frame.shape[:2]
        # MediaPipe expects RGB; OpenCV frames are BGR.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mesh.process(rgb)

        multi = getattr(results, "multi_face_landmarks", None)
        if not multi:
            return []

        n = self.config.num_landmarks
        faces: list[tuple[Bbox, list[Point], float | None]] = []
        for face_landmarks in multi:
            pts: list[Point] = [
                (lm.x * img_w, lm.y * img_h) for lm in face_landmarks.landmark[:n]
            ]
            if len(pts) < n:
                # Model returned fewer points than expected — skip defensively.
                logger.warning(
                    "Face has %d landmarks, expected %d; skipping.", len(pts), n
                )
                continue
            face_bbox = _bbox_from_points(pts, img_w, img_h)
            ear = compute_ear(
                pts, self.config.left_eye_ear_idx, self.config.right_eye_ear_idx
            )
            faces.append((face_bbox, pts, ear))
        return faces

    def analyze(
        self, frame: np.ndarray, person_bboxes: Sequence[Sequence[float]]
    ) -> list[FaceResult]:
        """Analyze one frame and bind faces to the given person boxes.

        Args:
            frame: A ``(H, W, 3)`` BGR image as returned by OpenCV.
            person_bboxes: Person boxes ``(x, y, w, h)`` from ``detection.py``,
                in image pixels.

        Returns:
            A list of :class:`FaceResult`, one per entry in ``person_bboxes`` and
            in the same order. Persons with no matching face have all-``None``
            fields.

        Raises:
            RuntimeError: If the analyzer has already been closed.
            TypeError: If ``frame`` is not a NumPy array.
            ValueError: If ``frame`` is empty/not 3-channel, or a person bbox is
                malformed.
        """
        if self._closed:
            raise RuntimeError("FaceAnalyzer is closed; create a new instance.")
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"frame must be a numpy.ndarray, got {type(frame)!r}.")
        if frame.size == 0:
            raise ValueError("frame is empty (zero-size array).")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"frame must be an (H, W, 3) image, got shape {frame.shape!r}."
            )

        boxes: list[Bbox] = [_coerce_bbox(b) for b in person_bboxes]
        if not boxes:
            return []

        faces = self._detect_faces(frame)

        # Greedy assignment: each person takes the still-unused face with the
        # highest containment above threshold. Deterministic in input order.
        used = [False] * len(faces)
        results: list[FaceResult] = []
        for person_box in boxes:
            best_i = -1
            best_score = 0.0
            for i, (face_bbox, _pts, _ear) in enumerate(faces):
                if used[i]:
                    continue
                score = _containment(face_bbox, person_box)
                if score > best_score:
                    best_score = score
                    best_i = i

            if best_i >= 0 and best_score >= self.config.assign_min_containment:
                used[best_i] = True
                face_bbox, pts, ear = faces[best_i]
                results.append(FaceResult(face_bbox=face_bbox, landmarks=pts, ear=ear))
            else:
                results.append(FaceResult(face_bbox=None, landmarks=None, ear=None))

        matched = sum(1 for r in results if r.landmarks is not None)
        logger.debug(
            "analyze: %d persons, %d faces detected, %d matched.",
            len(boxes),
            len(faces),
            matched,
        )
        return results
