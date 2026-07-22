"""Unit tests for :mod:`backend.detection`.

These tests exercise the real YOLOv11 model. They require ``ultralytics`` +
``torch`` to be installed (and, on first run, network access for Ultralytics to
download the COCO weights). When those deps are unavailable the model-backed
tests skip cleanly rather than fail — we never fake a green result.

Test coverage (as specified for Person A):
    1. The model loads without error.
    2. A person is detected on a supplied fixture image.
    3. A black (all-zero) frame yields no persons and no objects.
    4. JSONL output written by ``run_on_video`` matches ``schema.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# Skip the entire module if the detection stack is not installed. This keeps CI
# honest: absent deps => skipped (visibly), never a false pass.
pytest.importorskip("ultralytics")
pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")
jsonschema = pytest.importorskip("jsonschema")

from backend.detection import (  # noqa: E402 - imported after importorskip guards
    Detector,
    Obj,
    Person,
    run_on_video,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCHEMA_PATH = _REPO_ROOT / "schema.json"
_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def detector() -> Detector:
    """A single shared detector for the module (model load is expensive)."""
    return Detector()


@pytest.fixture(scope="module")
def schema() -> dict:
    """The frozen Stage 1 JSON schema loaded once."""
    with _SCHEMA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _find_fixture_image() -> Path | None:
    """Return a person-containing fixture image, or ``None`` if none is found.

    Resolution order:
        1. Any image a teammate drops into ``tests/fixtures/`` (preferred — use a
           real classroom frame here for a representative test).
        2. Ultralytics' bundled sample assets (``bus.jpg`` has four people,
           ``zidane.jpg`` has two), which ship with the package. This lets the
           test run for real out of the box on the target GPU machine without
           anyone having to supply an image manually.

    Returns:
        A path to a usable image, or ``None`` if neither source is available.
    """
    if _FIXTURE_DIR.is_dir():
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            for candidate in sorted(_FIXTURE_DIR.glob(pattern)):
                return candidate

    try:
        from ultralytics.utils import ASSETS
    except Exception:  # noqa: BLE001 - any import/attr failure => no fallback
        return None

    for name in ("bus.jpg", "zidane.jpg"):
        candidate = Path(ASSETS) / name
        if candidate.is_file():
            return candidate
    return None


def test_model_loads_without_error(detector: Detector) -> None:
    """The Ultralytics model loads and exposes a class-name map."""
    assert detector.device in ("cuda", "cpu")
    # `.names` is populated from the loaded model; person must be a known class.
    assert "person" in detector._names.values()


def test_detects_person_on_fixture(detector: Detector) -> None:
    """A supplied fixture image containing a person yields >= 1 Person."""
    fixture = _find_fixture_image()
    if fixture is None:
        pytest.skip(
            "No fixture image in tests/fixtures/ "
            "(drop a person/classroom .jpg/.png there to enable this test)."
        )

    frame = cv2.imread(str(fixture))
    assert frame is not None, f"Failed to read fixture image: {fixture}"

    persons, objects = detector.detect(frame)

    assert isinstance(persons, list)
    assert isinstance(objects, list)
    assert len(persons) >= 1, f"Expected >=1 person in {fixture.name}, found 0."
    for person in persons:
        assert isinstance(person, Person)
        x, y, w, h = person.bbox
        assert w > 0 and h > 0
        assert 0.0 <= person.confidence <= 1.0
        assert person.confidence >= detector.config.person_conf
    for obj in objects:
        assert isinstance(obj, Obj)
        assert obj.cls in detector.config.object_whitelist


def test_black_frame_returns_empty(detector: Detector) -> None:
    """An all-zero frame produces no persons and no objects."""
    black = np.zeros((720, 1280, 3), dtype=np.uint8)
    persons, objects = detector.detect(black)
    assert persons == []
    assert objects == []


def test_jsonl_output_matches_schema(
    detector: Detector, schema: dict, tmp_path: Path
) -> None:
    """`run_on_video` output is valid JSONL and conforms to schema.json."""
    # Build a short synthetic video (10 black frames). It will contain no
    # detections, but the per-frame records must still validate — empty
    # `persons`/`objects` arrays are legal under the schema.
    video_path = tmp_path / "clip.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width, height, fps, n_frames = 320, 240, 10.0, 10
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    assert writer.isOpened(), "Could not open VideoWriter (codec unavailable?)."
    try:
        for _ in range(n_frames):
            writer.write(np.zeros((height, width, 3), dtype=np.uint8))
    finally:
        writer.release()

    out_path = tmp_path / "stage1.jsonl"
    written = run_on_video(video_path, out_path, detector=detector)

    assert written >= 1
    assert out_path.is_file()

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == written

    validator = jsonschema.Draft202012Validator(schema)
    seen_ids: list[int] = []
    for line in lines:
        record = json.loads(line)
        # Raises jsonschema.ValidationError on any contract violation.
        validator.validate(record)
        seen_ids.append(record["frame_id"])
        for person in record["persons"]:
            assert person["track_id"] is None
            assert person["face"] is None
            assert person["head_pose"] is None

    # frame_ids are unique and monotonically increasing.
    assert seen_ids == sorted(seen_ids)
    assert len(set(seen_ids)) == len(seen_ids)


def test_detect_rejects_bad_input(detector: Detector) -> None:
    """Non-array and malformed frames raise explicit, typed errors."""
    with pytest.raises(TypeError):
        detector.detect("not-a-frame")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        detector.detect(np.zeros((0, 0, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        detector.detect(np.zeros((10, 10), dtype=np.uint8))  # missing channels
