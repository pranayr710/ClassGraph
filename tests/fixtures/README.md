# Test fixtures

Drop images here to turn skipped/failing fixture-dependent tests into real runs.

## `frontal_face.jpg` — enables 3 tests

A single clear, **frontal** face image satisfies every fixture-dependent test:

| Test | File | Needs |
|---|---|---|
| `test_single_face_returns_468_landmarks` | `test_face.py` | a face MediaPipe Face Mesh can detect |
| `test_ear_in_valid_range_on_face` | `test_face.py` | eyes open (asserts `0.05 <= ear <= 0.5`) |
| `test_frontal_face_returns_teacher` | `test_headpose.py` | facing the lens (asserts `\|yaw\| < 15`) |

Requirements: one face looking straight at the camera, eyes open, well lit, and
**large in the frame**. A plain selfie works well.

## Discovery order (matters if you add more than one image)

* `_find_frontal_fixture()` (head pose) prefers `frontal*.{jpg,jpeg,png}`, else
  any image here. It has **no fallback** — it skips rather than risk a flaky
  strict-yaw assertion on a non-frontal sample.
* `_find_face_fixture()` (face) takes the **alphabetically first** image here,
  else falls back to Ultralytics' bundled samples.
* `test_detects_person_on_fixture` (detection) likewise falls back to the
  Ultralytics assets, so it runs even with this directory empty.

Because the face tests take the alphabetically first image, a second fixture
named e.g. `classroom.jpg` would sort **before** `frontal_face.jpg` and be used
for them. Name additional images so they sort later (e.g. `zz_classroom.jpg`).

## Why the Ultralytics fallback is not enough for faces

`test_detects_person_on_fixture` falls back to `bus.jpg` / `zidane.jpg`, which
is fine for **person detection**. It is not fine for the **face** tests:
MediaPipe Face Mesh cannot detect a face in `zidane.jpg` at the configured
`min_detection_confidence=0.50`. Measured on this repo:

```
Face Mesh   conf=0.5 -> 0 faces
Face Mesh   conf=0.3 -> 0 faces
Face Mesh   conf=0.1 -> 1 face
Face Detection (standalone) conf=0.5 -> 1 face
```

The faces there are turned and partially occluded, so Face Mesh scores them
below threshold even though a face is present. Supply a real frontal fixture
rather than lowering the threshold.
