# ClassGraph — Challenges & Solutions Log

> Written for slide-building. Each entry: **what broke / what was wrong → how we found out → what we did → the number that proves it.**
> Every number below was measured on this project's real footage, not assumed. Session covers: GPU environment setup, two real pipeline bugs, detection tuning, a new posture-fallback module, repo hygiene, and a research phase feeding Stage 3 design.

---

## 1. Environment: from bare Python to a working GPU pipeline

**Problem:** The GPU machine had nothing installed — no PyTorch, no CUDA, no ML packages. The handoff's plan assumed a clean `pip install -r requirements.txt` would work.

**What went wrong along the way:**

| Issue | How it was found | Fix |
|---|---|---|
| `requirements.txt` pinned `sixdrepnet>=0.1.7`, which **has never existed on PyPI** (latest is 0.1.6) | The pin blocked the *entire* install — one bad line stopped numpy, opencv, ultralytics, mediapipe from installing too | Repinned to `>=0.1.6,<0.2` |
| MediaPipe failed with a cryptic DLL load error | Root-caused to a **missing Microsoft Visual C++ Redistributable** — the machine had zero MSVC runtime installed | Downloaded and installed the official redistributable (signature-verified before running) |
| Plain `pip install torch` installs a **CPU-only** build silently | Would have made the whole GPU purchase pointless — caught before it happened | Installed explicitly from PyTorch's `cu124` index |

**Result:** `torch.cuda.is_available() == True` on the RTX 4050 Laptop GPU, confirmed — not assumed. Every dependency verified importable, including `mediapipe`'s legacy Solutions API (`mp.solutions.face_mesh`), which the handoff had only verified on a *different* machine.

---

## 2. Bug #1 — Face detection found zero faces on real footage

**The problem:** Before any fix, running the pipeline on real classroom photos and a real video produced **0 faces, on every single frame** — despite people clearly having visible faces.

**Root cause:** `face.py` ran MediaPipe Face Mesh once on the **whole video frame**. Face Mesh's internal detector downscales its input to a small fixed size before looking for a face — so a face that's small *relative to the whole frame* gets destroyed before detection even runs. In a 3840×2160 frame, a perfectly visible face was invisible to the model.

**How it was proven, not guessed:**

| Test image | Whole-frame approach | Per-person-crop approach |
|---|---|---|
| 3840×2160 video, 1 student | 0 faces | **1 / 1** |
| 1920×1088 classroom CCTV, 20 students | 0 faces | **8 / 20** |

**The fix:** Crop to each detected person's bounding box *first*, then run Face Mesh on that crop. This restores the face's size relative to its input.

**A second-order finding while fixing it:** padding the crop for "safety margin" actually made things *worse* — it re-enlarges the frame relative to the face, undoing the fix. Measured: `pad=0.15` → 1/5 video frames; `pad=0.0` → 5/5. Shipped with **zero padding**, against initial intuition.

**Why it mattered:** this bug meant the entire face + gaze half of the engagement pipeline produced nothing on any real footage — it only ever "passed" because the test suite used synthetic frames and fake injected models.

---

## 3. Bug #2 — The gaze direction was backwards

**The problem:** After fixing face detection, gaze labels looked wrong: students visibly bowed over desks writing were being labeled `"back"` (looking backward/up) — and `"down"` never appeared at all, across 12 real classroom images.

**Root cause:** SixDRepNet (the head-pose model) reports pitch as **up-positive**. Our code's contract — and the `classify_gaze` logic — assumed **down-positive**. The two conventions were exactly inverted. Confirmed directly from the model's own source code (`draw_axis` function), not inferred from behavior.

**Measured effect of the fix, same 12 images:**

| Label | Before fix | After fix |
|---|---|---|
| `down` | 0 | **6** |
| `back` | 2 | **0** |
| `teacher` | 17 | 13 |

**Why it mattered:** "looking down at a phone" is arguably the single most important signal for an engagement system. The bug was silently inverting it — a student on their phone would have been unlabelable as `"down"` at all.

**Process note:** no existing test caught this, because the test's fake model fed values straight through and only checked that labels were *valid*, never that they were *correct*. A regression test pinning the sign convention was added.

---

## 4. Detection tuning — recovering the students the model was resizing away

**Problem:** Person detection was missing a large fraction of students, especially in back rows of wide classroom shots.

**Root cause:** YOLO resizes every frame to a fixed inference size (`imgsz`) before detecting anything. At the shipped default of 960px, a back-row student who was ~60px tall in a 1920px-wide frame shrank to ~30px — too small to detect.

**Measured sweep across 12 real classroom images:**

| `imgsz` | Persons found | Speed (ms/image) |
|---|---|---|
| 960 (old default) | 175 | 34 |
| **1280 (chosen)** | **236** | 50 |
| 1536 | 271 | 72 |
| 1920 | 301 | 86 |

Also lowered the person-confidence threshold (0.40 → 0.30) — distant students score lower, and for engagement statistics a missed student is worse than an extra box.

**Net effect:** 139 → 236 persons found across the same 12 images; faces recoverable went from 0 → 95 (combined with the crop fix above).

**A mistake caught along the way:** initially concluded that raising `object_conf` would remove "phantom laptop" false positives seen in one classroom image. Checking a *second* image revealed it was an actual **computer lab** — those laptops were real. The threshold was left alone rather than "fixing" a problem that was actually a labeling assumption error.

---

## 5. Building a face-independent fallback signal — `PostureAnalyzer`

**Problem underneath everything above:** even after both fixes and tuning, only ~40–45% of detected students ever have a usable face — an inherent limit of an overhead/rear-corner camera, not something more tuning fixes. A student bowed over a desk shows the camera the crown of their head; no face algorithm can recover a face that isn't in the frame.

**The idea tested:** if a face isn't visible, is *body posture* (head/shoulder/hip position) still detectable? MediaPipe Pose is another off-the-shelf, pretrained model — no training required, consistent with the rest of the pipeline.

**Measured, 167 faceless persons across 13 real classroom images:** 94 (56%) still yielded usable pose keypoints. Visually confirmed by rendering skeletons — the one standing teacher in a room full of seated writing students was the single clean "upright" case, correctly distinguished from every seated "bowed" student.

**A hypothesis that was tested and rejected, honestly:** initially tried classifying posture as "bowed" vs. "upright" using a simple geometric rule (nose position relative to shoulder line). Hand-checking a spread of 40 real crops showed this **did not hold up** — comparing the feature's distribution between students who had a visible face (presumably more upright/facing camera) and those who didn't showed the two populations almost completely overlap. **No fake classifier was shipped.** Instead, `PostureAnalyzer` returns raw geometry only (nose/shoulder/hip coordinates), explicitly documented as *not* a validated posture label — the same honesty pattern the eye-openness (EAR) module already used.

**A second idea tested and rejected:** chaining MediaPipe's dedicated face-detection model before Face Mesh, hoping it would recover more faces at distance. Measured across all 13 images: it performed *worse* (25% vs. the existing 42%) — documented as a rejected approach so it isn't re-attempted.

**Result after wiring it into the full pipeline** (real run, 321-frame video):

| | Before posture fallback | After |
|---|---|---|
| Persons with **some** signal (face or posture) | 265 / 321 (face only) | **321 / 321 (100%)** |
| Persons with **no** signal at all | 56 | **0** |

**Cost:** processing speed dropped from 11.0 → 7.8 FPS on the RTX 4050 (a fourth model now runs per person). Documented plainly, not hidden.

---

## 6. Schema change — done with the project owner's direct sign-off

**Problem:** the output contract (`schema.json`) was explicitly "frozen — do not edit without raising it with the team." Adding posture output meant changing it.

**Resolution:** the project owner authorized the change directly. Added a new, additive `posture` field to each person record (nullable, matching the existing `face`/`head_pose` pattern) rather than overwriting anything. Verified: 321/321 real output records still validate against the updated schema.

---

## 7. Repository hygiene issues

| Issue | What happened | Fix |
|---|---|---|
| Stale local `main` branch caused a rejected push | An earlier, separate history-rewrite (squashing commits, stripping AI co-authorship) left a duplicate local branch that a plain `git push origin main` grabbed by name instead of the actual work | Diagnosed via `git merge-base`, confirmed no data loss, deleted the stale branch, renamed the correct one to `main` |
| AI co-author trailer reappeared in a commit after being explicitly removed earlier | A default habit (auto-appending a `Co-Authored-By: Claude` line) conflicted with the project owner's explicit, repeated request to keep it out of GitHub's contributor list | Caught before/immediately after push in two separate instances; amended and force-pushed the correction each time; the policy was adopted going forward for this repo |
| Ruff (linter) flagged 80 issues codebase-wide | Version drift — the handoff's "ruff clean" claim was true for an older ruff release, not the one now installed | Fixed 78 by hand/auto-fix (redundant casts, stale suppressions, minor logging style); 2 left as a documented, deliberate exception (would require a new dependency for a stub-only nitpick) |

---

## 8. Research phase — grounding Stage 3 design decisions before building them

**The trigger:** two open design questions with no obvious answer — (a) how to avoid punishing normal, brief attention lapses, and (b) how to avoid mistaking a student asking a neighbor for help as "distracted."

**What was done:** four parallel deep-research passes across cognitive science, learning sciences (CSCL), computer-vision prior art, and AI ethics / sensor-fusion design — roughly 130 real, cited sources — synthesized into a shared reference document (published separately) with every claim tagged by evidence strength (well-supported / contested / unverified).

**Headline findings that will shape Stage 3:**

- A single ~2-second mental break was shown to **eliminate** vigilance decline entirely across a 50-minute task — brief lapses aren't attention failing, they're the mechanism that protects it.
- Real gaze-based lecture-attention systems get their best results aggregating over **~12-second windows**, never single frames — direct precedent for how ClassGraph should score attention.
- "Productive peer talk" is, by the learning-sciences field's own definition, a property of *what's said*, not of gaze or posture — no vision-only system in the literature claims to make that distinction, and the field's own answer when it needed to was to add a microphone, not a smarter camera heuristic.
- A **real, deployed** classroom emotion-monitoring system in China (live per-student scores on classroom screens) is a documented case of measurable student harm and public backlash — the clearest evidence for how *not* to present this kind of data.
- The EU AI Act makes inferring **emotion** from biometric data in an education setting a flat legal prohibition, not just "high-risk" — regardless of where ClassGraph is ultimately used, this sets the tone for how outputs should be framed.

**Nine concrete, source-traceable decisions came out of this** (rolling attention windows, a distinct "peer-oriented" category, equal engineering investment in the posture branch, per-student calibration, grounding the label taxonomy in an established construct, never using the word "emotion," defaulting reports to class-level trends rather than individual live scores, resetting tracking identity every session, and auditing face/gaze accuracy across skin tone) — full detail and sourcing in the published research artifact.

---

## Where things stand, in numbers

| Metric | Session start | Now |
|---|---|---|
| Automated tests passing | 10 (9 skipped) | **38 (0 skipped, 0 failed)** |
| CUDA confirmed working | No | **Yes — RTX 4050** |
| Faces found on real footage | 0 | Robust (42% face, up to 100% w/ posture fallback) |
| Persons found (12-image sample) | 139 | 236 |
| Gaze `"down"` label reachable | No (bug) | Yes |
| Real end-to-end run completed | Never | Yes — 321 frames, schema-valid |
| Stage 2 (tracking) | Not started | Done (ByteTrack) |
| Stage 3 (engagement scoring) design | No research basis | Grounded in ~130 sourced findings |
