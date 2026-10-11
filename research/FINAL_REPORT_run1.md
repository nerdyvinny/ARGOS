# FINAL REPORT: classroom vision for ARGOS

**Outcome: STOPPED at the 3-hour limit, targets NOT met.** One change kept (OpenVINO models), one reverted (tracker).
Processed FPS went from 0.25 to 0.33 (target 15). Person recall went from 26.9% to 28.0% (target 90%).
Overlap-test ID switches went from 1.17 to 0.87 per person-minute, which is under the target of 1.
Unit tests: 28/28 pass on 3 back-to-back runs.

## Final numbers vs baseline (real-time replay of the 5-min classroom clip)

| Metric | Baseline | Kept (round 1) | Target | Met? |
|---|---|---|---|---|
| Processed FPS avg / 5th pct | 0.25 / 0.12 | 0.33 / 0.22 | >= 15 | no |
| Delay ms avg / 95th pct | 3960 / 8065 | 3030 / 4502 | - | - |
| Person precision / recall | 51.0% / 26.9% | 60.5% / 28.0% | recall >= 90% | no |
| Person recall near (<2 m) / far (>6 m) | 65.0% / 16.9% | 65.0% / 18.3% | - | - |
| Phone recall (near) | 33.3% | 53.3% | >= 90% | no |
| Watch recall (near) | 21.4% | 28.6% | >= 90% | no |
| Headphones recall (near) | 0% | 7.1% | >= 90% | no |
| Paper / pen recall | 0% / 0% | 0% / 0% (no such class yet) | >= 90% | no |
| ID switches per person-minute, whole clip | 0.77 | 0.77 | - | - |
| ID switches per person-minute, overlap test (145-290 s) | 1.17 | 0.87 | <= 1 | **yes** |
| Dropout face / eyes / head / pose (near+mid people) | 36.8 / 36.8 / 36.8 / 45.5% | 36.8 / 36.8 / 36.8 / 40.9% | - | - |
| Alerts | none | 1 phone alert at 300.1 s (flagged for review) | - | - |

Reliable distance: no detector reached 90% recall at any distance. The best is near people (< 2 m) at 65%; people beyond ~6 m are found 18% of the time. Items were only tested within ~1.5 m. Distances are estimated from box height (+/- 30%, D9).

## VISION_LOG (full table)

| Round | Change | Video | FPS avg / p5 | Person P / R | Item recall (phone, watch, headph., paper, pen) | ID sw / p-min (overlap) | Dropout face/eyes/head/pose | Alerts | Result |
|---|---|---|---|---|---|---|---|---|---|
| 0 | Baseline | - | 0.25 / 0.12 | 51.0 / 26.9% | 33.3, 21.4, 0, 0, 0% | 0.77 (1.17) | 36.8/36.8/36.8/45.5% | none | baseline |
| 1 | OpenVINO copies of all 3 models | V4 (06:53, 08:01), V5 (05:10) | 0.33 / 0.22 | 60.5 / 28.0% | 53.3, 28.6, 7.1, 0, 0% | 0.77 (0.87) | 36.8/36.8/36.8/40.9% | phone @300 s | **kept**, commit ec40882 |
| 2a | Tracker + AI every 4th frame | V6 (04:53-07:43), V8 (01:01-02:05) | 1.22 / 0.29 | - / 0% | 0% | 0 (0) | 100% | none | bug (everyone hidden after a slow AI check), fixed -> 2b |
| 2b | Same, fixed | V6, V8 | 1.32 / 0.31 | 63.3 / 33.3% | 0% (items only on AI-check frames) | 0.58 (0.73) | 21.1/26.3/21.1/31.8% | none | **reverted**: items lost and round 1's phone alert stopped firing; patch in `research/round2_tracker.patch` |

## Kept change and where it came from
- **Round 1, OpenVINO** (videos 4 and 5: export to a CPU-optimised format). `vision/detector.py:20-27` (setting), `:48-53` (`fast_copy`), `:56-66` (`load_models` uses the OpenVINO folder, skips `set_classes` for the baked YOLO-World copy). New `tools/export_openvino.py` makes the copies. Falls back to the `.pt` files when no copy exists, so other laptops behave as before.

## Files changed on the branch (vs origin/main 95f7811)
- `vision/detector.py` (OpenVINO loading, see above)
- `tools/export_openvino.py` (new)
- `perf/eval_vision.py` (new test harness), `perf/record_classroom.py` (new 5-min recorder), `perf/peek_frames.py` (new), `perf/VISION_LOG.md` (new)
- `research/` : `video_notes.md`, `PLAN.md`, `DECISIONS.md`, `INSTALL_LOG.md`, `FINAL_REPORT.md`, `round2_tracker.patch`, `labels/teacher.py`, `labels/build_gt.py`, `labels/review.json`, `labels/gt.json`, `labels/teacher.json` (box numbers only, no images)
- `.gitignore` (sandbox folders, clips, frames, models, images)
No images of people, clips, models or logs are committed.

## Installed / downloaded (all inside `C:\Users\vobic\ARGOS-vision`, details in INSTALL_LOG.md)
- PyPI, pinned, into `.venv-sandbox`: ultralytics 8.4.175 (AGPL, already the project's library), torch 2.14.1, torchvision 0.29.1, opencv-python 5.0.0.93, numpy 2.5.4, openvino 2026.4.1 (telemetry opted out), transformers 5.19.0, lap 0.5.13, flask 3.1.2, yt-dlp 2026.8.19, ffmpeg-binaries 1.1.0 (+ their dependencies).
- Models in `models\`: the 3 existing project models copied from the main checkout (AGPL, already used by ARGOS); `PekingU/rtdetr_v2_r50vd` (Apache-2.0, 172 MB) used only to help draw ground-truth boxes; OpenVINO copies made locally.

## Decisions (full list in DECISIONS.md)
D1 asked once whether to run, then recorded first. D3 Gemini watch engine was overloaded, used the local engine. D5 existing AGPL models kept for the baseline, new downloads permissive only. D7 the clip came out at 3.85 fps. D8 ground truth = helper model + my eye check. D9 metres are estimates. D11 in-between keep rule (FPS can't jump to 15 in one step). D12 real-time replay harness. D14 round 2 reverted. D16 no alert thresholds touched.

## What still fails and why
1. **Speed.** The full pipeline (pose + phone + YOLO-World on 3 tiles plus head/wrist close-ups) takes ~3 s per frame even with OpenVINO on this 4-core CPU. 15 FPS needs the AI to run on a fraction of frames (round 2 did this and got 4x) and a slimmer AI pass: one full-frame pass instead of 3 tiles, close-ups only every few frames.
2. **Items vanish between AI checks** (why round 2 was reverted). Next step: hold each item's last box until the next AI check. That is a small change on top of `research/round2_tracker.patch`.
3. **Paper and pen are not detected at all.** No model in the project knows these classes. The next steps were adding "pen" and "sheet of paper" to the YOLO-World words (and re-exporting), or fine-tuning on frames from a separate clip.
4. **Far people (> 6 m) found 18% of the time.** The background students are 40-70 px tall; the 3-tile layout also crops the top 20% and bottom 15% of the frame (set for a 360 camera, `settings/config.py` CROP_TOP/CROP_BOTTOM), which on this wide camera throws away the desk.
5. **Face/eyes/head data missing 37% of the time** for near people facing the camera. Face landmarks (video 3, MediaPipe) were planned but not reached.
6. **Not built:** item-to-hand linking and "phone in hand" / "headphones on" review events, gaze, the per-person face/eye/limb tracking requirement beyond what YOLO pose gives.
7. **Test limits.** One recorded clip, only one person who moves, ground truth on 27 frames, and hand-drawn item boxes. Treat every percentage as approximate (e.g. 15 phone instances).

## Commands for you to review and merge (PowerShell)
```powershell
cd C:\Users\vobic\ARGOS-vision; git log --oneline main..feat/classroom-vision
```
```powershell
cd C:\Users\vobic\ARGOS-vision; git diff 95f7811..feat/classroom-vision --stat
```
```powershell
cd C:\Users\vobic\ARGOS-vision; .\.venv-sandbox\Scripts\python.exe -m unittest discover -s tests
```
To merge it yourself (I did not merge or push):
```powershell
cd C:\Users\vobic\ARGOS; git fetch origin; git checkout main; git pull --rebase; git merge --no-ff feat/classroom-vision
```
To try the reverted tracker: `git apply research\round2_tracker.patch` inside the worktree.
