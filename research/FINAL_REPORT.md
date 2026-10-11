# FINAL REPORT: branch fix/fps-ids-limbs

(The first run's report is kept as `FINAL_REPORT_run1.md`.)

**Conclusion: much better, but NOT all targets met.** The lag is fixed as far as this hardware allows: the
picture now shows every frame the camera sends (3.85 fps, up from 0.2) only ~70 ms late (down from 4.9 s).
Two limits are outside the code: **the camera itself only sends 4 frames per second** (raw mode; MJPG never
took effect), and **the laptop's only GPU (Intel UHD 620) can't run the models with its 2020 driver**. Both
need a fix from you (below). Tracking went from 29 numbers to 7-10 for 6 people, but ID switches (1.24-1.55
per person-minute) and getting the same number back after leaving (0-1 of 3) still miss their targets. Devices
went from 7% to 36-50% recall and false devices from 30 to 14-20 per 10 minutes, still short of 90% / 1.
Score is gone; per-person alert counts replace it. Limbs are tracked and drawn per person.
Unit tests: 38/38 pass.

## Before and after (clip 2, real-time replay; "after" = 3 back-to-back runs of the final code)

| Target | Goal | Before | After (3 runs) | Met? |
|---|---|---|---|---|
| Display FPS | camera's full rate | 0.2 | 3.85 / 3.85 / 3.85 (camera sends 3.85) | yes (camera max) |
| Processed FPS avg / p5 | >= 20 / >= 15 | 0.2 / 0.11 | 3.85 / 3.61 every run (only 3.85 frames/s exist); loop capacity 19.5-20.6 / 11.3-11.8 | **no** (camera limit) |
| Delay (replay, frame to screen) | <= 200 ms live stopwatch | 4,931 avg / 9,191 p95 ms | 70-74 avg / 108-114 p95 ms | replay yes; live stopwatch **not run** |
| GPU in use, proven | yes | not used | **not used**: trace shows GPU compute 0%, models `openvino:CPU`; GPU compile fails on old driver | **no** (driver) |
| Numbers / true headcount | <= 1.2 | 29 / 6 = 4.83 | 9, 7, 10 / 6 = 1.50, 1.17, 1.67 | **1 of 3 runs** |
| ID switches per person-min (overlap) | <= 0.5 | 1.55 | 1.55, 1.24, 1.24 | **no** |
| Same number after leaving <= 30 s | >= 90% | 0 of 3 | 1, 1, 0 of 3 | **no** |
| Watch-type false devices per 10 min | <= 1 | 30 | 20, 14, 20 | **no** |
| Device recall, >= half visible | >= 90% | 7.7% | 53.8%, 38.5%, 46.2% | **no** |
| Device precision | >= 90% | 6.2% | 41.2%, 41.7%, 37.5% | **no** |
| Device recall full / half-hidden / edge | report | 0% / 25% / 0% | 44-67% / 25% / 0% (n = 9 / 4 / 1) | reported |
| People P / R | (keep) | 57% / 26% | 77-82% / 35-37%; near 95%, far 7-10% | improved |
| Pen recall | (new) | 0% | 38.9% | new |
| Skeleton bound to every tracked person | yes | not drawn | yes; upper-body point dropout 15.2-16.7%, jitter 0.05-0.07% of person height | yes |
| Score removed (search) | gone | in 4 files + README | search finds no "score" in app, ui, rules, tests, settings, vision, README | yes |
| alert_count correct | yes | n/a | unit tests for counting, cooldown and merge pass; not checked against a clip with labeled true alerts | partly |
| unittest discover | passes | 32 OK | 38 OK | yes |

## The 6 bugs: proven cause and fix (details and all experiments in research/FIX_PLAN.md)

1. **Lag.** Proven causes: the AI ran in the same loop as the screen (screen updated 0.2/s, picture 4.9 s old);
   YOLO-World ran on tiles plus close-ups every frame (2.5-2.9 s of each 3.4-4.0 s check); every frame was cut
   into 3 tiles (pose 63 ms whole picture vs ~480 ms tiles); first-use compile took 18.7 s; the camera sends raw
   YUY2 at 4.0 fps; no GPU usable (Intel UHD 620, OpenVINO GPU build error on driver 27.20.100.8854; torch is a
   CPU build). Fixes: threaded loop (camera keeps only newest frame, AI worker with queue size 1, tracker moves
   people every frame) `app/runtime.py:19-193`, live camera uses it `app/main.py:135-187, 230`; fast detection
   mode (whole picture once, far tiles every 3rd check, YOLO-World on wrist/head close-ups every 2nd check and the
   whole picture every 4th, results reused, warm-up of every input size) `vision/detector.py:72-86, 329-370, 371-500`.
   Now the CPU is the limit: 92% busy, which slows each model ~3-4x vs the idle bench.
2. **Tracking.** Proven causes: IoU-only matching across slow checks, a number on the first frame, numbers
   retired after 10 s, body-part boxes becoming "people", far people counted as missed on checks that never
   looked at them, and re-link colours taken from the wrong area (seated vs walking 0.53 vs stranger 0.55).
   Fix kept: `vision/tracker.py:115-360` (optical flow between checks, 3 sightings before a number, 2-pass
   matching, 45 s lost pool with re-link by position, size and clothes colours, merge keeps the older number,
   events to the trace), brain uses tracker numbers `rules/brain.py:99-113`. Three further tracker changes were
   tried and dropped (FIX_LOG rounds 3a-3c).
3. **Watch false positives.** Proven cause: YOLO-World "smartwatch" boxes on wrists/hands with no confirmation
   (30 per 10 min). Fix: one `device` class with 2-of-3-checks confirmation and watch size/shape/near-wrist checks
   `vision/devices.py:44-110`, switch `settings/config.py:81`. Result 14-20 per 10 min.
4. **Partial phones.** Proven: half-hidden 25%, edge 0% recall (n = 4 and 1). Fix: phone model also runs on
   wrist close-ups (`vision/detector.py:453-500`). No fine-tuning (time); still the weakest spot.
5. **Score.** `score = looks + 2 x long looks + 5 x phone alerts` removed everywhere; `alert_count` +1 per alert
   event `rules/brain.py:37, 186`, `merge_people` `rules/brain.py:191-207`, dashboard `ui/dashboard.py:40, 89`,
   summary `app/main.py` (per person + total), README lines 7, 54, 64, tests `tests/test_brain.py`.
6. **Limbs.** Cause: body points were never bound to a person or drawn. Fix: tracker binds and smooths them
   (One Euro filter `vision/tracker.py:79-112`), app draws the skeleton in the person's colour
   `app/main.py:29-38` and in `draw()`.

Alert mapping (planned): the old "phone" alert is now "device" (`rules/brain.py:217`), firing on the same
things as before; paper and pen never alert (`rules/brain.py:300`). No thresholds changed.

## perf/FIX_LOG.md

| Round | Change | Bug | Display | AI/s | Delay avg/p95 | People P/R | Device P/R | FP/10min | Num/person | IDSW | Return | Limb drop | Result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | Baseline | - | 0.2 | 0.2 | 4931 / 9191 | 57% / 26% | 6% / 7% | 30 | 4.83 | 1.55 | 0/3 | n/a | baseline |
| 1 | Threaded loop + tracker + alert_count | 1,2,5 | 3.85 | 0.31 | 66 / 92 | 55% / 26% | 36% / 36% | 18 | 1.00 | 0.62 | 1/3 | 42% | kept |
| 2a | Fast detection | 1 | 3.85 | 0.93 | 62 / 85 | 80% / 31% | 43% / 21% | 8 | 1.50 | 0.62 | 1/3 | 20% | reverted |
| 2b | + reuse YOLO-World results | 1 | 3.85 | 0.95 | 62 / 82 | 91% / 29% | 33% / 43% | 24 | 1.17 | 0.93 | 0/3 | 21% | reverted |
| 2c | + far people not missed when far band unchecked | 1,2 | 3.85 | 0.96 | 69 / 102 | 80% / 37% | 40% / 43% | 18 | 1.17 | 0.93 | 2/3 | 16% | kept (exception) |
| 3a | Shirt colours + unsure revive | 2 | 3.85 | 0.93 | 75 / 110 | 72% / 35% | 37% / 50% | 24 | 1.67 | 0.93 | 0/3 | 19% | reverted |
| 3b | Body-part filter + unsure revive | 2 | 3.85 | 1.16 | 75 / 113 | 71% / 34% | 37% / 50% | 24 | 1.17 | 0.62 | 1/3 | 18% | reverted |
| 3c | Shirt colours + body-part filter | 2 | 3.85 | 1.12 | 78 / 121 | 75% / 37% | 44% / 50% | 18 | 1.67 | 0.93 | 2/3 | 17% | reverted, idea dropped |
| 4 | Device confirmation + watch checks + pen/paper words | 3,4 | 3.85 | 1.10 | 76 / 123 | 80% / 37% | 44% / 50% | 18 | 1.17 | 1.24 | 0/3 | 17% | kept |
| 5 | Skeleton per person colour + tracker tests | 6 | 3.85 | 1.11 | 74 / 113 | 80% / 37% | 41% / 50% | 20 | 1.50 | 1.55 | 1/3 | 17% | kept |

Rounds 4 and 5 use the same tracker code as 2c; their tracking spread is run-to-run noise (see FIX_LOG note).

## Commits on fix/fps-ids-limbs (local only)
- 526917e fix: threaded live loop, session tracker, alert_count instead of score
- 4450f25 perf: fast detection mode
- f3ea5a2 fix: confirm devices over 2 of 3 AI checks, watch sanity checks, pen/paper words
- 2e97590 feat: draw each person's skeleton in their own colour; tracker unit tests
- (docs commit with this report)

Files changed vs the start of the branch: 24 files, +1,903 / -29 lines. App code: `app/main.py`, `app/runtime.py`
(new), `app/trace.py` (new), `vision/detector.py`, `vision/tracker.py` (new), `vision/devices.py` (new),
`rules/brain.py`, `ui/dashboard.py`, `settings/config.py:79-83`, `README.md`. Tests: `tests/test_brain.py`,
`tests/test_main.py`, `tests/test_tracker.py` (new). Measuring tools: `perf/record_clip2.py`, `perf/camera_probe.py`,
`perf/bench_models.py`, `perf/replay_live.py`, `perf/score2.py`, `perf/eval_vision.py`. Labels (box numbers only):
`research/labels2/review.json`, `research/labels2/gt.json`.

## Installed / downloaded in this run (research/INSTALL_LOG.md)
- opencv-python 4.12.0.88 (tried for the camera, uninstalled; back to 5.0.0.93)
- openvino 2025.4.0 (tried for the GPU, uninstalled; back to 2026.4.1)
- YOLO-World OpenVINO copy re-exported locally with the words "pen" and "sheet of paper"
- No new models downloaded. Everything inside `C:\Users\vobic\ARGOS-vision`; sandbox size 3.0 GB.

## Decisions (research/DECISIONS.md, run 2 section)
D-R0 recorded a new clip as asked. D-R1 camera stuck at 4 fps, MJPG idea dropped after 3 failures (Media
Foundation not tried: could open the laptop camera). D-R2 GPU blocked by the driver. D-R3 processed-FPS target
impossible with a 4 fps camera; loop capacity reported instead. D-R4 round 2 kept on attempt 3 with logged
exceptions. D-R5 tracker changes dropped after 3 failures. D-R6 phone/watch alert is now "device"; paper/pen
never alert. D-R7 video files keep the old loop. D-R8 false-device rate estimated from 25 checked frames.
D-R9 no fine-tuning. D-R10 live stopwatch test not run.

## What still fails and why
- **15-30 fps**: the Arducam sends raw YUY2 at 4 fps. Try a USB 3 port and check in the Windows Camera app or
  AMCap whether it offers MJPG 1080p30; the old perf log measured 30 fps on this camera.
- **GPU**: update the Intel UHD Graphics 620 driver (Intel Driver & Support Assistant, or HP's support page for
  this laptop), then run `.\.venv-sandbox\Scripts\python.exe perf\bench_models.py`. The UHD 620 is an integrated
  chip; expect a modest gain, not a big one.
- **CPU is full (92%)**: with 3.85 frames/s the AI manages ~1 check per second. Smaller inputs or dropping
  YOLO-World would raise that but cost recall.
- **Tracking targets**: ID switches 1.24-1.55 per person-minute and 0-1 of 3 returns. Shirt-only colours are
  proven better on single frames (0.68-0.91 vs a stranger's 0.14) but didn't pass the replay gate. A longer
  clip with more returns is needed to tell a real gain from noise.
- **Devices**: half-hidden phones 25%, phone at the frame edge 0% (only 4 and 1 examples). Needs fine-tuning on
  frames from a separate clip with occlusion and edge-cut augmentation.
- **Headphones and paper**: 0% recall on this clip.
- **Far people** (> 6 m): found 7-10% of the time.
- **Test limits**: one 5-min clip, one person moving, 25 checked frames; small counts (14 devices, 3 returns).

## Commands for you (PowerShell)
Review:
```powershell
cd C:\Users\vobic\ARGOS-vision; git log --oneline feat/classroom-vision..fix/fps-ids-limbs
```
```powershell
cd C:\Users\vobic\ARGOS-vision; git diff feat/classroom-vision..fix/fps-ids-limbs --stat
```
```powershell
cd C:\Users\vobic\ARGOS-vision; .\.venv-sandbox\Scripts\python.exe -m unittest discover -s tests
```
Live stopwatch delay test (set `SHOW_STOPWATCH = True` in settings\config.py, point the Arducam at a phone stopwatch next to the screen, compare the two times in a photo):
```powershell
cd C:\Users\vobic\ARGOS-vision; .\.venv-sandbox\Scripts\python.exe -m app.main
```
Merge yourself (I did not merge or push). This branch includes feat/classroom-vision:
```powershell
cd C:\Users\vobic\ARGOS; git fetch origin; git checkout main; git pull --rebase; git merge --no-ff fix/fps-ids-limbs
```
