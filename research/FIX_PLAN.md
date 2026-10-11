# FIX_PLAN: what causes each bug (proven with data) and what was done

Conclusion first: the lag came from **four stacked causes**, in this order of size:
the AI blocking the screen, YOLO-World on every frame, tiles on every frame, and a camera that only sends 4 fps.
The graphics chip is **not used and cannot be** until its driver is updated. Tracking numbers improved a lot
(29 -> 7-9 numbers for 6 people) but the remaining number changes are within run-to-run noise. Devices went
from 7% to 50% recall.

## Bug 1: LAG

| Rank | Suspect | Experiment | Result | Verdict |
|---|---|---|---|---|
| 1 | AI blocks the display loop | Baseline replay: one loop does camera -> AI -> draw | screen updated 0.2 times/s; picture 4,931 ms old on average (9,191 ms at p95) | **confirmed, main cause** |
| 2 | Slow model run on everything every frame | `perf/bench_models.py`, then timing `detect_raw` on real frames | each check 3.4-4.0 s; YOLO-World 2.5-2.9 s of it (~75%) | **confirmed** |
| 3 | Tiling on every frame | same bench: full frame vs 3 tiles | pose 63 vs 295-480 ms, objects 56 vs 268-480 ms, YOLO-World 231 vs 890 ms | **confirmed** |
| 4 | Camera frames, not the AI | `perf/camera_probe.py` on the configured camera | camera sends raw **YUY2 at 4.0 fps**; MJPG request ignored (OpenCV 5.0 both orders, OpenCV 4.12 too: 1 fps, black). The new recorder with separate camera and writer threads still got 3.85 fps with 0 dropped writes | **confirmed: hard limit of 4 fps** |
| 5 | Models on CPU instead of the GPU | device list + OpenVINO compile on "GPU" | only GPU is the **Intel UHD 620** (integrated, driver 27.20.100.8854 from 2020). torch is a CPU build. OpenVINO sees the GPU but fails to build programs (CL_BUILD_PROGRAM_FAILURE) with OpenVINO 2026.4.1 and 2025.4.0 | **confirmed: CPU only; driver blocks GPU** |
| 6 | First-use compile | timing the first call | first AI check 18.7 s, later ones 3.4-4.0 s | confirmed, fixed by warm-up at load |
| 7 | Camera frames queuing (latency) | the old app read with buffer size 1 | no queue build-up; latency came from the AI blocking (rank 1) | ruled out |
| 8 | Heavy drawing / Python loops | trace: draw + tracker + rules per frame | 50-90 ms per frame total, capacity ~19-24 fps | ruled out as main cause |
| 9 | Pose/face models on the whole frame instead of crops | code read | there is no separate face/eye model; pose runs once per check | not applicable |

Fixes kept: threaded loop with newest-frame-only camera thread and an AI worker with queue size 1 (round 1);
whole-picture pose/objects, far tiles every 3rd check, YOLO-World only on wrist/head close-ups every 2nd check
and the whole picture every 4th, warm-up of every input size (round 2c).
Not possible here: GPU (driver), 30 fps display (camera sends 4).

## Bug 2: TRACKING (too many people, forgotten people, changed numbers)

| Rank | Suspect | Experiment | Result | Verdict |
|---|---|---|---|---|
| 1 | No motion model / IoU-only matching between slow checks | baseline: 29 numbers for 6 people | each slow check moved people too far for IoU 0.3 | **confirmed**; flow tracker + 2-pass matching fixed most (29 -> 6-9) |
| 2 | New tracks numbered after one frame | baseline code `rules/brain.py:99-132` | every unmatched box got a number at once | **confirmed**; now 3 sightings |
| 3 | Short lifetime / no re-linking | baseline: 0 of 3 returns got their number back | numbers retired after 10 s | **confirmed**; 45 s lost pool + re-link: 0-2 of 3 |
| 4 | Body parts boxed as extra people | per-number lifetimes in round 1/2 runs | arm / lower-body boxes inside the near person got numbers | **confirmed**; filter tried (3b, 3c) but net results not better -> dropped |
| 5 | Re-link colours compare the wrong area | offline test, seated vs walking vs a stranger | middle-of-box colours: user walking 0.53 vs stranger 0.55 (no separation); shirt-only: user 0.68-0.91 vs stranger 0.14 | **confirmed** but the replay with it (3a, 3c) was not better overall -> dropped after 3 attempts |
| 6 | Far people dropped between far-tile checks | round 2a/2b: far recall 0-2% | far people were "missed" on checks that never looked at them | **confirmed**, fixed in 2c (far recall 9.5%) |
| 7 | Duplicate boxes from weak NMS / separate face and body people | code read | only one person model; duplicates were body parts (rank 4) | ruled out |

What still fails: ID switches 0.62-1.55 per person-minute on the overlap section (target 0.5) and 0-2 of 3
returns kept their number (target 90%). The spread between identical runs is as large as the changes tried.

## Bugs 3 and 4: DEVICES (watch false positives, partial phones)

| Rank | Suspect | Experiment | Result | Verdict |
|---|---|---|---|---|
| 1 | YOLO-World "smartwatch" on wrists, sleeves, hands | per-frame device boxes in round 1 | false watch boxes on the near person's other wrist / hand / phone area | **confirmed** |
| 2 | No temporal confirmation | baseline: any single-check box counted | 30 false devices per 10 min | **confirmed**; 2-of-3 confirmation + watch size/shape/wrist checks: 18 per 10 min |
| 3 | Leftover region gating from desk boxes | search for desk/region/CROP in detector | no desk-box code left, but tiles mode cut off the top 20% and bottom 15% (`CROP_TOP/CROP_BOTTOM`) | **confirmed**; fast mode searches the whole picture |
| 4 | Detector never trained on half-hidden phones | recall by visibility | full 67%, half-hidden 25%, edge 0% (n = 9 / 4 / 1) | **confirmed**; wrist close-ups help full phones; no fine-tuning done (time) |

## Bug 5: SCORE

Cause: `rules/brain.py` computed `score = looks + 2 x long looks + 5 x phone alerts` on every frame; app,
dashboard and README showed it. Fix: removed everywhere (search for "score" finds nothing in app, ui, rules,
tests, settings, vision, README); `alert_count` per person, +1 per alert event (the cooldown already stops one
event counting twice), `merge_people` adds counts when two numbers merge; shown on screen, dashboard and in the
session summary (per person + total). 4 new unit tests.

## Bug 6: LIMBS

Cause: the pose model already gave 17 body points per person, but they were never bound to the person's number,
smoothed, or drawn (only the 5 face points were drawn). Fix: the tracker binds the points to the number, smooths
them with a One Euro filter, moves them with the box between checks; the app draws shoulders-elbows-wrists,
shoulders-hips, hips-knees-ankles in the person's own colour. Upper-body point dropout 16-17% (was 42% in round 1).
