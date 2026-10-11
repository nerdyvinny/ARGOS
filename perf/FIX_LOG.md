# FIX_LOG: branch fix/fps-ids-limbs

Test: `perf/clip2/classroom2.avi` (new 5-min clip, 1154 frames; the camera only delivers 3.85 fps, see FIX_PLAN).
Ground truth: 25 hand-checked frames (65 people, 57 items, devices tagged full / half / edge) + 97 frames for
ID switches; true headcount 6 (`research/labels2/gt.json`).
Harness: `perf/replay_live.py` plays the clip at recorded speed through the real threaded app loop;
`perf/score2.py` scores it. Baseline (round 0) used the old single loop via `perf/eval_vision.py`.

Columns: display FPS | AI checks per s | loop capacity (avg / p5 fps) | delay ms (avg / p95) |
people P / R | device P / R (>= half visible) | false devices per 10 min | numbers per real person |
overlap ID switches per person-min | same number after leaving | upper-body point dropout.

| Round | Change | Bug | Display | AI/s | Capacity | Delay | People P/R | Device P/R (>=half) | FP/10min | Num/person | IDSW | Return | Limb drop | Result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | Baseline (feat/classroom-vision, single loop) | - | 0.2 | 0.2 | 0.2 / 0.11 | 4931 / 9191 | 57% / 26% | 6% / 7% (8%) | 30 | 4.83 | 1.55 | 0/3 | n/a | baseline |
| 1 | Threaded loop (camera thread, AI worker, queue 1) + new tracker + alert_count | 1, 2, 5 | 3.85 | 0.31 | 22.4 / 14.6 | 66 / 92 | 55% / 26% | 36% / 36% (38%) | 18 | 1.00 | 0.62 | 1/3 | 42% | **kept** 526917e |
| 2a | Fast detection: whole picture + close-ups, far tiles every 3rd check, YOLO-World every 2nd/4th | 1 | 3.85 | 0.93 | 24.5 / 15.7 | 62 / 85 | 80% / 31% | 43% / 21% (23%) | 8 | 1.50 | 0.62 | 1/3 | 20% | reverted (device recall, numbers worse) |
| 2b | 2a + reuse YOLO-World results between runs | 1 | 3.85 | 0.95 | 24.3 / 16.8 | 62 / 82 | 91% / 29% | 33% / 43% (46%) | 24 | 1.17 | 0.93 | 0/3 | 21% | reverted (precision, tracking worse) |
| 2c | 2b + far people not counted missed on checks that skip the far band; 8 s to confirm | 1, 2 | 3.85 | 0.96 | 21.1 / 12.4 | 69 / 102 | 80% / 37% | 40% / 43% (46%) | 18 | 1.17 | 0.93 | 2/3 | 16% | **kept with exception** 4450f25 (see DECISIONS D-R2) |
| 3a | Shirt-only colours for re-linking + unsure detections may revive | 2 | 3.85 | 0.93 | 19.2 / 11.7 | 75 / 110 | 72% / 35% | 37% / 50% (54%) | 24 | 1.67 | 0.93 | 0/3 | 19% | reverted |
| 3b | Drop body-part boxes (85% inside a bigger person) + unsure revive | 2 | 3.85 | 1.16 | 19.6 / 11.7 | 75 / 113 | 71% / 34% | 37% / 50% (54%) | 24 | 1.17 | 0.62 | 1/3 | 18% | reverted |
| 3c | Shirt colours + body-part filter (no unsure revive) | 2 | 3.85 | 1.12 | 18.4 / 10.8 | 78 / 121 | 75% / 37% | 44% / 50% (54%) | 18 | 1.67 | 0.93 | 2/3 | 17% | reverted; **idea dropped after 3 failures** |
| 4 | Device confirmation (2 of last 3 checks) + watch size/shape/wrist checks + "pen", "sheet of paper" words | 3, 4 | 3.85 | 1.10 | 19.0 / 10.3 | 76 / 123 | 80% / 37% | 44% / 50% (54%) | 18 | 1.17 | 1.24* | 0/3* | 17% | **kept** f3ea5a2 (pen recall 0% -> 39%) |
| 5 | Skeleton drawn per person colour; tracker unit tests | 6 | 3.85 | 1.11 | 19.5 / 11.4 | 74 / 113 | 80% / 37% | 41% / 50% (54%) | 20 | 1.50* | 1.55* | 1/3* | 17% | **kept** 2e97590 |

\* Tracker code identical to round 2c. The spread (numbers 1.17-1.67, ID switches 0.62-1.55, returns 0/3-2/3)
across rounds 2c-5 is run-to-run noise: the AI worker's timing differs each run, and the overlap section has
only ~3.2 person-minutes and 3 returns, so one event moves the number a lot.

Unit tests: 38/38 pass (32 before + 6 new tracker tests).
