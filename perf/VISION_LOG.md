# VISION_LOG: classroom vision rounds

Test: `perf/clip/classroom.avi` (5 min, 1155 frames recorded at 3.85 fps, real classroom, 1 near person + 2-5 far people).
Ground truth: 27 hand-checked frames (93 people, 89 items) + 97 frames for ID switches (`research/labels/gt.json`).
Harness: `perf/eval_vision.py`, **real-time replay** (the clip plays at recorded speed; a slow program drops frames like the live camera; each frame is scored on the result that was on screen at that moment).
Laptop: i7-8650U, 4 cores, no GPU. Processed FPS = frames / processing time; p5 = 1000 / 95th-percentile frame ms.
Keep rule: accuracy up AND processed FPS >= 15 AND every baseline alert still fires.

| Round | Change | Video | FPS avg / p5 | Delay ms avg / p95 | Person P / R | Item recall (phone, watch, headph., paper, pen) | ID sw / person-min (overlap) | Dropout face / eyes / head / pose | Alerts | Result |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | Baseline (origin/main 95f7811, .pt models, IoU matching) | - | 0.25 / 0.12 | 3960 / 8065 | 51.0% / 26.9% | 33.3%, 21.4%, 0%, 0%, 0% | 0.77 (1.17) | 36.8% / 36.8% / 36.8% / 45.5% | none | baseline |
| 1 | OpenVINO copies of all 3 models (`vision/detector.py:26-70`, `tools/export_openvino.py`) | V4 06:53/08:01, V5 05:10 | 0.33 / 0.22 | 3030 / 4502 | 60.5% / 28.0% | 53.3%, 28.6%, 7.1%, 0%, 0% | 0.77 (0.87) | 36.8% / 36.8% / 36.8% / 40.9% | phone @300.1 s | **kept** (accuracy up, FPS up; see D11) - commit ec40882 |
| 2a | Tracker (optical flow + ByteTrack-style 2-pass matching + lost buffer) and AI on every 4th frame | V6 04:53-07:43, V8 01:01-02:05 | 1.22 / 0.29 | 819 / 3394 | - / 0% | 0% all | 0 (0) | 100% all | none | bug: a 4.6 s AI check made every track look >1 s lost, so nobody was shown. Fixed, re-ran as 2b |
| 2b | Same, tracks kept until the next AI check misses them; models warmed up at load | V6, V8 | 1.32 / 0.31 | 757 / 3200 | 63.3% / 33.3% | 0% all (items only on AI-check frames) | 0.58 (0.73) | 21.1% / 26.3% / 21.1% / 31.8% | none (round 1's phone alert lost) | **reverted**: item recall fell to 0 and the round-1 phone alert stopped firing. Saved as `research/round2_tracker.patch` |

Stopped after round 2b: 3-hour limit reached (started ~23:05, stopped ~02:05).
Unit tests on the kept code: `unittest discover -s tests` 28/28 OK on 3 back-to-back runs (after installing flask into the sandbox; it was missing, not a code fault).
