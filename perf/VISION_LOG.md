# VISION_LOG: classroom vision rounds

Test: `perf/clip/classroom.avi` (5 min, 1155 frames recorded at 3.85 fps, real classroom, 1 near person + 2-5 far people).
Ground truth: 27 hand-checked frames (93 people, 89 items) + 97 frames for ID switches (`research/labels/gt.json`).
Harness: `perf/eval_vision.py`, **real-time replay** (the clip plays at recorded speed; a slow program drops frames like the live camera; each frame is scored on the result that was on screen at that moment).
Laptop: i7-8650U, 4 cores, no GPU. Processed FPS = frames / processing time; p5 = 1000 / 95th-percentile frame ms.
Keep rule: accuracy up AND processed FPS >= 15 AND every baseline alert still fires.

| Round | Change | Video | FPS avg / p5 | Delay ms avg / p95 | Person P / R | Item recall (phone, watch, headph., paper, pen) | ID sw / person-min (overlap) | Dropout face / eyes / head / pose | Alerts | Result |
|---|---|---|---|---|---|---|---|---|---|---|
