# perf/fps-15 log

Camera: Arducam USB (config SOURCE = 1), 1920x1080. Laptop: i7-8650U, 4 cores, no GPU.
Each row: 60 s on the real camera. Processed p5 = 5th percentile = 1000 / (95th percentile frame ms).
Clip check: 60 s reference clip (231 frames), baseline alerts = phone @ 1.05 s, long_look @ 37.95 s (Desk 1).

| Round | Change | Camera FPS | Processed FPS avg | Processed FPS p5 | Clip check | Tests | Result |
|---|---|---|---|---|---|---|---|
| 0 | Baseline (origin/main) | 3.86 | 1.54 | 1.04 | (baseline) | 16/16 | - |
| 1 | Ask for MJPG 1920x1080 @ 30 while the camera opens | 30.36 | 1.45 | 1.06 | PASS | 17/17 | kept |
| 2 | Camera read on its own thread (newest picture only) + AI on a worker thread; screen redrawn on every camera picture | 30 | AI 0.95 (p5 0.64) / screen 27.72 (p5 22.93) | | PASS | 20/20 | reverted: AI checks fell 35% |
| 3 | Round 2 + screen redrawn at most 20/s, dashboard picture at most 10/s | 30 | AI 1.00 (p5 0.81) / screen 14.15 (p5 12.14) | | PASS | 20/20 | reverted: AI checks fell 31% |
| 4 | Round 3 + reader thread only grabs; a picture is unpacked only when the AI or screen asks for it | 30 | AI 0.97 (p5 0.87) / screen 14.92 (p5 13.40) | | PASS | 17/20 (test fake lacked retrieve) | reverted: AI checks fell 33% |
| final | Confirmation of kept code (round 1) | 30.57 | 1.58 | 1.28 | PASS | 17/17 | - |

## Why it stopped (round 4, 19:27, 37 min in)
Stop rule hit: the remaining options all need an install, a model swap or a threshold change.
- The two AI models take ~630 ms per picture on this 4-core CPU (95% of the frame). 15 processed FPS needs ~10x faster AI.
- Not yet tried, and each one breaks a rule: OpenVINO/ONNX (not installed), smaller YOLO input or fewer tiles (changes detection),
  phone model every 2nd picture (phone rule needs 3 sightings in 4 s, so it would miss alerts).
- Showing the video at 15+ FPS works (rounds 3-4), but any second thread on this laptop cuts the AI's checks by about a third.
