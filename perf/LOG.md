# perf/fps-15 log

Camera: Arducam USB (config SOURCE = 1), 1920x1080. Laptop: i7-8650U, 4 cores, no GPU.
Each row: 60 s on the real camera. Processed p5 = 5th percentile = 1000 / (95th percentile frame ms).
Clip check: 60 s reference clip (231 frames), baseline alerts = phone @ 1.05 s, long_look @ 37.95 s (Desk 1).

| Round | Change | Camera FPS | Processed FPS avg | Processed FPS p5 | Clip check | Tests | Result |
|---|---|---|---|---|---|---|---|
| 0 | Baseline (origin/main) | 3.86 | 1.54 | 1.04 | (baseline) | 16/16 | - |
| 1 | Ask for MJPG 1920x1080 @ 30 while the camera opens | 30.36 | 1.45 | 1.06 | PASS | 17/17 | kept |
