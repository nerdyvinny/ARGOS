# PLAN: classroom vision for ARGOS

## What the code does today (origin/main 95f7811)

| Part | Where | What it does | Weak spot for a classroom |
|---|---|---|---|
| Models | `vision/detector.py:43-51` | `yolo11n-pose` (people + 17 points), `yolo11n` (COCO phone only, class 67), `yolov8s-worldv2` (open words: smartwatch, earbuds, headphones, name tag, id card) | no pen, paper; PyTorch on CPU is slow |
| Tiles | `vision/detector.py:54-81`, config `TILES=3`, `CROP_TOP=0.20`, `CROP_BOTTOM=0.85` | cuts a 360-style band into 3 tiles at 640 | bottom 15% is thrown away, which is exactly where desk items sit on a normal wide camera |
| Close-ups | `vision/detector.py:191-268` | YOLO-World on a square around each head and wrist | runs on every frame, all close-ups, expensive |
| Following people | `rules/brain.py:99-132` | IoU >= 0.3 between this frame's box and the last box, greedy; unseen 10 s = new number | no motion model: a fast stand-up or a crossing swaps or drops numbers |
| Item -> person | `rules/brain.py:135-156` | item centre inside the person box grown 20%, closest box centre wins | no hands; "headphones on" vs "on the desk" not separated |
| Head pose | `rules/brain.py:66-94` | nose vs eye/ear x positions = left/right turn number | no pitch, no eyes/gaze, drops when face points < 0.5 |
| Separate head tool | `headTracker/headTracker.py:19-75` | yaw/pitch -> angle check | not wired into the app |
| Speed | `perf/LOG.md` | 1.45-1.58 processed FPS on this laptop (i7-8650U, no GPU) | needs ~10x for 15 FPS |

## Techniques from the videos, ranked

Ranked by expected gain for the stop targets (15 FPS, recall, ID switches), then FPS cost.

| # | Technique | Video (time) | Expected gain | FPS cost | Needs |
|---|---|---|---|---|---|
| 1 | Export models to a CPU-optimised runtime (OpenVINO on this Intel CPU; the video shows NCNN 4x on ARM) | V4 06:53/08:01, V5 05:10 | 2-3x FPS, same accuracy | negative (faster) | `openvino` pip (Apache-2.0) |
| 2 | Real tracker with a motion model and a lost-track buffer (ByteTrack style; SORT predict/update) instead of IoU-only matching | V6 04:53-07:43, V8 01:01-02:05 | fewer ID switches through stand-ups, crossings, short occlusions | ~1 ms | Ultralytics' own BYTETracker (already installed) |
| 3 | Detect every Nth frame, track in between (Kalman predict + optical flow) | V6 05:47 ("if we lose detection for one or two frames the tracker keeps it"), V8 02:05 | the only way to 15 processed FPS on this CPU | big saving | OpenCV (installed) |
| 4 | Smaller full-frame input + high-res crops only where needed (far people, hands, heads) | V5 06:13-08:27 (lower res = faster but loses distance) | FPS up, distance kept by the crops | crops cost per person | nothing new |
| 5 | More open-vocabulary words (pen, paper, wristwatch) | V2 01:17 + project's YOLO-World | recall for pen/paper/watch (0 today for pen/paper) | small per extra word | nothing new |
| 6 | Face landmarks for eyes/iris + head pose matrix on near people's head crops | V3 00:46-01:46 | face/eye/head dropout down; adds a gaze direction | ~10-20 ms per near face | `mediapipe` pip (Apache-2.0) + face_landmarker.task (Apache-2.0) |
| 7 | Fine-tune a small model on my own labeled frames | V1, V2 13:24, V7 05:51 | class recall on this room | training time on CPU; test leakage risk | labeled frames from a separate part of the clip |

Not used: SAHI (not in the videos), DeepSORT appearance model (V6 mentions it; adds an appearance network and the rules forbid identity matching), TensorRT (no NVIDIA GPU).

## Measuring

- Test clip: `perf/clip/classroom.avi` (5 min, 1155 frames, 3.85 fps from the camera, gitignored).
- Ground truth: `research/labels/gt.json` (boxes only, no images), labeled on a sample of frames.
- Harness: `perf/eval_vision.py` replays the clip through a pipeline and reports per-class / per-distance precision and recall, ID switches per person-minute, face/eye/head/pose dropout, processed FPS (avg and 5th percentile), delay, and the alert list.
- Keep rule: accuracy up, processed FPS >= 15, and every alert that fired at baseline still fires.
