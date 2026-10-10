# Tracking models: what we use now, and what could replace it

Research only. No app code was changed. Checked on 2026-10-10.

## Short version

- ARGOS uses **one** tracking model: `yolo11n-pose.pt` (Ultralytics, AGPL-3.0).
  It finds every person and gives 17 body points. There is no separate face
  detector, face-landmark model, or eye/gaze tracker.
- Speed on our laptop (CPU, no GPU), measured on `desks_check.jpg` (1920x1080),
  3 tiles at 640: **1.45 FPS** for pose alone, **0.75 FPS** for pose + phone model.
  The stream wasn't running, so this is a saved frame, not the live camera.
- 15 FPS is not reachable on this laptop with any model swap.
- The real weak spot: on the test frame the head-turn number worked for
  **0 of 2 people** (their face points were under the 0.5 confidence cutoff).

## What we use now

| Category | What it uses | Where | What the app does with it |
|---|---|---|---|
| Body pose | `yolo11n-pose.pt` | loaded `vision/detector.py:21`, run `:92` | body points per person, used to match people to desks |
| Face detection | none (uses the person box) | - | - |
| Head pose | nose vs eye/ear points from the pose model | `rules/brain.py:86`, used at `:204` | left/right turn number; "look away" if it moves 0.25 from that person's 2-minute normal |
| Eye / gaze | none | - | - |

## Hugging Face candidates

Every page below was opened. Info comes from each model card; `[not stated]`
means the card doesn't say. Licenses must allow a public repo. Face
recognition / identity models were left out on purpose.

### Face detection (new - we have none)

| Model | License | Format, size | CPU | Accuracy (from card) |
|---|---|---|---|---|
| [AdamCodd/YOLOv11n-face-detection](https://huggingface.co/AdamCodd/YOLOv11n-face-detection) | Apache-2.0 on card, but built on Ultralytics YOLO11 (AGPL, same as now) | ONNX 10.6 MB, PyTorch 5.5 MB | yes | WIDER FACE val AP: 0.94 easy, 0.92 medium, 0.81 hard |
| [qualcomm/Lightweight-Face-Detection](https://huggingface.co/qualcomm/Lightweight-Face-Detection) | BSD-3 | ONNX, 480x640; weights on Qualcomm AI Hub | [not stated] | [not stated] |

### Head pose / face landmarks

| Model | License | Format, size | CPU | Accuracy |
|---|---|---|---|---|
| [litert-community/6DRepNet-HeadPose-LiteRT](https://huggingface.co/litert-community/6DRepNet-HeadPose-LiteRT) | MIT | TFLite 157 MB, needs a face crop | yes, 155 ms on Raspberry Pi 5 CPU | [not stated] |
| [litert-community/RTMPose-Face-WFLW-LiteRT](https://huggingface.co/litert-community/RTMPose-Face-WFLW-LiteRT) | Apache-2.0 (training data terms [not stated]) | TFLite 33.6 MB, 98 points | yes, 71.6 ms on Pi 5 CPU | [not stated] |

### Eye / gaze - nothing good found

[litert-community/L2CS-Gaze360-LiteRT](https://huggingface.co/litert-community/L2CS-Gaze360-LiteRT)
(MIT) takes 541.7 ms per face on CPU and needs a close 448x448 face crop.
People in a 360 room shot are far too small for that.

### Body pose (vs current YOLO11n-pose)

| Model | License | Format | CPU | Accuracy |
|---|---|---|---|---|
| [qualcomm/HRNetPose](https://huggingface.co/qualcomm/HRNetPose) | MIT | ONNX, 256x192 | [not stated] | [not stated] |
| [mlboydaisuke/RTMPose-s-Body-ExecuTorch](https://huggingface.co/mlboydaisuke/RTMPose-s-Body-ExecuTorch) | Apache-2.0 | ExecuTorch 21.9 MB | yes (needs `executorch`) | [not stated] |

Both do one person per run, so they need a person finder first and get slower
with every extra person. YOLO11n-pose does everyone in one pass, so it stays
the baseline.

### Left out
- `iitolstykh/YOLO-Face-Person-Detector`: AGPL and a 137 MB large model, too slow on CPU.
- Qualcomm MediaPipe face/pose: built for one close-up person at 256x256.
- RTMPose-s TFLite: its own card says the CPU backend refused it.

## Next step (not done yet)

Benchmark `AdamCodd/YOLOv11n-face-detection` + `6DRepNet-HeadPose-LiteRT`
(face box -> head angle) on the live stream. Needs the `ai-edge-litert`
package and the RTMP stream running. Expected cost: about 150 ms extra per face.
