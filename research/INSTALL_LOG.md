# Install and download log

Everything below lives inside `C:\Users\vobic\ARGOS-vision` (venv `.venv-sandbox\`, models `models\`, both gitignored).
Python: 3.14.6 (the only Python on this machine), venv created with `py -3.14 -m venv .venv-sandbox`.

## pip packages (PyPI only, pinned)

| Package | Version | License | Size (approx) | Source | Why |
|---|---|---|---|---|---|
| pip (upgrade) | latest at install | MIT | 2 MB | https://pypi.org/project/pip/ | venv bootstrap |
| ultralytics | 8.4.175 | AGPL-3.0 | 1 MB | https://pypi.org/project/ultralytics/ | already the project's detector library (requirements.txt) |
| torch | 2.14.1 (CPU wheel) | BSD-3-Clause | ~120 MB | https://pypi.org/project/torch/ | ultralytics dependency |
| torchvision | 0.29.1 | BSD-3-Clause | ~4 MB | https://pypi.org/project/torchvision/ | ultralytics dependency |
| opencv-python | 5.0.0.93 | Apache-2.0 | ~40 MB | https://pypi.org/project/opencv-python/ | ultralytics dependency, camera |
| numpy | 2.5.4 | BSD-3-Clause | ~13 MB | https://pypi.org/project/numpy/ | dependency |
| (other ultralytics deps) | matplotlib 3.11.2, pillow 12.3.0, polars 2.0.0, psutil 7.2.2, pyyaml 6.0.3, requests 2.34.2 | permissive (BSD/MIT/Apache/PSF) | ~60 MB total | PyPI | pulled in by ultralytics |
| ffmpeg-binaries | 1.1.0 | MIT wrapper; bundled FFmpeg is LGPL/GPL | 91 MB wheel | https://pypi.org/project/ffmpeg-binaries/ | ffmpeg + ffprobe for the watch skill's local engine (tool only, not shipped) |
| patool | 4.1.0 | GPL-3.0 | <1 MB | PyPI | dependency of ffmpeg-binaries |
| yt-dlp | 2026.8.19 | Unlicense | 3 MB | https://pypi.org/project/yt-dlp/ | watch skill video download |
| transformers | 5.19.0 | Apache-2.0 | 12 MB | https://pypi.org/project/transformers/ | runs the labeling helper model (RT-DETRv2) |
| (transformers deps) | huggingface-hub 2.2.0, tokenizers 0.23.3, safetensors 0.8.0, regex, tqdm, typer, rich, hf-xet 1.7.0 | Apache-2.0 / MIT | ~20 MB | PyPI | dependency |
| openvino | 2026.4.1 | Apache-2.0 | ~40 MB | https://pypi.org/project/openvino/ | faster CPU inference (plan item 1) |
| openvino-telemetry | 2025.2.0 | Apache-2.0 | <1 MB | PyPI | pulled in by openvino; **opted out** with `opt_in_out --opt_out` right after install |
| lap | 0.5.13 | BSD-2-Clause | <1 MB | https://pypi.org/project/lap/ | linear assignment for the ByteTrack tracker |

## Models (all under 500 MB)

| Model | Repo / ID | License | Size | Source | Note |
|---|---|---|---|---|---|
| yolo11n-pose.pt | Ultralytics YOLO11 | AGPL-3.0 | 6.3 MB | copied from the main checkout (read only), originally github.com/ultralytics/assets | existing project model, needed for the baseline; see DECISIONS D5 |
| yolo11n.pt | Ultralytics YOLO11 | AGPL-3.0 | 5.6 MB | same | existing project model |
| yolov8s-worldv2.pt | Ultralytics YOLO-World v2 | AGPL-3.0 | 25.9 MB | same | existing project model |
| RT-DETRv2 R50 | `PekingU/rtdetr_v2_r50vd` | Apache-2.0 | 172 MB | https://huggingface.co/PekingU/rtdetr_v2_r50vd | labeling helper only (ground-truth boxes), not used by the app |

## Not installed
- Nothing into the main `.venv` or system Python.
- WhisperX / cloud transcription for the watch skill: not needed, every video had captions.

## Added later
| flask | 3.1.2 | BSD-3-Clause | ~1 MB (+ werkzeug, jinja2, itsdangerous, blinker) | https://pypi.org/project/flask/ | needed by `ui/dashboard.py` so `tests/test_main.py` can import |

## Run 2 (fix/fps-ids-limbs)
| Package | Version | License | Size | Source | Why / outcome |
|---|---|---|---|---|---|
| opencv-python | 4.12.0.88 | Apache-2.0 | ~40 MB | https://pypi.org/project/opencv-python/ | camera MJPG test; did not help, **uninstalled**, back to 5.0.0.93 |
| openvino | 2025.4.0 | Apache-2.0 | ~40 MB | https://pypi.org/project/openvino/ | GPU driver test; also failed, **uninstalled**, back to 2026.4.1 |
| (model) yolov8s-worldv2 OpenVINO copy | re-exported locally | AGPL-3.0 (existing model) | 49 MB | made from the existing .pt | added the words "pen" and "sheet of paper" |
No new models downloaded in run 2.
