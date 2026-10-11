# Times each model on one frame of the classroom clip, on each device OpenVINO offers
# (CPU, and the Intel GPU), for full-frame vs 3-tile input. Prints median ms over 10 runs.
#
#   .\.venv-sandbox\Scripts\python.exe perf\bench_models.py

import os
import sys
import time
import statistics
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO

CLIP = "perf/clip2/classroom2.avi"
MODELS = [("pose", "models/yolo11n-pose_openvino_model", "pose"),
          ("objects", "models/yolo11n_openvino_model", "detect"),
          ("world", "models/yolov8s-worldv2_openvino_model", "detect")]
RUNS = 10


def a_frame():
    cap = cv2.VideoCapture(CLIP)
    for i in range(300):
        ok, f = cap.read()
    return f


def tiles3(frame):
    h, w = frame.shape[:2]
    band = frame[int(h * 0.2):int(h * 0.85)]
    tw = w // 3
    extra = int(tw * 0.15)
    return [np.ascontiguousarray(band[:, max(0, i * tw - extra):min(w, (i + 1) * tw + extra)]) for i in range(3)]


def bench(model, images, device):
    model.predict(images, imgsz=640, device=device, verbose=False)   # warm-up / compile
    ms = []
    for i in range(RUNS):
        t0 = time.perf_counter()
        model.predict(images, imgsz=640, device=device, verbose=False)
        ms.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ms)


def main():
    frame = a_frame()
    for device in ["intel:cpu", "intel:gpu"]:
        for name, path, task in MODELS:
            try:
                model = YOLO(path, task=task)
                full = bench(model, frame, device)
                tiled = bench(model, tiles3(frame), device)
                print(device, name.ljust(8), "full frame", round(full), "ms | 3 tiles", round(tiled), "ms", flush=True)
            except Exception as e:
                print(device, name, "FAILED:", str(e)[:160], flush=True)


if __name__ == "__main__":
    main()
