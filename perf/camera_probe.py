# Short camera check (diagnosis only): which pixel format the camera really sends and how many
# frames per second arrive, for a few ways of opening it. Uses ONLY config.SOURCE.
#
#   .\.venv-sandbox\Scripts\python.exe perf\camera_probe.py

import os
import sys
import time
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from settings import config
from capture import camera

SECONDS = 4


def fourcc_text(cap):
    v = int(cap.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))


def measure(cap, label):
    for i in range(5):
        cap.read()
    n = 0
    start = time.time()
    shape = None
    while time.time() - start < SECONDS:
        ok, f = cap.read()
        if ok:
            n += 1
            shape = f.shape
    print(label, "| format", fourcc_text(cap), "| size", shape, "| fps", round(n / SECONDS, 2),
          "| reported fps", cap.get(cv2.CAP_PROP_FPS), flush=True)


def main():
    if config.SOURCE == 0:
        print("STOP: config.SOURCE is the laptop camera")
        return
    # 1) exactly how the app opens it today
    cap = camera.open_camera(config.SOURCE)
    if cap is None:
        print("STOP: configured camera could not be opened")
        return
    measure(cap, "A: app's open_camera (MJPG params at open)")
    cap.release()

    # 2) open plainly, then set MJPG + size + fps one by one (order matters on some DirectShow drivers)
    cap = cv2.VideoCapture(config.SOURCE, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
    measure(cap, "B: DSHOW, set FOURCC first then size")
    cap.release()
    # (No Media Foundation test: it can number cameras differently, so its "camera 1"
    #  might be the laptop camera, which we must never open.)


if __name__ == "__main__":
    main()
