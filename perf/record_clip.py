# TEMPORARY: records the 60 s reference clip from the real camera.
# Shows what to do on screen. Saves raw frames (no drawings) plus the time of each frame.
#
#   venv\Scripts\python.exe perf\record_clip.py

import os
import sys
import json
import time
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from settings import config
from capture import camera

SECONDS = 60
COUNTDOWN = 10
STEPS = [
    (0, "LOOK AHEAD at your desk"),
    (15, "LOOK AWAY to one side and HOLD it"),
    (30, "TURN YOUR HEAD left and right"),
    (45, "LEAVE the picture"),
]


def step_text(t):
    text = STEPS[0][1]
    for start, s in STEPS:
        if t >= start:
            text = s
    return text


def main():
    cap = camera.open_camera(config.SOURCE)
    if cap is None:
        return
    os.makedirs("perf/clip", exist_ok=True)
    writer = None
    times = []
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        start = time.time()
        while time.time() - start < COUNTDOWN:
            ok, frame = cap.read()
            left = int(COUNTDOWN - (time.time() - start)) + 1
            show = cv2.resize(frame, (1280, 720))
            cv2.putText(show, "Get in place - recording in " + str(left), (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
            cv2.imshow("Reference clip", show)
            cv2.waitKey(1)

        start = time.time()
        while time.time() - start < SECONDS:
            ok, frame = cap.read()
            if not ok:
                print("camera read failed")
                break
            t = time.time() - start
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter("perf/clip/reference.avi", cv2.VideoWriter_fourcc(*"MJPG"), 4, (w, h))
                writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)
            writer.write(frame)
            times.append(t)

            show = cv2.resize(frame, (1280, 720))
            cv2.putText(show, step_text(t) + "   " + str(int(SECONDS - t)) + "s", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.imshow("Reference clip", show)
            cv2.waitKey(1)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    with open("perf/clip/reference_times.json", "w") as f:
        json.dump(times, f)
    print("saved", len(times), "frames,", round(len(times) / SECONDS, 2), "fps")


if __name__ == "__main__":
    main()
