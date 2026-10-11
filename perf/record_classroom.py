# Records the 5-minute classroom test clip from the configured camera (config.SOURCE).
# Waits up to 10 minutes for a person to be in frame (checks every 30 s), then records.
# Saves raw frames (no drawings) plus the time of each frame. Never commit the clip.
#
#   .\.venv-sandbox\Scripts\python.exe perf\record_classroom.py

import os
import sys
import json
import time
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from settings import config
from capture import camera

SECONDS = 300
COUNTDOWN = 10
WAIT_LIMIT = 600
WAIT_CHECK = 30
OUT_DIR = "perf/clip"
STEPS = [
    (0, "SIT and look ahead at your desk"),
    (25, "Hold a PHONE in your hand, use it"),
    (50, "Hide the phone. Put HEADPHONES on"),
    (75, "Headphones off. Show your WATCH wrist"),
    (100, "Hold PAPER and a PEN, write"),
    (125, "TURN your head left / right, look away"),
    (150, "STAND UP, walk across the room"),
    (175, "Walk FAR from the camera, hold phone up"),
    (200, "Come back, cross in front of others / objects"),
    (225, "Hide behind something for a moment, come out"),
    (250, "LEAVE the picture"),
    (265, "COME BACK, sit, phone in hand again"),
    (285, "Put everything away, look ahead"),
]


def step_text(t):
    text = STEPS[0][1]
    for start, s in STEPS:
        if t >= start:
            text = s
    return text


def person_in_frame(model, frame):
    result = model.predict(frame, imgsz=640, conf=0.4, classes=[0], verbose=False)[0]
    return len(result.boxes) > 0


def wait_for_person(cap, model):
    start = time.time()
    while time.time() - start < WAIT_LIMIT:
        frame = None
        for i in range(5):
            ok, f = cap.read()
            if ok:
                frame = f
        if frame is not None and person_in_frame(model, frame):
            print("person in frame after", int(time.time() - start), "s", flush=True)
            return True
        print("nobody in frame yet,", int(time.time() - start), "s waited", flush=True)
        check_end = time.time() + WAIT_CHECK
        while time.time() < check_end:
            cap.read()   # keep the camera buffer fresh
    return False


def main():
    cap = camera.open_camera(config.SOURCE)
    if cap is None:
        print("STOP: configured camera could not be opened", flush=True)
        sys.exit(2)
    from ultralytics import YOLO
    model = YOLO("models/yolo11n.pt")
    os.makedirs(OUT_DIR, exist_ok=True)
    writer = None
    times = []
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not wait_for_person(cap, model):
            print("STOP: nobody in frame after 10 minutes", flush=True)
            sys.exit(3)
        start = time.time()
        while time.time() - start < COUNTDOWN:
            ok, frame = cap.read()
            if not ok:
                continue
            left = int(COUNTDOWN - (time.time() - start)) + 1
            show = cv2.resize(frame, (1280, 720))
            cv2.putText(show, "Recording in " + str(left) + " - first: " + STEPS[0][1], (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
            cv2.imshow("Classroom clip", show)
            cv2.waitKey(1)

        start = time.time()
        while time.time() - start < SECONDS:
            ok, frame = cap.read()
            if not ok:
                print("camera read failed", flush=True)
                break
            t = time.time() - start
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(OUT_DIR + "/classroom.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, (w, h))
                writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 90)
            writer.write(frame)
            times.append(t)

            show = cv2.resize(frame, (1280, 720))
            cv2.putText(show, step_text(t) + "   " + str(int(SECONDS - t)) + "s", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
            cv2.imshow("Classroom clip", show)
            cv2.waitKey(1)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    with open(OUT_DIR + "/classroom_times.json", "w") as f:
        json.dump(times, f)
    with open(OUT_DIR + "/classroom_steps.json", "w") as f:
        json.dump(STEPS, f)
    print("saved", len(times), "frames,", round(len(times) / SECONDS, 2), "fps", flush=True)


if __name__ == "__main__":
    main()
