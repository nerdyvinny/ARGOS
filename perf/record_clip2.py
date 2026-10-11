# Records the second 5-minute classroom test clip at the camera's full speed.
# The camera is read on its own thread and frames are written on another, so slow
# video encoding can't drop the camera rate (the first clip only got 3.85 fps).
# Waits up to 10 minutes for a person (checks every 30 s). Never commit the clip.
#
#   .\.venv-sandbox\Scripts\python.exe perf\record_clip2.py

import os
import sys
import json
import time
import queue
import threading
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from settings import config
from capture import camera

SECONDS = 300
COUNTDOWN = 10
WAIT_LIMIT = 600
WAIT_CHECK = 30
OUT_DIR = "perf/clip2"
STEPS = [
    (0, "SIT, look ahead. Watch on wrist, sleeve/wristband visible"),
    (25, "Hold PHONE fully visible, use it"),
    (45, "Hold phone HALF HIDDEN by your hand / under the desk edge"),
    (65, "Hold phone at the EDGE of the picture (half cut off)"),
    (85, "Phone away. HEADPHONES on"),
    (105, "Headphones off. PAPER and PEN, write"),
    (125, "Turn head left / right, look around"),
    (145, "STAND UP, walk across, cross in front of others"),
    (170, "LEAVE the picture and stay out ~20 s"),
    (195, "COME BACK, sit in the SAME seat"),
    (215, "LEAVE again on the other side ~25 s"),
    (245, "COME BACK, sit, phone half hidden in hand"),
    (270, "Arms up, stretch, move your arms (limbs test)"),
    (288, "Put everything away, look ahead"),
]


def step_text(t):
    text = STEPS[0][1]
    for start, s in STEPS:
        if t >= start:
            text = s
    return text


class Reader(threading.Thread):
    # keeps reading the camera so frames never pile up; hands each new frame to whoever asks
    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.count = 0
        self.running = True

    def run(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = f
                    self.count += 1

    def latest(self):
        with self.lock:
            return self.frame, self.count


def person_in_frame(model, frame):
    result = model.predict(frame, imgsz=640, conf=0.4, classes=[0], verbose=False)[0]
    return len(result.boxes) > 0


def wait_for_person(reader, model):
    start = time.time()
    while time.time() - start < WAIT_LIMIT:
        frame, _ = reader.latest()
        if frame is not None and person_in_frame(model, frame):
            print("person in frame after", int(time.time() - start), "s", flush=True)
            return True
        print("nobody in frame yet,", int(time.time() - start), "s waited", flush=True)
        time.sleep(WAIT_CHECK)
    return False


def writer_loop(q, path, size, fps):
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    out.set(cv2.VIDEOWRITER_PROP_QUALITY, 85)
    while True:
        f = q.get()
        if f is None:
            break
        out.write(f)
    out.release()


def main():
    cap = camera.open_camera(config.SOURCE)
    if cap is None:
        print("STOP: configured camera could not be opened", flush=True)
        sys.exit(2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    reader = Reader(cap)
    reader.start()
    from ultralytics import YOLO
    model = YOLO("models/yolo11n.pt")
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        if not wait_for_person(reader, model):
            print("STOP: nobody in frame after 10 minutes", flush=True)
            sys.exit(3)
        del model
        start = time.time()
        while time.time() - start < COUNTDOWN:
            frame, _ = reader.latest()
            if frame is not None:
                show = cv2.resize(frame, (960, 540))
                cv2.putText(show, "Recording in " + str(int(COUNTDOWN - (time.time() - start)) + 1) + ": " + STEPS[0][1],
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow("Classroom clip 2", show)
            cv2.waitKey(30)

        frame, last_count = reader.latest()
        h, w = frame.shape[:2]
        q = queue.Queue(maxsize=240)
        wt = threading.Thread(target=writer_loop, args=(q, OUT_DIR + "/classroom2.avi", (w, h), config.CAMERA_FPS))
        wt.start()
        times = []
        dropped = 0
        start = time.time()
        shown = 0
        while time.time() - start < SECONDS:
            frame, count = reader.latest()
            if count == last_count:
                time.sleep(0.002)
                continue
            last_count = count
            t = time.time() - start
            try:
                q.put_nowait(frame)
                times.append(t)
            except queue.Full:
                dropped += 1
            shown += 1
            if shown % 3 == 0:   # preview at ~10 fps, small, so it costs little
                show = cv2.resize(frame, (960, 540))
                cv2.putText(show, step_text(t) + "  " + str(int(SECONDS - t)) + "s", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Classroom clip 2", show)
                cv2.waitKey(1)
        q.put(None)
        wt.join()
    finally:
        reader.running = False
        time.sleep(0.1)
        cap.release()
        cv2.destroyAllWindows()

    json.dump(times, open(OUT_DIR + "/classroom2_times.json", "w"))
    json.dump(STEPS, open(OUT_DIR + "/classroom2_steps.json", "w"))
    print("saved", len(times), "frames,", round(len(times) / SECONDS, 2), "fps, writer-full drops", dropped, flush=True)


if __name__ == "__main__":
    main()
