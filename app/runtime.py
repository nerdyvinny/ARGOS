# The live loop, split into 3 parts so a slow AI can't make the picture lag:
#
#   camera thread   -> keeps only the NEWEST picture (old ones are dropped, never queued)
#   AI worker       -> takes the newest picture when it is free (queue size 1), runs the models
#   main loop       -> every new picture: move everyone with the tracker (cheap), apply the
#                      AI's latest result when one is ready, run the rules, draw, show
#
# The same loop replays a recorded clip (ClipSource) at its real speed, so the numbers
# from a replay are what the live camera would give.

import threading
import time
import cv2

from settings import config
from app.trace import Timer


class CameraSource(threading.Thread):
    # reads the camera as fast as it delivers; only the newest picture is kept
    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap = cap
        self.cond = threading.Condition()
        self.item = None
        self.ended = False
        self.running = True
        self.start_time = time.time()

    def run(self):
        n = 0
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                break
            with self.cond:
                self.item = (n, time.time() - self.start_time, frame, 0.0)
                self.cond.notify_all()
            n += 1
        with self.cond:
            self.ended = True
            self.cond.notify_all()

    def latest(self, after, timeout=1.0):
        with self.cond:
            if self.item is None or self.item[0] <= after:
                self.cond.wait(timeout)
            if self.item is None or self.item[0] <= after:
                return None
            return self.item

    def stop(self):
        self.running = False


class ClipSource(CameraSource):
    # plays a recorded clip at the times it was recorded (a fake camera)
    def __init__(self, path, times):
        super().__init__(None)
        self.path = path
        self.times = times

    def run(self):
        cap = cv2.VideoCapture(self.path)
        self.start_time = time.time()
        for n, t in enumerate(self.times):
            if not self.running:
                break
            wait = self.start_time + t - time.time()
            if wait > 0:
                time.sleep(wait)
            t0 = time.perf_counter()
            ok, frame = cap.read()
            decode_ms = (time.perf_counter() - t0) * 1000
            if not ok:
                break
            with self.cond:
                self.item = (n, t, frame, decode_ms)
                self.cond.notify_all()
        cap.release()
        with self.cond:
            self.ended = True
            self.cond.notify_all()


class DetectWorker(threading.Thread):
    # runs the AI on the newest picture it is given; a newer picture replaces a waiting one
    def __init__(self, detect_fn):
        super().__init__(daemon=True)
        self.detect_fn = detect_fn
        self.cond = threading.Condition()
        self.request = None
        self.result = None
        self.busy = False
        self.running = True

    def submit(self, item):
        with self.cond:
            self.request = item
            self.cond.notify_all()

    def run(self):
        while self.running:
            with self.cond:
                while self.request is None and self.running:
                    self.cond.wait(0.5)
                if not self.running:
                    break
                item = self.request
                self.request = None
                self.busy = True
            stages = {}
            people, items = self.detect_fn(item[2], stages)
            with self.cond:
                self.result = (item[0], item[1], people, items, stages)
                self.busy = False

    def take(self):
        with self.cond:
            r = self.result
            self.result = None
            return r

    def waiting(self):
        with self.cond:
            return int(self.request is not None) + int(self.busy)

    def stop(self):
        self.running = False


def run(source, detector, brain, tracker, devices, trace=None, show=None, on_frame=None, on_alert=None):
    # source: CameraSource or ClipSource (already started). show(frame, people_states, matches, items, stages)
    # draws and displays and gives back False to stop. on_frame(record) gets every shown frame.
    worker = DetectWorker(detector.detect_raw)
    worker.start()
    people_states = []
    last = -1
    session_start = time.time()
    try:
        while True:
            item = source.latest(last)
            if item is None:
                if source.ended:
                    break
                continue
            index, t_capture, frame, decode_ms = item
            last = index
            now = t_capture
            stages = {"decode": decode_ms}
            if not worker.busy:
                worker.submit(item)
            with Timer(stages, "tracker"):
                tracker.predict(frame, now)
                result = worker.take()
                fresh = result is not None
                if fresh:
                    r_index, r_time, found_people, found_items, det_stages = result
                    stages.update(det_stages)
                    tracker.update(found_people, now)
                    devices.update(found_items, tracker.current(), now)
                for kept, gone in tracker.take_merges():
                    brain.merge_people(people_states, "Person " + str(kept), "Person " + str(gone))
                    if trace is not None:
                        trace.event("number_changed", kept=kept, merged=gone, reason="merge, older number kept")
            people = tracker.current()
            items = devices.current()
            with Timer(stages, "brain"):
                alerts, matches = brain.update_all(people_states, people, items, now)
            if on_alert is not None:
                for a in alerts:
                    on_alert(a, frame, people_states, now)
            keep_going = True
            if show is not None:
                with Timer(stages, "draw"):
                    keep_going = show(frame, people_states, matches, items, stages)
            t_shown = time.time() - session_start
            if trace is not None:
                trace.row(index, t_capture, t_shown, fresh, stages, worker.waiting())
            if on_frame is not None:
                on_frame({"i": index, "t": t_capture, "shown": t_shown, "fresh": fresh,
                          "loop_ms": sum(v for k, v in stages.items() if k in ("tracker", "brain", "draw")),
                          "stages": dict(stages),
                          "people": [{"id": name, "box": p["box"], "points": p.get("points", [])} for name, p in matches.items()],
                          "items": [{"label": it["label"], "box": it["box"], "kind": it.get("kind")} for it in items],
                          "alerts": [{"kind": a["kind"], "person": a["person"], "time": round(now, 2)} for a in alerts]})
            if not keep_going:
                break
    finally:
        worker.stop()
        source.stop()
        tracker.reset()   # session over: forget every person, number and clothes histogram
    return people_states
