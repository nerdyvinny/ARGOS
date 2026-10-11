# The main program. Run it during the test:
#
#   python -m app.main              (uses SOURCE from config.py = the USB camera)
#   python -m app.main test.mp4     (use a video file - great for practicing)
#
# Press Q in the video window to stop.

import os
import sys
import csv
import time
import threading
import webbrowser
import cv2
import numpy as np
from settings import config
from capture import camera
from rules import brain
from vision import detector
from ui import dashboard
from app import clips

COLORS = {
    "green": (0, 200, 0),
    "yellow": (0, 220, 255),
    "red": (0, 0, 255),
    "empty": (120, 120, 120),
}


# limbs (COCO body points): shoulders-elbows-wrists, shoulders-hips, hips-knees-ankles
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)]
SCREEN_WIDTH = 1280   # the window and the web page show the picture this wide
DASHBOARD_FPS = 10    # pictures per second sent to the web page
PERSON_COLORS = [(255, 128, 0), (0, 200, 255), (200, 0, 255), (0, 255, 128), (255, 0, 128),
                 (128, 255, 0), (0, 128, 255), (255, 255, 0)]


def person_color(name):
    # each person number always gets the same colour
    digits = "".join(c for c in name if c.isdigit())
    return PERSON_COLORS[int(digits or 0) % len(PERSON_COLORS)]


def get_source():
    if len(sys.argv) > 1:
        s = sys.argv[1]
        if s.isdigit():
            return int(s)
        return s
    return config.SOURCE


def exam_time_text(seconds):
    seconds = int(seconds)
    return str(seconds // 60).zfill(2) + ":" + str(seconds % 60).zfill(2)


def draw(frame, people_states, matches, phones, width=None):
    # width: shrink the picture to this width FIRST and draw on the small copy
    # (drawing on the full 1920x1080 picture and shrinking after was half the time of every frame)
    s = 1.0
    if width is not None and frame.shape[1] > width:
        s = width / frame.shape[1]
        img = cv2.resize(frame, (width, int(frame.shape[0] * s)), interpolation=cv2.INTER_LINEAR)
    else:
        img = frame.copy()

    def p(x, y):
        return (int(x * s), int(y * s))

    for d in people_states:
        if d["name"] not in matches:
            continue   # only draw people we can see right now
        x1, y1, x2, y2 = d["box"]
        color = COLORS.get(d["status"], (255, 255, 255))
        thick = 4 if d["status"] == "red" else 2
        cv2.rectangle(img, p(x1, y1), p(x2, y2), color, thick)

        tx, ty = p(x1, y1)
        text = d["name"] + "  alerts " + str(d["alert_count"])
        cv2.putText(img, text, (tx + 5, ty + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if d["deviation"] is not None:
            turn = "turn " + str(round(d["deviation"], 2))
            cv2.putText(img, turn, (tx + 5, ty + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # body skeleton of each matched person, in that person's own colour (bound to their number)
    for name in matches:
        pts = matches[name]["points"]
        color = person_color(name)
        for a, b in SKELETON:
            if a < len(pts) and b < len(pts) and pts[a][2] >= config.KEYPOINT_CONFIDENCE and pts[b][2] >= config.KEYPOINT_CONFIDENCE:
                cv2.line(img, p(pts[a][0], pts[a][1]), p(pts[b][0], pts[b][1]), color, 3)

    # face points of each matched person (nose = red, eyes/ears = white)
    for name in matches:
        pts = matches[name]["points"]
        for k in range(min(5, len(pts))):
            if pts[k][2] >= config.KEYPOINT_CONFIDENCE:
                c = (0, 0, 255) if k == 0 else (255, 255, 255)
                cv2.circle(img, p(pts[k][0], pts[k][1]), 4, c, -1)

    for ph in phones:
        x1, y1, x2, y2 = ph["box"]
        cv2.rectangle(img, p(x1, y1), p(x2, y2), (255, 0, 255), 3)
        lx, ly = p(x1, y1)
        cv2.putText(img, ph.get("label", "phone").upper(), (lx, ly - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    return img


def resize_to_width(img, width):
    h, w = img.shape[:2]
    if w <= width:
        return img
    scale = width / w
    return cv2.resize(img, (width, int(h * scale)))


def save_evidence(drawn_frame, person_state, alert):
    # save a close up of the person (a bit bigger than their box)
    h, w = drawn_frame.shape[:2]
    x1, y1, x2, y2 = person_state["box"]
    pad_x = (x2 - x1) * 0.5
    pad_y = (y2 - y1) * 0.5
    x1 = int(max(0, x1 - pad_x))
    y1 = int(max(0, y1 - pad_y))
    x2 = int(min(w, x2 + pad_x))
    y2 = int(min(h, y2 + pad_y))
    crop = drawn_frame[y1:y2, x1:x2]
    if crop.size == 0:
        crop = drawn_frame

    name = person_state["name"].replace(" ", "_")
    file_name = name + "_" + alert["kind"] + "_" + str(int(time.time() * 1000)) + ".jpg"
    cv2.imwrite(os.path.join(config.EVIDENCE_FOLDER, file_name), crop)
    return file_name


# ---------------- alert pictures (live camera) ----------------
# The old picture was the newest camera picture when the alert fired. A device alert fires a
# few AI checks after the phone was first seen, so the phone was often already gone, and nothing
# was drawn on it. Now we pick the AI-checked picture that shows the moment, and draw on it.

EVIDENCE_HEIGHT = 720   # the saved picture is 720 high: whole room on the left, zoom on the right


def near_person(box, person_box):
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    return brain.point_in_box(cx, cy, brain.grow_box(person_box, 0.3))


def pick_evidence(history, alert, person_box):
    # history: the last AI-checked pictures {"t", "frame", "people", "items"} (oldest first)
    # gives back {"entry": picture, "item": device or None}, or None if there is nothing yet
    if not history:
        return None
    if alert["kind"] == "device":
        best = None
        for e in history:
            for it in e["items"]:
                if it.get("label") in brain.NOT_ALERT_ITEMS or not near_person(it["box"], person_box):
                    continue
                if best is None or it.get("conf", 0) > best["item"].get("conf", 0):
                    best = {"entry": e, "item": it}
        if best is not None:
            return best
    # looks (and a device we can't find): the newest AI picture is the moment of the stare
    return {"entry": history[-1], "item": None}


def evidence_picture(pick, person_box, caption):
    frame = pick["entry"]["frame"]
    item = pick["item"]
    h, w = frame.shape[:2]
    s = EVIDENCE_HEIGHT / h
    room = cv2.resize(frame, (int(w * s), EVIDENCE_HEIGHT), interpolation=cv2.INTER_AREA)

    def sb(b, scale, ox=0, oy=0):
        return (int((b[0] - ox) * scale), int((b[1] - oy) * scale)), (int((b[2] - ox) * scale), int((b[3] - oy) * scale))

    a, b = sb(person_box, s)
    cv2.rectangle(room, a, b, (0, 0, 255), 3)
    if item is not None:
        a, b = sb(item["box"], s)
        cv2.rectangle(room, a, b, (255, 0, 255), 3)
        label = str(item.get("kind") or item.get("label", "device")).upper() + " " + str(round(item.get("conf", 0), 2))
        cv2.putText(room, label, (a[0], max(20, a[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.rectangle(room, (0, 0), (room.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(room, caption, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # zoom: the person and the device together, a bit bigger than both
    z = list(person_box)
    if item is not None:
        z = [min(z[0], item["box"][0]), min(z[1], item["box"][1]), max(z[2], item["box"][2]), max(z[3], item["box"][3])]
    pad_x = (z[2] - z[0]) * 0.25
    pad_y = (z[3] - z[1]) * 0.25
    x1, y1 = int(max(0, z[0] - pad_x)), int(max(0, z[1] - pad_y))
    x2, y2 = int(min(w, z[2] + pad_x)), int(min(h, z[3] + pad_y))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return room
    zs = EVIDENCE_HEIGHT / crop.shape[0]
    zw = min(int(crop.shape[1] * zs), 1280)
    zoom = cv2.resize(crop, (zw, EVIDENCE_HEIGHT))
    zsx = zw / crop.shape[1]
    if item is not None:
        a = (int((item["box"][0] - x1) * zsx), int((item["box"][1] - y1) * zs))
        b = (int((item["box"][2] - x1) * zsx), int((item["box"][3] - y1) * zs))
        cv2.rectangle(zoom, a, b, (255, 0, 255), 3)
    return np.hstack([room, zoom])


def save_alert_picture(history, person_state, alert, frame):
    # saves the alert picture; gives back the file name
    name = person_state["name"].replace(" ", "_")
    file_name = name + "_" + alert["kind"] + "_" + str(int(time.time() * 1000)) + ".jpg"
    pick = pick_evidence(history, alert, person_state["box"])
    if pick is None:
        return save_evidence(frame, person_state, alert)   # no AI picture yet: old close-up
    caption = person_state["name"] + " - " + alert["kind"] + " - " + alert.get("exam_time", "")
    img = evidence_picture(pick, person_state["box"], caption)
    cv2.imwrite(os.path.join(config.EVIDENCE_FOLDER, file_name), img)
    return file_name


def write_csv(alert):
    new_file = not os.path.exists(config.ALERTS_CSV)
    with open(config.ALERTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["clock", "exam_time", "person", "kind", "message", "image", "clip"])
        writer.writerow([alert["clock"], alert["exam_time"], alert["person"], alert["kind"],
                         alert["message"], alert["image"], alert.get("clip", "")])


class Window(threading.Thread):
    # Shows pictures in the OpenCV window on its own thread. Updating the window (imshow + waitKey)
    # took ~6-15 ms per picture; the screen loop now just hands the newest picture over.
    def __init__(self, title):
        super().__init__(daemon=True)
        self.title = title
        self.picture = None
        self.quit = False
        self.cond = threading.Condition()

    def show(self, picture):
        with self.cond:
            self.picture = picture
            self.cond.notify()

    def run(self):
        while not self.quit:
            with self.cond:
                if self.picture is None:
                    self.cond.wait(0.1)
                picture = self.picture
                self.picture = None
            if picture is not None:
                cv2.imshow(self.title, picture)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.quit = True
        cv2.destroyAllWindows()


def run_live(cap):
    cv2.setNumThreads(getattr(config, "SCREEN_CV_THREADS", -1))   # -1 = OpenCV's default
    # The threaded loop (app/runtime.py): newest camera picture only, AI in a worker,
    # tracker + rules + drawing on every picture. Gives back the people states for the summary.
    from app import runtime
    from app.trace import Trace
    from vision.tracker import Tracker
    from vision.devices import DeviceMemory

    # the AI worker starts first: in its own process it loads the models itself (about a minute)
    worker = runtime.make_worker(detector)
    worker.start()
    model_devices = getattr(worker, "devices", None) or detector.model_devices()
    trace = Trace(devices=model_devices) if getattr(config, "TRACE", True) else None
    tracker = Tracker(config.PERSON_CONFIDENCE, events=trace.event if trace else None)
    devices = DeviceMemory(confirm=getattr(config, "DEVICE_CONFIRM", False))
    source = runtime.CameraSource(cap)
    clips.setup(DASHBOARD_FPS)
    session_start = time.time()

    def on_alert(a, frame, people_states, now, history=None):
        person_state = None
        for d in people_states:
            if d["name"] == a["person"]:
                person_state = d
        a["clock"] = time.strftime("%H:%M:%S")
        a["exam_time"] = exam_time_text(now)
        a["image"] = save_alert_picture(history, person_state, a, frame) if person_state else ""
        a["clip"] = clips.start_clip(a)
        write_csv(a)
        if config.DASHBOARD:
            dashboard.add_alerts([a])
        print("[ALERT]", a["exam_time"], a["message"])

    last_dashboard = [0.0]

    def show(frame, people_states, matches, items, stages):
        small = draw(frame, people_states, matches, items, width=SCREEN_WIDTH)
        if getattr(config, "SHOW_STOPWATCH", False):
            # on-screen clock for the delay test: film a stopwatch and compare the two times
            cv2.putText(small, "%.3f" % (time.time() - session_start), (20, small.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        # the web page doesn't need 30 pictures a second: making a JPEG every frame cost ~8 ms
        if config.DASHBOARD and time.time() - last_dashboard[0] >= 1.0 / DASHBOARD_FPS:
            last_dashboard[0] = time.time()
            clips.push(small)
            dashboard.set_people(people_states)
            ok_jpg, jpg = cv2.imencode(".jpg", small)
            if ok_jpg:
                dashboard.set_frame(jpg.tobytes())
        if config.SHOW_WINDOW:
            window.show(small)
            if window.quit:
                return False
        return True

    window = Window("Exam Watch (press Q to quit)")
    if config.SHOW_WINDOW:
        window.start()
    source.start()
    try:
        return runtime.run(source, detector, brain, tracker, devices, trace=trace, show=show, on_alert=on_alert,
                           worker=worker)
    finally:
        window.quit = True        # the window thread closes its own window
        if window.is_alive():
            window.join(timeout=2)
        if trace is not None:
            trace.close()


def main():
    source = get_source()

    cap = camera.open_camera(source)
    if cap is None:
        return

    people_states = []   # set early so the summary still works after an early Ctrl+C
    total_alerts = 0
    try:
        is_file = isinstance(source, str)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 1:
            fps = 30
        if not is_file:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # always use the newest picture

        # a video file: skip pictures so we check it at the live camera's speed
        file_step = 1
        if is_file:
            file_step = max(1, round(fps / config.VIDEO_FILE_FPS))
            clips.setup(fps / file_step)

        ok, frame = cap.read()
        if not ok:
            print("Could not read from the camera/video")
            return

        h, w = frame.shape[:2]
        print("Video size:", w, "x", h)

        if is_file or not getattr(config, "AI_PROCESS", False):
            detector.load_models()   # (with AI_PROCESS the AI process loads its own copy)
        os.makedirs(config.EVIDENCE_FOLDER, exist_ok=True)

        if config.DASHBOARD:
            dashboard.start()
            url = "http://localhost:" + str(config.DASHBOARD_PORT)
            print("Dashboard:", url)
            if config.OPEN_BROWSER:
                webbrowser.open(url)

        if not is_file:
            # live camera: camera, AI and screen run separately so the picture never lags
            people_states = run_live(cap)
            total_alerts = sum(d["alert_count"] for d in people_states)

        start = time.time()
        frame_number = 0
        if is_file:
            total_alerts = 0

        while is_file:
            if frame_number > 0:
                for i in range(file_step - 1):
                    cap.grab()
                ok, frame = cap.read()
                if not ok:
                    print("Video ended")
                    break
            frame_number = frame_number + 1

            # for a video file use the video's own clock, so slow computers still get the right seconds
            if is_file:
                now = (1 + (frame_number - 1) * file_step) / fps
            else:
                now = time.time() - start

            people, phones = detector.detect(frame)
            alerts, matches = brain.update_all(people_states, people, phones, now)
            drawn = draw(frame, people_states, matches, phones)

            for a in alerts:
                person_state = None
                for d in people_states:
                    if d["name"] == a["person"]:
                        person_state = d
                a["clock"] = time.strftime("%H:%M:%S")
                a["exam_time"] = exam_time_text(now)
                a["image"] = save_evidence(drawn, person_state, a)
                a["clip"] = clips.start_clip(a)
                write_csv(a)
                total_alerts = total_alerts + 1
                print("[ALERT]", a["exam_time"], a["message"])

            small = resize_to_width(drawn, 1280)
            clips.push(small)

            if config.DASHBOARD:
                dashboard.add_alerts(alerts)
                dashboard.set_people(people_states)
                ok_jpg, jpg = cv2.imencode(".jpg", small)
                if ok_jpg:
                    dashboard.set_frame(jpg.tobytes())

            if config.SHOW_WINDOW:
                cv2.imshow("Exam Watch (press Q to quit)", small)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            if frame_number % 100 == 0:
                speed = frame_number / (time.time() - start)
                print("frames:", frame_number, " speed:", round(speed, 1), "fps")

    except KeyboardInterrupt:
        print("Stopped (Ctrl+C)")
    finally:
        # always let go of the camera, even after Ctrl+C or a crash
        cap.release()
        if config.SHOW_WINDOW:
            cv2.destroyAllWindows()

    print("")
    print("===== SUMMARY =====")
    print("Total alerts:", total_alerts)
    print("Alerts per person (total " + str(sum(d["alert_count"] for d in people_states)) + "):")
    for d in sorted(people_states, key=lambda x: x["alert_count"], reverse=True):
        print(d["name"], "| alerts", d["alert_count"], "| looks", d["total_looks"],
              "| long looks", d["total_long_looks"], "| phone alerts", d["total_phone_alerts"])


if __name__ == "__main__":
    main()
