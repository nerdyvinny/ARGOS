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
import webbrowser
import cv2
from settings import config
from capture import camera
from rules import brain
from vision import detector
from ui import dashboard

COLORS = {
    "green": (0, 200, 0),
    "yellow": (0, 220, 255),
    "red": (0, 0, 255),
    "empty": (120, 120, 120),
}


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


def draw(frame, people_states, matches, phones):
    img = frame.copy()

    for d in people_states:
        if d["name"] not in matches:
            continue   # only draw people we can see right now
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        color = COLORS.get(d["status"], (255, 255, 255))
        thick = 4 if d["status"] == "red" else 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)

        text = d["name"] + "  alerts " + str(d["alert_count"])
        cv2.putText(img, text, (x1 + 5, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if d["deviation"] is not None:
            turn = "turn " + str(round(d["deviation"], 2))
            cv2.putText(img, turn, (x1 + 5, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # face points of each matched person (nose = red, eyes/ears = white)
    for name in matches:
        p = matches[name]
        pts = p["points"]
        for k in range(min(5, len(pts))):
            if pts[k][2] >= config.KEYPOINT_CONFIDENCE:
                c = (0, 0, 255) if k == 0 else (255, 255, 255)
                cv2.circle(img, (int(pts[k][0]), int(pts[k][1])), 4, c, -1)

    for ph in phones:
        x1, y1, x2, y2 = [int(v) for v in ph["box"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.putText(img, ph.get("label", "phone").upper(), (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

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


def write_csv(alert):
    new_file = not os.path.exists(config.ALERTS_CSV)
    with open(config.ALERTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["clock", "exam_time", "person", "kind", "message", "image"])
        writer.writerow([alert["clock"], alert["exam_time"], alert["person"], alert["kind"],
                         alert["message"], alert["image"]])


def run_live(cap):
    # The threaded loop (app/runtime.py): newest camera picture only, AI in a worker,
    # tracker + rules + drawing on every picture. Gives back the people states for the summary.
    from app import runtime
    from app.trace import Trace
    from vision.tracker import Tracker
    from vision.devices import DeviceMemory

    trace = Trace(devices=detector.model_devices()) if getattr(config, "TRACE", True) else None
    tracker = Tracker(config.PERSON_CONFIDENCE, events=trace.event if trace else None)
    devices = DeviceMemory(confirm=getattr(config, "DEVICE_CONFIRM", False))
    source = runtime.CameraSource(cap)
    session_start = time.time()

    def on_alert(a, frame, people_states, now):
        person_state = None
        for d in people_states:
            if d["name"] == a["person"]:
                person_state = d
        a["clock"] = time.strftime("%H:%M:%S")
        a["exam_time"] = exam_time_text(now)
        a["image"] = save_evidence(frame, person_state, a) if person_state else ""
        write_csv(a)
        if config.DASHBOARD:
            dashboard.add_alerts([a])
        print("[ALERT]", a["exam_time"], a["message"])

    def show(frame, people_states, matches, items, stages):
        drawn = draw(frame, people_states, matches, items)
        if getattr(config, "SHOW_STOPWATCH", False):
            # on-screen clock for the delay test: film a stopwatch and compare the two times
            cv2.putText(drawn, "%.3f" % (time.time() - session_start), (20, drawn.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        small = resize_to_width(drawn, 1280)
        if config.DASHBOARD:
            dashboard.set_people(people_states)
            ok_jpg, jpg = cv2.imencode(".jpg", small)
            if ok_jpg:
                dashboard.set_frame(jpg.tobytes())
        if config.SHOW_WINDOW:
            cv2.imshow("Exam Watch (press Q to quit)", small)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False
        return True

    source.start()
    try:
        return runtime.run(source, detector, brain, tracker, devices, trace=trace, show=show, on_alert=on_alert)
    finally:
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

        ok, frame = cap.read()
        if not ok:
            print("Could not read from the camera/video")
            return

        h, w = frame.shape[:2]
        print("Video size:", w, "x", h)

        detector.load_models()
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
                write_csv(a)
                total_alerts = total_alerts + 1
                print("[ALERT]", a["exam_time"], a["message"])

            small = resize_to_width(drawn, 1280)

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
