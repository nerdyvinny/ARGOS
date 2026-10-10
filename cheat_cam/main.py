# The main program. Run it during the test:
#
#   python main.py              (uses SOURCE from config.py = the USB camera)
#   python main.py test.mp4     (use a video file - great for practicing)
#
# Press Q in the video window to stop.

import os
import sys
import csv
import time
import webbrowser
import cv2
import config
import camera
import brain
import detector
import dashboard

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


def draw(frame, desk_states, matches, phones):
    img = frame.copy()

    for d in desk_states:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        color = COLORS.get(d["status"], (255, 255, 255))
        thick = 4 if d["status"] == "red" else 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)

        text = d["name"] + "  score " + str(d["score"])
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
        cv2.putText(img, "PHONE", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    return img


def resize_to_width(img, width):
    h, w = img.shape[:2]
    if w <= width:
        return img
    scale = width / w
    return cv2.resize(img, (width, int(h * scale)))


def save_evidence(drawn_frame, desk_state, alert):
    # save a close up of the desk (a bit bigger than the desk box)
    h, w = drawn_frame.shape[:2]
    x1, y1, x2, y2 = desk_state["box"]
    pad_x = (x2 - x1) * 0.5
    pad_y = (y2 - y1) * 0.5
    x1 = int(max(0, x1 - pad_x))
    y1 = int(max(0, y1 - pad_y))
    x2 = int(min(w, x2 + pad_x))
    y2 = int(min(h, y2 + pad_y))
    crop = drawn_frame[y1:y2, x1:x2]
    if crop.size == 0:
        crop = drawn_frame

    name = desk_state["name"].replace(" ", "_")
    file_name = name + "_" + alert["kind"] + "_" + str(int(time.time() * 1000)) + ".jpg"
    cv2.imwrite(os.path.join(config.EVIDENCE_FOLDER, file_name), crop)
    return file_name


def write_csv(alert):
    new_file = not os.path.exists(config.ALERTS_CSV)
    with open(config.ALERTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["clock", "exam_time", "desk", "kind", "message", "image"])
        writer.writerow([alert["clock"], alert["exam_time"], alert["desk"], alert["kind"],
                         alert["message"], alert["image"]])


def main():
    source = get_source()

    if not os.path.exists(config.DESKS_FILE):
        print("No", config.DESKS_FILE, "found. Run: python setup_desks.py")
        return

    cap = camera.open_camera(source)
    if cap is None:
        return

    is_file = isinstance(source, str)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1:
        fps = 30
    if not is_file:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # always use the newest picture

    ok, frame = cap.read()
    if not ok:
        print("Could not read from the camera/video")
        return

    h, w = frame.shape[:2]
    print("Video size:", w, "x", h)

    desk_states = brain.load_desks(w, h)
    print("Loaded", len(desk_states), "desks")

    detector.load_models()
    os.makedirs(config.EVIDENCE_FOLDER, exist_ok=True)

    if config.DASHBOARD:
        dashboard.start()
        url = "http://localhost:" + str(config.DASHBOARD_PORT)
        print("Dashboard:", url)
        if config.OPEN_BROWSER:
            webbrowser.open(url)

    start = time.time()
    frame_number = 0
    total_alerts = 0

    while True:
        if frame_number > 0:
            ok, frame = cap.read()
            if not ok:
                print("Video ended")
                break
        frame_number = frame_number + 1

        # for a video file use the video's own clock, so slow computers still get the right seconds
        if is_file:
            now = frame_number / fps
        else:
            now = time.time() - start

        people, phones = detector.detect(frame)
        alerts, matches = brain.update_all(desk_states, people, phones, now)
        drawn = draw(frame, desk_states, matches, phones)

        for a in alerts:
            desk_state = None
            for d in desk_states:
                if d["name"] == a["desk"]:
                    desk_state = d
            a["clock"] = time.strftime("%H:%M:%S")
            a["exam_time"] = exam_time_text(now)
            a["image"] = save_evidence(drawn, desk_state, a)
            write_csv(a)
            total_alerts = total_alerts + 1
            print("[ALERT]", a["exam_time"], a["message"])

        small = resize_to_width(drawn, 1280)

        if config.DASHBOARD:
            dashboard.add_alerts(alerts)
            dashboard.set_desks(desk_states)
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

    cap.release()
    if config.SHOW_WINDOW:
        cv2.destroyAllWindows()

    print("")
    print("===== SUMMARY =====")
    print("Total alerts:", total_alerts)
    for d in sorted(desk_states, key=lambda x: x["score"], reverse=True):
        print(d["name"], "| score", d["score"], "| looks", d["total_looks"],
              "| long looks", d["total_long_looks"], "| phone alerts", d["total_phone_alerts"])


if __name__ == "__main__":
    main()
