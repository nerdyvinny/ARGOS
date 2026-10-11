# TEMPORARY: runs the REAL app loop on the camera for 60 s and counts
#   - video FPS: pictures shown on the screen / dashboard
#   - AI FPS:    pictures the AI finished checking
# Evidence and alerts.csv go to a temporary folder.
#
#   venv\Scripts\python.exe perf\measure_app.py [seconds]

import os
import sys
import time
import tempfile
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from settings import config
from capture import camera
from rules import brain
from vision import detector
from app import main as app_main


def p5_fps(stamps):
    # 5th percentile speed = 1 / (95th percentile gap between pictures)
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    if not gaps:
        return None
    return round(1 / float(np.percentile(gaps, 95)), 2)


def main():
    seconds = 60
    if len(sys.argv) > 1:
        seconds = float(sys.argv[1])

    tmp = tempfile.mkdtemp()
    config.EVIDENCE_FOLDER = os.path.join(tmp, "evidence")
    config.ALERTS_CSV = os.path.join(tmp, "alerts.csv")
    config.DASHBOARD = True
    config.OPEN_BROWSER = False
    os.makedirs(config.EVIDENCE_FOLDER)

    counts = {"shown": 0, "checked": 0}
    stamps = {"shown": [], "checked": []}
    real_show = app_main.show
    real_update_all = brain.update_all
    deadline = [None]

    def counting_show(drawn, people_states):
        if deadline[0] is None:
            deadline[0] = time.time() + seconds
        counts["shown"] = counts["shown"] + 1
        stamps["shown"].append(time.time())
        if real_show(drawn, people_states):
            return True
        return time.time() > deadline[0]

    def counting_update_all(people_states, people, phones, now):
        if deadline[0] is not None and time.time() <= deadline[0]:
            counts["checked"] = counts["checked"] + 1
            stamps["checked"].append(time.time())
        return real_update_all(people_states, people, phones, now)

    app_main.show = counting_show
    brain.update_all = counting_update_all

    cap = camera.open_camera(config.SOURCE)
    if cap is None:
        return
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = cap.read()
        people_states = []
        detector.load_models()
        detector.detect(frame)     # warm up
        from ui import dashboard
        dashboard.start()
        alert_counts = {"alerts": 0}
        app_main.run_live_camera(cap, people_states, alert_counts)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("video FPS (screen/dashboard):", round(counts["shown"] / seconds, 2), "| p5", p5_fps(stamps["shown"]))
    print("AI FPS (processed):", round(counts["checked"] / seconds, 2), "| p5", p5_fps(stamps["checked"]))
    print("alerts:", alert_counts["alerts"])


if __name__ == "__main__":
    main()
