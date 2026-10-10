# TEMPORARY: replays the reference clip through the AI + rules and lists the alerts.
#
#   venv\Scripts\python.exe perf\replay_clip.py save        -> perf/clip/baseline_alerts.json
#   venv\Scripts\python.exe perf\replay_clip.py check       -> compares with the baseline
#
# Every picture is checked, using the time it was recorded at, so the result
# does not depend on how fast this computer is.

import os
import sys
import json
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rules import brain
from vision import detector

CLIP = "perf/clip/reference.avi"
TIMES = "perf/clip/reference_times.json"
BASELINE = "perf/clip/baseline_alerts.json"
TOLERANCE = 1.0


def replay():
    with open(TIMES) as f:
        times = json.load(f)
    cap = cv2.VideoCapture(CLIP)
    detector.load_models()
    desk_states = None
    alerts = []
    frames = 0
    try:
        for t in times:
            ok, frame = cap.read()
            if not ok:
                break
            if desk_states is None:
                h, w = frame.shape[:2]
                desk_states = brain.load_desks(w, h)
            people, phones = detector.detect(frame)
            new, matches = brain.update_all(desk_states, people, phones, t)
            for a in new:
                alerts.append({"desk": a["desk"], "kind": a["kind"], "time": round(t, 2)})
            frames = frames + 1
    finally:
        cap.release()
    print("replayed", frames, "frames,", len(alerts), "alerts")
    for a in alerts:
        print("  ", a["time"], a["desk"], a["kind"])
    return alerts


def compare(baseline, now):
    # every baseline alert must come back within 1 s, and nothing new may appear
    unused = list(now)
    missing = []
    for b in baseline:
        found = None
        for a in unused:
            if a["desk"] == b["desk"] and a["kind"] == b["kind"] and abs(a["time"] - b["time"]) <= TOLERANCE:
                found = a
                break
        if found is None:
            missing.append(b)
        else:
            unused.remove(found)
    return missing, unused


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    alerts = replay()
    if mode == "save":
        with open(BASELINE, "w") as f:
            json.dump(alerts, f, indent=1)
        print("saved", BASELINE)
        return
    with open(BASELINE) as f:
        baseline = json.load(f)
    missing, extra = compare(baseline, alerts)
    if not missing and not extra:
        print("CLIP CHECK: PASS (" + str(len(baseline)) + " baseline alerts matched)")
    else:
        print("CLIP CHECK: FAIL | missing:", missing, "| new:", extra)
        sys.exit(1)


if __name__ == "__main__":
    main()
