# Replays classroom clip 2 through the REAL threaded app loop (app/runtime.py) with a fake
# camera that plays the clip at its recorded speed. Writes perf/runs/<name>.json (records for
# the scorer, no images) and perf/trace.csv, then prints the scores.
#
#   .\.venv-sandbox\Scripts\python.exe perf\replay_live.py --name c2_r01_runtime

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from settings import config

CLIP = "perf/clip2/classroom2"


def point_models_at_sandbox():
    for name in ["POSE_MODEL", "OBJECT_MODEL", "WORLD_MODEL"]:
        value = getattr(config, name, None)
        if value and not os.path.dirname(value) and os.path.exists(os.path.join("models", value)):
            setattr(config, name, os.path.join("models", value))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seconds", type=float, default=None, help="only the first N seconds")
    args = ap.parse_args()
    if not hasattr(config, "WORLD_MODEL"):
        config.WORLD_MODEL = "yolov8s-worldv2.pt"
    point_models_at_sandbox()

    from vision import detector
    from vision.tracker import Tracker
    from vision.devices import DeviceMemory
    from rules import brain
    from app import runtime
    from app.trace import Trace
    from app import main as app_main

    if not getattr(config, "AI_PROCESS", False):
        detector.load_models()   # (with AI_PROCESS the AI process loads its own copy)
    times = json.load(open(CLIP + "_times.json"))
    if args.seconds:
        times = [t for t in times if t <= args.seconds]
    trace = Trace(devices=detector.model_devices())
    tracker = Tracker(config.PERSON_CONFIDENCE, events=trace.event)
    devices = DeviceMemory(confirm=getattr(config, "DEVICE_CONFIRM", False))
    source = runtime.ClipSource(CLIP + ".avi", times)
    records = []

    def on_frame(rec):
        rec["delay_ms"] = (time.time() - (source.start_time + rec["t"])) * 1000
        records.append(rec)
        if len(records) % 200 == 0:
            print("shown", len(records), "frames, clip time", round(rec["t"], 1), "s", flush=True)

    def show(frame, people_states, matches, items, stages):
        drawn = app_main.draw(frame, people_states, matches, items)
        app_main.resize_to_width(drawn, 1280)
        return True

    # start the AI first: in its own process it loads the models (1-2 min); starting the clip before
    # that skipped the first ~2 minutes of the clip and halved every score
    worker = runtime.make_worker(detector)
    worker.start()
    source.start()
    runtime.run(source, detector, brain, tracker, devices, trace=trace, show=show, on_frame=on_frame,
                worker=worker)
    trace.devices = detector.model_devices()
    trace.close()
    meta = {"camera_fps": len(times) / times[-1], "model_devices": detector.model_devices(),
            "config": {k: getattr(config, k) for k in ["DETECT_MODE", "DEVICE_CONFIRM"] if hasattr(config, k)}}
    os.makedirs("perf/runs", exist_ok=True)
    json.dump({"frames": records, "meta": meta}, open("perf/runs/" + args.name + ".json", "w"))
    from perf import score2
    s = score2.score(args.name)
    json.dump(s, open("perf/runs/" + args.name + ".score.json", "w"), indent=1)
    score2.show(args.name, s)
    print("model devices:", meta["model_devices"])


if __name__ == "__main__":
    main()
