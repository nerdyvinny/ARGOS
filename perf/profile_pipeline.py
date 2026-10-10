# TEMPORARY speed test. Not part of the app.
#
# Runs the same steps as app/main.py on the real camera and times each step.
#
#   venv\Scripts\python.exe perf\profile_pipeline.py            (60 s camera-only + 60 s full)
#   venv\Scripts\python.exe perf\profile_pipeline.py 20         (shorter run, for quick checks)
#
# It never saves evidence pictures or writes alerts.csv.

import os
import sys
import time
import json
import shutil
import platform
import subprocess
import threading
import numpy as np
import psutil
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from settings import config
from capture import camera
from rules import brain
from vision import detector
from ui import dashboard
from app import main as app_main


class CpuSampler:
    # samples CPU % once a second in the background
    def __init__(self):
        self.system = []
        self.process = []
        self.running = True
        self.proc = psutil.Process()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        psutil.cpu_percent(None)
        self.proc.cpu_percent(None)
        while self.running:
            time.sleep(1.0)
            self.system.append(psutil.cpu_percent(None))
            # process % is per core, so divide by core count to get "% of whole CPU"
            self.process.append(self.proc.cpu_percent(None) / psutil.cpu_count())

    def start(self):
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()


def stats(times_ms):
    if len(times_ms) == 0:
        return None, None
    a = np.array(times_ms)
    return float(a.mean()), float(np.percentile(a, 95))


def gpu_report():
    report = {}
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
            report["nvidia-smi"] = out.stdout.strip() or out.stderr.strip()
        except Exception as e:
            report["nvidia-smi"] = "error: " + str(e)
    else:
        report["nvidia-smi"] = "not found (no NVIDIA driver)"

    try:
        import onnxruntime
        report["onnxruntime providers"] = onnxruntime.get_available_providers()
    except ImportError:
        report["onnxruntime providers"] = "onnxruntime not installed"

    try:
        import openvino
        report["openvino"] = openvino.__version__
    except ImportError:
        report["openvino"] = "not installed"

    import torch
    report["torch"] = torch.__version__ + " | cuda available: " + str(torch.cuda.is_available()) + \
        " | threads: " + str(torch.get_num_threads())
    report["cpu"] = platform.processor() + " | cores " + str(psutil.cpu_count(False)) + \
        " | threads " + str(psutil.cpu_count())
    return report


def camera_only(cap, seconds):
    # how fast does the camera hand us new pictures when we do nothing else?
    read_ms = []
    start = time.perf_counter()
    count = 0
    while time.perf_counter() - start < seconds:
        t = time.perf_counter()
        ok, frame = cap.read()
        read_ms.append((time.perf_counter() - t) * 1000)
        if not ok:
            print("camera read failed")
            break
        count = count + 1
    elapsed = time.perf_counter() - start
    return count / elapsed, read_ms


def full_pipeline(cap, seconds, desk_states):
    names = ["capture/decode", "tile+crop (resize/convert)", "body pose + face/eye points (pose model)",
             "phone (object model)", "rules (brain)", "draw boxes", "resize to 1280",
             "jpeg encode for dashboard", "display (imshow+waitKey)"]
    times = {}
    for n in names:
        times[n] = []
    frame_times = []
    alerts_seen = 0
    people_counts = []

    start = time.perf_counter()
    frames = 0
    while time.perf_counter() - start < seconds:
        f0 = time.perf_counter()

        t = time.perf_counter()
        ok, frame = cap.read()
        times["capture/decode"].append((time.perf_counter() - t) * 1000)
        if not ok:
            print("camera read failed")
            break

        t = time.perf_counter()
        tiles = detector.make_tiles(frame)
        times["tile+crop (resize/convert)"].append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        people = detector.find_people(tiles)
        times["body pose + face/eye points (pose model)"].append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        phones = detector.find_phones(tiles)
        times["phone (object model)"].append((time.perf_counter() - t) * 1000)
        people_counts.append(len(people))

        now = time.perf_counter() - start
        t = time.perf_counter()
        alerts, matches = brain.update_all(desk_states, people, phones, now)
        times["rules (brain)"].append((time.perf_counter() - t) * 1000)
        alerts_seen = alerts_seen + len(alerts)

        t = time.perf_counter()
        drawn = app_main.draw(frame, desk_states, matches, phones)
        times["draw boxes"].append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        small = app_main.resize_to_width(drawn, 1280)
        times["resize to 1280"].append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        ok_jpg, jpg = cv2.imencode(".jpg", small)
        if ok_jpg:
            dashboard.set_frame(jpg.tobytes())
        dashboard.set_desks(desk_states)
        times["jpeg encode for dashboard"].append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        cv2.imshow("ARGOS profile (press Q to stop early)", small)
        key = cv2.waitKey(1) & 0xFF
        times["display (imshow+waitKey)"].append((time.perf_counter() - t) * 1000)

        frame_times.append((time.perf_counter() - f0) * 1000)
        frames = frames + 1
        if key == ord("q"):
            break

    elapsed = time.perf_counter() - start
    return frames / elapsed, names, times, frame_times, alerts_seen, people_counts


def main():
    seconds = 60
    if len(sys.argv) > 1:
        seconds = float(sys.argv[1])

    print("Source from config:", config.SOURCE)
    gpu = gpu_report()
    for k in gpu:
        print(k + ":", gpu[k])

    cap = camera.open_camera(config.SOURCE)
    if cap is None:
        print("camera did not open")
        return
    try:
        if not isinstance(config.SOURCE, str):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # same as app/main.py
        ok, frame = cap.read()
        h, w = frame.shape[:2]
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_text = "".join([chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)])
        print("Opened:", w, "x", h, "| format", repr(fourcc_text), "| driver says", cap.get(cv2.CAP_PROP_FPS), "fps")

        desk_states = brain.load_desks(w, h)
        detector.load_models()

        # warm up the models so the first slow run is not counted
        for i in range(3):
            ok, frame = cap.read()
            detector.detect(frame)

        print("\n--- camera only,", seconds, "s ---")
        cpu = CpuSampler()
        cpu.start()
        cam_fps, read_ms = camera_only(cap, seconds)
        cpu.stop()
        cam_cpu = (float(np.mean(cpu.system)), float(np.mean(cpu.process)))
        print("camera FPS:", round(cam_fps, 2))

        print("\n--- full pipeline,", seconds, "s ---")
        cpu = CpuSampler()
        cpu.start()
        proc_fps, names, times, frame_times, alerts_seen, people_counts = full_pipeline(cap, seconds, desk_states)
        cpu.stop()
        pipe_cpu = (float(np.mean(cpu.system)), float(np.mean(cpu.process)))
    finally:
        cap.release()
        cv2.destroyAllWindows()

    total_mean, total_p95 = stats(frame_times)
    print("\n{:<44} {:>9} {:>9} {:>7}".format("stage", "avg ms", "p95 ms", "share"))
    rows = []
    for n in names:
        mean, p95 = stats(times[n])
        share = mean / total_mean * 100
        rows.append({"stage": n, "avg_ms": mean, "p95_ms": p95, "share_pct": share})
        print("{:<44} {:>9.1f} {:>9.1f} {:>6.0f}%".format(n, mean, p95, share))
    print("{:<44} {:>9} {:>9}".format("360 flattening", "n/a", "n/a") + "   (not used: tiles only)")
    print("{:<44} {:>9.1f} {:>9.1f}".format("WHOLE FRAME", total_mean, total_p95))

    print("\ncamera FPS (camera only):", round(cam_fps, 2))
    print("processed FPS (full pipeline):", round(proc_fps, 2))
    print("CPU % camera-only: system", round(cam_cpu[0]), "| this program", round(cam_cpu[1]))
    print("CPU % full pipeline: system", round(pipe_cpu[0]), "| this program", round(pipe_cpu[1]))
    print("people seen per frame (avg):", round(float(np.mean(people_counts)), 2), "| alerts:", alerts_seen)

    result = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": config.SOURCE, "width": w, "height": h, "format": fourcc_text,
        "seconds": seconds, "camera_fps": cam_fps, "processed_fps": proc_fps,
        "camera_read_ms": stats(read_ms), "stages": rows,
        "frame_ms": [total_mean, total_p95], "cpu_camera_only": cam_cpu, "cpu_pipeline": pipe_cpu,
        "people_avg": float(np.mean(people_counts)), "alerts": alerts_seen, "gpu": gpu,
    }
    os.makedirs("perf/results", exist_ok=True)
    out = os.path.join("perf", "results", "profile_" + time.strftime("%Y%m%d_%H%M%S") + ".json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
