# Per-frame timing trace -> perf/trace.csv (gitignored).
# One row per displayed frame: time in each stage, which device each model really ran on,
# queue lengths, CPU and GPU use, and tracker events (created, confirmed, lost, revived,
# merged, number changed) with the reason and IDs.
#
# GPU use comes from Windows' own counters ("GPU Engine" utilization), read by typeperf in
# the background once a second. Nothing is sent anywhere.

import csv
import os
import subprocess
import threading
import time

import psutil

TRACE_PATH = "perf/trace.csv"
STAGES = ["capture_wait", "decode", "resize", "pose", "objects", "world", "tracker", "brain", "draw", "display"]


class GpuMeter(threading.Thread):
    # sums Windows "GPU Engine" utilization per engine type (3D, Compute, Copy, VideoDecode)
    def __init__(self):
        super().__init__(daemon=True)
        self.values = {}
        self.proc = None

    def run(self):
        try:
            self.proc = subprocess.Popen(
                ["typeperf", r"\GPU Engine(*)\Utilization Percentage", "-si", "1"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            return
        header = None
        for line in self.proc.stdout:
            parts = [p.strip('"') for p in line.strip().split('","')]
            if len(parts) < 2:
                continue
            if header is None and "GPU Engine" in line:
                header = parts
                continue
            if header is None or len(parts) != len(header):
                continue
            sums = {}
            for name, value in zip(header[1:], parts[1:]):
                try:
                    v = float(value)
                except ValueError:
                    continue
                kind = name.split("engtype_")[-1].split(")")[0] if "engtype_" in name else "other"
                sums[kind] = sums.get(kind, 0.0) + v
            self.values = sums

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()


class Trace:
    def __init__(self, path=TRACE_PATH, devices=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["frame", "t_capture", "t_shown", "delay_ms", "fresh_detection"] +
                             [s + "_ms" for s in STAGES] +
                             ["devices", "det_queue", "cpu_pct", "gpu_3d", "gpu_compute", "gpu_other", "events"])
        self.devices = devices or {}
        self.gpu = GpuMeter()
        self.gpu.start()
        psutil.cpu_percent(None)
        self.lock = threading.Lock()
        self.pending_events = []

    def event(self, kind, **info):
        # tracker events: created / confirmed / lost / revived / merged / number_changed
        with self.lock:
            self.pending_events.append(kind + ":" + ",".join(k + "=" + str(v) for k, v in info.items()))

    def row(self, frame, t_capture, t_shown, fresh, stages, det_queue):
        with self.lock:
            events = self.pending_events
            self.pending_events = []
        g = self.gpu.values
        other = sum(v for k, v in g.items() if k not in ("3D", "Compute"))
        self.writer.writerow([frame, round(t_capture, 4), round(t_shown, 4), round((t_shown - t_capture) * 1000, 1),
                              int(bool(fresh))] +
                             [round(stages.get(s, 0.0), 2) for s in STAGES] +
                             [";".join(k + "=" + v for k, v in self.devices.items()), det_queue,
                              psutil.cpu_percent(None), round(g.get("3D", 0.0), 1), round(g.get("Compute", 0.0), 1),
                              round(other, 1), "|".join(events)])

    def close(self):
        self.gpu.stop()
        self.file.close()


class Timer:
    # with Timer(stages, "pose"): ...   adds the milliseconds to stages["pose"]
    def __init__(self, stages, name):
        self.stages = stages
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.stages[self.name] = self.stages.get(self.name, 0.0) + (time.perf_counter() - self.t0) * 1000
        return False
