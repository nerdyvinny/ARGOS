# Replays the classroom test clip through the app's real detect -> brain path and scores it
# against the hand-checked ground truth in research/labels/gt.json.
#
#   .\.venv-sandbox\Scripts\python.exe perf\eval_vision.py --name baseline
#
# Writes perf/runs/<name>.json (numbers + alert list, no images) and prints a summary.
# Processed FPS = frames / time spent in detect + brain (clip decoding is not counted,
# the live app decodes on the camera side). p5 FPS = 1000 / 95th-percentile frame ms.

import os
import sys
import json
import time
import argparse
import inspect
import statistics
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from settings import config

CLIP = "perf/clip/classroom.avi"
TIMES = "perf/clip/classroom_times.json"
GT = "research/labels/gt.json"
CLASSES = ["person", "phone", "watch", "headphones", "paper", "pen"]
LABEL_MAP = {"phone": "phone", "smartwatch": "watch", "watch": "watch", "headphones": "headphones",
             "earbuds": "headphones", "paper": "paper", "pen": "pen"}
OVERLAP_TEST = (145.0, 290.0)     # seconds of the clip with standing up, walking, crossing, hiding, leaving
KP = 0.5                          # keypoint confidence used for the dropout numbers (same as config)
CLIP_FPS = 3.85                   # measured camera rate of the recording

# Distance bands from box height at 1080p. The camera's focal length was estimated at ~900 px
# (see research/DECISIONS.md), so a seated person (~1.0 m visible) is about 900 / h metres away.
FOCAL_PX = 900.0
SEATED_VISIBLE_M = 1.0


def person_distance_m(box):
    h = max(1.0, box[3] - box[1])
    return FOCAL_PX * SEATED_VISIBLE_M / h


def band_of_distance(m):
    if m < 2.0:
        return "near(<2m)"
    if m < 6.0:
        return "mid(2-6m)"
    return "far(>6m)"


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0


def item_hit(pred, gt):
    # hand-drawn item boxes are rough, so an item counts if the boxes overlap a bit
    # or the predicted centre falls inside the ground-truth box grown by 30%
    if iou(pred, gt) >= 0.2:
        return True
    w, h = gt[2] - gt[0], gt[3] - gt[1]
    cx, cy = (pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2
    return gt[0] - 0.3 * w <= cx <= gt[2] + 0.3 * w and gt[1] - 0.3 * h <= cy <= gt[3] + 0.3 * h


def person_hit(pred, gt):
    return iou(pred, gt) >= 0.3


def match(preds, gts, hit):
    # greedy one-to-one matching; gives back list of (pred_index, gt_index)
    pairs = []
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            if hit(p["box"], g["box"]):
                pairs.append((iou(p["box"], g["box"]), i, j))
    pairs.sort(reverse=True)
    used_p, used_g, out = set(), set(), []
    for s, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        out.append((i, j))
    return out


def face_flags(person):
    from rules import brain
    pts = person.get("points", [])

    def ok(k):
        return k < len(pts) and pts[k][2] >= KP
    face = person.get("face_ok", ok(0) and (ok(1) or ok(2)))
    eyes = person.get("eyes_ok", ok(1) and ok(2))
    head = person.get("head_ok", brain.head_yaw(pts) is not None)
    pose = person.get("pose_ok", ok(5) and ok(6) and (ok(7) or ok(8) or ok(9) or ok(10)))
    return {"face": bool(face), "eyes": bool(eyes), "head": bool(head), "pose": bool(pose)}


def next_frame_index(times, last, clock):
    # Live camera rule: when the program is free at `clock`, it takes the newest picture
    # the camera has already made (older ones are dropped). If none is new, it waits for the next.
    j = last + 1
    while j + 1 < len(times) and times[j + 1] <= clock:
        j += 1
    return j


def run_clip(limit=None):
    # Real-time replay: the clip plays at its recorded speed and a slow program skips pictures,
    # exactly like the live camera. Each result is shown from the moment it is finished.
    from vision import detector
    from rules import brain
    detector.load_models()
    takes_time = "now" in inspect.signature(detector.detect).parameters
    times = json.load(open(TIMES))
    if limit is not None:
        times = times[:limit]
    cap = cv2.VideoCapture(CLIP)
    people_states = []
    frames = []
    alerts = []
    clock = 0.0          # simulated wall clock (seconds since the clip started)
    pos = -1             # index of the last picture pulled from the video file
    last = -1
    while last + 1 < len(times):
        j = next_frame_index(times, last, clock)
        while pos < j - 1:
            cap.grab()
            pos += 1
        ok, frame = cap.read()
        pos += 1
        if not ok:
            break
        now = times[j]
        clock = max(clock, now)
        t0 = time.perf_counter()
        if takes_time:
            people, phones = detector.detect(frame, now)   # tracker uses the clip's clock
        else:
            people, phones = detector.detect(frame)
        new_alerts, matches = brain.update_all(people_states, people, phones, now)
        ms = (time.perf_counter() - t0) * 1000
        clock = clock + ms / 1000.0
        for a in new_alerts:
            alerts.append({"kind": a["kind"], "person": a["person"], "time": round(now, 2), "message": a["message"]})
        frames.append({
            "i": j, "t": now, "done": clock, "ms": ms,
            "people": [{"id": name, "box": p["box"], "flags": face_flags(p)} for name, p in matches.items()],
            "items": [{"label": LABEL_MAP.get(ph.get("label", "phone"), ph.get("label")), "box": ph["box"],
                       "person": ph.get("person")} for ph in phones],
        })
        if len(frames) % 50 == 0:
            print("processed", len(frames), "at clip time", round(now, 1), "s,", round(ms), "ms", flush=True)
        last = j
    return frames, alerts


def shown_at(frames, t):
    # the result on screen at clip time t = the newest one finished by then
    best = None
    for f in frames:
        if f["done"] <= t + 1e-6:
            best = f
        else:
            break
    return best


def gt_band(g):
    return g.get("band") or band_of_distance(person_distance_m(g["box"]))


def score_flags(flags, preds, gts, pairs):
    hit_g = {j for _, j in pairs}
    for pi, gj in pairs:
        gg = gts[gj]
        if gt_band(gg).startswith("far"):
            continue   # faces of people >6 m away are a few pixels; not scored
        for k, v in preds[pi]["flags"].items():
            if k in ("face", "eyes", "head") and gg.get("facing") == "back":
                continue
            flags[k][1] += 1
            if not v:
                flags[k][0] += 1
    # a person we missed entirely also drops all their face/pose data
    for j, gg in enumerate(gts):
        if j in hit_g or gt_band(gg).startswith("far"):
            continue
        for k in flags:
            if k in ("face", "eyes", "head") and gg.get("facing") == "back":
                continue
            flags[k][0] += 1
            flags[k][1] += 1


EMPTY = {"people": [], "items": []}


def score(frames, gt):
    det = {c: {"tp": 0, "fp": 0, "fn": 0, "bands": {}} for c in CLASSES}
    flags = {k: [0, 0] for k in ["face", "eyes", "head", "pose"]}   # [missing, total]
    for key, g in gt["frames"].items():
        f = shown_at(frames, g["t"]) or EMPTY
        for c in CLASSES:
            if c == "person":
                preds = [{"box": p["box"], "flags": p["flags"]} for p in f["people"]]
                gts = g["people"]
                pairs = match(preds, gts, person_hit)
                score_flags(flags, preds, gts, pairs)
            else:
                preds = [it for it in f["items"] if it["label"] == c]
                gts = [it for it in g["items"] if it["label"] == c]
                pairs = match(preds, gts, item_hit)
            hit_g = {j for _, j in pairs}
            det[c]["tp"] += len(pairs)
            det[c]["fp"] += len(preds) - len(pairs)
            det[c]["fn"] += len(gts) - len(pairs)
            for j, gg in enumerate(gts):
                # every labeled item is on the near desk or in the near person's hands (< 2 m)
                band = gt_band(gg) if c == "person" else gg.get("band", "near(<2m)")
                d = det[c]["bands"].setdefault(band, [0, 0])
                d[1] += 1
                if j in hit_g:
                    d[0] += 1
    out = {}
    for c in CLASSES:
        d = det[c]
        p = d["tp"] / (d["tp"] + d["fp"]) if d["tp"] + d["fp"] else None
        r = d["tp"] / (d["tp"] + d["fn"]) if d["tp"] + d["fn"] else None
        out[c] = {"precision": p, "recall": r, "tp": d["tp"], "fp": d["fp"], "fn": d["fn"],
                  "recall_by_band": {b: (v[0] / v[1], v[1]) for b, v in sorted(d["bands"].items())}}
    drop = {k: (v[0] / v[1] if v[1] else None, v[1]) for k, v in flags.items()}
    return out, drop


def id_switches(frames, gt, t_range=None):
    # For each ground-truth person, follow which app number covers them; a change = 1 switch.
    times = json.load(open(TIMES))
    dense = gt.get("dense", {})
    keys = sorted(int(k) for k in dense)
    spacing = (keys[1] - keys[0]) / CLIP_FPS if len(keys) > 1 else 1.0
    last = {}
    switches = 0
    samples = 0
    for k in keys:
        if t_range and not (t_range[0] <= times[k] <= t_range[1]):
            continue
        f = shown_at(frames, times[k]) or EMPTY
        gts = dense[str(k)]
        preds = [{"box": p["box"], "id": p["id"]} for p in f["people"]]
        for pi, gj in match(preds, gts, person_hit):
            gid = gts[gj]["gid"]
            pid = preds[pi]["id"]
            if gid in last and last[gid] != pid:
                switches += 1
            last[gid] = pid
        samples += len(gts)
    return switches, samples * spacing / 60.0


def summarize(frames, alerts, gt):
    ms = [f["ms"] for f in frames]
    p95 = sorted(ms)[int(0.95 * (len(ms) - 1))]
    det, drop = score(frames, gt)
    sw_all, min_all = id_switches(frames, gt)
    sw_ov, min_ov = id_switches(frames, gt, OVERLAP_TEST)
    return {
        "frames": len(frames),
        "fps_avg": len(frames) / (sum(ms) / 1000.0),
        "fps_p5": 1000.0 / p95,
        "delay_ms_avg": statistics.mean(ms),
        "delay_ms_p95": p95,
        "detection": det,
        "dropout": drop,
        "id_switches": sw_all, "person_minutes": min_all,
        "idsw_per_person_min": sw_all / min_all if min_all else None,
        "overlap_id_switches": sw_ov, "overlap_person_minutes": min_ov,
        "overlap_idsw_per_person_min": sw_ov / min_ov if min_ov else None,
        "alerts": alerts,
    }


def fmt(x, pct=True):
    if x is None:
        return "-"
    return str(round(100 * x, 1)) + "%" if pct else str(round(x, 2))


def print_summary(s):
    print("processed FPS avg", fmt(s["fps_avg"], False), "p5", fmt(s["fps_p5"], False),
          "| delay ms avg", fmt(s["delay_ms_avg"], False), "p95", fmt(s["delay_ms_p95"], False))
    for c, d in s["detection"].items():
        bands = ", ".join(b + " " + fmt(v[0]) + " (n=" + str(v[1]) + ")" for b, v in d["recall_by_band"].items())
        print(" ", c.ljust(10), "P", fmt(d["precision"]).rjust(6), "R", fmt(d["recall"]).rjust(6), "|", bands)
    print("  dropout", {k: (fmt(v[0]), v[1]) for k, v in s["dropout"].items()})
    print("  ID switches", s["id_switches"], "over", fmt(s["person_minutes"], False), "person-min =",
          fmt(s["idsw_per_person_min"], False), "| overlap test", s["overlap_id_switches"], "/",
          fmt(s["overlap_person_minutes"], False), "=", fmt(s["overlap_idsw_per_person_min"], False))
    print("  alerts", [(a["kind"], a["time"], a["person"]) for a in s["alerts"]])


def point_models_at_sandbox():
    # the models live in models/ (gitignored) inside the sandbox
    for name in ["POSE_MODEL", "OBJECT_MODEL", "WORLD_MODEL"]:
        value = getattr(config, name, None)
        if value and not os.path.dirname(value) and os.path.exists(os.path.join("models", value)):
            setattr(config, name, os.path.join("models", value))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rescore", action="store_true", help="score an existing run again (after label fixes)")
    ap.add_argument("--no-openvino", action="store_true", help="use the plain .pt models (the original code path)")
    args = ap.parse_args()
    if args.no_openvino:
        config.USE_OPENVINO = False
    gt = json.load(open(GT))
    path = "perf/runs/" + args.name + ".json"
    if args.rescore:
        old = json.load(open(path))
        frames, alerts = old["frames"], old["summary"]["alerts"]
    else:
        if not hasattr(config, "WORLD_MODEL"):
            config.WORLD_MODEL = "yolov8s-worldv2.pt"
        point_models_at_sandbox()
        frames, alerts = run_clip(args.limit)
    s = summarize(frames, alerts, gt)
    os.makedirs("perf/runs", exist_ok=True)
    json.dump({"summary": s, "frames": frames}, open(path, "w"))
    print_summary(s)


if __name__ == "__main__":
    main()
