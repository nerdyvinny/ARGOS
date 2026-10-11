# Scores a run of the classroom clip 2 against research/labels2/gt.json and prints every target.
# Works on both kinds of run files:
#   - perf/runs/c2_*.json from perf/replay_live.py (the threaded app loop, real time)
#   - perf/runs/c2_r00_baseline.json from perf/eval_vision.py (the old single loop)
#
#   .\.venv-sandbox\Scripts\python.exe perf\score2.py c2_r01_runtime [c2_r00_baseline ...]

import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perf.eval_vision import item_hit, person_hit, match, band_of_distance, person_distance_m

GT = "research/labels2/gt.json"
TIMES = "perf/clip2/classroom2_times.json"
HEADCOUNT = 6                      # A (near), L1-L3 (seated far back), M1 and M2 (walking past)
OVERLAP = (145.0, 270.0)           # stand up, cross, leave twice, come back
LABEL_SPACING_S = 12.0             # the 25 checked frames are ~12 s apart
UPPER_BODY = [5, 6, 7, 8, 9, 10]   # shoulders, elbows, wrists
KP = 0.5


def load(name):
    d = json.load(open("perf/runs/" + name + ".json"))
    return d["frames"], d.get("summary", {}), d.get("meta", {})


def picker(frames):
    # result on screen when picture i was the newest: for the threaded loop it is the record
    # of picture i (or the last one before it); for the old loop, the last finished result.
    threaded = len(frames) > 0 and "shown" in frames[0]
    times = json.load(open(TIMES))
    by_i = {f["i"]: f for f in frames}

    def pick(i):
        if threaded:
            for k in range(i, -1, -1):
                if k in by_i:
                    return by_i[k]
            return None
        best = None
        for f in frames:
            if f["done"] <= times[i] + 1e-6:
                best = f
            else:
                break
        return best
    return pick, threaded


def detection_scores(frames, gt, pick):
    out = {}
    for c in ["person", "device", "headphones", "paper", "pen"]:
        tp = fp = fn = 0
        by_vis = {}
        for key, g in gt["frames"].items():
            f = pick(int(key)) or {"people": [], "items": []}
            if c == "person":
                preds = [{"box": p["box"]} for p in f["people"]]
                gts = g["people"]
                pairs = match(preds, gts, person_hit)
            else:
                preds = [it for it in f["items"] if it["label"] == c]
                gts = [it for it in g["items"] if it["label"] == c]
                pairs = match(preds, gts, item_hit)
            hit = {j for _, j in pairs}
            tp += len(pairs)
            fp += len(preds) - len(pairs)
            fn += len(gts) - len(pairs)
            for j, gg in enumerate(gts):
                k = band_of_distance(person_distance_m(gg["box"])) if c == "person" else gg.get("vis", "full")
                d = by_vis.setdefault(k, [0, 0])
                d[1] += 1
                d[0] += int(j in hit)
        r = {"precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
             "tp": tp, "fp": fp, "fn": fn, "recall_by": {k: (v[0] / v[1], v[1]) for k, v in sorted(by_vis.items())}}
        if c == "device":
            # each false device box on a checked frame stands for one ~12 s stretch of the clip
            r["fp_per_10min"] = fp * 600.0 / (len(gt["frames"]) * LABEL_SPACING_S)
            half = [v for k, v in by_vis.items() if k in ("full", "half")]
            n = sum(v[1] for v in half)
            r["recall_half_visible_or_more"] = sum(v[0] for v in half) / n if n else None
        out[c] = r
    return out


def tracking_scores(frames, gt, pick):
    times = json.load(open(TIMES))
    ids = set()
    for f in frames:
        for p in f["people"]:
            ids.add(p["id"])
    dense = gt["dense"]
    keys = sorted(int(k) for k in dense)
    spacing = (keys[1] - keys[0]) / 3.85
    last, switches, samples = {}, 0, 0
    seq = {}   # gid -> list of (time, app number or None)
    for k in keys:
        f = pick(k) or {"people": []}
        gts = dense[str(k)]
        preds = [{"box": p["box"], "id": p["id"]} for p in f["people"]]
        pairs = match(preds, gts, person_hit)
        got = {gts[gj]["gid"]: preds[pi]["id"] for pi, gj in pairs}
        for g in gts:
            seq.setdefault(g["gid"], []).append((times[k], got.get(g["gid"])))
        if OVERLAP[0] <= times[k] <= OVERLAP[1]:
            for gid, pid in got.items():
                if gid in last and last[gid] != pid:
                    switches += 1
                last[gid] = pid
            samples += len(gts)
    minutes = samples * spacing / 60.0
    # same number back after leaving: the near person A left the picture twice
    a = seq.get("A", [])
    present = [t for t, pid in a]
    gaps = [(t0, t1) for t0, t1 in zip(present, present[1:]) if 8.0 <= t1 - t0 <= 40.0]
    returns, kept = 0, 0
    for t0, t1 in gaps:
        before = [pid for t, pid in a if t <= t0 and pid is not None]
        after = [pid for t, pid in a if t >= t1 and pid is not None]
        if before and after:
            returns += 1
            kept += int(before[-1] == after[0])
    return {"unique_numbers": len(ids), "headcount": HEADCOUNT, "numbers_per_person": len(ids) / HEADCOUNT,
            "overlap_switches": switches, "overlap_person_min": minutes,
            "overlap_idsw_per_person_min": switches / minutes if minutes else None,
            "returns": returns, "returns_same_number": kept, "absences": gaps}


def limb_scores(frames, gt, pick, threaded):
    if not threaded:
        return {"dropout": None, "jitter_pct_of_height": None, "note": "old loop does not record body points"}
    missing = total = 0
    for key, g in gt["frames"].items():
        f = pick(int(key)) or {"people": []}
        gts = [p for p in g["people"] if not band_of_distance(person_distance_m(p["box"])).startswith("far")]
        preds = [{"box": p["box"], "points": p.get("points", [])} for p in f["people"]]
        pairs = match(preds, gts, person_hit)
        hit = {j for _, j in pairs}
        for pi, gj in pairs:
            pts = preds[pi]["points"]
            for k in UPPER_BODY:
                total += 1
                missing += int(not (k < len(pts) and pts[k][2] >= KP))
        for j in range(len(gts)):
            if j not in hit:
                total += len(UPPER_BODY)
                missing += len(UPPER_BODY)
    # jitter: how much the wrists/elbows of a nearly still near person move between frames
    shakes = []
    prev = {}
    for f in frames:
        for p in f["people"]:
            b = p["box"]
            h = max(1.0, b[3] - b[1])
            if h < 450:
                continue
            c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            old = prev.get(p["id"])
            if old is not None and math.hypot(c[0] - old[0][0], c[1] - old[0][1]) < 0.02 * h:
                pts, opts = p.get("points", []), old[1]
                for k in UPPER_BODY:
                    if k < len(pts) and k < len(opts) and pts[k][2] >= KP and opts[k][2] >= KP:
                        shakes.append(math.hypot(pts[k][0] - opts[k][0], pts[k][1] - opts[k][1]) / h)
            prev[p["id"]] = (c, p.get("points", []))
    return {"dropout": missing / total if total else None, "points_checked": total,
            "jitter_pct_of_height": 100 * statistics.median(shakes) if shakes else None}


def speed_scores(frames, threaded, summary, meta):
    if not threaded:
        return {"display_fps": summary.get("fps_avg"), "processed_fps": summary.get("fps_avg"),
                "processed_p5": summary.get("fps_p5"), "detect_fps": summary.get("fps_avg"),
                "capacity_fps": summary.get("fps_avg"), "capacity_p5": summary.get("fps_p5"),
                "delay_ms_avg": summary.get("delay_ms_avg"), "delay_ms_p95": summary.get("delay_ms_p95")}
    duration = frames[-1]["t"] - frames[0]["t"]
    loop = sorted(f["loop_ms"] for f in frames)
    gaps = sorted(b["t"] - a["t"] for a, b in zip(frames, frames[1:]))
    delays = sorted(f["delay_ms"] for f in frames if "delay_ms" in f)
    return {"camera_fps": meta.get("camera_fps"),
            "display_fps": len(frames) / duration,
            "processed_fps": len(frames) / duration,
            "processed_p5": 1.0 / gaps[int(0.95 * (len(gaps) - 1))],
            "detect_fps": sum(1 for f in frames if f["fresh"]) / duration,
            "capacity_fps": 1000.0 / statistics.mean(loop),
            "capacity_p5": 1000.0 / loop[int(0.95 * (len(loop) - 1))],
            "delay_ms_avg": statistics.mean(delays) if delays else None,
            "delay_ms_p95": delays[int(0.95 * (len(delays) - 1))] if delays else None}


def score(name):
    frames, summary, meta = load(name)
    gt = json.load(open(GT))
    pick, threaded = picker(frames)
    alerts = summary.get("alerts") or [a for f in frames for a in f.get("alerts", [])]
    return {"speed": speed_scores(frames, threaded, summary, meta), "detection": detection_scores(frames, gt, pick),
            "tracking": tracking_scores(frames, gt, pick), "limbs": limb_scores(frames, gt, pick, threaded),
            "alerts": [(a["kind"], a["time"], a["person"]) for a in alerts], "meta": meta}


def pct(x):
    return "-" if x is None else str(round(100 * x, 1)) + "%"


def num(x, d=2):
    return "-" if x is None else str(round(x, d))


def show(name, s):
    sp, de, tr, li = s["speed"], s["detection"], s["tracking"], s["limbs"]
    print("==", name, "==")
    print("  speed: display", num(sp.get("display_fps")), "fps | processed", num(sp.get("processed_fps")), "avg /",
          num(sp.get("processed_p5")), "p5 | AI checks", num(sp.get("detect_fps")), "/s | loop capacity",
          num(sp.get("capacity_fps")), "avg /", num(sp.get("capacity_p5")), "p5 | delay ms",
          num(sp.get("delay_ms_avg"), 0), "avg /", num(sp.get("delay_ms_p95"), 0), "p95")
    p = de["person"]
    print("  people: P", pct(p["precision"]), "R", pct(p["recall"]), "|",
          ", ".join(k + " " + pct(v[0]) + " (n=" + str(v[1]) + ")" for k, v in p["recall_by"].items()))
    d = de["device"]
    print("  device: P", pct(d["precision"]), "R", pct(d["recall"]), "| >=half visible", pct(d["recall_half_visible_or_more"]),
          "|", ", ".join(k + " " + pct(v[0]) + " (n=" + str(v[1]) + ")" for k, v in d["recall_by"].items()),
          "| false devices per 10 min", num(d["fp_per_10min"], 1))
    for c in ["headphones", "paper", "pen"]:
        print("  " + c + ": P", pct(de[c]["precision"]), "R", pct(de[c]["recall"]))
    print("  tracking: numbers", tr["unique_numbers"], "for", tr["headcount"], "people =", num(tr["numbers_per_person"]),
          "| overlap ID switches", tr["overlap_switches"], "/", num(tr["overlap_person_min"]), "person-min =",
          num(tr["overlap_idsw_per_person_min"]), "| same number after leaving", tr["returns_same_number"], "of", tr["returns"])
    print("  limbs: upper-body point dropout", pct(li["dropout"]), "| jitter", num(li.get("jitter_pct_of_height")), "% of height")
    print("  alerts:", s["alerts"])


if __name__ == "__main__":
    for n in sys.argv[1:]:
        s = score(n)
        json.dump(s, open("perf/runs/" + n + ".score.json", "w"), indent=1)
        show(n, s)
