# Builds research/labels/gt.json from the helper model's boxes (teacher.json) and my
# corrections after looking at each review frame (review.json). Boxes only, no images.
#
#   .\.venv-sandbox\Scripts\python.exe research\labels\build_gt.py
#
# review.json, per review frame index:
#   "drop":   teacher box numbers that are wrong
#   "gid":    teacher box number -> person letter (same letter = same real person all clip long)
#   "facing": teacher box number or added letter -> "front" | "side" | "back"
#   "add":    people the helper missed: [letter, x1, y1, x2, y2, facing]
#   "items":  [label, x1, y1, x2, y2]   (label: phone, watch, headphones, paper, pen)
# Dense frames (people only) get letters by following each letter's last box (IoU), and
# hand-added people (far, seated, they barely move) are copied from the nearest review frame.

import json

import sys
FOLDER = sys.argv[1] if len(sys.argv) > 1 else "research/labels"
TEACHER = FOLDER + "/teacher.json"
REVIEW = FOLDER + "/review.json"
OUT = FOLDER + "/gt.json"


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0


def review_people(teacher_frame, rv):
    people = []
    # numbers are the ones drawn on the review picture (all helper boxes, people and phones)
    for n, d in enumerate(teacher_frame["dets"]):
        if d["label"] != "person" or n in rv.get("drop", []):
            continue
        gid = rv.get("gid", {}).get(str(n))
        if gid is None:
            continue
        people.append({"gid": gid, "box": d["box"], "facing": rv.get("facing", {}).get(str(n), "front"),
                       "added": False})
    for a in rv.get("add", []):
        people.append({"gid": a[0], "box": [float(v) for v in a[1:5]], "facing": a[5] if len(a) > 5 else "front",
                       "added": True})
    return people


def main():
    teacher = json.load(open(TEACHER))
    review = json.load(open(REVIEW))
    frames = {}
    for key, rv in review.items():
        tf = teacher[key]
        frames[key] = {"t": tf["t"],
                       "people": review_people(tf, rv),
                       "items": [{"label": it[0], "box": [float(v) for v in it[1:5]],
                                  "vis": it[5] if len(it) > 5 else "full",       # full / half / edge
                                  "kind": it[6] if len(it) > 6 else it[0]}       # phone / watch for devices
                                 for it in rv.get("items", [])]}
    review_keys = sorted(int(k) for k in frames)

    dense = {}
    last = {}
    for key in sorted(teacher, key=int):
        k = int(key)
        if key in frames:
            people = [{"gid": p["gid"], "box": p["box"]} for p in frames[key]["people"]]
            for p in people:
                last[p["gid"]] = p["box"]
            dense[key] = people
            continue
        nearest = min(review_keys, key=lambda r: abs(r - k))
        people = []
        used = set()
        for d in [d for d in teacher[key]["dets"] if d["label"] == "person"]:
            best, best_gid = 0.3, None
            for gid, box in last.items():
                if gid in used:
                    continue
                s = iou(d["box"], box)
                if s > best:
                    best, best_gid = s, gid
            if best_gid is not None:
                used.add(best_gid)
                last[best_gid] = d["box"]
                people.append({"gid": best_gid, "box": d["box"]})
        for p in frames[str(nearest)]["people"]:
            if p["added"] and p["gid"] not in used:
                people.append({"gid": p["gid"], "box": p["box"]})
        dense[key] = people

    json.dump({"frames": frames, "dense": dense}, open(OUT, "w"), indent=0)
    n_people = sum(len(f["people"]) for f in frames.values())
    n_items = sum(len(f["items"]) for f in frames.values())
    print("review frames", len(frames), "people", n_people, "items", n_items, "dense frames", len(dense))


if __name__ == "__main__":
    main()
