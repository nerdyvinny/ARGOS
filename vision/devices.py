# Keeps detected items steady between AI checks and decides which ones are real.
#
#   - "phone" and "smartwatch" are one class now: "device" (kind = phone or watch).
#     Headphones, paper and pen stay their own classes.
#   - Between AI checks the last result is held, so items don't flash on and off.
#   - With confirm on: an item is only reported after it was seen in CONFIRM_HITS of the last
#     CONFIRM_WINDOW AI checks (temporal confirmation), and a "watch" must pass size/shape checks
#     and sit near a wrist (this removes most wristband / sleeve / shadow false alarms).

DEVICE_KIND = {"phone": "phone", "cell phone": "phone", "smartwatch": "watch", "watch": "watch", "device": "phone"}
CONFIRM_HITS = 2
CONFIRM_WINDOW = 3
WATCH_MAX_AREA = 0.03       # a watch is at most 3% of its person's box
WATCH_ASPECT = (0.3, 3.3)   # width / height
WATCH_WRIST_DISTANCE = 0.5  # watch centre within this many person-box widths of a wrist point
WRISTS = [9, 10]


def normalise(item):
    label = item.get("label", "phone")
    out = dict(item)
    if label in DEVICE_KIND:
        out["label"] = "device"
        out["kind"] = DEVICE_KIND[label]
    return out


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0


def same_place(a, b):
    if iou(a, b) >= 0.2:
        return True
    size = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1])
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 <= size


def watch_is_plausible(item, people):
    b = item["box"]
    w, h = b[2] - b[0], b[3] - b[1]
    if h <= 0 or not (WATCH_ASPECT[0] <= w / h <= WATCH_ASPECT[1]):
        return False
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    for p in people:
        pb = p["box"]
        pw, ph = pb[2] - pb[0], pb[3] - pb[1]
        if pw <= 0 or w * h > WATCH_MAX_AREA * pw * ph:
            continue
        for k in WRISTS:
            pts = p.get("points", [])
            if k < len(pts) and pts[k][2] >= 0.3:
                if ((pts[k][0] - cx) ** 2 + (pts[k][1] - cy) ** 2) ** 0.5 <= WATCH_WRIST_DISTANCE * pw:
                    return True
    return False


class DeviceMemory:
    def __init__(self, confirm=False):
        self.confirm = confirm
        self.entries = []   # {"label", "kind", "box", "conf", "history": [True/False ...]}
        self.held = []

    def update(self, items, people, now):
        items = [normalise(it) for it in items]
        if not self.confirm:
            self.held = items          # just hold the last AI result until the next one
            return
        items = [it for it in items if it.get("kind") != "watch" or watch_is_plausible(it, people)]
        used = set()
        for e in self.entries:
            match = None
            for i, it in enumerate(items):
                if i in used or it["label"] != e["label"]:
                    continue
                if same_place(it["box"], e["box"]):
                    match = i
                    break
            if match is None:
                e["history"].append(False)
            else:
                used.add(match)
                e["box"] = list(items[match]["box"])
                e["conf"] = items[match].get("conf", 0)
                e["kind"] = items[match].get("kind")
                e["history"].append(True)
            e["history"] = e["history"][-CONFIRM_WINDOW:]
        for i, it in enumerate(items):
            if i not in used:
                self.entries.append({"label": it["label"], "kind": it.get("kind"), "box": list(it["box"]),
                                     "conf": it.get("conf", 0), "history": [True]})
        self.entries = [e for e in self.entries if any(e["history"])]

    def current(self):
        if not self.confirm:
            return [dict(it) for it in self.held]
        out = []
        for e in self.entries:
            if sum(e["history"]) >= CONFIRM_HITS and e["history"][-1]:
                out.append({"label": e["label"], "kind": e["kind"], "box": list(e["box"]), "conf": e["conf"]})
        return out

    def reset(self):
        self.entries = []
        self.held = []
