# Follows people from picture to picture and gives each one a session number.
#
# Ideas from the SORT / ByteTrack videos plus the bug list:
#   - Between AI checks, every box (and its body points) moves with optical flow, so the
#     AI does not have to run on every picture.
#   - A new person only gets a number after being seen CONFIRM_HITS times (no numbers for
#     one-frame flickers).
#   - Matching is done in 2 passes: sure detections first, then unsure ones may only keep an
#     existing person alive (they never start a new one).
#   - Someone who disappears goes to a "lost" pool for LOST_SECONDS. When a person shows up
#     again near where they left, with a similar size and similar clothes colours, they get
#     their OLD number back.
#   - Two numbers that turn out to be the same person (boxes on top of each other for a
#     while) are merged: the older number stays.
#   - Body points are smoothed with a One Euro filter (steady when still, quick when moving).
#
# Session only: the clothes-colour histograms live in memory and reset() throws them away.
# No face recognition, no face data, nothing saved to disk.

import math
import cv2
import numpy as np

FLOW_WIDTH = 640           # optical flow runs on a small copy of the picture
FLOW_POINTS = 25           # corner points followed inside each box
MATCH_IOU = 0.25           # boxes must overlap this much to be the same person
CONFIRM_HITS = 3           # sightings before a new person gets a number
TENTATIVE_SECONDS = 8.0    # a not-yet-confirmed person missing this long is dropped (far people are checked every ~3 s)
SMALL_PERSON = 0.3         # boxes shorter than this x picture height are 'far' people
LOST_SECONDS = 45.0        # a numbered person is remembered this long after leaving
REVIVE_DISTANCE = 2.5      # a returning person may be this many box heights from where they left
REVIVE_SIZE = (0.5, 2.0)   # ...with a box height between half and double the old one
REVIVE_COLOUR = 0.45       # ...and clothes colours at least this similar (0..1, 1 = same)
MERGE_IOU = 0.7            # two numbered boxes this much on top of each other ...
MERGE_UPDATES = 3          # ... for this many AI checks in a row are one person
SHOW_MISSES = 1            # a person the AI missed once is still shown (flow-moved)


# ---------------- small helpers ----------------

def box_area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = box_area([x1, y1, x2, y2])
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0


def center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def torso_colours(frame, box):
    # hue/saturation histogram of the middle of the box (clothes), for re-finding a person
    if frame is None:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx1, cx2 = int(max(0, x1 + bw * 0.25)), int(min(w, x2 - bw * 0.25))
    cy1, cy2 = int(max(0, y1 + bh * 0.25)), int(min(h, y1 + bh * 0.7))
    if cx2 - cx1 < 6 or cy2 - cy1 < 6:
        return None
    patch = cv2.cvtColor(frame[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([patch], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)
    return hist


def colour_similarity(a, b):
    if a is None or b is None:
        return 0.5   # unknown: neither helps nor blocks
    return 1.0 - cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA)


class OneEuro:
    # One Euro filter for one number: steady when it barely moves, follows fast moves quickly
    def __init__(self, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x = None
        self.dx = 0.0
        self.t = None

    @staticmethod
    def alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x is None or self.t is None or t <= self.t:
            self.x, self.t = x, t
            return x
        dt = t - self.t
        dx = (x - self.x) / dt
        a_d = self.alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * abs(self.dx)
        a = self.alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x


def no_events(kind, **info):
    pass


# ---------------- the tracker ----------------

class Tracker:
    def __init__(self, new_confidence, events=None):
        self.new_confidence = new_confidence   # a detection this sure may start a new person
        self.events = events or no_events
        self.reset()

    def reset(self):
        # call when the session ends: forgets every person, number and colour histogram
        self.active = []
        self.lost = []
        self.next_key = 1          # internal key for every track, numbered or not
        self.next_number = 1       # the session numbers people see ("Person N")
        self.prev_gray = None
        self.frame = None
        self.scale = 1.0
        self.merges = []           # (kept number, merged number) pairs for the brain
        self.far_checked = True

    # ---------- every picture ----------

    def predict(self, frame, now):
        h, w = frame.shape[:2]
        self.scale = FLOW_WIDTH / w
        small = cv2.resize(frame, (FLOW_WIDTH, int(h * self.scale)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is not None:
            for t in self.active:
                self.flow_move(t, self.prev_gray, gray)
        self.prev_gray = gray
        self.frame = frame

    def flow_move(self, track, old_gray, new_gray):
        s = self.scale
        b = [v * s for v in track["box"]]
        x1, y1 = max(0, int(b[0])), max(0, int(b[1]))
        x2, y2 = min(old_gray.shape[1], int(b[2])), min(old_gray.shape[0], int(b[3]))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return
        mask = np.zeros_like(old_gray)
        mask[y1:y2, x1:x2] = 255
        pts = cv2.goodFeaturesToTrack(old_gray, FLOW_POINTS, 0.01, 4, mask=mask)
        if pts is None or len(pts) < 4:
            return
        new_pts, status, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, pts, None, winSize=(21, 21), maxLevel=3)
        good = status.reshape(-1) == 1
        if good.sum() < 4:
            return
        moves = (new_pts[good] - pts[good]).reshape(-1, 2)
        dx = float(np.median(moves[:, 0])) / s
        dy = float(np.median(moves[:, 1])) / s
        track["box"] = [track["box"][0] + dx, track["box"][1] + dy, track["box"][2] + dx, track["box"][3] + dy]
        for p in track["points"]:
            p[0] += dx
            p[1] += dy

    # ---------- when the AI looked ----------

    def update(self, people, now, far_checked=True):
        # far_checked: did this AI check look closely at the far band? If not, far people
        # it didn't see are not counted as missed (they were simply not looked for).
        self.far_checked = far_checked
        sure = [p for p in people if p["conf"] >= self.new_confidence]
        unsure = [p for p in people if p["conf"] < self.new_confidence]
        left = self.match(sure, self.active, now)
        rest = [t for t in self.active if t["seen"] != now]
        self.match(unsure, rest, now)
        left = self.revive(left, now)
        for d in left:
            self.add(d, now)
        self.age_tracks(now)
        self.merge_doubles(now)

    def match(self, dets, tracks, now):
        pairs = []
        for i, d in enumerate(dets):
            for t in tracks:
                s = iou(d["box"], t["box"])
                if s >= MATCH_IOU:
                    pairs.append((s, i, t["key"]))
        pairs.sort(reverse=True)
        used_d, used_t = set(), set()
        by_key = {t["key"]: t for t in tracks}
        for s, i, key in pairs:
            if i in used_d or key in used_t:
                continue
            used_d.add(i)
            used_t.add(key)
            self.refresh(by_key[key], dets[i], now)
        return [d for i, d in enumerate(dets) if i not in used_d]

    def refresh(self, t, d, now):
        t["box"] = list(d["box"])
        t["points"] = self.smooth_points(t, d.get("points", []), now)
        t["conf"] = d["conf"]
        t["seen"] = now
        t["hits"] += 1
        t["misses"] = 0
        if t["hits"] % 5 == 1:
            c = torso_colours(self.frame, t["box"])
            if c is not None:
                t["colours"] = c if t["colours"] is None else 0.8 * t["colours"] + 0.2 * c
        if t["number"] is None and t["hits"] >= CONFIRM_HITS:
            t["number"] = self.next_number
            self.next_number += 1
            self.events("confirmed", number=t["number"], key=t["key"], reason="seen " + str(t["hits"]) + " times")

    def smooth_points(self, t, points, now):
        if len(t["filters"]) != len(points):
            t["filters"] = [(OneEuro(), OneEuro()) for p in points]
        out = []
        for (fx, fy), p in zip(t["filters"], points):
            if p[2] < 0.3:
                out.append([p[0], p[1], p[2]])   # unsure point: don't drag the filter with it
                continue
            out.append([fx(p[0], now), fy(p[1], now), p[2]])
        return out

    def revive(self, dets, now):
        # a sure detection with no match may be a numbered person coming back
        left = []
        for d in dets:
            best, best_value = None, 0.0
            dx, dy = center(d["box"])
            dh = d["box"][3] - d["box"][1]
            colours = torso_colours(self.frame, d["box"])
            for t in self.lost:
                tx, ty = center(t["box"])
                th = t["box"][3] - t["box"][1]
                if th <= 0 or not (REVIVE_SIZE[0] <= dh / th <= REVIVE_SIZE[1]):
                    continue
                dist = math.hypot(tx - dx, ty - dy) / th
                if dist > REVIVE_DISTANCE:
                    continue
                sim = colour_similarity(colours, t["colours"])
                if sim < REVIVE_COLOUR:
                    continue
                value = sim - 0.1 * dist
                if value > best_value:
                    best, best_value = t, value
            if best is None:
                left.append(d)
                continue
            self.lost.remove(best)
            self.active.append(best)
            gone = round(now - best["seen"], 1)
            self.refresh(best, d, now)
            self.events("revived", number=best["number"], key=best["key"], reason="back after " + str(gone) + "s")
        return left

    def add(self, d, now):
        t = {"key": self.next_key, "number": None, "box": list(d["box"]), "conf": d["conf"],
             "points": [list(p) for p in d.get("points", [])], "filters": [], "seen": now, "born": now,
             "hits": 1, "misses": 0, "colours": torso_colours(self.frame, d["box"]), "overlap": {}}
        self.next_key += 1
        self.active.append(t)
        self.events("created", key=t["key"], reason="new detection conf " + str(round(d["conf"], 2)))
        if CONFIRM_HITS <= 1:
            t["number"] = self.next_number
            self.next_number += 1

    def age_tracks(self, now):
        keep = []
        for t in self.active:
            if t["seen"] == now:
                keep.append(t)
                continue
            if not self.far_checked and self.is_small(t):
                keep.append(t)
                continue
            t["misses"] += 1
            if t["number"] is None:
                if now - t["seen"] <= TENTATIVE_SECONDS:
                    keep.append(t)
                continue
            if t["misses"] > SHOW_MISSES:
                self.lost.append(t)
                self.events("lost", number=t["number"], key=t["key"], reason="missed " + str(t["misses"]) + " checks")
            else:
                keep.append(t)
        self.active = keep
        self.lost = [t for t in self.lost if now - t["seen"] <= LOST_SECONDS]

    def is_small(self, t):
        if self.frame is None:
            return False
        return t["box"][3] - t["box"][1] < SMALL_PERSON * self.frame.shape[0]

    def merge_doubles(self, now):
        numbered = [t for t in self.active if t["number"] is not None]
        drop = set()
        for i in range(len(numbered)):
            for j in range(i + 1, len(numbered)):
                a, b = numbered[i], numbered[j]
                if a["key"] in drop or b["key"] in drop:
                    continue
                if iou(a["box"], b["box"]) >= MERGE_IOU:
                    a["overlap"][b["key"]] = a["overlap"].get(b["key"], 0) + 1
                else:
                    a["overlap"].pop(b["key"], None)
                if a["overlap"].get(b["key"], 0) >= MERGE_UPDATES:
                    keep, gone = (a, b) if a["number"] < b["number"] else (b, a)
                    drop.add(gone["key"])
                    self.merges.append((keep["number"], gone["number"]))
                    self.events("merged", kept=keep["number"], merged=gone["number"], reason="same person")
        self.active = [t for t in self.active if t["key"] not in drop]

    # ---------- result ----------

    def current(self):
        # numbered people on screen now (seen by the last AI check, or missed once and flow-moved)
        out = []
        for t in self.active:
            if t["number"] is None:
                continue
            out.append({"box": list(t["box"]), "conf": t["conf"], "points": [list(p) for p in t["points"]],
                        "track_id": t["number"], "fresh": t["misses"] == 0})
        return out

    def take_merges(self):
        m = self.merges
        self.merges = []
        return m
