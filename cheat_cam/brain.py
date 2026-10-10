# The "brain" decides who is cheating.
# Every desk has its own memory (a dictionary) that we update every frame.
#
# Big idea:
#   - We do NOT need to know WHO the student is. The desk IS the ID.
#   - Every person has a different "normal" head angle (because the camera sees
#     each desk from a different side). So we learn each person's normal angle
#     and only care when they turn AWAY from their own normal.

import json
import statistics
import config


# ---------------- desks ----------------

def load_desks(frame_w, frame_h):
    with open(config.DESKS_FILE) as f:
        data = json.load(f)

    # if the camera picture size changed since setup, stretch the boxes to fit
    sx = 1.0
    sy = 1.0
    if data.get("frame_width") and data.get("frame_height"):
        sx = frame_w / data["frame_width"]
        sy = frame_h / data["frame_height"]

    states = []
    for d in data["desks"]:
        b = d["box"]
        box = [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
        states.append(new_desk_state(d["name"], box))
    return states


def new_desk_state(name, box):
    return {
        "name": name,
        "box": box,
        "present": False,
        "recent_yaws": [],       # last few head angles (for smoothing)
        "yaw_history": [],       # (time, angle) for the last 2 minutes
        "yaw_now": None,
        "baseline": None,
        "deviation": None,
        "away_start": None,      # when they started looking away
        "away_counted": False,
        "long_alerted": False,
        "look_events": [],       # times of each "look away"
        "phone_times": [],       # times a phone was seen
        "last_alert": {},        # kind -> time (for the cooldown)
        "last_any_alert": None,
        "total_looks": 0,
        "total_phone_alerts": 0,
        "total_long_looks": 0,
        "score": 0,
        "status": "empty",
    }


# ---------------- small math helpers ----------------

def box_area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def overlap_area(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return box_area([x1, y1, x2, y2])


def point_in_box(x, y, b):
    return b[0] <= x <= b[2] and b[1] <= y <= b[3]


def grow_box(b, amount):
    w = b[2] - b[0]
    h = b[3] - b[1]
    return [b[0] - w * amount, b[1] - h * amount, b[2] + w * amount, b[3] + h * amount]


def head_yaw(points):
    # Gives a number for how much the head is turned left/right.
    # About 0 = facing the camera, negative = turned one way, positive = the other way.
    # Idea: the nose sits in the middle of the eyes/ears when you look straight.
    # When you turn your head, the nose moves to one side.
    if len(points) < 5:
        return None
    nose = points[0]
    if nose[2] < config.KEYPOINT_CONFIDENCE:
        return None

    xs = []
    for k in [1, 2, 3, 4]:   # eyes and ears
        if points[k][2] >= config.KEYPOINT_CONFIDENCE:
            xs.append(points[k][0])
    if len(xs) < 2:
        return None

    center = sum(xs) / len(xs)
    width = max(xs) - min(xs)
    if width < 1:
        return None

    yaw = (nose[0] - center) / width
    if yaw > 1:
        yaw = 1
    if yaw < -1:
        yaw = -1
    return yaw


# ---------------- matching people / phones to desks ----------------

def match_people_to_desks(people, desk_states):
    # gives back a dictionary: desk name -> person
    matches = {}
    best_overlap = {}
    for p in people:
        p_area = box_area(p["box"])
        if p_area <= 0:
            continue
        best_desk = None
        best = 0
        for d in desk_states:
            o = overlap_area(p["box"], d["box"]) / p_area
            if o > best:
                best = o
                best_desk = d["name"]
        if best_desk is None or best < 0.3:
            continue   # this person is not at any desk (maybe the proctor walking)
        # if 2 people match the same desk, keep the one that fits better
        if best_desk not in matches or best > best_overlap[best_desk]:
            matches[best_desk] = p
            best_overlap[best_desk] = best
    return matches


def match_phones_to_desks(phones, desk_states, matches):
    # gives back {desk name: "phone" or "smartwatch"} for desks that have a device
    desks_with_phone = {}
    for ph in phones:
        b = ph["box"]
        cx = (b[0] + b[2]) / 2
        cy = (b[1] + b[3]) / 2
        found = None
        # first: is the phone near a person we know?
        for name in matches:
            if point_in_box(cx, cy, grow_box(matches[name]["box"], 0.2)):
                found = name
                break
        # second: is the phone inside a desk box?
        if found is None:
            for d in desk_states:
                if point_in_box(cx, cy, d["box"]):
                    found = d["name"]
                    break
        if found is not None:
            desks_with_phone[found] = ph.get("label", "phone")
    return desks_with_phone


# ---------------- the main brain update ----------------

def make_alert(state, kind, message, now, alerts):
    # gives back True if a new alert was made (False if it is still cooling down)
    last = state["last_alert"].get(kind)
    if last is not None and now - last < config.ALERT_COOLDOWN_SECONDS:
        return False
    state["last_alert"][kind] = now
    state["last_any_alert"] = now
    alerts.append({"desk": state["name"], "kind": kind, "message": message, "time": now})
    return True


def update_desk(state, person, phone_seen, now, device="phone"):
    alerts = []

    # ----- phone check (works even if the person is hidden) -----
    if phone_seen:
        state["phone_times"].append(now)
        state["last_device"] = device
    state["phone_times"] = [t for t in state["phone_times"] if now - t <= config.PHONE_WINDOW_SECONDS]
    if len(state["phone_times"]) >= config.PHONE_MIN_HITS:
        if make_alert(state, "phone", state.get("last_device", "phone").capitalize() + " seen at " + state["name"], now, alerts):
            state["total_phone_alerts"] = state["total_phone_alerts"] + 1

    # ----- head turning check -----
    if person is None:
        state["present"] = False
        state["away_start"] = None
        state["long_alerted"] = False
        state["yaw_now"] = None
        state["deviation"] = None
    else:
        state["present"] = True
        yaw = head_yaw(person["points"])
        if yaw is not None:
            state["recent_yaws"].append(yaw)
            if len(state["recent_yaws"]) > config.YAW_SMOOTH_FRAMES:
                state["recent_yaws"].pop(0)
            smooth = statistics.median(state["recent_yaws"])
            state["yaw_now"] = smooth

            state["yaw_history"].append((now, smooth))
            state["yaw_history"] = [h for h in state["yaw_history"] if now - h[0] <= config.BASELINE_SECONDS]

            if len(state["yaw_history"]) >= config.BASELINE_MIN_SAMPLES:
                baseline = statistics.median([h[1] for h in state["yaw_history"]])
                dev = smooth - baseline
                state["baseline"] = baseline
                state["deviation"] = dev

                if abs(dev) > config.YAW_THRESHOLD:
                    if state["away_start"] is None:
                        state["away_start"] = now
                        state["away_counted"] = False
                    how_long = now - state["away_start"]

                    if how_long >= config.LOOK_AWAY_SECONDS and not state["away_counted"]:
                        state["look_events"].append(now)
                        state["away_counted"] = True
                        state["total_looks"] = state["total_looks"] + 1

                    if how_long >= config.LONG_LOOK_SECONDS and not state["long_alerted"]:
                        side = "left" if dev < 0 else "right"
                        state["long_alerted"] = True
                        if make_alert(state, "long_look",
                                      state["name"] + " stared " + side + " for " + str(round(how_long, 1)) + "s",
                                      now, alerts):
                            state["total_long_looks"] = state["total_long_looks"] + 1
                else:
                    state["away_start"] = None
                    state["long_alerted"] = False

    # ----- too many look aways -----
    state["look_events"] = [t for t in state["look_events"] if now - t <= config.LOOK_COUNT_WINDOW]
    if len(state["look_events"]) >= config.LOOK_COUNT_LIMIT:
        minutes = round(config.LOOK_COUNT_WINDOW / 60, 1)
        if make_alert(state, "looking",
                      state["name"] + " looked at other desks " + str(len(state["look_events"])) +
                      " times in " + str(minutes) + " min", now, alerts):
            state["look_events"] = []   # start counting again

    # ----- score and color -----
    state["score"] = state["total_looks"] + 2 * state["total_long_looks"] + 5 * state["total_phone_alerts"]

    if state["last_any_alert"] is not None and now - state["last_any_alert"] < 60:
        state["status"] = "red"
    elif not state["present"]:
        state["status"] = "empty"
    elif state["away_start"] is not None or len(state["look_events"]) > 0:
        state["status"] = "yellow"
    else:
        state["status"] = "green"

    return alerts


def update_all(desk_states, people, phones, now):
    matches = match_people_to_desks(people, desk_states)
    phone_desks = match_phones_to_desks(phones, desk_states, matches)
    all_alerts = []
    for d in desk_states:
        person = matches.get(d["name"])
        phone_seen = d["name"] in phone_desks
        device = phone_desks.get(d["name"], "phone")
        all_alerts = all_alerts + update_desk(d, person, phone_seen, now, device)
    return all_alerts, matches
