# The "brain" decides who is cheating.
# Every person has their own memory (a dictionary) that we update every frame.
#
# Big idea:
#   - We do NOT need to know WHO the student is. We follow each person's box
#     from one picture to the next and call them "Person 1", "Person 2"...
#   - Every person has a different "normal" head angle (because the camera sees
#     each person from a different side). So we learn each person's normal angle
#     and only care when they turn AWAY from their own normal.

import statistics
from settings import config


def new_person_state(name, box):
    return {
        "name": name,
        "box": box,
        "last_seen": None,       # when we last saw this person
        "present": False,
        "recent_yaws": [],       # last few head angles (for smoothing)
        "yaw_history": [],       # (time, angle) for the last 2 minutes
        "yaw_now": None,
        "last_yaw_time": None,   # when we last saw the head angle
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
        "alert_count": 0,        # alerts for this person this session (each event counted once)
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


# ---------------- following people / matching phones ----------------

def match_people_by_track(people, people_states, now):
    # The tracker (vision/tracker.py) already gave every person a session number,
    # so "Person N" is simply tracker number N.
    matches = {}
    names = {s["name"]: s for s in people_states}
    for p in people:
        name = "Person " + str(p["track_id"])
        if name not in names:
            names[name] = new_person_state(name, p["box"])
            people_states.append(names[name])
        matches[name] = p
        names[name]["box"] = p["box"]
        names[name]["last_seen"] = now
    return matches


def match_people(people, people_states, now):
    # gives back a dictionary: person name -> detected person
    # Each detected person goes to the remembered person whose last box overlaps it most.
    # Someone nobody matches becomes a new "Person N".
    if len(people) > 0 and all("track_id" in p for p in people):
        return match_people_by_track(people, people_states, now)
    matches = {}
    taken = set()
    pairs = []
    for i in range(len(people)):
        for s in people_states:
            if s["last_seen"] is not None and now - s["last_seen"] > config.FORGET_PERSON_SECONDS:
                continue   # gone too long, don't reuse this name
            a = people[i]["box"]
            b = s["box"]
            inter = overlap_area(a, b)
            union = box_area(a) + box_area(b) - inter
            if union > 0 and inter / union >= 0.3:
                pairs.append((inter / union, i, s["name"]))
    # best overlaps first, so each person and each name is used once
    pairs.sort(reverse=True)
    for overlap, i, name in pairs:
        if i in taken or name in matches:
            continue
        matches[name] = people[i]
        taken.add(i)
    for i in range(len(people)):
        if i not in taken:
            name = "Person " + str(len(people_states) + 1)
            people_states.append(new_person_state(name, people[i]["box"]))
            matches[name] = people[i]
    for s in people_states:
        if s["name"] in matches:
            s["box"] = matches[s["name"]]["box"]
            s["last_seen"] = now
    return matches


def match_phones_to_people(phones, matches, labels=None):
    # gives back a set of person names that have a phone
    people_with_phone = set()
    for ph in phones:
        b = ph["box"]
        cx = (b[0] + b[2]) / 2
        cy = (b[1] + b[3]) / 2
        found = None
        # is the phone near a person? (if near 2 people, take the closest one)
        closest = None
        for name in matches:
            pb = matches[name]["box"]
            if point_in_box(cx, cy, grow_box(pb, 0.2)):
                dist = (cx - (pb[0] + pb[2]) / 2) ** 2 + (cy - (pb[1] + pb[3]) / 2) ** 2
                if closest is None or dist < closest:
                    closest = dist
                    found = name
        if found is not None:
            people_with_phone.add(found)
            if labels is not None:
                labels[found] = ph.get("label", "phone")   # "phone", "smartwatch", "earbuds"...
    return people_with_phone


# ---------------- the main brain update ----------------

def make_alert(state, kind, message, now, alerts):
    # gives back True if a new alert was made (False if it is still cooling down)
    last = state["last_alert"].get(kind)
    if last is not None and now - last < config.ALERT_COOLDOWN_SECONDS:
        return False
    state["last_alert"][kind] = now
    state["last_any_alert"] = now
    state["alert_count"] = state["alert_count"] + 1   # one event = one count (the cooldown stops repeats)
    alerts.append({"person": state["name"], "kind": kind, "message": message, "time": now})
    return True


def merge_people(people_states, kept_name, gone_name):
    # The tracker found that two numbers were the same person: keep the older number,
    # add the newer number's alert count and totals to it, and forget the newer one.
    kept = None
    gone = None
    for s in people_states:
        if s["name"] == kept_name:
            kept = s
        if s["name"] == gone_name:
            gone = s
    if kept is None or gone is None or kept is gone:
        return
    for key in ["alert_count", "total_looks", "total_phone_alerts", "total_long_looks"]:
        kept[key] = kept[key] + gone[key]
    people_states.remove(gone)


def update_person(state, person, phone_seen, now, device="phone"):
    alerts = []

    # ----- phone check -----
    if phone_seen:
        state["phone_times"].append(now)
        state["last_device"] = device
    state["phone_times"] = [t for t in state["phone_times"] if now - t <= config.PHONE_WINDOW_SECONDS]
    if len(state["phone_times"]) >= config.PHONE_MIN_HITS:
        if make_alert(state, "device", state.get("last_device", "device").capitalize() + " seen at " + state["name"], now, alerts):
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
            # the face was hidden for a while (head down writing): the old angles and the
            # look-away timer are out of date, so start fresh
            if state["last_yaw_time"] is not None and now - state["last_yaw_time"] > config.LOOK_AWAY_SECONDS:
                state["recent_yaws"] = []
                state["away_start"] = None
                state["long_alerted"] = False
            state["last_yaw_time"] = now

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
                        # only mark it as alerted when the alert really went out (not blocked by the cooldown)
                        if make_alert(state, "long_look",
                                      state["name"] + " stared " + side + " for " + str(round(how_long, 1)) + "s",
                                      now, alerts):
                            state["long_alerted"] = True
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

    # ----- color -----
    if state["last_any_alert"] is not None and now - state["last_any_alert"] < 60:
        state["status"] = "red"
    elif not state["present"]:
        state["status"] = "empty"
    elif state["away_start"] is not None or len(state["look_events"]) > 0:
        state["status"] = "yellow"
    else:
        state["status"] = "green"

    return alerts


# items that never trigger the device alert (they are reported, not alerted on)
NOT_ALERT_ITEMS = ["paper", "pen"]


def update_all(people_states, people, phones, now):
    matches = match_people(people, people_states, now)
    phones = [ph for ph in phones if ph.get("label") not in NOT_ALERT_ITEMS]
    device_labels = {}
    phone_people = match_phones_to_people(phones, matches, device_labels)
    all_alerts = []
    for s in people_states:
        person = matches.get(s["name"])
        phone_seen = s["name"] in phone_people
        device = device_labels.get(s["name"], "phone")
        all_alerts = all_alerts + update_person(s, person, phone_seen, now, device)
    return all_alerts, matches
