# This file runs the AI models on a picture.
# It gives back:
#   people = list of {"box": [x1,y1,x2,y2], "conf": 0.9, "points": [[x,y,conf], ... 17 points]}
#   phones = list of {"box": [x1,y1,x2,y2], "conf": 0.5, "label": "phone"}
#            (label can also be "smartwatch", "earbuds" or "headphones")
#
# Body points (from YOLO pose, COCO order):
#   0 nose, 1 left eye, 2 right eye, 3 left ear, 4 right ear,
#   5 left shoulder, 6 right shoulder, ... 9 left wrist, 10 right wrist ...
#
# Devices COCO doesn't know (smartwatch, earbuds, headphones):
#   We use YOLO-World, a YOLO model you can give any list of words to look for.
#   It looks at every tile, and also at a close-up square around each person's
#   head and wrists (earbuds and watches are tiny, close-ups help a lot).
#   It also looks for name tags / ID cards, so a name tag the phone model
#   thinks is a phone gets thrown out.
#   Turn it off with WORLD_CHECK = False in settings/config.py.

import os
import numpy as np
from ultralytics import YOLO
from settings import config

# OpenVINO copies of the models run 2-3x faster on an Intel CPU (make them with tools\export_openvino.py).
# When a "<name>_openvino_model" folder is next to the .pt file we use it, otherwise the .pt file.
USE_OPENVINO = getattr(config, "USE_OPENVINO", True)
AI_THREADS = getattr(config, "AI_THREADS", None)   # CPU threads the AI may use (None = all of them)

pose_model = None
object_model = None
world_model = None

# YOLO-World settings (you can override any of these in settings/config.py)
WORLD_CHECK = getattr(config, "WORLD_CHECK", True)
WORLD_MODEL = getattr(config, "WORLD_MODEL", "yolov8s-worldv2.pt")
WORLD_CONFIDENCE = getattr(config, "WORLD_CONFIDENCE", 0.15)   # YOLO-World confidences are low, 0.15 is normal
# word to look for -> label we give it. None = not a device (used to throw out fake phones)
WORLD_WORDS = {
    "smartwatch": "smartwatch",
    "wireless earbuds": "earbuds",
    "headphones": "headphones",
    "name tag": None,
    "id card": None,
    "pen": "pen",                 # reported, never alerted on (see rules/brain.py NOT_ALERT_ITEMS)
    "sheet of paper": "paper",
}
HEAD_POINTS = [0, 1, 2, 3, 4]   # nose, eyes, ears
WRISTS = [9, 10]


def fast_copy(name):
    # gives back the OpenVINO folder for this model if it was made, else the .pt name
    folder = name.replace(".pt", "_openvino_model")
    if USE_OPENVINO and os.path.isdir(folder):
        return folder
    return name


def load_models():
    global pose_model, object_model, world_model
    print("Loading AI models (first time it downloads them)...")
    pose_model = YOLO(fast_copy(config.POSE_MODEL), task="pose")
    object_model = YOLO(fast_copy(config.OBJECT_MODEL), task="detect")
    if WORLD_CHECK:
        world_file = fast_copy(WORLD_MODEL)
        world_model = YOLO(world_file, task="detect")
        if world_file == WORLD_MODEL:
            world_model.set_classes(list(WORLD_WORDS.keys()))   # the OpenVINO copy already has the words
    warm_up()
    if AI_THREADS:
        for m in [pose_model, object_model, world_model]:
            limit_threads(m, AI_THREADS)
        warm_up()   # the rebuilt models compile again on first use
    print("Models loaded")


def limit_threads(m, threads):
    # OpenVINO uses every CPU thread by default, so the screen loop got almost none (it ran at 15 fps
    # with a 30 fps camera). Rebuild the OpenVINO model so it uses only `threads` CPU threads.
    if m is None:
        return
    backend = getattr(getattr(getattr(m, "predictor", None), "model", None), "backend", None)
    compile_fn = getattr(backend, "compile_model", None)
    if compile_fn is None or not hasattr(backend, "ov_compiled_model"):
        return   # not an OpenVINO model (e.g. a .pt file): nothing to limit
    folder = str(getattr(m, "model_name", "") or getattr(m, "ckpt_path", "") or "")
    if not os.path.isdir(folder):
        return
    xml = [f for f in os.listdir(folder) if f.endswith(".xml")]
    if not xml:
        return
    settings = dict(compile_fn.keywords.get("config", {}))
    settings["INFERENCE_NUM_THREADS"] = threads
    # straight on the CPU: "AUTO" hides the thread setting, and the laptop's GPU can't run the models anyway
    backend.ov_compiled_model = compile_fn.func(os.path.join(folder, xml[0]), device_name="CPU", config=settings)


def warm_up():
    # OpenVINO builds its program the first time it sees each input size (the first real
    # check took 18.7 s without this). Run every size the fast mode uses once, on blank pictures.
    shapes = [((config.CAMERA_HEIGHT, config.CAMERA_WIDTH), config.IMAGE_SIZE),
              ((config.CAMERA_HEIGHT // 2, config.CAMERA_WIDTH // 2 + config.CAMERA_WIDTH // 20), config.IMAGE_SIZE),
              ((200, 200), 320)]
    for m in [pose_model, object_model, world_model]:
        if m is None:
            continue
        for (h, w), size in shapes:
            m(np.zeros((h, w, 3), dtype=np.uint8), imgsz=size, verbose=False)


def make_tiles(frame):
    # Cut the picture into pieces. Gives back a list of (x_offset, y_offset, piece)
    h, w = frame.shape[:2]
    top = int(h * config.CROP_TOP)
    bottom = int(h * config.CROP_BOTTOM)
    if bottom <= top:
        top = 0
        bottom = h

    band = frame[top:bottom]

    tiles = []
    if config.TILES <= 1:
        tiles.append((0, top, np.ascontiguousarray(band)))
        return tiles

    tile_w = int(w / config.TILES)
    extra = int(tile_w * config.TILE_OVERLAP)
    for i in range(config.TILES):
        x1 = i * tile_w - extra
        x2 = (i + 1) * tile_w + extra
        if x1 < 0:
            x1 = 0
        if x2 > w:
            x2 = w
        piece = np.ascontiguousarray(band[:, x1:x2])
        tiles.append((x1, top, piece))
    return tiles


def box_area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def overlap_of_smaller(a, b):
    # how much of the smaller box is inside the other box (0 to 1)
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = box_area([x1, y1, x2, y2])
    smaller = min(box_area(a), box_area(b))
    if smaller <= 0:
        return 0
    return inter / smaller


def remove_doubles(things):
    # The same person can show up in 2 tiles (because of the overlap).
    # Keep the biggest box and throw away boxes from ANOTHER tile that are mostly inside it.
    # (Boxes from the same tile are different people, e.g. a student sitting behind another.)
    things = sorted(things, key=lambda t: box_area(t["box"]), reverse=True)
    kept = []
    for t in things:
        is_double = False
        for k in kept:
            if t["tile"] != k["tile"] and overlap_of_smaller(t["box"], k["box"]) > 0.6:
                is_double = True
                break
        if not is_double:
            kept.append(t)
    return kept


def find_people(tiles):
    images = [t[2] for t in tiles]
    results = pose_model(images, conf=config.PERSON_CONFIDENCE, imgsz=config.IMAGE_SIZE, verbose=False)

    people = []
    for i in range(len(results)):
        r = results[i]
        ox = tiles[i][0]
        oy = tiles[i][1]
        if r.boxes is None or len(r.boxes) == 0:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        kpts = None
        if r.keypoints is not None:
            kpts = r.keypoints.data.cpu().numpy()

        for j in range(len(boxes)):
            b = boxes[j]
            box = [float(b[0] + ox), float(b[1] + oy), float(b[2] + ox), float(b[3] + oy)]

            points = []
            if kpts is not None and j < len(kpts):
                for p in kpts[j]:
                    if len(p) >= 3:
                        points.append([float(p[0] + ox), float(p[1] + oy), float(p[2])])
                    else:
                        points.append([float(p[0] + ox), float(p[1] + oy), 1.0])

            people.append({"box": box, "conf": float(confs[j]), "points": points, "tile": i})

    return remove_doubles(people)


def find_phones(tiles):
    images = [t[2] for t in tiles]
    results = object_model(images, conf=config.PHONE_CONFIDENCE, imgsz=config.IMAGE_SIZE,
                           classes=[config.PHONE_CLASS_ID], verbose=False)

    phones = []
    for i in range(len(results)):
        r = results[i]
        ox = tiles[i][0]
        oy = tiles[i][1]
        if r.boxes is None or len(r.boxes) == 0:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        for j in range(len(boxes)):
            b = boxes[j]
            box = [float(b[0] + ox), float(b[1] + oy), float(b[2] + ox), float(b[3] + oy)]
            phones.append({"box": box, "conf": float(confs[j]), "tile": i, "label": "phone"})

    return remove_doubles(phones)


def remove_doubles_per_label(things):
    # Like remove_doubles, but only compares boxes with the same label.
    # So a phone held next to your headphones is NOT thrown away as a "double".
    groups = {}
    for t in things:
        label = t.get("label", "phone")
        if label not in groups:
            groups[label] = []
        groups[label].append(t)
    kept = []
    for label in groups:
        kept = kept + remove_doubles(groups[label])
    return kept


def close_up_squares(people):
    # a square around each head and each wrist, in full-picture numbers
    squares = []
    for p in people:
        b = p["box"]
        person_w = b[2] - b[0]
        pts = p["points"]

        head = [pts[k] for k in HEAD_POINTS if k < len(pts) and pts[k][2] >= config.KEYPOINT_CONFIDENCE]
        if len(head) >= 2:
            xs = [q[0] for q in head]
            ys = [q[1] for q in head]
            side = max((max(xs) - min(xs)) * 2.5, person_w * 0.45)
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            squares.append([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2])

        for k in WRISTS:
            if k < len(pts) and pts[k][2] >= config.KEYPOINT_CONFIDENCE:
                side = person_w * 0.4
                x, y = pts[k][0], pts[k][1]
                squares.append([x - side / 2, y - side / 2, x + side / 2, y + side / 2])
    return squares


def cut_square(frame, sq):
    # cut a square out of the full picture (the part inside the picture)
    h, w = frame.shape[:2]
    x1 = max(0, int(sq[0]))
    y1 = max(0, int(sq[1]))
    x2 = min(w, int(sq[2]))
    y2 = min(h, int(sq[3]))
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None, 0, 0
    return np.ascontiguousarray(frame[y1:y2, x1:x2]), x1, y1


def find_world_things(frame, tiles, people):
    # runs YOLO-World on all tiles + close-ups in one go
    # gives back (devices, not_devices): devices have a label, not_devices are name tags etc.
    images = []
    offsets = []
    tile_ids = []
    for i in range(len(tiles)):
        images.append(tiles[i][2])
        offsets.append((tiles[i][0], tiles[i][1]))
        tile_ids.append(i)
    for sq in close_up_squares(people):
        piece, ox, oy = cut_square(frame, sq)
        if piece is None:
            continue
        images.append(piece)
        offsets.append((ox, oy))
        tile_ids.append(None)   # close-ups are not tiles

    results = world_model(images, conf=WORLD_CONFIDENCE, imgsz=config.IMAGE_SIZE, verbose=False)

    words = list(WORLD_WORDS.keys())
    devices = []
    not_devices = []
    for i in range(len(results)):
        r = results[i]
        ox, oy = offsets[i]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        for j in range(len(boxes)):
            b = boxes[j]
            box = [float(b[0] + ox), float(b[1] + oy), float(b[2] + ox), float(b[3] + oy)]
            label = WORLD_WORDS[words[int(classes[j])]]
            thing = {"box": box, "conf": float(confs[j]), "tile": tile_ids[i], "label": label}
            if label is None:
                not_devices.append(thing)
            else:
                devices.append(thing)
    return devices, not_devices


# ---------- device filter (round 3) ----------
# On the 25 hand-checked frames every false device was YOLO-World calling something a "smartwatch"
# (at its 0.15 cut-off), and in the live test a pen was taken for a phone. The rules below took false
# devices from 12 to 7 there while keeping all 8 real watches. Numbers can be changed in settings/config.py.
PHONE_CONFIDENCE_FULL = getattr(config, "PHONE_CONFIDENCE_FULL", 0.35)        # phone on the whole picture
PHONE_CONFIDENCE_CLOSEUP = getattr(config, "PHONE_CONFIDENCE_CLOSEUP", 0.45)  # close-ups are blown up: be surer
PHONE_MAX_ASPECT = getattr(config, "PHONE_MAX_ASPECT", 3.0)                   # long side / short side; a pen is ~10
WORLD_DEVICE_CONFIDENCE = getattr(config, "WORLD_DEVICE_CONFIDENCE", 0.35)    # smartwatch / earbuds / headphones
PEN_VETO_RATIO = 0.8           # a pen box at the same spot this sure (x the phone's confidence) = it's a pen
REPORT_ONLY = ["pen", "paper"]  # never alerted on, so the filter leaves them alone


def is_real_phone(ph, others):
    if ph["conf"] < (PHONE_CONFIDENCE_CLOSEUP if ph.get("src") == "closeup" else PHONE_CONFIDENCE_FULL):
        return False
    w = ph["box"][2] - ph["box"][0]
    h = ph["box"][3] - ph["box"][1]
    if min(w, h) <= 0 or max(w, h) / min(w, h) > PHONE_MAX_ASPECT:
        return False      # long and thin: a pen, not a phone
    for o in others:
        if o.get("label") == "pen" and overlap_of_smaller(ph["box"], o["box"]) > 0.5                 and o["conf"] >= ph["conf"] * PEN_VETO_RATIO:
            return False  # YOLO-World is about as sure it is a pen
    return True


def filter_devices(items, others):
    # items: phones + YOLO-World things; others: YOLO-World things (pens are used to throw out fake phones)
    kept = []
    for it in items:
        label = it.get("label")
        if label in REPORT_ONLY:
            kept.append(it)
        elif label == "phone":
            if is_real_phone(it, others):
                kept.append(it)
        elif it.get("conf", 0) >= WORLD_DEVICE_CONFIDENCE:
            kept.append(it)
    return kept


def drop_fake_phones(phones, not_devices):
    # a "phone" that is really a name tag / ID card: YOLO-World is surer it's a tag
    kept = []
    for ph in phones:
        fake = False
        for nd in not_devices:
            if overlap_of_smaller(ph["box"], nd["box"]) > 0.6 and nd["conf"] >= ph["conf"]:
                fake = True
                break
        if not fake:
            kept.append(ph)
    return kept


def detect(frame):
    tiles = make_tiles(frame)
    people = find_people(tiles)
    phones = find_phones(tiles)
    if WORLD_CHECK and world_model is not None:
        devices, not_devices = find_world_things(frame, tiles, people)
        phones = drop_fake_phones(phones, not_devices) + devices
        phones = remove_doubles_per_label(phones)
    return people, phones


# ---------------- for the threaded loop (app/runtime.py) ----------------

def detect_raw(frame, stages):
    # Same models as detect(), run by the AI worker. Fills stages with the ms of each model.
    # Gives back (people detections, item detections); the tracker and the device memory
    # turn these into numbered people and steady items.
    from app.trace import Timer
    if DETECT_MODE == "fast":
        return detect_fast(frame, stages)
    with Timer(stages, "resize"):
        tiles = make_tiles(frame)
    with Timer(stages, "pose"):
        people = find_people(tiles)
    with Timer(stages, "objects"):
        phones = find_phones(tiles)
    if WORLD_CHECK and world_model is not None:
        with Timer(stages, "world"):
            devices, not_devices = find_world_things(frame, tiles, people)
        phones = drop_fake_phones(phones, not_devices) + devices
        phones = remove_doubles_per_label(phones)
    return people, phones


def model_devices():
    # which device each model REALLY runs on (asked from OpenVINO after the first run), for the trace
    out = {}
    for name, m in [("pose", pose_model), ("objects", object_model), ("world", world_model)]:
        if m is None:
            continue
        backend = getattr(getattr(m, "predictor", None), "model", None)
        compiled = getattr(backend, "ov_compiled_model", None)
        if compiled is not None:
            try:
                out[name] = "openvino:" + "+".join(compiled.get_property("EXECUTION_DEVICES"))
            except Exception:
                out[name] = "openvino:unknown"
        else:
            out[name] = "torch:" + str(getattr(backend, "device", "cpu"))
    return out


# ---------------- fast mode: no tiles every frame, slow models only where and when needed ----------------
# (bench on this laptop: full-frame pose 63 ms vs 3 tiles 295-480 ms; YOLO-World was ~75% of each check)

DETECT_MODE = getattr(config, "DETECT_MODE", "fast")      # "fast" (normal camera) or "tiles" (360 camera)
LOW_PERSON_CONFIDENCE = 0.25    # unsure people are passed on so the tracker can keep someone alive
FAR_EVERY = 3                   # far-away band tiles: every 3rd AI check (seated people barely move)
FAR_BAND = (0.25, 0.75)         # part of the picture height where far people are
FAR_MAX_HEIGHT = 0.3            # people from the far tiles are kept only if smaller than this x picture height
WORLD_CROPS_EVERY = 2           # YOLO-World on wrist/head close-ups: every 2nd AI check
WORLD_FULL_EVERY = 4            # YOLO-World on the whole picture (headphones, paper...): every 4th check
CROP_SIZE = 320                 # close-ups are small, so the models look at them at 320
fast_count = 0
last_world_crops = []           # YOLO-World results are reused on the checks where it doesn't run
last_world_full = []
far_checked = [True]            # did the last check look at the far tiles? (read by the loop)


def result_people(r, ox, oy, tile):
    people = []
    if r.boxes is None or len(r.boxes) == 0:
        return people
    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    kpts = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None
    for j in range(len(boxes)):
        b = boxes[j]
        points = []
        if kpts is not None and j < len(kpts):
            points = [[float(p[0] + ox), float(p[1] + oy), float(p[2]) if len(p) >= 3 else 1.0] for p in kpts[j]]
        people.append({"box": [float(b[0] + ox), float(b[1] + oy), float(b[2] + ox), float(b[3] + oy)],
                       "conf": float(confs[j]), "points": points, "tile": tile})
    return people


def result_things(r, ox, oy, label_of):
    things = []
    if r.boxes is None or len(r.boxes) == 0:
        return things
    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy()
    for j in range(len(boxes)):
        label = label_of(int(classes[j]))
        if label is None and label_of is not world_label:
            continue
        b = boxes[j]
        things.append({"box": [float(b[0] + ox), float(b[1] + oy), float(b[2] + ox), float(b[3] + oy)],
                       "conf": float(confs[j]), "label": label, "tile": None})
    return things


def world_label(c):
    words = list(WORLD_WORDS.keys())
    return WORLD_WORDS[words[c]] if c < len(words) else None


def far_tiles(frame):
    h, w = frame.shape[:2]
    y1, y2 = int(h * FAR_BAND[0]), int(h * FAR_BAND[1])
    half = w // 2
    extra = int(half * 0.1)
    return [(0, y1, np.ascontiguousarray(frame[y1:y2, 0:half + extra])),
            (half - extra, y1, np.ascontiguousarray(frame[y1:y2, half - extra:w]))]


def find_people_fast(frame):
    global fast_count
    h = frame.shape[0]
    far_checked[0] = False
    r = pose_model(frame, conf=LOW_PERSON_CONFIDENCE, imgsz=config.IMAGE_SIZE, verbose=False)[0]
    people = result_people(r, 0, 0, 0)
    if fast_count % FAR_EVERY == 0:
        far_checked[0] = True
        tiles = far_tiles(frame)
        results = pose_model([t[2] for t in tiles], conf=LOW_PERSON_CONFIDENCE, imgsz=config.IMAGE_SIZE, verbose=False)
        for i in range(len(results)):
            for p in result_people(results[i], tiles[i][0], tiles[i][1], i + 1):
                if p["box"][3] - p["box"][1] > FAR_MAX_HEIGHT * h:
                    continue
                if any(overlap_of_smaller(p["box"], q["box"]) > 0.5 for q in people):
                    continue
                people.append(p)
    return people


def detect_fast(frame, stages):
    global fast_count
    from app.trace import Timer
    with Timer(stages, "pose"):
        people = find_people_fast(frame)
    sure_people = [p for p in people if p["conf"] >= config.PERSON_CONFIDENCE]
    squares = close_up_squares(sure_people)
    crops = []
    for sq in squares:
        piece, ox, oy = cut_square(frame, sq)
        if piece is not None:
            crops.append((ox, oy, piece))
    with Timer(stages, "objects"):
        r = object_model(frame, conf=config.PHONE_CONFIDENCE, imgsz=config.IMAGE_SIZE,
                         classes=[config.PHONE_CLASS_ID], verbose=False)[0]
        phones = [dict(t, src="full") for t in result_things(r, 0, 0, lambda c: "phone")]
        if crops:
            rs = object_model([c[2] for c in crops], conf=config.PHONE_CONFIDENCE, imgsz=CROP_SIZE,
                              classes=[config.PHONE_CLASS_ID], verbose=False)
            for c, rr in zip(crops, rs):
                phones = phones + [dict(t, src="closeup") for t in result_things(rr, c[0], c[1], lambda k: "phone")]
    global last_world_crops, last_world_full
    if WORLD_CHECK and world_model is not None:
        with Timer(stages, "world"):
            if fast_count % WORLD_FULL_EVERY == 0:
                rr = world_model(frame, conf=WORLD_CONFIDENCE, imgsz=config.IMAGE_SIZE, verbose=False)[0]
                last_world_full = result_things(rr, 0, 0, world_label)
            if fast_count % WORLD_CROPS_EVERY == 0:
                last_world_crops = []
                if crops:
                    rs = world_model([c[2] for c in crops], conf=WORLD_CONFIDENCE, imgsz=CROP_SIZE, verbose=False)
                    for c, rr in zip(crops, rs):
                        last_world_crops = last_world_crops + result_things(rr, c[0], c[1], world_label)
    found = last_world_full + last_world_crops
    devices = [t for t in found if t["label"] is not None]
    not_devices = [t for t in found if t["label"] is None]
    phones = filter_devices(drop_fake_phones(phones, not_devices) + devices, devices)
    phones = remove_doubles_per_label(phones)
    stages["far_checked"] = 1 if far_checked[0] else 0
    fast_count = fast_count + 1
    return people, phones
