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
    print("Models loaded")


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
