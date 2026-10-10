# This file runs the AI models on a picture.
# It gives back:
#   people = list of {"box": [x1,y1,x2,y2], "conf": 0.9, "points": [[x,y,conf], ... 17 points]}
#   phones = list of {"box": [x1,y1,x2,y2], "conf": 0.5}
#
# Body points (from YOLO pose, COCO order):
#   0 nose, 1 left eye, 2 right eye, 3 left ear, 4 right ear,
#   5 left shoulder, 6 right shoulder, ... 9 left wrist, 10 right wrist ...

import numpy as np
from ultralytics import YOLO
from settings import config

pose_model = None
object_model = None


def load_models():
    global pose_model, object_model
    print("Loading AI models (first time it downloads them)...")
    pose_model = YOLO(config.POSE_MODEL)
    object_model = YOLO(config.OBJECT_MODEL)
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
    # Keep the biggest box and throw away boxes that are mostly inside it.
    things = sorted(things, key=lambda t: box_area(t["box"]), reverse=True)
    kept = []
    for t in things:
        is_double = False
        for k in kept:
            if overlap_of_smaller(t["box"], k["box"]) > 0.6:
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

            people.append({"box": box, "conf": float(confs[j]), "points": points})

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
            phones.append({"box": box, "conf": float(confs[j])})

    return remove_doubles(phones)


def detect(frame):
    tiles = make_tiles(frame)
    people = find_people(tiles)
    phones = find_phones(tiles)
    return people, phones
