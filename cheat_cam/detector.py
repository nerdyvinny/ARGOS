# This file runs the AI models on a picture.
# It gives back:
#   people = list of {"box": [x1,y1,x2,y2], "conf": 0.9, "points": [[x,y,conf], ... 17 points]}
#   phones = list of {"box": [x1,y1,x2,y2], "conf": 0.5}
#
# Body points (COCO order, same for both model types):
#   0 nose, 1 left eye, 2 right eye, 3 left ear, 4 right ear,
#   5 left shoulder, 6 right shoulder, ... 9 left wrist, 10 right wrist ...
#
# Two ways to run (pick with DETECTOR_BACKEND in config.py, default "huggingface"):
#   "huggingface" = models from the Hugging Face Hub (transformers library)
#       step 1: RT-DETR finds people AND phones in one go (trained on COCO)
#       step 2: ViTPose looks at each person box and finds the 17 body points
#   "yolo" = the old Ultralytics YOLO11 models (backup if Hugging Face is slow)
#
# Smartwatch check (Hugging Face only, turn on with WATCH_CHECK = True in config.py):
#   COCO has no "smartwatch", so we cut a small square around each wrist point
#   and ask CLIP "is this a wrist with a smartwatch or a bare wrist?".
#   A watch is added to the phones list with "label": "smartwatch",
#   so brain.py treats it like a phone (same timer and alert).
#
# Phone double-check (Hugging Face only, turn off with PHONE_CHECK = False in config.py):
#   RT-DETR sometimes calls any small rectangle a phone (name tags, ID cards, paper).
#   So every "phone" box is also shown to CLIP. We only keep it if CLIP also says
#   phone (or smartwatch). This uses the same CLIP model as the watch check.
#
# Earbud / headphone check (Hugging Face only, turn off with EAR_CHECK = False in config.py):
#   COCO has no earbuds or headphones either. We cut a square around each person's head
#   (from the nose, eye and ear points) and ask CLIP if they wear earbuds or headphones.
#   A hit is added to the phones list with "label": "earbuds" or "headphones".

import numpy as np
from PIL import Image
import config

# Hugging Face models. They download by themselves the first time.
# You can change these in config.py (same names) without touching this file.
HF_OBJECT_MODEL = getattr(config, "HF_OBJECT_MODEL", "PekingU/rtdetr_r18vd_coco_o365")
HF_POSE_MODEL = getattr(config, "HF_POSE_MODEL", "usyd-community/vitpose-base-simple")
BACKEND = getattr(config, "DETECTOR_BACKEND", "huggingface")

# smartwatch check
WATCH_CHECK = getattr(config, "WATCH_CHECK", False)   # off by default to keep it fast
HF_WATCH_MODEL = getattr(config, "HF_WATCH_MODEL", "openai/clip-vit-base-patch32")
WATCH_CONFIDENCE = getattr(config, "WATCH_CONFIDENCE", 0.6)   # how sure CLIP has to be
WATCH_BOX_SIZE = getattr(config, "WATCH_BOX_SIZE", 0.35)      # wrist square = this much of the person box width
# what CLIP chooses between. The first one is the "watch" answer.
WATCH_TEXTS = [
    "a photo of a wrist wearing a smartwatch",
    "a photo of a bare wrist",
    "a photo of a wrist in a sleeve",
    "a photo of a hand holding a pen",
]

# phone double-check
PHONE_CHECK = getattr(config, "PHONE_CHECK", True)
# what CLIP chooses between for every "phone" box. Only the first two count as a device.
PHONE_TEXTS = [
    "a photo of a smartphone",
    "a photo of a smartwatch",
    "a photo of a name tag",
    "a photo of an ID card",
    "a photo of a piece of paper",
    "a photo of a pen",
    "a photo of a hand",
]
DEVICE_LABELS = ["phone", "smartwatch"]

# earbud / headphone check
EAR_CHECK = getattr(config, "EAR_CHECK", True)
EAR_CONFIDENCE = getattr(config, "EAR_CONFIDENCE", 0.5)   # how sure CLIP has to be
# what CLIP chooses between for every head. Only the first two count as a device.
EAR_TEXTS = [
    "a photo of a person wearing wireless earbuds",
    "a photo of a person wearing headphones",
    "a photo of a person with nothing in their ears",
    "a photo of a person with long hair covering their ears",
    "a photo of a person wearing a hat",
]
EAR_LABELS = ["earbuds", "headphones"]
FACE_POINTS = [0, 1, 2, 3, 4]   # nose, eyes, ears

LEFT_WRIST = 9
RIGHT_WRIST = 10

# names the object model uses for the things we want
PERSON_NAME = "person"
PHONE_NAME = "cell phone"

# YOLO models (only used when BACKEND = "yolo")
pose_model = None
object_model = None

# Hugging Face models (only used when BACKEND = "huggingface")
hf_device = "cpu"
hf_object_processor = None
hf_object_model = None
hf_pose_processor = None
hf_pose_model = None
person_label = None
phone_label = None
hf_watch_processor = None
hf_watch_model = None


def load_models():
    print("Loading AI models (first time it downloads them)...")
    if BACKEND == "yolo":
        load_yolo_models()
    else:
        load_hf_models()
    print("Models loaded")


def load_yolo_models():
    global pose_model, object_model
    from ultralytics import YOLO
    pose_model = YOLO(config.POSE_MODEL)
    object_model = YOLO(config.OBJECT_MODEL)


def load_hf_models():
    global hf_device, hf_object_processor, hf_object_model, hf_pose_processor, hf_pose_model
    global person_label, phone_label, hf_watch_processor, hf_watch_model
    import os
    # on a Mac, if the graphics chip can't do one step, let it do that step on the cpu instead of crashing
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch
    from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation
    from transformers import CLIPModel

    if torch.cuda.is_available():
        hf_device = "cuda"   # NVIDIA graphics card
    elif torch.backends.mps.is_available():
        hf_device = "mps"    # Mac with an M1/M2/M3/M4 chip (much faster than cpu)
    else:
        hf_device = "cpu"
    print("Using", hf_device)

    print("Object model:", HF_OBJECT_MODEL)
    hf_object_processor = AutoProcessor.from_pretrained(HF_OBJECT_MODEL)
    hf_object_model = RTDetrForObjectDetection.from_pretrained(HF_OBJECT_MODEL).to(hf_device)
    hf_object_model.eval()

    print("Pose model:", HF_POSE_MODEL)
    hf_pose_processor = AutoProcessor.from_pretrained(HF_POSE_MODEL)
    hf_pose_model = VitPoseForPoseEstimation.from_pretrained(HF_POSE_MODEL).to(hf_device)
    hf_pose_model.eval()

    if WATCH_CHECK or PHONE_CHECK or EAR_CHECK:
        print("CLIP model (watch / phone check):", HF_WATCH_MODEL)
        hf_watch_processor = AutoProcessor.from_pretrained(HF_WATCH_MODEL)
        hf_watch_model = CLIPModel.from_pretrained(HF_WATCH_MODEL).to(hf_device)
        hf_watch_model.eval()

    # find the label numbers for "person" and "cell phone"
    person_label = find_label(hf_object_model, PERSON_NAME)
    phone_label = find_label(hf_object_model, PHONE_NAME)
    if person_label is None or phone_label is None:
        print("WARNING: this object model does not know 'person' or 'cell phone'. Use a COCO model.")


def find_label(model, name):
    for label_id, label_name in model.config.id2label.items():
        if label_name.lower() == name:
            return int(label_id)
    return None


def make_tiles(frame):
    # Cut the picture into pieces. Gives back a list of (x_offset, y_offset, piece)
    h, w = frame.shape[:2]
    top = int(h * config.CROP_TOP)
    bottom = int(h * config.CROP_BOTTOM)
    if bottom <= top:
        top = 0
        bottom = h

    band = frame[top:bottom]

    # Tiles only help for a really wide 360 picture.
    # A normal camera picture (the band is less than 3x wider than tall) is checked in one go,
    # otherwise we'd run the AI 3 times on the same picture for nothing.
    band_h = bottom - top
    one_tile = config.TILES <= 1 or w < 3 * band_h

    tiles = []
    if one_tile:
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


# ---------------- Hugging Face version ----------------

def hf_find_people_and_phones(tiles):
    import torch

    # OpenCV pictures are BGR, Hugging Face wants RGB
    images = []
    for t in tiles:
        rgb = t[2][:, :, ::-1]
        images.append(Image.fromarray(np.ascontiguousarray(rgb)))

    # step 1: find people and phones in every tile (one batch)
    inputs = hf_object_processor(images=images, return_tensors="pt").to(hf_device)
    with torch.inference_mode():
        outputs = hf_object_model(**inputs)
    sizes = [(img.height, img.width) for img in images]
    lowest = min(config.PERSON_CONFIDENCE, config.PHONE_CONFIDENCE)
    results = hf_object_processor.post_process_object_detection(outputs, target_sizes=sizes, threshold=lowest)

    people = []
    phones = []
    # for step 2: per tile, the person boxes inside that tile
    tile_person_boxes = []
    tile_person_confs = []
    for i in range(len(results)):
        r = results[i]
        ox = tiles[i][0]
        oy = tiles[i][1]
        boxes_here = []
        confs_here = []
        for score, label, b in zip(r["scores"].tolist(), r["labels"].tolist(), r["boxes"].tolist()):
            if label == phone_label and score >= config.PHONE_CONFIDENCE:
                box = [b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy]
                phones.append({"box": box, "conf": float(score)})
            elif label == person_label and score >= config.PERSON_CONFIDENCE:
                boxes_here.append(b)
                confs_here.append(float(score))
        tile_person_boxes.append(boxes_here)
        tile_person_confs.append(confs_here)

    # step 2: body points for every person (one batch, only tiles that have people)
    pose_images = []
    pose_boxes = []      # ViTPose wants [x, y, width, height]
    pose_tile_ids = []
    for i in range(len(images)):
        if len(tile_person_boxes[i]) == 0:
            continue
        xywh = []
        for b in tile_person_boxes[i]:
            xywh.append([b[0], b[1], b[2] - b[0], b[3] - b[1]])
        pose_images.append(images[i])
        pose_boxes.append(xywh)
        pose_tile_ids.append(i)

    if len(pose_images) == 0:
        return people, remove_doubles(phones)

    inputs = hf_pose_processor(pose_images, boxes=pose_boxes, return_tensors="pt").to(hf_device)
    with torch.inference_mode():
        outputs = hf_pose_model(**inputs)
    pose_results = hf_pose_processor.post_process_pose_estimation(outputs, boxes=pose_boxes)

    for k in range(len(pose_results)):
        i = pose_tile_ids[k]
        ox = tiles[i][0]
        oy = tiles[i][1]
        for j in range(len(pose_results[k])):
            person = pose_results[k][j]
            b = tile_person_boxes[i][j]
            box = [b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy]
            points = []
            xy = person["keypoints"].tolist()
            conf = person["scores"].tolist()
            for p in range(len(xy)):
                points.append([float(xy[p][0] + ox), float(xy[p][1] + oy), float(conf[p])])
            people.append({"box": box, "conf": tile_person_confs[i][j], "points": points})

    people = remove_doubles(people)
    if PHONE_CHECK:
        phones = hf_check_phones(tiles, phones)
    if WATCH_CHECK:
        phones = phones + hf_find_watches(tiles, people)
    if EAR_CHECK:
        phones = phones + hf_find_ear_devices(tiles, people)
    return people, remove_doubles(phones)


def hf_find_ear_devices(tiles, people):
    # cut out every head and ask CLIP about earbuds / headphones
    import torch

    crops = []
    crop_boxes = []
    for person in people:
        pts = []
        for k in FACE_POINTS:
            if k < len(person["points"]) and person["points"][k][2] >= config.KEYPOINT_CONFIDENCE:
                pts.append(person["points"][k])
        if len(pts) < 2:
            continue   # can't see the face well enough
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        b = person["box"]
        # head square: a bit wider than the face points, at least 40% of the person box
        side = max((max(xs) - min(xs)) * 2.2, (b[2] - b[0]) * 0.4)
        if side < 24:
            continue   # person too tiny to see earbuds
        crop_box = [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2]
        piece = cut_from_tiles(tiles, crop_box, clip=True)
        if piece is None:
            continue
        crops.append(Image.fromarray(np.ascontiguousarray(piece[:, :, ::-1])))
        crop_boxes.append(crop_box)

    found = []
    if len(crops) == 0:
        return found

    inputs = hf_watch_processor(text=EAR_TEXTS, images=crops, return_tensors="pt", padding=True).to(hf_device)
    with torch.inference_mode():
        outputs = hf_watch_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)   # one row per head, one column per text

    for i in range(len(crops)):
        best = int(probs[i].argmax())
        conf = float(probs[i][best])
        if best < len(EAR_LABELS) and conf >= EAR_CONFIDENCE:
            found.append({"box": crop_boxes[i], "conf": conf, "label": EAR_LABELS[best]})
    return found


def hf_check_phones(tiles, phones):
    # ask CLIP about every "phone" box, keep only real devices
    import torch

    if len(phones) == 0:
        return phones

    crops = []
    for ph in phones:
        b = ph["box"]
        # a bit of extra room around the box so CLIP can see what it is
        pad_x = (b[2] - b[0]) * 0.3
        pad_y = (b[3] - b[1]) * 0.3
        piece = cut_from_tiles(tiles, [b[0] - pad_x, b[1] - pad_y, b[2] + pad_x, b[3] + pad_y], clip=True)
        if piece is None:
            crops.append(None)
        else:
            crops.append(Image.fromarray(np.ascontiguousarray(piece[:, :, ::-1])))

    images = [c for c in crops if c is not None]
    if len(images) == 0:
        return phones

    inputs = hf_watch_processor(text=PHONE_TEXTS, images=images, return_tensors="pt", padding=True).to(hf_device)
    with torch.inference_mode():
        outputs = hf_watch_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)   # one row per box, one column per text

    kept = []
    row = 0
    for i in range(len(phones)):
        if crops[i] is None:
            kept.append(phones[i])   # couldn't cut it out, trust RT-DETR
            continue
        best = int(probs[row].argmax())
        row = row + 1
        if best < len(DEVICE_LABELS):
            ph = dict(phones[i])
            ph["label"] = DEVICE_LABELS[best]
            kept.append(ph)
    return kept


def hf_find_watches(tiles, people):
    import torch

    crops = []
    crop_boxes = []
    for person in people:
        b = person["box"]
        side = (b[2] - b[0]) * WATCH_BOX_SIZE
        if side < 16:
            continue   # person too tiny, can't see a watch anyway
        for k in [LEFT_WRIST, RIGHT_WRIST]:
            if k >= len(person["points"]):
                continue
            x, y, c = person["points"][k]
            if c < config.KEYPOINT_CONFIDENCE:
                continue
            crop_box = [x - side / 2, y - side / 2, x + side / 2, y + side / 2]
            piece = cut_from_tiles(tiles, crop_box)
            if piece is None:
                continue
            rgb = np.ascontiguousarray(piece[:, :, ::-1])
            crops.append(Image.fromarray(rgb))
            crop_boxes.append(crop_box)

    watches = []
    if len(crops) == 0:
        return watches

    inputs = hf_watch_processor(text=WATCH_TEXTS, images=crops, return_tensors="pt", padding=True).to(hf_device)
    with torch.inference_mode():
        outputs = hf_watch_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)   # one row per wrist, one column per text

    for i in range(len(crops)):
        watch_prob = float(probs[i][0])
        if watch_prob >= WATCH_CONFIDENCE:
            watches.append({"box": crop_boxes[i], "conf": watch_prob, "label": "smartwatch"})
    return watches


def cut_from_tiles(tiles, box, clip=False):
    # find a tile that has this box inside it and cut the piece out
    # clip=True: if the box sticks out of the tile a bit, just cut the part that's inside
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    for ox, oy, piece in tiles:
        h, w = piece.shape[:2]
        x1 = int(box[0] - ox)
        y1 = int(box[1] - oy)
        x2 = int(box[2] - ox)
        y2 = int(box[3] - oy)
        if clip:
            if not (0 <= cx - ox < w and 0 <= cy - oy < h):
                continue
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
        elif x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            continue
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        return piece[y1:y2, x1:x2]
    return None


# ---------------- YOLO version (backup) ----------------

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
    if BACKEND == "yolo":
        people = find_people(tiles)
        phones = find_phones(tiles)
    else:
        people, phones = hf_find_people_and_phones(tiles)
    return people, phones
