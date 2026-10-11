# Helper for ground-truth labeling (NOT part of the app).
# Runs a bigger, separate detector (RT-DETRv2-R50, Apache-2.0) on sampled clip frames,
# on the full frame and on 2x3 overlapping tiles, so far-away people are found too.
# Writes boxes only (no images) to research/labels/teacher.json and, for my visual review,
# small annotated pictures to research/labels/frames/ (gitignored, never committed).
#
#   .\.venv-sandbox\Scripts\python.exe research\labels\teacher.py

import json
import cv2
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection

MODEL_DIR = "models/rtdetr_v2_r50vd"
CLIP = "perf/clip/classroom.avi"
TIMES = "perf/clip/classroom_times.json"
REVIEW_EVERY = 24      # frames I check by eye (~6 s apart)
DENSE_EVERY = 12       # frames used for ID-switch counting (~3 s apart, people only)
KEEP = {"person": 0.5, "cell phone": 0.35}

torch.set_num_threads(4)
processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
model = AutoModelForObjectDetection.from_pretrained(MODEL_DIR).eval()


def tiles(w, h):
    # full frame only: on this CPU each pass takes ~7 s, so far people are added by hand in review
    return [(0, 0, w, h)]


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def inside_ratio(small, big):
    x1, y1, x2, y2 = max(small[0], big[0]), max(small[1], big[1]), min(small[2], big[2]), min(small[3], big[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a = (small[2] - small[0]) * (small[3] - small[1])
    return inter / a if a > 0 else 0


def detect(frame):
    h, w = frame.shape[:2]
    found = []
    for (x1, y1, x2, y2) in tiles(w, h):
        crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        inputs = processor(images=crop, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs)
        res = processor.post_process_object_detection(out, threshold=0.3, target_sizes=[(y2 - y1, x2 - x1)])[0]
        for s, l, b in zip(res["scores"], res["labels"], res["boxes"]):
            name = model.config.id2label[int(l)]
            if name in KEEP and float(s) >= KEEP[name]:
                bb = [float(b[0]) + x1, float(b[1]) + y1, float(b[2]) + x1, float(b[3]) + y1]
                # a box cut by a tile edge is a partial person: skip it unless the tile is the full frame
                edge = (x1 > 0 and bb[0] - x1 < 3) or (x2 < w and x2 - bb[2] < 3) or \
                       (y1 > 0 and bb[1] - y1 < 3) or (y2 < h and y2 - bb[3] < 3)
                if edge and (x2 - x1) < w:
                    continue
                found.append({"label": "person" if name == "person" else "phone", "box": bb, "conf": float(s)})
    found.sort(key=lambda d: -d["conf"])
    kept = []
    for d in found:
        dup = False
        for k in kept:
            if k["label"] == d["label"] and (iou(k["box"], d["box"]) > 0.5 or inside_ratio(d["box"], k["box"]) > 0.8):
                dup = True
                break
        if not dup:
            kept.append(d)
    return kept


def draw(frame, dets):
    img = frame.copy()
    for x in range(0, 1920, 200):
        cv2.line(img, (x, 0), (x, 1080), (80, 80, 80), 1)
        cv2.putText(img, str(x), (x + 3, 1070), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    for y in range(0, 1080, 200):
        cv2.line(img, (0, y), (1920, y), (80, 80, 80), 1)
        cv2.putText(img, str(y), (3, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    for i, d in enumerate(dets):
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        c = (0, 255, 0) if d["label"] == "person" else (255, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 3)
        cv2.putText(img, str(i) + d["label"][0], (x1 + 4, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, c, 3)
    return cv2.resize(img, (1280, 720))


def main():
    times = json.load(open(TIMES))
    cap = cv2.VideoCapture(CLIP)
    out = {}
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % DENSE_EVERY == 0 or i % REVIEW_EVERY == 0:
            dets = detect(frame)
            if i % REVIEW_EVERY != 0:
                dets = [d for d in dets if d["label"] == "person"]
            out[str(i)] = {"t": round(times[i], 3), "review": i % REVIEW_EVERY == 0, "dets": dets}
            if i % REVIEW_EVERY == 0:
                cv2.imwrite("research/labels/frames/f%04d.jpg" % i, draw(frame, dets))
            if i % 40 == 0:
                print("frame", i, len(dets), flush=True)
        i += 1
    json.dump(out, open("research/labels/teacher.json", "w"))
    print("done", len(out), flush=True)


if __name__ == "__main__":
    main()
