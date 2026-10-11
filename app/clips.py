# Short video clips for alerts, so the proctor can play back what happened.
# We keep the last few seconds of pictures in memory. When an alert happens we keep
# collecting a few more seconds, then save everything as a small video in the evidence folder.

import os
import time
import threading
from collections import deque
import cv2
from settings import config

SECONDS_BEFORE = 4   # how much of the past goes into a clip
SECONDS_AFTER = 3    # how much we keep recording after the alert

fps = 10
recent = deque(maxlen=SECONDS_BEFORE * fps)
waiting = []          # clips still collecting their "after" pictures
lock = threading.Lock()


def setup(frames_per_second):
    # call once at the start: how many pictures per second push() will get
    global fps, recent
    fps = max(1, int(round(frames_per_second)))
    with lock:
        recent = deque(maxlen=SECONDS_BEFORE * fps)


def start_clip(alert):
    # remember that this alert needs a clip; gives back the file name it will have
    name = alert["person"].replace(" ", "_")
    file_name = name + "_" + alert["kind"] + "_" + str(int(time.time() * 1000)) + ".webm"
    with lock:
        waiting.append({"file": file_name, "frames": list(recent), "left": SECONDS_AFTER * fps})
    return file_name


def push(frame):
    # give every shown picture to the clip recorder (already small, with boxes drawn)
    finished = []
    with lock:
        recent.append(frame)
        for clip in waiting:
            clip["frames"].append(frame)
            clip["left"] = clip["left"] - 1
            if clip["left"] <= 0:
                finished.append(clip)
        for clip in finished:
            waiting.remove(clip)
    for clip in finished:
        # saving a video takes a while, so do it on the side and don't slow the camera down
        threading.Thread(target=save_clip, args=(clip,), daemon=True).start()


def save_clip(clip):
    frames = clip["frames"]
    if not frames:
        return
    h, w = frames[0].shape[:2]
    path = os.path.join(config.EVIDENCE_FOLDER, clip["file"])
    temp_path = path + ".part.webm"   # the page only sees the file once it is complete
    # VP8 inside .webm plays in every browser (OpenCV's usual mp4 codec does not)
    writer = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*"VP80"), fps, (w, h))
    if not writer.isOpened():
        print("[CLIP] could not open the video writer for", clip["file"])
        return
    for f in frames:
        if f.shape[:2] != (h, w):
            f = cv2.resize(f, (w, h))
        writer.write(f)
    writer.release()
    os.replace(temp_path, path)
