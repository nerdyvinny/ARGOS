# Opens the camera (or a video file) the same way for every program.
#
# We ONLY want the wide USB camera (Arducam), never the laptop camera.
# The laptop camera can only do 1280 wide, the Arducam can do 1920 wide.
# So after opening a camera we check how wide the picture is. Too small = wrong camera = stop.

import cv2
import sys
from settings import config


def open_camera(source):
    # gives back an opened cap, or None if something is wrong

    # a video file: just open it
    if isinstance(source, str):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Could not open the video file:", source)
            return None
        return cap

    # camera 0 is the laptop camera: never open it, not even for a moment
    if source == 0:
        print("Camera 0 is the laptop camera - we never use it. Use the USB camera (1), or try: python -m app.main 2")
        return None

    # a camera number: use DirectShow (Windows' default camera system can't read the Arducam)
    if sys.platform == "win32":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)   # Windows needs DirectShow for the Arducam
    else:
        cap = cv2.VideoCapture(source)                  # Mac / Linux: the normal way works
    if not cap.isOpened():
        print("Could not open camera", source, "- is the USB camera plugged in?")
        print("Also close the Windows Camera app (or Zoom/Teams) - only one program can use a camera at a time.")
        return None

    # Do NOT ask this camera for MJPG (compressed video). We tested it: MJPG makes the
    # Arducam send only black pictures until you unplug it and plug it back in.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    # the first few pictures are dark while the camera adjusts, so skip them
    frame = None
    for i in range(10):
        ok, f = cap.read()
        if ok:
            frame = f
    if frame is None:
        print("Camera", source, "opened but gave no picture")
        cap.release()
        return None

    h, w = frame.shape[:2]
    if w < config.MIN_CAMERA_WIDTH:
        print("Camera", source, "is only", w, "x", h, "- that looks like the LAPTOP camera, not the USB camera.")
        print("Unplug and replug the USB camera, or try another number: python -m app.main 2")
        cap.release()
        return None

    if frame.mean() < 2:
        print("WARNING: camera", source, "is sending a black picture. Check the camera (see README).")

    return cap
