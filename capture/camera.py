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

    # Ask for MJPG (compressed video) at 30 pictures a second. Without it the Arducam sends
    # raw video, and USB 2.0 is only fast enough for about 4 raw 1920x1080 pictures a second.
    # MJPG must be asked for WHILE the camera opens. Asking after it opened is ignored, and
    # (in our first test) asking that way made the Arducam send black pictures.
    settings = [cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"),
                cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH,
                cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT,
                cv2.CAP_PROP_FPS, config.CAMERA_FPS]

    # a camera number: use DirectShow (Windows' default camera system can't read the Arducam)
    if sys.platform == "win32":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW, settings)   # Windows needs DirectShow for the Arducam
    else:
        cap = cv2.VideoCapture(source, cv2.CAP_ANY, settings)     # Mac / Linux: the normal way works
    if not cap.isOpened():
        print("Could not open camera", source, "- is the USB camera plugged in?")
        print("Also close the Windows Camera app (or Zoom/Teams) - only one program can use a camera at a time.")
        return None

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
