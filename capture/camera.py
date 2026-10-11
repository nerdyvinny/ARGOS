# Opens the camera (or a video file) the same way for every program.
#
# We ONLY want the wide USB camera (Arducam), never the laptop camera.
# The laptop camera can only do 1280 wide, the Arducam can do 1920 wide.
# So after opening a camera we check how wide the picture is. Too small = wrong camera = stop.

import cv2
import sys
import time
from settings import config

SPEED_TEST_PICTURES = 15   # pictures timed after opening to measure the real camera speed
MIN_GOOD_FPS = 20          # slower than this = the camera fell back to raw video


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

    size = [cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH,
            cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT,
            cv2.CAP_PROP_FPS, config.CAMERA_FPS]

    if sys.platform == "win32":
        # Windows: use Media Foundation. Through DirectShow the Arducam only offers raw video
        # (1920x1080 at 5 fps max), and asking DirectShow for MJPG makes it send black pictures
        # until it is replugged. Media Foundation unpacks the camera's compressed video for us:
        # 1920x1080 at 30 fps (measured 30.0 fps on 2026-10-11).
        # Media Foundation numbers cameras the same way here (0 = laptop, 1 = Arducam).
        cap, frame = try_open(source, cv2.CAP_MSMF, size)
        if cap is None:
            # right after another program lets go of the camera it can need a moment
            time.sleep(2)
            cap, frame = try_open(source, cv2.CAP_MSMF, size)
        if cap is None:
            print("Media Foundation could not read camera", source, "- trying DirectShow (slow: about 4 fps)")
            cap, frame = try_open(source, cv2.CAP_DSHOW, size)
    else:
        # Mac / Linux: the normal way works; ask for MJPG so the picture comes at 30 fps
        mjpg = [cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")] + size
        cap, frame = try_open(source, cv2.CAP_ANY, mjpg)
    if cap is None:
        print("Could not get a picture from camera", source, "- is the USB camera plugged in?")
        print("Also close the Windows Camera app (or Zoom/Teams) - only one program can use a camera at a time.")
        return None

    h, w = frame.shape[:2]
    if w < config.MIN_CAMERA_WIDTH:
        print("Camera", source, "is only", w, "x", h, "- that looks like the LAPTOP camera, not the USB camera.")
        print("Unplug and replug the USB camera, or try another number: python -m app.main 2")
        cap.release()
        return None

    if frame.mean() < 2:
        print("WARNING: camera", source, "is sending a black picture. Check the camera (see README).")

    check_speed(cap, w, h)
    return cap


def try_open(source, backend, settings):
    # opens the camera with one camera system; gives back (cap, picture) or (None, None)
    cap = cv2.VideoCapture(source, backend, settings)
    if not cap.isOpened():
        return None, None
    # the first few pictures are dark while the camera adjusts, so skip them
    frame = None
    for i in range(10):
        ok, f = cap.read()
        if ok:
            frame = f
    if frame is None:
        cap.release()
        return None, None
    return cap, frame


def format_name(cap):
    # the 4 letters of the picture format the camera really sends (MJPG = fast, YUY2 = raw/slow)
    try:
        v = int(cap.get(cv2.CAP_PROP_FOURCC))
    except (TypeError, ValueError):
        return "????"
    return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))


def check_speed(cap, w, h):
    # time a few pictures so we KNOW how fast the camera really is (it can silently fall back to 4 fps)
    start = time.time()
    count = 0
    while count < SPEED_TEST_PICTURES and time.time() - start < 4:
        ok, f = cap.read()
        if ok:
            count = count + 1
    seconds = time.time() - start
    fps = count / seconds if seconds > 0 else 0
    try:
        system = cap.getBackendName()   # MSMF = Media Foundation, DSHOW = DirectShow
    except Exception:
        system = "?"
    print("Camera:", system, format_name(cap).strip(), str(w) + "x" + str(h), "@", round(fps, 1), "fps")
    if fps < MIN_GOOD_FPS:
        print("WARNING: the camera only sends", round(fps, 1), "pictures a second (it should be about 30).")
        print("  It is sending raw video instead of MJPG. To fix it:")
        print("  1. close this program and the Windows Camera app")
        print("  2. unplug the USB camera, wait 5 seconds, plug it into a port ON the laptop (not a hub)")
        print("  3. start again")
    return fps
