# Run this ONCE before the test starts.
# It grabs one picture from the camera and lets you draw a box on every desk.
#
#   python -m tools.setup_desks            (uses SOURCE from config.py)
#   python -m tools.setup_desks test.mp4   (use a video file instead)
#
# How to draw:
#   drag a box over a desk (include the chair, where the person sits)
#   press ENTER or SPACE to save that box, then draw the next one
#   press ESC when you are done with all desks

import sys
import json
import cv2
from settings import config
from capture import camera


def get_source():
    if len(sys.argv) > 1:
        s = sys.argv[1]
        if s.isdigit():
            return int(s)
        return s
    return config.SOURCE


def main():
    source = get_source()
    cap = camera.open_camera(source)
    if cap is None:
        return

    # skip a few frames because the first ones from a webcam are often dark
    frame = None
    for i in range(10):
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()

    if frame is None:
        print("Could not read a picture from the camera/video")
        return

    # make the picture fit on the screen
    h, w = frame.shape[:2]
    scale = 1.0
    if w > 1600:
        scale = 1600 / w
    small = cv2.resize(frame, (int(w * scale), int(h * scale)))

    print("Draw a box on each desk. ENTER = save box, ESC = finished.")
    boxes = cv2.selectROIs("Draw desks", small, showCrosshair=False)
    cv2.destroyAllWindows()

    desks = []
    number = 1
    for b in boxes:
        x, y, bw, bh = [int(v) for v in b]
        if bw == 0 or bh == 0:
            continue
        # turn the box back into full size picture numbers
        x1 = int(x / scale)
        y1 = int(y / scale)
        x2 = int((x + bw) / scale)
        y2 = int((y + bh) / scale)
        desks.append({"name": "Desk " + str(number), "box": [x1, y1, x2, y2]})
        number = number + 1

    with open(config.DESKS_FILE, "w") as f:
        json.dump({"frame_width": w, "frame_height": h, "desks": desks}, f, indent=2)

    print("Saved", len(desks), "desks to", config.DESKS_FILE)

    # save a picture so you can check the boxes
    check = frame.copy()
    for d in desks:
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(check, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(check, d["name"], (x1 + 5, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imwrite("desks_check.jpg", check)
    print("Saved desks_check.jpg so you can see the boxes")


if __name__ == "__main__":
    main()
