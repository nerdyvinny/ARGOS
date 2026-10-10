# ============================================
#  SETTINGS - change these to tune the system
# ============================================

# Where the video comes from.
# 1 = the wide USB camera (Arducam). 0 is the laptop camera - NEVER use that one.
# or put a file name like "test_video.mp4"
SOURCE = 1

# Video files are checked at about the live camera's speed (pictures per second),
# so the rules act the same on a practice video as in the real room.
VIDEO_FILE_FPS = 2

# Picture size we ask the USB camera for
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

# Safety check: the laptop camera can only do 1280 wide.
# If the camera we opened is smaller than this, it's the wrong camera and we stop.
MIN_CAMERA_WIDTH = 1920

# AI models (they download by themselves the first time)
POSE_MODEL = "yolo11n-pose.pt"   # finds people + face points (nose, eyes, ears)
OBJECT_MODEL = "yolo11n.pt"      # finds phones
PHONE_CLASS_ID = 67              # "cell phone" in the COCO dataset

# How sure the AI has to be
PERSON_CONFIDENCE = 0.4
PHONE_CONFIDENCE = 0.25          # lower because phones are small
KEYPOINT_CONFIDENCE = 0.5        # how sure a face point has to be to use it

# Image size the AI looks at (bigger = sees small people better but slower)
# Speed test on our laptop (no GPU), 1920x1080 camera, pictures per second:
#   TILES 3 + 960 = 1.2 fps | TILES 3 + 640 = 2.4 fps | TILES 2 + 640 = 4.0 fps | TILES 1 + 640 = 7.4 fps
# The USB camera only gives ~4 fps anyway, so faster than 4 doesn't help.
IMAGE_SIZE = 640

# ---------- 360 camera stuff ----------
# A 360 picture is really wide, so people look tiny.
# We cut it into pieces (tiles) and check each piece.
# Set TILES = 1 for a normal wide camera.
TILES = 3
TILE_OVERLAP = 0.15              # 15% overlap so people on the edge are not cut in half

# Cut off the ceiling and the floor (0.0 to 1.0 of the picture height)
CROP_TOP = 0.20
CROP_BOTTOM = 0.85

# ---------- desks ----------
DESKS_FILE = "desks.json"        # made by setup_desks.py

# ---------- looking at other desks ----------
YAW_SMOOTH_FRAMES = 5            # average the head angle over a few frames
BASELINE_SECONDS = 120           # each person's "normal" head angle = median of last 2 min
BASELINE_MIN_SAMPLES = 15        # need this many samples before we judge someone
YAW_THRESHOLD = 0.25             # how far the head has to turn from normal
LOOK_AWAY_SECONDS = 2.0          # turned this long = 1 "look away"
LOOK_COUNT_WINDOW = 300          # count look aways in the last 5 minutes
LOOK_COUNT_LIMIT = 4             # this many look aways = alert
LONG_LOOK_SECONDS = 6.0          # one really long look = alert right away

# ---------- phone ----------
PHONE_WINDOW_SECONDS = 4.0       # look at the last 4 seconds (we only get ~2 pictures a second,
                                 # so 2 seconds was too short to ever see the phone 3 times)
PHONE_MIN_HITS = 3               # phone seen this many times in that window = alert

# ---------- alerts ----------
ALERT_COOLDOWN_SECONDS = 30      # don't spam the same alert for the same desk
EVIDENCE_FOLDER = "evidence"
ALERTS_CSV = "alerts.csv"

# ---------- screen / dashboard ----------
SHOW_WINDOW = True               # show an OpenCV window
DASHBOARD = True                 # run the web page at http://localhost:5000
DASHBOARD_PORT = 5000
OPEN_BROWSER = True              # open the web page in your browser by itself when main.py starts
