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
CAMERA_FPS = 30                  # pictures per second we ask the camera for (MJPG, see capture/camera.py)

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

# ---------- following people ----------
FORGET_PERSON_SECONDS = 10       # someone unseen this long gets a new "Person N" when they come back

# ---------- looking at other desks ----------
YAW_SMOOTH_SECONDS = 1.0         # head angle = middle value of the last second of AI checks
FACE_GONE_SECONDS = 3.0          # face unseen this long (head down writing) = start the look timer fresh
BASELINE_SECONDS = 120           # each person's "normal" head angle = median of last 2 min
BASELINE_MIN_SAMPLES = 15        # need this many samples before we judge someone
YAW_THRESHOLD = 0.25             # how far the head has to turn from normal
LOOK_AWAY_SECONDS = 1.0          # turned this long = 1 "look away"
LOOK_COUNT_WINDOW = 120          # count look aways in the last 2 minutes
LOOK_COUNT_LIMIT = 3             # this many look aways = alert
LONG_LOOK_SECONDS = 3.0          # one long stare = alert right away

# ---------- phone ----------
PHONE_WINDOW_SECONDS = 4.0       # look at the last 4 seconds of AI checks
PHONE_MIN_HITS = 1               # confirmed device seen on this many AI checks in that window = alert.
                                 # 1 because vision/devices.py already needs it in 2 of the last 3 checks
                                 # (with only real AI checks counting, 3 hits in 4 s almost never happened)

# ---------- alerts ----------
ALERT_COOLDOWN_SECONDS = 30      # don't spam the same alert for the same person
EVIDENCE_FOLDER = "evidence"
ALERTS_CSV = "alerts.csv"

# ---------- screen / dashboard ----------
SHOW_WINDOW = True               # show an OpenCV window
DASHBOARD = True                 # run the web page at http://localhost:5000
DASHBOARD_PORT = 5000
OPEN_BROWSER = True              # open the web page in your browser by itself when main.py starts

# ---------- speed / threaded loop ----------
DETECT_MODE = "fast"             # "fast": whole picture + close-ups (normal wide camera); "tiles": the 360 layout above
DEVICE_CONFIRM = True            # device must be seen in 2 of the last 3 AI checks (+ watch size/shape checks)
TRACE = True                     # write perf/trace.csv (time per stage, devices, CPU/GPU use, tracker events)
SHOW_STOPWATCH = False           # draw a clock on the picture for the delay test

# ---------- speed on a CPU-only laptop (round 3, measured on a 30 fps replay) ----------
AI_PROCESS = True                # the AI runs in its own process, so it can't block the screen loop
AI_LOW_PRIORITY = False          # low priority starved the AI (0.34 checks/s); normal + 3 threads: 0.86
AI_THREADS = 3                   # CPU threads the AI may use (the screen loop keeps the rest)
SCREEN_CV_THREADS = 1            # OpenCV in the screen loop runs on 1 thread (its helper threads waited
                                 # behind the AI: one resize took 75 ms instead of 3 ms)
