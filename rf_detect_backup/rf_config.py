# ============================================
#  RF SETTINGS - change these to tune the monitor
# ============================================

import os

# the folder this file is in. All RF files are saved here, no matter where you run from.
FOLDER = os.path.dirname(os.path.abspath(__file__))

# ---------- the radio ----------
SAMPLE_RATE = 2.4e6          # RTL-SDR: 2.4 MHz is the most it does reliably (use 2.048e6 if you get dropouts)
GAIN = 40                    # in dB. Use "auto" to let the radio decide. Lower = shorter range (see RANGE below)
PPM = 0                      # frequency correction, leave 0

# Python needs rtlsdr.dll (the RTL-SDR driver). SDR++ comes with one, so we borrow it.
# Put the folder where you unzipped SDR++ here. If it is wrong, the program tells you.
RTLSDR_DLL_FOLDER = os.path.join(os.path.expanduser("~"), "Downloads", "sdrpp_windows_x64")

# ---------- how we measure ----------
SAMPLES_PER_STEP = 262144    # samples grabbed at each frequency (about 0.1 second)
FFT_SIZE = 1024              # must divide SAMPLES_PER_STEP evenly
CHUNKS = 16                  # each grab is cut into this many pieces so we can catch short bursts
USABLE_FRACTION = 0.8        # only use the middle 80% of each window (the edges are weak)
DC_NOTCH_HZ = 60000          # ignore this much around the center (the radio makes a fake spike there)

# ---------- baseline (empty room) ----------
BASELINE_SWEEPS = 5          # how many times we sweep all bands to learn "normal"
BASELINE_FILE = os.path.join(FOLDER, "rf_baseline.json")
SIM_BASELINE_FILE = os.path.join(FOLDER, "rf_baseline_sim.json")   # used with --simulate, so the real one is safe

# ---------- deciding "a phone is transmitting" ----------
THRESHOLD_DB = 3.0           # average power this many dB above baseline = suspicious
PEAK_THRESHOLD_DB = 6.0      # a short burst this many dB above the baseline peak = suspicious
CONFIRM_HITS = 2             # must be suspicious in this many...
CONFIRM_WINDOW = 5           # ...of the last this many sweeps before we call it ACTIVE

LOG_FILE = os.path.join(FOLDER, "rf_log.csv")

# ============================================================
#  RANGE: only count phones IN FRONT of the camera, IN THIS ROOM
# ============================================================
# Radio has no "distance" setting. A phone's signal just gets weaker the farther
# away it is and the more walls are in the way. So we limit range by STRENGTH:
# anything weaker than a phone at the back desk is treated as "not in this room".
#
#   1. HARDWARE: a flat metal sheet (baking tray, or foil glued on cardboard)
#      about 11 cm BEHIND the antenna, facing the room, makes phones behind the camera
#      read weaker (about 6-10 dB). (11 cm = a quarter wavelength at ~700 MHz.)
#      A directional "patch" or "Yagi" antenna for 600-900 MHz, pointed the same
#      way as the camera, works even better.
#   2. GAIN: lowering GAIN (top of this file) makes far, weak phones sink into the
#      radio's own noise while close phones still show. Re-run baseline after.
#
# Be honest about the limit: a phone with bad service shouts louder than one with good
# service (up to 1000x), so a far phone CAN sometimes look like a near one.

# ---------- camera link (PLACEHOLDER: not hooked up yet) ----------
CAMERA_CONFIRM = False       # True = only say ACTIVE when the camera also sees something
CAMERA_FOV_DEGREES = 120     # how wide the camera sees. Only used as a note for now.

# ---------- web page (shown at http://localhost:5050) ----------
WEB_PORT = 5050              # not 5000, that one is the camera dashboard
WATERFALL_FILE = os.path.join(FOLDER, "rf_waterfall.html")   # live: frequency vs time
WATERFALL_ROWS = 60          # how many sweeps (rows) the waterfall remembers
HEAT_MAX_DB = 12.0           # this many dB above normal (or more) shows as full red

# ============================================================
#  CELL PHONE BANDS (heard by the RTL-SDR)
# ============================================================
# Phones TRANSMIT on the "uplink" part of each band, so these are uplink ranges (MHz).
# 4G (LTE) and 5G use the same frequencies, so one entry covers both.
# Smaller carriers rent from the big ones, so they show up in the same bands:
#   on T-Mobile: Metro, Mint, Google Fi, US Mobile   on Verizon: Visible, Total, Straight Talk
#   on AT&T: Cricket, H2O                            Boost Mobile: own network + AT&T/T-Mobile
# Carriers change bands over time, so we report by band. CHECK these for your area.
#
# The RTL-SDR can only tune up to about 1766 MHz, so these can NOT be seen with it:
#   PCS 1850-1915 (all carriers), T-Mobile 2.5 GHz (B41/n41), C-band 3.7 GHz (n77),
#   AT&T WCS 2.3 GHz (B30), CBRS 3.5 GHz, Wi-Fi and Bluetooth (see BLUETOOTH below).
# More bands = slower sweeps (about 0.15 s per window). Turn off bands nobody uses
# in your area with "enabled": False.
BANDS = [
    {"name": "B71 600MHz",   "start": 663.0,  "end": 698.0,  "carriers": "T-Mobile, Boost",   "enabled": True},
    {"name": "B12/17 700",   "start": 699.0,  "end": 716.0,  "carriers": "AT&T, T-Mob, USC",  "enabled": True},
    {"name": "B13 700",      "start": 777.0,  "end": 787.0,  "carriers": "Verizon",           "enabled": True},
    {"name": "B14 FirstNet", "start": 788.0,  "end": 798.0,  "carriers": "AT&T FirstNet",     "enabled": True},
    {"name": "B26 800",      "start": 817.0,  "end": 824.0,  "carriers": "T-Mobile (Sprint)", "enabled": True},
    {"name": "B5 850",       "start": 824.0,  "end": 849.0,  "carriers": "AT&T, Verizon, USC", "enabled": True},
    # top edge of the RTL-SDR: weaker here, and some dongles can't lock up this high.
    # If these rows show "PLL not locked" or junk numbers, set them to False.
    {"name": "B70 AWS-4",    "start": 1695.0, "end": 1710.0, "carriers": "Boost (Dish)",      "enabled": False},
    {"name": "B66 AWS",      "start": 1710.0, "end": 1765.0, "carriers": "all big 3",         "enabled": False},
]

# ============================================================
#  BLUETOOTH (separate: uses the LAPTOP's Bluetooth, not the RTL-SDR)
# ============================================================
# Bluetooth is at 2.4 GHz (2402-2480 MHz), too high for the RTL-SDR. The laptop's
# own Bluetooth listens on the 3 "advertising" channels (37, 38, 39 = 2402, 2426,
# 2480 MHz) where headphones announce themselves. Windows picks the channels, not us.
# Needs Python 3.12:  py -3.12 -m pip install bleak
BT_CHECK = True              # also check Bluetooth while "monitor" runs
BT_SCAN_SECONDS = 5          # how long each Bluetooth scan listens
BT_FORGET_SECONDS = 30       # stop showing a device we haven't heard for this long
BT_IGNORE_NAMES = []         # names to never flag, e.g. ["Proctor AirPods"]

# RANGE for Bluetooth: devices weaker than BT_MIN_RSSI are ignored (probably another room).
# To set it: put headphones (or the JBL speaker) in the NEXT room, run
# "py -3.12 rf_detect\rf_monitor.py bluetooth", read its dBm, and set this ~5 higher.
# Example: next room shows -88  ->  BT_MIN_RSSI = -83
BT_MIN_RSSI = -85            # in dBm. Closer to 0 = shorter range. -100 = hear everything.
BT_NEAR_RSSI = -60           # stronger than this = very close (about 1-2 m). Very rough!
BT_MEDIUM_RSSI = -75         # stronger than this = somewhere in the room
BT_FILE = os.path.join(FOLDER, "rf_bluetooth.html")
BT_LOG_FILE = os.path.join(FOLDER, "bt_log.csv")
