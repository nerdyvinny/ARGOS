# ============================================
#  RF SETTINGS - change these to tune the monitor
# ============================================

import os

# the folder this file is in. All RF files are saved here, no matter where you run from.
FOLDER = os.path.dirname(os.path.abspath(__file__))

# ---------- the radio ----------
SAMPLE_RATE = 2.4e6          # RTL-SDR: 2.4 MHz is the most it does reliably (use 2.048e6 if you get dropouts)
GAIN = 40                    # in dB. Use "auto" to let the radio decide.
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

# ---------- heatmaps (shown at http://localhost:5050) ----------
WEB_PORT = 5050              # not 5000, that one is the camera dashboard
WATERFALL_FILE = os.path.join(FOLDER, "rf_waterfall.html")   # live: frequency vs time
WATERFALL_ROWS = 60          # how many sweeps (rows) the waterfall remembers
SURVEY_FILE = os.path.join(FOLDER, "rf_survey.html")         # desk map made by "survey" mode
SURVEY_SWEEPS = 3            # sweeps taken at each desk (more = steadier, but slower)
HEAT_MAX_DB = 12.0           # this many dB above normal (or more) shows as full red

# Desk names and positions for the survey map. We reuse the camera's desks.json
# (made by tools/setup_desks.py) if it exists. If not, we make a simple grid.
DESKS_FILE = os.path.join(os.path.dirname(FOLDER), "desks.json")
GRID_ROWS = 3
GRID_COLS = 4

# ---------- bands to watch ----------
# Phones TRANSMIT on the "uplink" part of each band, so these are uplink ranges (MHz).
# Carriers share bands and change them over time, so we report by band, not by carrier.
# The "carriers" text is only a hint. CHECK THESE for your area before trusting them.
# The RTL-SDR can only tune up to about 1766 MHz, so the 1.9 GHz (PCS) and 2.5 GHz
# (T-Mobile Band 41) bands, plus Wi-Fi and Bluetooth, can NOT be seen with it.
BANDS = [
    {"name": "B71 600MHz",  "start": 663.0,  "end": 698.0,  "carriers": "T-Mobile",          "enabled": True},
    {"name": "B12/17 700",  "start": 699.0,  "end": 716.0,  "carriers": "AT&T, T-Mobile",    "enabled": True},
    {"name": "B13 700",     "start": 777.0,  "end": 787.0,  "carriers": "Verizon",           "enabled": True},
    {"name": "B14 FirstNet", "start": 788.0, "end": 798.0,  "carriers": "AT&T FirstNet",    "enabled": True},
    {"name": "B5 850",      "start": 824.0,  "end": 849.0,  "carriers": "AT&T, Verizon",     "enabled": True},
    {"name": "B4/66 AWS",   "start": 1710.0, "end": 1755.0, "carriers": "all (weak: top edge of radio)", "enabled": False},
]
