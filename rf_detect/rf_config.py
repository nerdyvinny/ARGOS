# ============================================
#  RF SETTINGS - change these to tune the monitor
# ============================================

# ---------- the radio ----------
SAMPLE_RATE = 2.4e6          # RTL-SDR: 2.4 MHz is the most it does reliably (use 2.048e6 if you get dropouts)
GAIN = 40                    # in dB. Use "auto" to let the radio decide.
PPM = 0                      # frequency correction, leave 0

# ---------- how we measure ----------
SAMPLES_PER_STEP = 262144    # samples grabbed at each frequency (about 0.1 second)
FFT_SIZE = 1024              # must divide SAMPLES_PER_STEP evenly
CHUNKS = 16                  # each grab is cut into this many pieces so we can catch short bursts
USABLE_FRACTION = 0.8        # only use the middle 80% of each window (the edges are weak)
DC_NOTCH_HZ = 60000          # ignore this much around the center (the radio makes a fake spike there)

# ---------- baseline (empty room) ----------
BASELINE_SWEEPS = 5          # how many times we sweep all bands to learn "normal"
BASELINE_FILE = "rf_baseline.json"

# ---------- deciding "a phone is transmitting" ----------
THRESHOLD_DB = 3.0           # average power this many dB above baseline = suspicious
PEAK_THRESHOLD_DB = 6.0      # a short burst this many dB above the baseline peak = suspicious
CONFIRM_HITS = 2             # must be suspicious in this many...
CONFIRM_WINDOW = 5           # ...of the last this many sweeps before we call it ACTIVE

LOG_FILE = "rf_log.csv"

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