# RF phone-activity monitor (uses an RTL-SDR)
#
# It does NOT decode or listen to anything. It only measures how much radio
# power there is in the phone "uplink" bands, like the waterfall in SDR++ does.
# Phones that are transmitting make the power go UP compared to the empty room.
#
# Run it from anywhere (Windows cmd), for example from the repo root:
#   py rf_detect\rf_monitor.py scan          one sweep, shows raw levels (good first test)
#   py rf_detect\rf_monitor.py baseline      learn what the EMPTY room looks like
#   py rf_detect\rf_monitor.py fovtest       range test: farthest desk vs behind camera vs next room
#   py rf_detect\rf_monitor.py monitor       watch and compare against the baseline
#   py rf_detect\rf_monitor.py bluetooth     only look for Bluetooth headphones (no RTL-SDR needed)
#
# Setup: the antenna sits at the FRONT of the room next to the camera, facing the
# students, with a metal sheet behind it (see rf_config.py). Order: baseline, fovtest, monitor.
# The camera link is still a placeholder (see camera_confirms below).
#
# "monitor" also checks Bluetooth for headphones in the background.
# Bluetooth needs Python 3.12, so for that use  py -3.12  instead of  py
# (first time:  py -3.12 -m pip install bleak numpy pyrtlsdr==0.2.93 "setuptools<81")
# (newer pyrtlsdr does not work with the rtlsdr.dll that comes with SDR++)
#
# The web page opens in your browser at http://localhost:5050
#   /waterfall   (scan / monitor)       which band is busy, over time
#   /bluetooth   (bluetooth / monitor)  headphones switched on nearby
# The page stays up after the sweeps finish. Press Ctrl+C to quit.
#
# Add --simulate to try everything with a fake radio (no hardware needed).
# Add --sweeps 20 to stop after 20 sweeps (or 20 Bluetooth scans).
# Add --no-web to skip the web page (the tests use this).
# Close SDR++ first! Only one program can use the radio at a time.

import os
import sys
import csv
import json
import time
import numpy as np

# Let Python find rf_config.py, rf_web.py and bt_scan.py (they sit next to this file),
# no matter which folder you run from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rf_config as config
import rf_web
import bt_scan


# ---------------- the radio (real or fake) ----------------

def find_rtlsdr_driver():
    # The rtlsdr Python package needs rtlsdr.dll. Look for it in this folder
    # and in the SDR++ folder (set in rf_config.py), and tell Windows about it.
    for folder in [config.FOLDER, config.RTLSDR_DLL_FOLDER]:
        if folder and os.path.exists(os.path.join(folder, "rtlsdr.dll")):
            os.environ["PATH"] = folder + os.pathsep + os.environ["PATH"]
            if sys.platform == "win32":
                os.add_dll_directory(folder)     # so libusb-1.0.dll next to it is found too
            return folder
    return None


class RealRadio:
    def __init__(self):
        folder = find_rtlsdr_driver()
        if folder is None:
            print("Could not find rtlsdr.dll. Set RTLSDR_DLL_FOLDER in rf_config.py")
            print("to your SDR++ folder, or copy rtlsdr.dll + libusb-1.0.dll into rf_detect.")
        from rtlsdr import RtlSdr   # imported here so --simulate works without it
        self.sdr = RtlSdr()
        self.sdr.sample_rate = config.SAMPLE_RATE
        if config.PPM:
            self.sdr.freq_correction = config.PPM
        self.sdr.gain = config.GAIN
        if config.GAIN == "auto":
            print("WARNING: GAIN is \"auto\". The radio then changes its own gain while it measures,")
            print("so the dB numbers drift and can't be compared to the baseline. Use a number (e.g. 40).")
        try:
            self.sdr.set_agc_mode(False)      # the RTL2832's own digital AGC: off, so levels stay comparable
        except Exception as e:
            print("Could not turn off the RTL2832 AGC:", e)

    def read(self, center_hz):
        return read_window(self.sdr, center_hz)

    def close(self):
        self.sdr.close()


def read_window(sdr, center_hz):
    # Tune, throw away the junk the radio makes right after tuning, then grab the samples.
    # The tuner needs a few ms to lock, and samples from the OLD frequency are still on
    # their way over USB. This happens after EVERY retune, so we throw away the same
    # amount every time (the old code threw away extra only on the very first read).
    sdr.set_center_freq(center_hz)
    sdr.read_samples(config.SETTLE_SAMPLES)
    return sdr.read_samples(config.SAMPLES_PER_STEP)


class FakeRadio:
    # Pretend radio for testing. It tries to act like the real one:
    #  - quiet noise everywhere, plus the small fake spike in the center every RTL-SDR has
    #  - junk for a few ms after EVERY retune (how long is a GUESS, see DECISIONS.md;
    #    "scan" measures it on the real radio)
    #  - nothing can be heard above TUNER_MAX_MHZ
    #  - when phone_on is True: a phone on a 10 MHz carrier in B12/17 (704-714 MHz) sends
    #    SHORT bursts (1-5 ms) on a few "resource blocks" (180 kHz to 1 MHz wide), about
    #    4 per 0.1 s, like a phone that is streaming. Most of the band is quiet most of the time.
    # (The old fake phone was a loud 8 ms burst across the WHOLE window in every grab. That was
    #  too easy, so it hid the bug where real short bursts were averaged away.)
    # phone_strength is how loud a burst is: 4 = 16x the noise power per Hz = +12 dB above the floor.
    # fovtest changes it to pretend the phone moved further away.
    def __init__(self):
        self.rng = np.random.default_rng(1)
        self.phone_on = False
        self.phone_strength = 4.0
        self.center = None
        self.since_tune = 0            # samples read since the last retune
        self.junk_samples = 8192       # junk after each retune (3.4 ms, a guess)
        self.sdr = self                # so code that talks to "radio.sdr" works with the fake too

    def read(self, center_hz):
        return read_window(self, center_hz)

    # the next two act like pyrtlsdr's RtlSdr, so read_window() works the same on both
    def set_center_freq(self, hz):
        self.center = hz
        self.since_tune = 0

    def read_samples(self, n):
        x = self.rng.normal(0, 0.01, n) + 1j * self.rng.normal(0, 0.01, n)
        x = x + (0.002 + 0.001j)                          # the fake spike at the center
        if self.phone_on and self.center <= config.TUNER_MAX_MHZ * 1e6:
            self.add_phone_bursts(x)
        if self.since_tune < self.junk_samples:           # junk right after tuning
            bad = min(n, self.junk_samples - self.since_tune)
            x[:bad] = x[:bad] * 10
        self.since_tune = self.since_tune + n
        return x

    def add_phone_bursts(self, x):
        n = len(x)
        count = self.rng.poisson(4 * n / config.SAMPLES_PER_STEP)
        for _ in range(count):
            width = self.rng.choice([180e3, 360e3, 1.08e6])            # 1, 2 or 6 resource blocks
            low = self.rng.uniform(704e6, 714e6 - width) - self.center  # where it lands, from our center
            if low + width < -config.SAMPLE_RATE / 2 or low > config.SAMPLE_RATE / 2:
                continue                                                # not in this window
            length = min(n - 1, int(self.rng.uniform(1, 5) * 1e-3 * config.SAMPLE_RATE))
            start = int(self.rng.integers(0, n - length))
            burst = self.rng.normal(0, 0.01, length) + 1j * self.rng.normal(0, 0.01, length)
            spectrum = np.fft.fft(burst)
            freqs = np.fft.fftfreq(length, 1.0 / config.SAMPLE_RATE)
            spectrum[(freqs < low) | (freqs > low + width)] = 0          # keep only its resource blocks
            x[start:start + length] += np.fft.ifft(spectrum) * self.phone_strength

    def close(self):
        pass


# ---------------- measuring ----------------

def make_steps(bands):
    # Cut each band into windows the radio can see at once.
    # Gives a list of {"band": name, "center": Hz}
    width = config.SAMPLE_RATE * config.USABLE_FRACTION
    steps = []
    for b in bands:
        start = b["start"] * 1e6
        end = b["end"] * 1e6
        center = start + width / 2
        while True:
            # the RTL-SDR can't tune above TUNER_MAX_MHZ: skip those windows
            # (cant_hear() below reports them, so they are never silently "ok")
            if center + width / 2 <= config.TUNER_MAX_MHZ * 1e6:
                steps.append({"band": b["name"], "center": center})
            if center + width / 2 >= end:
                break
            center = center + width
    return steps


def cant_hear(band, steps):
    # Gives back a short text about the part of this band the radio CAN'T hear, or None.
    width = config.SAMPLE_RATE * config.USABLE_FRACTION
    tops = [s["center"] + width / 2 for s in steps if s["band"] == band["name"]]
    if not tops:
        return "CAN'T HEAR: above the %d MHz RTL-SDR limit" % config.TUNER_MAX_MHZ
    if max(tops) < band["end"] * 1e6:
        return "can't hear %.0f-%.0f MHz" % (max(tops) / 1e6, band["end"])
    return None


def bin_mask():
    # which FFT bins we keep (middle part only, without the fake spike in the center)
    n = config.FFT_SIZE
    freqs = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / config.SAMPLE_RATE))
    half_usable = config.SAMPLE_RATE * config.USABLE_FRACTION / 2
    keep = (np.abs(freqs) <= half_usable) & (np.abs(freqs) >= config.DC_NOTCH_HZ)
    return keep


def measure(samples):
    # Gives back (average_db, peak_db) for one grab of samples.
    # dB here is "relative to the radio's full scale" (dBFS), not a calibrated number.
    # Only compare numbers to each other, never to the real world.
    n = config.FFT_SIZE
    segments = len(samples) // n
    samples = np.asarray(samples[:segments * n]).reshape(segments, n)
    window = np.hanning(n)
    spectrum = np.fft.fftshift(np.fft.fft(samples * window, axis=1), axes=1)
    power = (np.abs(spectrum) ** 2) / np.sum(window ** 2)
    power = power[:, bin_mask()]

    average = power.mean()                               # everything, averaged as power (not dB)

    # Peak: phones send in short bursts (often 1 ms) that only use a slice of the
    # band (a few "resource blocks", 180 kHz each). If we average a 1 ms burst with
    # lots of quiet time or quiet frequencies, it disappears. So we cut the spectrum of
    # every FFT slice (0.43 ms) into small pieces about PEAK_BINS wide, and keep the
    # loudest piece. That is about the size of one burst, so a burst shows at (almost)
    # full strength. Pure noise also has a loudest piece, about 2 dB above the
    # average; the baseline learns that, so it doesn't cause alarms.
    pieces = max(1, power.shape[1] // config.PEAK_BINS)
    peak = 0.0
    for part in np.array_split(power, pieces, axis=1):
        peak = max(peak, part.mean(axis=1).max())        # loudest 0.43 ms slice of this piece

    return to_db(average), to_db(peak)


def to_db(x):
    return 10 * np.log10(max(x, 1e-20))


def sweep(radio, steps):
    # Measure every step once. Gives back a dictionary: step key -> {"avg": dB, "peak": dB}
    result = {}
    for s in steps:
        avg, peak = measure(radio.read(s["center"]))
        result[step_key(s)] = {"avg": float(avg), "peak": float(peak)}
    return result


def step_key(step):
    return step["band"] + "|" + str(round(step["center"] / 1e6, 3))


def step_heat(steps, now, baseline):
    # One number per frequency window (for the waterfall and fovtest): how many dB above normal.
    # Uses whichever is bigger: the average jump or the short-burst jump.
    # Without a baseline we compare to the quietest window in this sweep instead.
    values = []
    quietest = min(now[k]["avg"] for k in now)
    for s in steps:
        k = step_key(s)
        if baseline is not None and k in baseline:
            d_avg = now[k]["avg"] - baseline[k]["avg"]
            d_peak = now[k]["peak"] - baseline[k]["peak"]
            values.append(max(d_avg, d_peak))
        else:
            values.append(now[k]["avg"] - quietest)
    return values


# ---------------- baseline ----------------

def baseline_file(simulate):
    # the fake radio gets its own baseline file so it never overwrites the real one
    if simulate:
        return config.SIM_BASELINE_FILE
    return config.BASELINE_FILE


def measure_settings(simulate):
    # Everything that changes the dB numbers. A baseline (or fovtest) only makes sense
    # with the SAME settings: e.g. 10 dB less GAIN makes every reading 10 dB lower, so
    # nothing would ever look "above normal". So we save these with the file and check them.
    return {
        "version": 2,                    # 2 = how measure() works since the burst fix. Change it if measure() changes
        "simulate": simulate,            # a fake-radio file must never be used for the real radio
        "sample_rate": config.SAMPLE_RATE,
        "gain": config.GAIN,
        "ppm": config.PPM,
        "samples_per_step": config.SAMPLES_PER_STEP,
        "settle_samples": config.SETTLE_SAMPLES,
        "fft_size": config.FFT_SIZE,
        "peak_bins": config.PEAK_BINS,
        "usable_fraction": config.USABLE_FRACTION,
        "dc_notch_hz": config.DC_NOTCH_HZ,
    }


def make_baseline(radio, steps, file_name, simulate=False):
    sweeps = []
    for i in range(config.BASELINE_SWEEPS):
        print("Baseline sweep", i + 1, "of", config.BASELINE_SWEEPS, "(keep the room quiet, phones OFF)")
        sweeps.append(sweep(radio, steps))

    baseline = {}
    for key in sweeps[0]:
        avgs = [s[key]["avg"] for s in sweeps]
        peaks = [s[key]["peak"] for s in sweeps]
        baseline[key] = {"avg": float(np.median(avgs)), "peak": float(np.median(peaks))}

    with open(file_name, "w") as f:
        json.dump({"settings": measure_settings(simulate), "values": baseline}, f, indent=2)
    print("Saved", file_name, "with", len(baseline), "frequency windows")

    # A quick check you can compare with FIX_REPORT.md: in an empty room the loudest
    # burst-sized piece is only about +2 dB above the average (that's just noise).
    # Much more than that means junk after retuning, or something really transmitting.
    print("Empty-room check: loudest burst-sized piece minus average (pure noise is about +2 dB):")
    names = []
    for s in steps:
        if s["band"] not in names:
            names.append(s["band"])
    for name in names:
        spread = [baseline[step_key(s)]["peak"] - baseline[step_key(s)]["avg"] for s in steps if s["band"] == name]
        print("   %-14s +%.1f dB (worst window +%.1f dB)" % (name, float(np.median(spread)), max(spread)))
    return baseline


def settle_check(sdr, center_hz):
    # How long does the junk after tuning REALLY last on this radio? ("scan" shows it)
    # We tune away and back, and this time KEEP the first samples to look at them.
    # It can only see junk that is louder or quieter than normal (3 dB or more).
    sdr.set_center_freq(center_hz + 5e6)
    sdr.read_samples(config.SETTLE_SAMPLES)
    sdr.set_center_freq(center_hz)
    raw = np.asarray(sdr.read_samples(4 * config.SETTLE_SAMPLES))
    n = config.FFT_SIZE
    pieces = len(raw) // n
    power = np.mean(np.abs(raw[:pieces * n].reshape(pieces, n)) ** 2, axis=1)
    normal = np.median(power[pieces // 2:])           # by the second half it has surely settled
    junk = 0
    while junk < pieces and abs(to_db(power[junk]) - to_db(normal)) > 3.0:
        junk = junk + 1
    junk_ms = 1000.0 * junk * n / config.SAMPLE_RATE
    settle_ms = 1000.0 * config.SETTLE_SAMPLES / config.SAMPLE_RATE
    print("")
    print("Retune check: junk after tuning lasted about %.1f ms. We throw away %.1f ms." % (junk_ms, settle_ms))
    if junk >= pieces // 2:
        print("   WARNING: it never settled. Something is wrong with the radio (or a strong signal is switching on/off).")
    elif junk * n > config.SETTLE_SAMPLES:
        print("   WARNING: not enough! Raise SETTLE_SAMPLES in rf_config.py to at least %d, then redo baseline." % (2 * junk * n))
    else:
        print("   OK")
    return junk_ms


def load_baseline(file_name, simulate=False):
    # Gives back the saved numbers, or None if there is no file or it can't be trusted.
    # Used for the baseline file and the fovtest file (same kind of .json file).
    if not os.path.exists(file_name):
        return None
    with open(file_name) as f:
        data = json.load(f)
    name = os.path.basename(file_name)
    if not isinstance(data, dict) or "settings" not in data or "values" not in data:
        print(name, "is from an older version of this program, so it is NOT used. Make it again.")
        return None
    wanted = measure_settings(simulate)
    if data["settings"] != wanted:
        print(name, "was made with different settings, so it is NOT used. Make it again. Differences:")
        for key in wanted:
            if data["settings"].get(key) != wanted[key]:
                print("   %s: file has %s, now %s" % (key, data["settings"].get(key), wanted[key]))
        return None
    return data["values"]


# ---------------- reporting ----------------

def band_readings(bands, steps, now, baseline):
    # For each band find the window that is most above normal.
    rows = []
    for b in bands:
        keys = [step_key(s) for s in steps if s["band"] == b["name"]]
        avg_delta = None
        peak_delta = None
        level = None
        for k in keys:
            if k not in now:
                continue
            if level is None or now[k]["avg"] > level:
                level = now[k]["avg"]
            if baseline is not None and k in baseline:
                d_avg = now[k]["avg"] - baseline[k]["avg"]
                d_peak = now[k]["peak"] - baseline[k]["peak"]
                if avg_delta is None or d_avg > avg_delta:
                    avg_delta = d_avg
                if peak_delta is None or d_peak > peak_delta:
                    peak_delta = d_peak
        rows.append({"band": b, "level": level, "avg_delta": avg_delta, "peak_delta": peak_delta,
                     "cant_hear": cant_hear(b, steps)})
    return rows


def is_suspicious(row, fov):
    # fov = alarm levels from "fovtest" (or None). See rf_config.py.
    if row["avg_delta"] is None:
        return False
    name = row["band"]["name"]
    if fov is not None and name in fov:
        # fovtest found a level that a phone IN FRONT of the camera goes over,
        # but a phone BEHIND it doesn't. Use that instead of the normal rule.
        return max(row["avg_delta"], row["peak_delta"]) >= fov[name]["threshold"]
    return row["avg_delta"] >= config.THRESHOLD_DB or row["peak_delta"] >= config.PEAK_THRESHOLD_DB


def band_status(row, history, camera):
    # Gives back (text to show, True if this band counts as ACTIVE).
    # camera = what camera_confirms() said: True, False, or None (not linked)
    name = row["band"]["name"]
    if row.get("cant_hear") and row.get("level") is None:
        return row["cant_hear"], False                  # the whole band is out of the radio's range
    note = ""
    if row.get("cant_hear"):
        note = "  (" + row["cant_hear"] + ")"           # part of the band is out of range
    if row["avg_delta"] is None:
        return "(no baseline)" + note, False
    if sum(history[name]) >= config.CONFIRM_HITS:
        if config.CAMERA_CONFIRM and camera is False:
            return "RF only, camera sees nothing (ignored)" + note, False
        if camera is True:
            return "ACTIVE  <-- transmitting (camera agrees)" + note, True
        return "ACTIVE  <-- transmitting" + note, True
    if history[name][-1]:
        return "maybe" + note, False
    return "ok" + note, False


def print_table(rows, history, sweep_number, camera):
    print("")
    print("==== CELL BANDS (RTL-SDR) sweep", sweep_number, "at", time.strftime("%H:%M:%S"), "====")
    print("%-14s %-16s %9s %9s %9s  %s" % ("band", "carriers", "level dB", "avg +dB", "peak +dB", "status"))
    for r in rows:
        name = r["band"]["name"]
        status, active = band_status(r, history, camera)
        avg_text = "" if r["avg_delta"] is None else "%+.1f" % r["avg_delta"]
        peak_text = "" if r["peak_delta"] is None else "%+.1f" % r["peak_delta"]
        level_text = "" if r["level"] is None else "%.1f" % r["level"]
        print("%-14s %-16s %9s %9s %9s  %s" % (name, r["band"]["carriers"][:16], level_text, avg_text, peak_text, status))


def write_log(rows, history, camera):
    new_file = not os.path.exists(config.LOG_FILE)
    with open(config.LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["time", "band", "level_db", "avg_delta_db", "peak_delta_db", "active"])
        for r in rows:
            name = r["band"]["name"]
            status, active = band_status(r, history, camera)
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), name, "" if r["level"] is None else round(r["level"], 2),
                        "" if r["avg_delta"] is None else round(r["avg_delta"], 2),
                        "" if r["peak_delta"] is None else round(r["peak_delta"], 2),
                        int(active)])


# ---------------- camera link (PLACEHOLDER: not hooked up yet) ----------------

def camera_confirms():
    # PLACEHOLDER. Gives back:
    #   True  = the camera sees something that explains the RF (a phone, someone on their phone)
    #   False = the camera sees nothing suspicious in its view
    #   None  = the camera is not connected yet (what it always says for now)
    #
    # The idea: the antenna and camera both face the students, so a phone the
    # camera can see should also be the one making the RF. If the RF goes up but
    # the camera sees nothing, the phone is probably outside the camera's view
    # (behind it, in the hallway, next room) and we can ignore it.
    #
    # TODO (with the team) - how to connect it, pick ONE:
    #   a) app/main.py saves what it sees every second to a small file, e.g.
    #      camera_status.json = {"time": 1700000000, "phone_people": ["Person 4"]}
    #      and here we read it:
    #          data = json.load(open("camera_status.json"))
    #          if time.time() - data["time"] > 5: return None     # camera stopped
    #          return len(data["phone_people"]) > 0
    #   b) run the RF monitor inside app/main.py and ask it directly, e.g.
    #          return len(latest_phone_detections) > 0             # from vision/detector.py
    #
    # Also TODO: when the camera says yes, tell rules/brain.py so that person's
    # score goes up a little (Person 3's file, ask first).
    return None


# ---------------- fovtest: only count phones in front of the camera ----------------

def fov_file(simulate):
    if simulate:
        return config.SIM_FOV_FILE
    return config.FOV_FILE


def measure_spot(radio, steps, bands, baseline):
    # How far above normal each band is with the test phone at one spot.
    # Takes the loudest window of each band, a few sweeps, and uses the middle value
    # (the median) so one random blip doesn't count.
    readings = {}
    for b in bands:
        readings[b["name"]] = []
    for _ in range(config.FOV_TEST_SWEEPS):
        values = step_heat(steps, sweep(radio, steps), baseline)
        for b in bands:
            band_values = [v for s, v in zip(steps, values) if s["band"] == b["name"]]
            readings[b["name"]].append(max(band_values))
    result = {}
    for name in readings:
        result[name] = float(np.median(readings[name]))
    return result


def run_fovtest(radio, steps, bands, baseline, simulate):
    # Sets the alarm level so a phone at the farthest desk counts, but a phone
    # behind the camera or in the next room doesn't. See RANGE in rf_config.py.
    bands = [b for b in bands if any(s["band"] == b["name"] for s in steps)]   # skip bands the radio can't hear
    print("")
    print("RANGE TEST. You need one test phone. Turn its Wi-Fi OFF, then start a")
    print("phone call or stream a video on mobile data, so it keeps transmitting.")

    print("")
    print("STEP 1: put the phone at the FARTHEST desk the camera can still see.")
    if simulate:
        radio.phone_on = True
        radio.phone_strength = 4.0          # fake: about +12 dB
    else:
        input("Press ENTER when it's there and transmitting...")
    front = measure_spot(radio, steps, bands, baseline)

    print("")
    print("STEP 2: move the SAME phone BEHIND the camera, to the closest spot you want")
    print("ignored (for example the teacher's desk, or the door behind you).")
    if simulate:
        radio.phone_strength = 2.0          # fake: about +6 dB, the metal sheet makes it weaker
    else:
        input("Press ENTER when it's there and transmitting...")
    behind = measure_spot(radio, steps, bands, baseline)

    print("")
    print("STEP 3: take the SAME phone into the NEXT classroom (or the hallway), right")
    print("on the other side of the wall - the closest spot from another room.")
    next_room = None
    if simulate:
        radio.phone_strength = 1.6          # fake: about +4 dB, the wall makes it weaker
        next_room = measure_spot(radio, steps, bands, baseline)
    else:
        answer = input("Press ENTER when it's there and transmitting (s = skip this step)...")
        if answer.strip().lower() != "s":
            next_room = measure_spot(radio, steps, bands, baseline)

    print("")
    print("%-14s %9s %9s %9s %7s  %s" % ("band", "front dB", "behind dB", "next room", "gap", "result"))
    levels = {}
    for b in bands:
        name = b["name"]
        f = front[name]
        bh = behind[name]
        # the loudest spot we want IGNORED
        ignore = bh
        next_text = "skipped"
        if next_room is not None:
            ignore = max(bh, next_room[name])
            next_text = "%+.1f" % next_room[name]
        if f < config.THRESHOLD_DB:
            result = "phone not seen in this band (normal rules stay)"
        elif f - ignore < 3.0:
            result = "too close to tell apart (see RANGE in rf_config.py)"
        else:
            threshold = max((f + ignore) / 2, config.THRESHOLD_DB)   # halfway between "count" and "ignore"
            levels[name] = {"front": round(f, 2), "behind": round(bh, 2), "threshold": round(threshold, 2)}
            if next_room is not None:
                levels[name]["next_room"] = round(next_room[name], 2)
            result = "OK: alarm at +%.1f dB" % threshold
        print("%-14s %+9.1f %+9.1f %9s %+7.1f  %s" % (name, f, bh, next_text, f - ignore, result))

    if not levels:
        print("")
        print("No band could tell this room from the other spots, so nothing was saved.")
        print("Check the phone was transmitting. Then try: a metal sheet ~11 cm behind the")
        print("antenna, the antenna closer to the students, or a lower GAIN (see rf_config.py).")
        return
    with open(fov_file(simulate), "w") as f:
        json.dump({"settings": measure_settings(simulate), "values": levels}, f, indent=2)
    print("")
    print("Saved", fov_file(simulate), "- monitor will use these alarm levels.")
    print("Bands not set here still use the normal rules. Re-run this if you move the antenna or camera.")


# ---------------- main ----------------

def keep_page_open(address):
    # the web page only works while this program runs, so wait here
    if address is None:
        return
    print("")
    print("Done measuring. The web page is still at", address, " Press Ctrl+C to quit.")
    while True:
        time.sleep(1)


def start_web(mode):
    # start the web page and open the right tab for this mode
    pages = {"/waterfall": config.WATERFALL_FILE, "/bluetooth": config.BT_FILE}
    page = "/waterfall"
    if mode == "bluetooth":
        page = "/bluetooth"
    return rf_web.start_web_server(config.WEB_PORT, pages, page)


def show_bluetooth(watcher):
    bt_scan.print_table(watcher)
    rf_web.write_bluetooth(config.BT_FILE, watcher.current(), watcher.total, watcher.problem,
                               watcher.too_far, config.BT_MIN_RSSI)


def run_bluetooth(simulate, max_scans, web):
    # Bluetooth only: doesn't need the RTL-SDR
    if simulate:
        print("(SIMULATION: fake Bluetooth, no hardware)")
    watcher = bt_scan.BluetoothWatcher(simulate)
    address = None
    if web:
        address = start_web("bluetooth")
    print("Scanning for Bluetooth headphones,", config.BT_SCAN_SECONDS, "seconds per scan. Press Ctrl+C to stop")
    try:
        scan_number = 0
        while max_scans is None or scan_number < max_scans:
            scan_number = scan_number + 1
            try:
                new = watcher.scan()
            except Exception as e:
                print(bt_scan.explain_problem(e))
                return
            for d in new:
                print("NEW headphones:", d["name"], "-", d["distance"], "-", d["why"])
            show_bluetooth(watcher)
        keep_page_open(address)
    except KeyboardInterrupt:
        print("\nStopped")


def get_options():
    args = sys.argv[1:]
    mode = args[0] if args and not args[0].startswith("--") else "monitor"
    simulate = "--simulate" in args
    web = "--no-web" not in args
    sweeps = None
    if "--sweeps" in args:
        sweeps = int(args[args.index("--sweeps") + 1])
    return mode, simulate, sweeps, web


def main():
    mode, simulate, max_sweeps, web = get_options()
    if mode not in ("scan", "baseline", "fovtest", "monitor", "bluetooth"):
        print("Use: py rf_monitor.py scan | baseline | fovtest | monitor | bluetooth   [--simulate] [--sweeps N] [--no-web]")
        return

    if mode == "bluetooth":
        run_bluetooth(simulate, max_sweeps, web)
        return

    bands = [b for b in config.BANDS if b["enabled"]]
    steps = make_steps(bands)
    print("Watching", len(bands), "bands in", len(steps), "frequency windows")
    for b in bands:
        text = cant_hear(b, steps)
        if text:
            print("  %s (%.0f-%.0f MHz): %s. A phone using only this part is MISSED." % (b["name"], b["start"], b["end"], text))

    if simulate:
        radio = FakeRadio()
        print("(SIMULATION: fake radio, no hardware)")
    else:
        try:
            radio = RealRadio()
        except Exception as e:
            print("Could not open the RTL-SDR:", e)
            print("Is SDR++ still open? Is the radio plugged in? Is pyrtlsdr installed (py -3.12 -m pip install pyrtlsdr==0.2.93 \"setuptools<81\")?")
            return

    try:
        if mode == "baseline":
            make_baseline(radio, steps, baseline_file(simulate), simulate)
            return

        baseline = None
        if mode in ("monitor", "fovtest"):
            baseline = load_baseline(baseline_file(simulate), simulate)
            if baseline is None and simulate:
                print("No fake baseline yet, making one first")
                baseline = make_baseline(radio, steps, baseline_file(simulate), simulate)
            if baseline is None:
                print("No usable", baseline_file(simulate), "yet. Run: py rf_monitor.py baseline  (in the EMPTY room)")
                return
            missing = [step_key(s) for s in steps if step_key(s) not in baseline]
            if missing:
                print("WARNING:", len(missing), "frequency windows have no baseline (bands were changed?).")
                print("They show \"(no baseline)\" and can never alarm. Run baseline again.")
                return

        if mode == "fovtest":
            run_fovtest(radio, steps, bands, baseline, simulate)
            return

        address = None
        if web:
            address = start_web(mode)

        history = {}
        for b in bands:
            history[b["name"]] = [False] * config.CONFIRM_WINDOW
        waterfall = []

        fov = None
        if mode == "monitor" and config.USE_FOV_TEST:
            fov = load_baseline(fov_file(simulate), simulate)     # same kind of .json file, so we reuse the loader
            if fov is None:
                print("No front/back test yet, so phones BEHIND the camera count too. Run: fovtest")
            else:
                print("Only counting phones in front of the camera for:", ", ".join(fov))
        if config.CAMERA_CONFIRM:
            print("CAMERA_CONFIRM is on, but the camera link is still a placeholder (see camera_confirms)")

        watcher = None
        if mode == "monitor" and config.BT_CHECK:
            watcher = bt_scan.BluetoothWatcher(simulate)
            watcher.start_background()          # scans Bluetooth while we sweep the radio

        sweep_number = 0
        print("Press Ctrl+C to stop")
        while max_sweeps is None or sweep_number < max_sweeps:
            sweep_number = sweep_number + 1
            if simulate and mode == "monitor" and sweep_number == 4:
                radio.phone_on = True            # test only: a fake phone starts transmitting
            now = sweep(radio, steps)
            rows = band_readings(bands, steps, now, baseline)
            for r in rows:
                name = r["band"]["name"]
                history[name] = history[name][1:] + [is_suspicious(r, fov)]
            camera = camera_confirms()                   # placeholder: always None for now
            print_table(rows, history, sweep_number, camera)

            waterfall.append((time.strftime("%H:%M:%S"), step_heat(steps, now, baseline)))
            waterfall = waterfall[-config.WATERFALL_ROWS:]
            rf_web.write_waterfall(config.WATERFALL_FILE, steps, waterfall, config.HEAT_MAX_DB, baseline is not None)

            if watcher is not None:
                show_bluetooth(watcher)
            if mode == "monitor" and not simulate:    # keep fake data out of the real log
                write_log(rows, history, camera)
            if mode == "scan":
                settle_check(radio.sdr, steps[0]["center"])
                break

        if mode == "monitor":
            active = [r["band"]["name"] for r in rows if band_status(r, history, camera)[1]]
            print("")
            print("Bands that look ACTIVE right now:", active if active else "none")
        keep_page_open(address)
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        radio.close()


if __name__ == "__main__":
    main()
