# RF phone-activity monitor (uses an RTL-SDR)
#
# It does NOT decode or listen to anything. It only measures how much radio
# power there is in the phone "uplink" bands, like the waterfall in SDR++ does.
# Phones that are transmitting make the power go UP compared to the empty room.
#
# Run it from anywhere (Windows cmd), for example from the repo root:
#   py rf_detect\rf_monitor.py scan          one sweep, shows raw levels (good first test)
#   py rf_detect\rf_monitor.py baseline      learn what the EMPTY room looks like
#   py rf_detect\rf_monitor.py monitor       watch and compare against the baseline
#   py rf_detect\rf_monitor.py survey        walk the antenna to each desk, makes a desk heatmap
#
# The heatmaps open in your browser at http://localhost:5050
#   /waterfall   (scan / monitor)  which band is busy, over time
#   /survey      (survey)          which desk is the loudest
# The page stays up after the sweeps finish. Press Ctrl+C to quit.
#
# Add --simulate to try everything with a fake radio (no hardware needed).
# Add --sweeps 20 to stop after 20 sweeps.
# Add --no-web to skip the heatmap page (the tests use this).
# Close SDR++ first! Only one program can use the radio at a time.

import os
import sys
import csv
import json
import time
import numpy as np

# Let Python find rf_config.py and rf_heatmap.py (they sit next to this file),
# no matter which folder you run from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rf_config as config
import rf_heatmap


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
        self.last_center = None

    def read(self, center_hz):
        self.sdr.center_freq = center_hz
        if self.last_center is None:
            self.sdr.read_samples(8192)       # first read after a retune is junk
        self.last_center = center_hz
        self.sdr.read_samples(4096)           # throw away a little so the radio settles
        return self.sdr.read_samples(config.SAMPLES_PER_STEP)

    def close(self):
        self.sdr.close()


class FakeRadio:
    # Pretend radio for testing. Quiet noise everywhere.
    # When phone_on is True, the 700 MHz band gets short bursts of extra power.
    # phone_strength is how loud (4 = about +12 dB). Survey mode makes it
    # louder at desks close to the fake phone.
    def __init__(self):
        self.rng = np.random.default_rng(1)
        self.phone_on = False
        self.phone_strength = 4.0

    def read(self, center_hz):
        n = config.SAMPLES_PER_STEP
        noise = self.rng.normal(0, 0.01, n) + 1j * self.rng.normal(0, 0.01, n)
        mhz = center_hz / 1e6
        if self.phone_on and 699 <= mhz <= 716:
            start = int(self.rng.integers(0, n - 20000))
            noise[start:start + 20000] *= self.phone_strength
        return noise

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
            steps.append({"band": b["name"], "center": center})
            if center + width / 2 >= end:
                break
            center = center + width
    return steps


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

    seg_power = power.mean(axis=1)                       # power of each small time slice
    average = seg_power.mean()
    chunk_power = seg_power[:(segments // config.CHUNKS) * config.CHUNKS]
    chunk_power = chunk_power.reshape(config.CHUNKS, -1).mean(axis=1)
    peak = chunk_power.max()                             # loudest piece = catches short bursts

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
    # One number per frequency window for the heatmaps: how many dB above normal.
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


def make_baseline(radio, steps, file_name):
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
        json.dump(baseline, f, indent=2)
    print("Saved", file_name, "with", len(baseline), "frequency windows")
    return baseline


def load_baseline(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name) as f:
        return json.load(f)


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
        rows.append({"band": b, "level": level, "avg_delta": avg_delta, "peak_delta": peak_delta})
    return rows


def is_suspicious(row):
    if row["avg_delta"] is None:
        return False
    return row["avg_delta"] >= config.THRESHOLD_DB or row["peak_delta"] >= config.PEAK_THRESHOLD_DB


def print_table(rows, history, sweep_number):
    print("")
    print("---- sweep", sweep_number, "at", time.strftime("%H:%M:%S"), "----")
    print("%-14s %-16s %9s %9s %9s  %s" % ("band", "carriers", "level dB", "avg +dB", "peak +dB", "status"))
    for r in rows:
        name = r["band"]["name"]
        status = "ok"
        if r["avg_delta"] is None:
            status = "(no baseline)"
        elif sum(history[name]) >= config.CONFIRM_HITS:
            status = "ACTIVE  <-- transmitting"
        elif history[name][-1]:
            status = "maybe"
        avg_text = "" if r["avg_delta"] is None else "%+.1f" % r["avg_delta"]
        peak_text = "" if r["peak_delta"] is None else "%+.1f" % r["peak_delta"]
        print("%-14s %-16s %9.1f %9s %9s  %s" % (name, r["band"]["carriers"][:16], r["level"], avg_text, peak_text, status))


def write_log(rows, history):
    new_file = not os.path.exists(config.LOG_FILE)
    with open(config.LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["time", "band", "level_db", "avg_delta_db", "peak_delta_db", "active"])
        for r in rows:
            name = r["band"]["name"]
            active = sum(history[name]) >= config.CONFIRM_HITS
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), name, round(r["level"], 2),
                        "" if r["avg_delta"] is None else round(r["avg_delta"], 2),
                        "" if r["peak_delta"] is None else round(r["peak_delta"], 2),
                        int(active)])


# ---------------- survey (desk heatmap) ----------------

def run_survey(radio, steps, baseline, simulate):
    # With one antenna we can't tell direction, but we CAN tell distance:
    # the signal is strongest close to the phone. So we carry the antenna to
    # each desk, measure, and color the desk map.
    desks, ratio = rf_heatmap.load_desk_layout(config.DESKS_FILE, config.GRID_ROWS, config.GRID_COLS)
    if os.path.exists(config.DESKS_FILE):
        print("Using", len(desks), "desks from", config.DESKS_FILE)
    else:
        print("No desks.json, using a", config.GRID_ROWS, "x", config.GRID_COLS, "grid (change GRID_ROWS/GRID_COLS in rf_config.py)")

    fake_phone = desks[min(2, len(desks) - 1)]       # simulate only: where the fake phone is
    results = {}
    rf_heatmap.write_survey(config.SURVEY_FILE, desks, ratio, results, config.HEAT_MAX_DB)

    for d in desks:
        if simulate:
            print("")
            print("(SIMULATION) antenna at", d["name"])
            distance = ((d["x"] - fake_phone["x"]) ** 2 + (d["y"] - fake_phone["y"]) ** 2) ** 0.5
            radio.phone_on = True
            radio.phone_strength = 1 + 3 / (1 + 6 * distance)     # louder when closer
        else:
            print("")
            answer = input("Put the antenna at " + d["name"] + " and press ENTER (s = skip, q = finish): ")
            if answer.strip().lower() == "q":
                break
            if answer.strip().lower() == "s":
                continue

        # average the loudest window over a few sweeps, so one random blip doesn't win
        strengths = []
        best_db = None
        best_band = ""
        for _ in range(config.SURVEY_SWEEPS):
            values = step_heat(steps, sweep(radio, steps), baseline)
            loudest = int(np.argmax(values))
            strengths.append(values[loudest])
            if best_db is None or values[loudest] > best_db:
                best_db = values[loudest]
                best_band = steps[loudest]["band"]
        results[d["name"]] = {"db": float(np.mean(strengths)), "band": best_band}
        print("  %s: %+.1f dB above normal (loudest band: %s)" % (d["name"], results[d["name"]]["db"], best_band))
        rf_heatmap.write_survey(config.SURVEY_FILE, desks, ratio, results, config.HEAT_MAX_DB)

    if results:
        ranked = sorted(results, key=lambda name: results[name]["db"], reverse=True)
        print("")
        print("Loudest desks:")
        for name in ranked[:3]:
            print("  %-10s %+.1f dB" % (name, results[name]["db"]))
        print("Saved", config.SURVEY_FILE)


# ---------------- main ----------------

def keep_page_open(address):
    # the heatmap page only works while this program runs, so wait here
    if address is None:
        return
    print("")
    print("Done measuring. Heatmap is still at", address, " Press Ctrl+C to quit.")
    while True:
        time.sleep(1)


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
    if mode not in ("scan", "baseline", "monitor", "survey"):
        print("Use: py rf_monitor.py scan | baseline | monitor | survey   [--simulate] [--sweeps N] [--no-web]")
        return

    bands = [b for b in config.BANDS if b["enabled"]]
    steps = make_steps(bands)
    print("Watching", len(bands), "bands in", len(steps), "frequency windows")

    if simulate:
        radio = FakeRadio()
        print("(SIMULATION: fake radio, no hardware)")
    else:
        try:
            radio = RealRadio()
        except Exception as e:
            print("Could not open the RTL-SDR:", e)
            print("Is SDR++ still open? Is the radio plugged in? Is pyrtlsdr installed (py -m pip install pyrtlsdr)?")
            return

    try:
        if mode == "baseline":
            make_baseline(radio, steps, baseline_file(simulate))
            return

        baseline = None
        if mode in ("monitor", "survey"):
            baseline = load_baseline(baseline_file(simulate))
            if baseline is None and simulate:
                print("No fake baseline yet, making one first")
                baseline = make_baseline(radio, steps, baseline_file(simulate))
            if baseline is None:
                print("No", config.BASELINE_FILE, "yet. Run: py rf_monitor.py baseline  (in the EMPTY room)")
                return

        address = None
        if web:
            page = "/survey" if mode == "survey" else "/waterfall"
            address = rf_heatmap.start_web_server(config.WEB_PORT, config.WATERFALL_FILE, config.SURVEY_FILE, page)

        if mode == "survey":
            run_survey(radio, steps, baseline, simulate)
            keep_page_open(address)
            return

        history = {}
        for b in bands:
            history[b["name"]] = [False] * config.CONFIRM_WINDOW
        waterfall = []

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
                history[name] = history[name][1:] + [is_suspicious(r)]
            print_table(rows, history, sweep_number)

            waterfall.append((time.strftime("%H:%M:%S"), step_heat(steps, now, baseline)))
            waterfall = waterfall[-config.WATERFALL_ROWS:]
            rf_heatmap.write_waterfall(config.WATERFALL_FILE, steps, waterfall, config.HEAT_MAX_DB, baseline is not None)

            if mode == "monitor" and not simulate:    # keep fake data out of the real log
                write_log(rows, history)
            if mode == "scan":
                break

        if mode == "monitor":
            active = [n for n in history if sum(history[n]) >= config.CONFIRM_HITS]
            print("")
            print("Bands that look ACTIVE right now:", active if active else "none")
        keep_page_open(address)
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        radio.close()


if __name__ == "__main__":
    main()
