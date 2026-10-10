# RF phone-activity monitor (uses an RTL-SDR)
#
# It does NOT decode or listen to anything. It only measures how much radio
# power there is in the phone "uplink" bands, like the waterfall in SDR++ does.
# Phones that are transmitting make the power go UP compared to the empty room.
#
#   python rf_monitor.py scan                one sweep, shows raw levels (good first test)
#   python rf_monitor.py baseline            learn what the EMPTY room looks like
#   python rf_monitor.py monitor             watch and compare against the baseline
#
# Add --simulate to try everything with a fake radio (no hardware needed).
# Add --sweeps 20 to stop after 20 sweeps.
# Close SDR++ first! Only one program can use the radio at a time.

import os
import sys
import csv
import json
import time
import numpy as np
import rf_config as config


# ---------------- the radio (real or fake) ----------------

class RealRadio:
    def __init__(self):
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
    def __init__(self):
        self.rng = np.random.default_rng(1)
        self.phone_on = False

    def read(self, center_hz):
        n = config.SAMPLES_PER_STEP
        noise = self.rng.normal(0, 0.01, n) + 1j * self.rng.normal(0, 0.01, n)
        mhz = center_hz / 1e6
        if self.phone_on and 699 <= mhz <= 716:
            start = int(self.rng.integers(0, n - 20000))
            noise[start:start + 20000] *= 4     # about +12 dB for a short moment
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


# ---------------- baseline ----------------

def make_baseline(radio, steps):
    sweeps = []
    for i in range(config.BASELINE_SWEEPS):
        print("Baseline sweep", i + 1, "of", config.BASELINE_SWEEPS, "(keep the room quiet, phones OFF)")
        sweeps.append(sweep(radio, steps))

    baseline = {}
    for key in sweeps[0]:
        avgs = [s[key]["avg"] for s in sweeps]
        peaks = [s[key]["peak"] for s in sweeps]
        baseline[key] = {"avg": float(np.median(avgs)), "peak": float(np.median(peaks))}

    with open(config.BASELINE_FILE, "w") as f:
        json.dump(baseline, f, indent=2)
    print("Saved", config.BASELINE_FILE, "with", len(baseline), "frequency windows")


def load_baseline():
    if not os.path.exists(config.BASELINE_FILE):
        return None
    with open(config.BASELINE_FILE) as f:
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


def band_status(hits):
    # hits = True/False for each of the last few sweeps (newest last)
    # ACTIVE only if it is loud RIGHT NOW and was also loud recently.
    # If it went quiet again we say "recent" so it does not look like it is still on.
    if hits[-1] and sum(hits) >= config.CONFIRM_HITS:
        return "ACTIVE"
    if hits[-1]:
        return "maybe"
    if sum(hits) > 0:
        return "recent"
    return "ok"


def print_table(rows, history, sweep_number):
    print("")
    print("---- sweep", sweep_number, "at", time.strftime("%H:%M:%S"), "----")
    print("%-14s %-16s %9s %9s %9s  %s" % ("band", "carriers", "level dB", "avg +dB", "peak +dB", "status"))
    for r in rows:
        name = r["band"]["name"]
        if r["avg_delta"] is None:
            status = "(no baseline)"
        else:
            status = band_status(history[name])
            if status == "ACTIVE":
                status = "ACTIVE  <-- transmitting"
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
            active = band_status(history[name]) == "ACTIVE"
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), name, round(r["level"], 2),
                        "" if r["avg_delta"] is None else round(r["avg_delta"], 2),
                        "" if r["peak_delta"] is None else round(r["peak_delta"], 2),
                        int(active)])


# ---------------- main ----------------

def get_options():
    args = sys.argv[1:]
    mode = args[0] if args and not args[0].startswith("--") else "monitor"
    simulate = "--simulate" in args
    sweeps = None
    if "--sweeps" in args:
        sweeps = int(args[args.index("--sweeps") + 1])
    return mode, simulate, sweeps


def main():
    mode, simulate, max_sweeps = get_options()
    if mode not in ("scan", "baseline", "monitor"):
        print("Use: python rf_monitor.py scan | baseline | monitor   [--simulate] [--sweeps N]")
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
            print("Is SDR++ still open? Is pyrtlsdr installed? See README.md")
            return

    try:
        if mode == "baseline":
            make_baseline(radio, steps)
            return

        baseline = None
        if mode == "monitor":
            baseline = load_baseline()
            if baseline is None:
                print("No", config.BASELINE_FILE, "yet. Run: python rf_monitor.py baseline  (in the EMPTY room)")
                return

        history = {}
        for b in bands:
            history[b["name"]] = [False] * config.CONFIRM_WINDOW

        sweep_number = 0
        print("Press Ctrl+C to stop")
        while max_sweeps is None or sweep_number < max_sweeps:
            sweep_number = sweep_number + 1
            if simulate and mode == "monitor" and sweep_number == 4:
                radio.phone_on = True            # test only: a fake phone starts transmitting
            rows = band_readings(bands, steps, sweep(radio, steps), baseline)
            for r in rows:
                name = r["band"]["name"]
                history[name] = history[name][1:] + [is_suspicious(r)]
            print_table(rows, history, sweep_number)
            if mode == "monitor":
                write_log(rows, history)
            if mode == "scan":
                break

        if mode == "monitor":
            active = [n for n in history if band_status(history[n]) == "ACTIVE"]
            print("")
            print("Bands that look ACTIVE right now:", active if active else "none")
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        radio.close()


if __name__ == "__main__":
    main()
