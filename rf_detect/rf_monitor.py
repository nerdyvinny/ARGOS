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
try:
    from rf_detect import rf_config as config   # python -m rf_detect.rf_monitor   (from the repo root)
except ImportError:
    import rf_config as config                  # python rf_monitor.py             (from inside rf_detect)


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