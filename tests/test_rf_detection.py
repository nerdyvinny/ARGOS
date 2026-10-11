# Detection tests for rf_detect/rf_monitor.py, using SYNTHETIC radio samples (no hardware).
# Run from the repo root:  py -3.12 -m unittest tests.test_rf_detection -v
#
# How it works: we put a pretend "rtlsdr" package in place of the real one, so the
# REAL RealRadio code in rf_monitor.py runs, but the samples come from SyntheticSdr below:
#   - a noise floor (like the radio's own hiss), stored as 8-bit numbers like a real RTL-SDR
#   - a small fake spike at the center (the DC spike every RTL-SDR has)
#   - optional junk right after every retune (ASSUMPTION: we don't know the real length,
#     see DECISIONS.md. The hardware checklist in FIX_REPORT.md measures it)
#   - optional phone bursts: 1-5 ms long, a set bandwidth, a set number of dB above the floor
#   - nothing can be heard when tuned above TUNER_MAX (like the R820T tuner)
#
# "dB above the floor" here means: the burst's power per Hz is that many dB above the
# noise floor's power per Hz, in the part of the spectrum the burst covers.

import os
import sys
import json
import types
import shutil
import tempfile
import subprocess
import unittest
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "rf_detect"))
import rf_config as config
import rf_monitor

FS = config.SAMPLE_RATE
R820T_MAX_HZ = 1766e6          # the synthetic tuner can't hear anything above this


class SyntheticSdr:
    # Looks like pyrtlsdr's RtlSdr to rf_monitor.py.
    def __init__(self, seed=1, junk_samples=0, junk_amp=10.0, noise=0.03, phones=None):
        self.rng = np.random.default_rng(seed)
        self.sample_rate = FS
        self.gain = 40
        self.freq_correction = 0
        self.junk_samples = junk_samples      # samples of junk after each retune
        self.junk_amp = junk_amp              # how much louder the junk is (amplitude)
        self.noise = noise                    # noise floor (amplitude of each of I and Q)
        self.phones = phones or []            # list of dicts, see phone() below
        self._center = None
        self.since_tune = 0                   # samples read since the last retune
        self.discards = []                    # samples thrown away before each measured read
        self.reads_this_tune = []

    # pyrtlsdr lets you do  sdr.center_freq = x  or  sdr.set_center_freq(x)
    def set_center_freq(self, hz):
        if self._center is not None:
            self.discards.append(sum(self.reads_this_tune[:-1]) if self.reads_this_tune else 0)
        self._center = float(hz)
        self.since_tune = 0
        self.reads_this_tune = []

    def get_center_freq(self):
        return self._center

    center_freq = property(get_center_freq, set_center_freq)

    def set_agc_mode(self, enabled):
        self.agc = enabled

    def set_manual_gain_enabled(self, enabled):
        pass

    def get_gain(self):
        return self.gain

    def close(self):
        pass

    def read_samples(self, n):
        rng = self.rng
        # gain changes everything (noise and signals) the same way, like a real tuner
        scale = 10 ** ((float(self.gain) - 40) / 20)
        x = (rng.normal(0, self.noise, n) + 1j * rng.normal(0, self.noise, n))
        x = x + (0.02 + 0.01j)                                  # DC spike
        if self._center <= R820T_MAX_HZ:
            for p in self.phones:
                if p.get("on", True):
                    self.add_bursts(x, p)
        if self.since_tune < self.junk_samples:                 # junk right after a retune
            bad = min(n, self.junk_samples - self.since_tune)
            x[:bad] = x[:bad] * self.junk_amp
        x = x * scale
        self.since_tune += n
        self.reads_this_tune.append(n)
        # store as 8-bit numbers (0..255), like the real radio, then back to -1..1
        i = np.clip(np.round(x.real * 127.5 + 127.5), 0, 255)
        q = np.clip(np.round(x.imag * 127.5 + 127.5), 0, 255)
        return (i - 127.5) / 127.5 + 1j * (q - 127.5) / 127.5

    def add_bursts(self, x, p):
        # p: {"low": MHz, "high": MHz (the carrier), "bw": Hz per burst, "snr": dB,
        #     "ms": (shortest, longest), "per_read": bursts per 0.1 s}
        n = len(x)
        expected = p["per_read"] * n / config.SAMPLES_PER_STEP
        count = self.rng.poisson(expected)
        for _ in range(count):
            # where in the carrier this burst lands (like a phone's resource blocks)
            lo_hz = self.rng.uniform(p["low"] * 1e6, p["high"] * 1e6 - p["bw"])
            f_lo = lo_hz - self._center
            f_hi = f_lo + p["bw"]
            if f_hi < -FS / 2 or f_lo > FS / 2:
                continue                                        # not in this window
            length = int(self.rng.uniform(p["ms"][0], p["ms"][1]) * 1e-3 * FS)
            if length >= n:
                length = n - 1
            start = int(self.rng.integers(0, n - length))
            b = self.rng.normal(0, self.noise, length) + 1j * self.rng.normal(0, self.noise, length)
            spec = np.fft.fft(b)
            f = np.fft.fftfreq(length, 1.0 / FS)
            spec[(f < f_lo) | (f > f_hi)] = 0                   # keep only the burst's bandwidth
            b = np.fft.ifft(spec) * np.sqrt(10 ** (p["snr"] / 10))
            x[start:start + length] += b


def phone(snr, bw=1.08e6, ms=(1, 5), per_read=3, low=704.0, high=714.0):
    # default: a 10 MHz AT&T/T-Mobile carrier in B12/17, bursts of 6 resource blocks
    return {"low": low, "high": high, "bw": bw, "snr": snr, "ms": ms, "per_read": per_read, "on": True}


def install_fake_rtlsdr(sdr):
    # rf_monitor.RealRadio does "from rtlsdr import RtlSdr" then RtlSdr()
    module = types.ModuleType("rtlsdr")
    module.RtlSdr = lambda: sdr
    sys.modules["rtlsdr"] = module


def band(name):
    for b in config.BANDS:
        if b["name"] == name:
            return dict(b)
    raise KeyError(name)


def hearable(bands):
    return [b for b in bands if b["start"] < R820T_MAX_HZ / 1e6]


def run_monitor(radio, bands, baseline, sweeps, fov=None, turn_on_at=None):
    # The same steps as the monitor loop in rf_monitor.main().
    # Gives back {band name: [(sweep number, status text, active), ...]}
    steps = rf_monitor.make_steps(bands)
    history = {}
    for b in bands:
        history[b["name"]] = [False] * config.CONFIRM_WINDOW
    result = {b["name"]: [] for b in bands}
    for number in range(1, sweeps + 1):
        if turn_on_at is not None and number == turn_on_at:
            for p in radio.sdr.phones:
                p["on"] = True
        now = rf_monitor.sweep(radio, steps)
        rows = rf_monitor.band_readings(bands, steps, now, baseline)
        for r in rows:
            name = r["band"]["name"]
            history[name] = history[name][1:] + [rf_monitor.is_suspicious(r, fov)]
        for r in rows:
            status, active = rf_monitor.band_status(r, history, None)
            result[r["band"]["name"]].append((number, status, active))
    return result


class Synthetic(unittest.TestCase):
    # base class: a temp folder for baseline files, and config restored after each test

    def setUp(self):
        self.folder = tempfile.mkdtemp()
        self.saved_gain = config.GAIN

    def tearDown(self):
        config.GAIN = self.saved_gain
        shutil.rmtree(self.folder, ignore_errors=True)
        sys.modules.pop("rtlsdr", None)

    def radio(self, **kwargs):
        sdr = SyntheticSdr(**kwargs)
        install_fake_rtlsdr(sdr)
        return rf_monitor.RealRadio()

    def baseline(self, radio, bands):
        # learn the empty room with every phone off
        saved = [p["on"] for p in radio.sdr.phones]
        for p in radio.sdr.phones:
            p["on"] = False
        steps = rf_monitor.make_steps(bands)
        path = os.path.join(self.folder, "baseline.json")
        rf_monitor.make_baseline(radio, steps, path)
        for p, on in zip(radio.sdr.phones, saved):
            p["on"] = on
        return rf_monitor.load_baseline(path), path


# ---------------------------------------------------------------- acceptance

class BurstDetection(Synthetic):

    def check_detected(self, snr, bw, ms, seed, junk):
        bands = [band("B12/17 700")]
        radio = self.radio(seed=seed, junk_samples=junk, phones=[phone(snr, bw=bw, ms=ms)])
        baseline, _ = self.baseline(radio, bands)
        self.assertIsNotNone(baseline)
        result = run_monitor(radio, bands, baseline, config.CONFIRM_WINDOW)
        active = [n for n, status, a in result["B12/17 700"] if a]
        label = "%d dB, %.0f kHz wide, %s ms, seed %d, junk %d" % (snr, bw / 1e3, ms, seed, junk)
        self.assertTrue(active, "never ACTIVE within CONFIRM_WINDOW sweeps: " + label
                        + " " + str(result["B12/17 700"]))
        return active[0]

    # acceptance: a burst 6 dB or more above the floor reaches ACTIVE within CONFIRM_WINDOW sweeps
    def test_6db_and_up_bursts_reach_active(self):
        seed = 10
        for snr in (6, 10, 20):
            for ms in ((1, 1), (5, 5)):
                for bw in (180e3, 1.08e6, 4.5e6):        # 1 resource block, 6 RBs, a full 5 MHz carrier
                    seed += 1
                    self.check_detected(snr, bw, ms, seed, junk=0)

    # same, with junk after every retune (the assumed real-radio behavior)
    def test_6db_bursts_reach_active_with_retune_junk(self):
        for seed, bw in ((101, 180e3), (102, 1.08e6), (103, 4.5e6)):
            self.check_detected(6, bw, (1, 5), seed, junk=8192)

    # acceptance: 50+ noise-only sweeps over every hearable band = 0 false ACTIVEs
    def test_noise_only_has_no_false_active(self):
        bands = hearable([b for b in config.BANDS if b["enabled"]])
        radio = self.radio(seed=7, junk_samples=8192)
        baseline, _ = self.baseline(radio, bands)
        result = run_monitor(radio, bands, baseline, 50)
        false_active = [(name, n) for name in result for n, s, a in result[name] if a]
        maybes = [(name, n) for name in result for n, s, a in result[name] if s == "maybe"]
        print("\n  noise only, 50 sweeps:", len(false_active), "ACTIVE,", len(maybes), "single 'maybe' hits")
        self.assertEqual(false_active, [])

    # acceptance: a band above 1766 MHz is reported as unhearable, not silently "ok"
    def test_band_above_tuner_limit_is_reported_unhearable(self):
        pcs = {"name": "PCS test", "start": 1850.0, "end": 1915.0, "carriers": "all", "enabled": True}
        bands = [band("B12/17 700"), pcs]
        radio = self.radio(seed=5, phones=[phone(20, low=1860.0, high=1870.0)])
        steps = rf_monitor.make_steps(bands)
        for s in steps:
            self.assertLessEqual(s["center"] + FS * config.USABLE_FRACTION / 2, R820T_MAX_HZ,
                                 "a window above the tuner limit was swept: " + str(s))
        baseline, _ = self.baseline(radio, bands)
        result = run_monitor(radio, bands, baseline, 3)
        for n, status, active in result["PCS test"]:
            self.assertIn("CAN'T HEAR", status)
            self.assertFalse(active)


# ---------------------------------------------------------------- the pipeline, piece by piece

class Measuring(unittest.TestCase):

    # averages must be taken in linear power, then turned into dB
    def test_average_is_linear_power(self):
        n = config.SAMPLES_PER_STEP
        rng = np.random.default_rng(3)
        x = rng.normal(0, 0.01, n) + 1j * rng.normal(0, 0.01, n)
        x[: n // 2] *= 10                                       # first half 20 dB louder
        avg, peak = rf_monitor.measure(x)
        quiet_avg, _ = rf_monitor.measure(np.concatenate([x[n // 2:], x[n // 2:]]))
        # linear average of +20 dB and +0 dB = 10*log10((100+1)/2) = +17.0 dB, NOT +10 dB
        self.assertAlmostEqual(avg - quiet_avg, 17.0, delta=0.3)

    # a 1 ms burst must not be averaged away: the peak shows (almost) its full strength
    def test_short_burst_shows_in_peak(self):
        n = config.SAMPLES_PER_STEP
        rng = np.random.default_rng(4)
        quiet = rng.normal(0, 0.01, n) + 1j * rng.normal(0, 0.01, n)
        loud = quiet.copy()
        length = int(1e-3 * FS)
        loud[100000:100000 + length] *= np.sqrt(10)             # +10 dB, wideband, 1 ms
        _, peak_quiet = rf_monitor.measure(quiet)
        _, peak_loud = rf_monitor.measure(loud)
        self.assertGreater(peak_loud - peak_quiet, 6.0)

    # the DC notch and USABLE_FRACTION keep the right bins, and windows don't leave gaps
    def test_bins_and_windows_line_up(self):
        keep = rf_monitor.bin_mask()
        freqs = np.fft.fftshift(np.fft.fftfreq(config.FFT_SIZE, 1.0 / FS))
        self.assertTrue(np.all(np.abs(freqs[keep]) >= config.DC_NOTCH_HZ))
        self.assertTrue(np.all(np.abs(freqs[keep]) <= FS * config.USABLE_FRACTION / 2))
        self.assertEqual(int(np.sum(keep[freqs < 0])), int(np.sum(keep[freqs > 0])))   # same on both sides
        bin_hz = FS / config.FFT_SIZE
        steps = rf_monitor.make_steps([band("B71 600MHz")])
        top_kept = freqs[keep].max()
        for a, b in zip(steps, steps[1:]):
            # distance between the last kept bin of one window and the first of the next.
            # Inside a window bins are 1 bin apart, so under 2 bins apart = less than one bin missing
            gap = (b["center"] - top_kept) - (a["center"] + top_kept)
            self.assertLess(gap, 2 * bin_hz)


class ConfirmWindow(unittest.TestCase):

    def status_after(self, hits):
        row = {"band": {"name": "B"}, "avg_delta": 0.0, "peak_delta": 0.0}
        history = {"B": [False] * config.CONFIRM_WINDOW}
        for h in hits:
            history["B"] = history["B"][1:] + [h]
        return rf_monitor.band_status(row, history, None)[1]

    # CONFIRM_HITS hits that are CONFIRM_WINDOW-1 sweeps apart still count, CONFIRM_WINDOW apart don't
    def test_no_off_by_one(self):
        w = config.CONFIRM_WINDOW
        self.assertTrue(self.status_after([True] + [False] * (w - 2) + [True]))
        self.assertFalse(self.status_after([True] + [False] * (w - 1) + [True]))
        self.assertTrue(self.status_after([True] * config.CONFIRM_HITS))
        self.assertFalse(self.status_after([True] * (config.CONFIRM_HITS - 1)))


class Retune(Synthetic):

    # junk is thrown away after EVERY retune, the same amount each time, and more than the junk
    def test_settles_after_every_retune(self):
        bands = [band("B12/17 700")]
        radio = self.radio(seed=2, junk_samples=8192)
        rf_monitor.sweep(radio, rf_monitor.make_steps(bands))
        radio.sdr.set_center_freq(1e9)                          # record the last window too
        discards = radio.sdr.discards
        self.assertEqual(len(set(discards)), 1, "different amounts thrown away: " + str(discards))
        self.assertGreaterEqual(discards[0], 8192)


class SavedFiles(Synthetic):

    # a baseline made at one GAIN must not be used at another (all the dB numbers would shift)
    def test_baseline_from_other_gain_is_refused(self):
        bands = [band("B12/17 700")]
        config.GAIN = 40
        radio = self.radio(seed=3)
        _, path = self.baseline(radio, bands)
        config.GAIN = 30
        self.assertIsNone(rf_monitor.load_baseline(path))

    # a fake-radio baseline must never be used for the real radio
    def test_sim_baseline_is_never_used_for_the_real_radio(self):
        script = os.path.join("rf_detect", "rf_monitor.py")
        result = subprocess.run([sys.executable, script, "baseline", "--simulate", "--no-web"],
                                cwd=REPO, capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr)
        copy = os.path.join(self.folder, "copied_baseline.json")
        shutil.copy(config.SIM_BASELINE_FILE, copy)
        self.assertIsNone(rf_monitor.load_baseline(copy))      # real mode (simulate not set)

    # files from the old version (no settings saved in them) are not trusted
    def test_old_files_are_not_trusted(self):
        old = os.path.join(self.folder, "old_fov.json")
        with open(old, "w") as f:
            json.dump({"B12/17 700": {"front": 3.0, "behind": 0.0, "threshold": 1.5}}, f)
        self.assertIsNone(rf_monitor.load_baseline(old))


# ---------------------------------------------------------------- the simulator, end to end

class SimulatedMonitor(unittest.TestCase):

    def test_monitor_simulate_20_sweeps(self):
        script = os.path.join("rf_detect", "rf_monitor.py")
        result = subprocess.run([sys.executable, script, "monitor", "--simulate", "--sweeps", "20", "--no-web"],
                                cwd=REPO, capture_output=True, text=True, timeout=600)
        self.assertEqual(result.returncode, 0, result.stderr)
        parts = result.stdout.split("==== CELL BANDS (RTL-SDR) sweep ")[1:]
        self.assertEqual(len(parts), 20)
        first_active = None
        for part in parts:
            number = int(part.split()[0])
            table = part.split("==== BLUETOOTH")[0]
            b12 = [line for line in table.splitlines() if line.startswith("B12/17 700")][0]
            if number < 4:
                self.assertNotIn("ACTIVE", table, "ACTIVE before the fake phone turned on")
            if "ACTIVE" in b12 and first_active is None:
                first_active = number
            self.assertIn("CAN'T HEAR", table)                 # the PCS row is listed, not hidden
        self.assertIsNotNone(first_active, "fake phone never detected")
        self.assertLess(first_active, 4 + config.CONFIRM_WINDOW)


if __name__ == "__main__":
    unittest.main()
