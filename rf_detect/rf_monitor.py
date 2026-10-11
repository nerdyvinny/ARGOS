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
    # phone_strength is how loud (4 = about +12 dB). fovtest changes it to
    # pretend the phone moved further away.
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
    if row["avg_delta"] is None:
        return "(no baseline)", False
    if sum(history[name]) >= config.CONFIRM_HITS:
        if config.CAMERA_CONFIRM and camera is False:
            return "RF only, camera sees nothing (ignored)", False
        if camera is True:
            return "ACTIVE  <-- transmitting (camera agrees)", True
        return "ACTIVE  <-- transmitting", True
    if history[name][-1]:
        return "maybe", False
    return "ok", False


def print_table(rows, history, sweep_number, camera):
    print("")
    print("==== CELL BANDS (RTL-SDR) sweep", sweep_number, "at", time.strftime("%H:%M:%S"), "====")
    print("%-14s %-16s %9s %9s %9s  %s" % ("band", "carriers", "level dB", "avg +dB", "peak +dB", "status"))
    for r in rows:
        name = r["band"]["name"]
        status, active = band_status(r, history, camera)
        avg_text = "" if r["avg_delta"] is None else "%+.1f" % r["avg_delta"]
        peak_text = "" if r["peak_delta"] is None else "%+.1f" % r["peak_delta"]
        print("%-14s %-16s %9.1f %9s %9s  %s" % (name, r["band"]["carriers"][:16], r["level"], avg_text, peak_text, status))


def write_log(rows, history, camera):
    new_file = not os.path.exists(config.LOG_FILE)
    with open(config.LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["time", "band", "level_db", "avg_delta_db", "peak_delta_db", "active"])
        for r in rows:
            name = r["band"]["name"]
            status, active = band_status(r, history, camera)
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), name, round(r["level"], 2),
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
    #      camera_status.json = {"time": 1700000000, "phone_desks": ["Desk 4"]}
    #      and here we read it:
    #          data = json.load(open("camera_status.json"))
    #          if time.time() - data["time"] > 5: return None     # camera stopped
    #          return len(data["phone_desks"]) > 0
    #   b) run the RF monitor inside app/main.py and ask it directly, e.g.
    #          return len(latest_phone_detections) > 0             # from vision/detector.py
    #
    # Also TODO: when the camera says yes, tell rules/brain.py so that desk's
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
        json.dump(levels, f, indent=2)
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
            make_baseline(radio, steps, baseline_file(simulate))
            return

        baseline = None
        if mode in ("monitor", "fovtest"):
            baseline = load_baseline(baseline_file(simulate))
            if baseline is None and simulate:
                print("No fake baseline yet, making one first")
                baseline = make_baseline(radio, steps, baseline_file(simulate))
            if baseline is None:
                print("No", config.BASELINE_FILE, "yet. Run: py rf_monitor.py baseline  (in the EMPTY room)")
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
            fov = load_baseline(fov_file(simulate))     # same kind of .json file, so we reuse the loader
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
