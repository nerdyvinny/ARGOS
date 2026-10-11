# Bluetooth headphone check. Uses the LAPTOP's own Bluetooth, not the RTL-SDR
# (Bluetooth is at 2.4 GHz, which the RTL-SDR can't reach).
#
# What it CAN see: headphones/earbuds that are switched on send out little
# "I'm here" messages (advertisements). AirPods send them while in use, and
# most earbuds with Google Fast Pair or their name in the message do too.
# What it CAN'T see: the actual audio link between a phone and its headphones.
# That is encrypted and hops channels, so no laptop can watch it.
# Signal strength (RSSI) gives a rough idea of how close the device is.
#
# Needs the "bleak" library:  py -3.12 -m pip install bleak
# (it does not install on Python 3.15 yet, so use Python 3.12)

import os
import csv
import time
import random
import asyncio
import threading

import rf_config as config


# ---------------- deciding "this looks like headphones" ----------------

HEADPHONE_WORDS = ["airpods", "beats", "buds", "headphone", "headset", "earbud", "earphone",
                   "wh-", "wf-", "jabra", "bose", "jbl", "soundcore", "skullcandy", "sennheiser"]

APPLE_ID = 0x004C            # Apple's company number in Bluetooth messages
APPLE_AIRPODS_TYPE = 0x07    # Apple message type 7 = AirPods / Beats status

# Bluetooth "service" numbers that mean audio. Written as the 4 hex digits
# from the middle of the long ID, e.g. 0000fe2c-0000-1000-... -> "fe2c"
AUDIO_SERVICES = {
    "fe2c": "Google Fast Pair (usually earbuds)",
    "184e": "LE Audio",
    "1850": "LE Audio",
    "1853": "LE Audio",
    "110b": "audio speaker/headphones",
    "111e": "hands-free headset",
}


def why_headphones(dev):
    # Gives back a short reason if this device looks like headphones, else None.
    # dev is a plain dictionary (see scan_real below).
    name = dev["name"].lower()
    for word in HEADPHONE_WORDS:
        if word in name:
            return "name has '" + word + "'"

    apple = dev["manufacturer_data"].get(APPLE_ID)
    if apple and apple[0] == APPLE_AIRPODS_TYPE:
        return "Apple AirPods/Beats signal"

    for uuid in list(dev["service_uuids"]) + list(dev["service_data"].keys()):
        short = uuid.lower()[4:8]
        if short in AUDIO_SERVICES:
            return AUDIO_SERVICES[short]
    return None


def how_close(rssi):
    # very rough! walls, bodies and bags all change this a lot
    if rssi >= config.BT_NEAR_RSSI:
        return "very close (1-2 m)"
    if rssi >= config.BT_MEDIUM_RSSI:
        return "in the room"
    return "far / weak"


def explain_problem(error):
    # turn a Python error into something a person can act on
    text = str(error)
    if isinstance(error, ImportError):
        return "Bluetooth library missing. Run:  py -3.12 -m pip install bleak   (then use py -3.12)"
    if "powered on" in text or "not available" in text.lower() or "radio" in text.lower():
        return "Bluetooth is OFF. Turn it on: Windows Settings > Bluetooth & devices."
    return "Bluetooth scan failed: " + text


# ---------------- scanning (real or fake) ----------------

async def scan_async(seconds):
    from bleak import BleakScanner     # imported here so --simulate works without it
    found = await BleakScanner.discover(timeout=seconds, return_adv=True)
    devices = []
    for address, (device, adv) in found.items():
        devices.append({
            "address": address,
            "name": adv.local_name or device.name or "",
            "rssi": adv.rssi,
            "manufacturer_data": dict(adv.manufacturer_data),
            "service_uuids": list(adv.service_uuids),
            "service_data": dict(adv.service_data),
        })
    return devices


def scan_real(seconds):
    return asyncio.run(scan_async(seconds))


class FakeBluetooth:
    # Pretend Bluetooth for testing: a mouse, a phone, AirPods close by,
    # and Galaxy Buds that show up a bit later, further away.
    def __init__(self):
        self.rng = random.Random(1)
        self.scans = 0

    def device(self, address, name, rssi, manufacturer_data=None, service_uuids=None):
        return {"address": address, "name": name, "rssi": rssi + self.rng.randint(-3, 3),
                "manufacturer_data": manufacturer_data or {}, "service_uuids": service_uuids or [],
                "service_data": {}}

    def scan(self):
        time.sleep(1)
        self.scans = self.scans + 1
        devices = [
            self.device("AA:00:00:00:00:01", "Logitech Mouse", -70),
            self.device("AA:00:00:00:00:02", "", -65, {APPLE_ID: bytes([0x10, 0x05, 0x01])}),   # an iPhone, not headphones
            self.device("AA:00:00:00:00:03", "", -55, {APPLE_ID: bytes([0x07, 0x19, 0x01, 0x0E, 0x20])}),
        ]
        if self.scans >= 3:
            devices.append(self.device("AA:00:00:00:00:04", "Galaxy Buds2 Pro", -78))
        devices.append(self.device("AA:00:00:00:00:05", "Sony WF-1000XM5", -95))   # next room: too weak, ignored
        return devices


# ---------------- keeping track over time ----------------

class BluetoothWatcher:
    def __init__(self, simulate):
        self.simulate = simulate
        self.fake = FakeBluetooth() if simulate else None
        self.headphones = {}       # address -> info about that device
        self.total = 0             # how many Bluetooth devices of any kind were in the last scan
        self.too_far = 0           # headphones ignored in the last scan for being too weak (other room?)
        self.problem = None        # text to show if Bluetooth isn't working
        self.lock = threading.Lock()

    def scan(self):
        # One scan. Gives back the headphones that are NEW this time.
        if self.simulate:
            devices = self.fake.scan()
        else:
            devices = scan_real(config.BT_SCAN_SECONDS)

        now = time.time()
        new = []
        with self.lock:
            self.total = len(devices)
            self.too_far = 0
            for dev in devices:
                if dev["name"] and dev["name"] in config.BT_IGNORE_NAMES:
                    continue
                why = why_headphones(dev)
                if why is None:
                    continue
                if dev["rssi"] < config.BT_MIN_RSSI:
                    self.too_far = self.too_far + 1     # too weak: probably another room
                    continue
                name = dev["name"]
                if name == "" and why == "Apple AirPods/Beats signal":
                    name = "AirPods / Beats"        # they don't send a name, but we know the type
                if name == "":
                    name = "(no name)"
                old = self.headphones.get(dev["address"])
                info = {"address": dev["address"], "name": name,
                        "rssi": dev["rssi"], "why": why, "distance": how_close(dev["rssi"]),
                        "first_seen": now if old is None else old["first_seen"], "last_seen": now}
                self.headphones[dev["address"]] = info
                if old is None:
                    new.append(info)

            # forget devices we haven't heard from for a while
            for address in list(self.headphones):
                if now - self.headphones[address]["last_seen"] > config.BT_FORGET_SECONDS:
                    del self.headphones[address]

        if not self.simulate:              # keep fake data out of the real log
            for info in new:
                write_log(info)
        return new

    def current(self):
        # headphones seen recently, strongest (closest) first
        with self.lock:
            found = list(self.headphones.values())
        return sorted(found, key=lambda d: d["rssi"], reverse=True)

    def start_background(self):
        # keep scanning in the background while the RF monitor runs
        thread = threading.Thread(target=self.run_forever, daemon=True)
        thread.start()

    def run_forever(self):
        while True:
            try:
                self.scan()
                self.problem = None
            except ImportError as e:
                self.problem = explain_problem(e)
                return                      # no point trying again
            except Exception as e:
                self.problem = explain_problem(e)
                time.sleep(5)


def write_log(info):
    new_file = not os.path.exists(config.BT_LOG_FILE)
    with open(config.BT_LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["time", "address", "name", "rssi", "distance", "why"])
        line = [time.strftime("%Y-%m-%d %H:%M:%S"), info["address"], info["name"],
                info["rssi"], info["distance"], info["why"]]
        w.writerow(line)
        print("BT LOG:", ",".join(str(x) for x in line))


def print_table(watcher):
    found = watcher.current()
    print("")
    print("==== BLUETOOTH (laptop) at", time.strftime("%H:%M:%S"), "====", watcher.total, "devices nearby,", len(found), "look like headphones")
    if watcher.too_far:
        print("   (ignored %d weaker than %d dBm - probably another room, see BT_MIN_RSSI)" % (watcher.too_far, config.BT_MIN_RSSI))
    if watcher.problem:
        print("  ", watcher.problem)
    for d in found:
        print("  %-22s %4d dBm  %-20s %s" % (d["name"][:22], d["rssi"], d["distance"], d["why"]))
