# Reads what the RF monitor (rf_detect/rf_monitor.py) hears, so the camera dashboard
# can cross-check it. The RF monitor is its own program (it needs the RTL-SDR and Bluetooth);
# every sweep it writes rf_detect/rf_status.json and we read that file here.
#
# A "close signal" is:
#   - a phone band that the RF monitor calls ACTIVE (it already only counts phones
#     in front of the camera, if you ran its fovtest), or
#   - Bluetooth headphones/earbuds stronger than NEAR_RSSI (about 1-2 m away).

import os
import json
import time

FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RF_STATUS_FILE = os.path.join(FOLDER, "rf_detect", "rf_status.json")
CAMERA_STATUS_FILE = os.path.join(FOLDER, "rf_detect", "camera_status.json")

NEAR_RSSI = -60          # Bluetooth stronger than this counts as close (same as BT_NEAR_RSSI in rf_config.py)
STALE_SECONDS = 15       # RF file older than this = the RF monitor isn't running

active_before = set()    # bands that were already active last time (so we only alert when one starts)
near_before = set()      # Bluetooth addresses that were already close last time


def read_status():
    # Gives back the RF monitor's latest status, or None if it isn't running.
    try:
        with open(RF_STATUS_FILE) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if time.time() - data.get("time", 0) > STALE_SECONDS:
        return None
    return data


def summary(status):
    # Short version for the web page header.
    if status is None:
        return {"connected": False, "radio": False, "active_bands": [], "near": []}
    near = [d["name"] + " (" + str(d["rssi"]) + " dBm)" for d in status.get("bluetooth", []) if d["rssi"] >= NEAR_RSSI]
    return {"connected": True, "radio": status.get("radio", False),
            "active_bands": status.get("active_bands", []), "near": near}


def new_close_signals(status):
    # Gives back the close signals that STARTED since the last check, as a list of
    # {"type": "phone" or "bluetooth", "text": "..."}.
    global active_before, near_before
    if status is None:
        active_before = set()
        near_before = set()
        return []
    events = []
    active_now = set(status.get("active_bands", []))
    for band in sorted(active_now - active_before):
        events.append({"type": "phone", "text": "Phone signal in band " + band})
    near_now = {}
    for d in status.get("bluetooth", []):
        if d["rssi"] >= NEAR_RSSI:
            near_now[d["address"]] = d
    for address in sorted(set(near_now) - near_before):
        d = near_now[address]
        events.append({"type": "bluetooth", "text": "Bluetooth " + d["name"] + " close by (" + str(d["rssi"]) + " dBm)"})
    active_before = active_now
    # Bluetooth strength jumps a few dBm between scans. Keep remembering a device until it is
    # gone from the list completely, so one pair of earbuds near the line doesn't alert again and again.
    still_heard = set(d["address"] for d in status.get("bluetooth", []))
    near_before = (near_before & still_heard) | set(near_now)
    return events


def write_camera_status(phone_people):
    # Tell the RF monitor who the camera saw with a device (its camera_confirms() reads this).
    data = {"time": time.time(), "phone_people": phone_people}
    temp = CAMERA_STATUS_FILE + ".tmp"
    try:
        with open(temp, "w") as f:
            json.dump(data, f)
        os.replace(temp, CAMERA_STATUS_FILE)
    except OSError as e:
        print("[RF] could not write", CAMERA_STATUS_FILE, e)
