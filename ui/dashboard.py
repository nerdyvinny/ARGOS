# Tiny web page for the proctor.  Open http://localhost:5000
# It shows the live video, every person's color, and the list of alerts.
# Each alert has a short video clip to play back. Device alerts (phone, watch...)
# also get Confirm / Dismiss buttons, because a person has to decide whose device it is.
#
# It also has a Radio panel: buttons that run the radio detector (rf_detect/rf_monitor.py)
# in the background, step by step: 1. Scan  2. Baseline  3. Monitor.
# While the radio monitor runs, a phone band that turns active or Bluetooth headphones close by
# make an "RF" alert that says whether the camera saw a device at the same time,
# plus a text notification on the side.

import os
import sys
import csv
import json
import atexit
import time
import threading
import subprocess
from flask import Flask, Response, jsonify, request, send_from_directory
from settings import config
from app import clips

app = Flask(__name__)
lock = threading.Lock()

latest_jpeg = None
alerts_list = []
people_info = []
next_id = [1]
REVIEW_CHOICES = ("confirmed", "dismissed")
REVIEWED_KINDS = ("device", "rf")   # only alerts about devices get Confirm / Dismiss
MATCH_SECONDS = 15                  # RF and camera count as "the same moment" if this close in time
rf_times = []                       # (time, text) of recent close RF signals
CLOSE_RSSI = -60                    # Bluetooth this strong (dBm) = headphones close by
RF_FRESH_SECONDS = 15               # rf_status.json older than this = radio monitor not running

# ---------- radio (RF) part ----------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the repo's top folder
RF_FOLDER = os.path.join(ROOT, "rf_detect")
RF_SCRIPT = os.path.join(RF_FOLDER, "rf_monitor.py")
RF_STATUS_FILE = os.path.join(RF_FOLDER, "rf_status.json")         # rf_monitor.py writes it every sweep
CAMERA_STATUS_FILE = os.path.join(ROOT, "camera_status.json")      # we write it, rf_monitor.py reads it
RF_PYTHON = ["py", "-3.12"]     # the radio part needs Python 3.12 (see README)
RF_STEPS = ["scan", "baseline", "monitor"]

rf_process = None       # the running radio program (or None)
rf_mode = ""            # which step it is running
rf_stopped = False      # True when the Stop button ended it
rf_output = []          # the last lines it printed
rf_steps_done = {}      # step -> "done" / "problem" / "stopped"
rf_status = {}          # the newest rf_status.json (only while monitor runs)
active_bands_before = set()
headphones_seen = set()
close_headphones = set()   # headphones already alerted as close (until they leave the list)

notifications = []      # text only, newest last
last_camera_write = 0


def set_frame(jpeg_bytes):
    global latest_jpeg
    with lock:
        latest_jpeg = jpeg_bytes


def add_alerts(new_alerts):
    with lock:
        for a in new_alerts:
            item = dict(a)   # a copy, so we don't change the caller's alert
            item["id"] = next_id[0]
            next_id[0] = next_id[0] + 1
            item["t"] = time.time()
            item["status"] = "pending" if item.get("kind") in REVIEWED_KINDS else None
            if item.get("kind") == "device" and "rf" not in item:
                item["rf"] = recent_rf_text(item["t"])
            alerts_list.append(item)
        # only keep the last 200
        while len(alerts_list) > 200:
            alerts_list.pop(0)
    for a in new_alerts:
        if a.get("kind") == "device":      # phone (or watch) seen by the camera
            notify("camera", "Camera: " + a["message"], "exam time " + str(a.get("exam_time", "")))


def set_people(people_states):
    global people_info
    info = []
    for d in people_states:
        info.append({
            "name": d["name"],
            "status": d["status"],
            "alert_count": d["alert_count"],
            "looks": d["total_looks"],
            "long_looks": d["total_long_looks"],
            "phones": d["total_phone_alerts"],
        })
    with lock:
        people_info = info


def recent_rf_text(when):
    # the RF signal that happened close to this time, or "" (call with the lock held)
    for t, text in reversed(rf_times):
        if abs(when - t) <= MATCH_SECONDS:
            return text
    return ""


def make_rf_alert(event):
    # A close RF signal started: look at what the camera saw around the same moment.
    now = time.time()
    with lock:
        rf_times.append((now, event["text"]))
        del rf_times[:-50]
        recent = [a for a in alerts_list if a.get("kind") == "device" and now - a["t"] <= MATCH_SECONDS]
        for a in recent:
            if not a.get("rf"):
                a["rf"] = event["text"]          # the camera's device alert now has RF backing it up
        frame = latest_jpeg
    people = sorted(set(a["person"] for a in recent))
    if people:
        person = people[0] if len(people) == 1 else "Several"
        message = event["text"] + ". Camera saw a device at " + ", ".join(people)
    else:
        person = "Room"
        message = event["text"] + ". Camera sees no device, check the room"
    alert = {"person": person, "kind": "rf", "message": message, "rf": event["text"],
             "clock": time.strftime("%H:%M:%S"), "exam_time": "", "image": ""}
    if frame is not None:
        alert["image"] = "RF_" + str(int(now * 1000)) + ".jpg"
        with open(os.path.join(config.EVIDENCE_FOLDER, alert["image"]), "wb") as f:
            f.write(frame)
    alert["clip"] = clips.start_clip(alert)
    add_alerts([alert])
    print("[RF]", message)


def clip_ready(file_name):
    return bool(file_name) and os.path.exists(os.path.join(config.EVIDENCE_FOLDER, file_name))


def save_review(alert):
    path = os.path.join(config.EVIDENCE_FOLDER, "reviews.csv")
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["reviewed_at", "exam_time", "person", "message", "decision", "image", "clip"])
        writer.writerow([time.strftime("%H:%M:%S"), alert.get("exam_time", ""), alert.get("person", ""),
                         alert.get("message", ""), alert["status"], alert.get("image", ""), alert.get("clip", "")])


def set_phones_seen(count):
    # Called by app/main.py for every picture. About once a second we save how many
    # phones the camera sees, so the radio monitor can check it (camera_confirms).
    global last_camera_write
    now = time.time()
    if now - last_camera_write < 1:
        return
    last_camera_write = now
    temp = CAMERA_STATUS_FILE + ".tmp"
    try:
        with open(temp, "w") as f:
            json.dump({"time": now, "phones_seen": count}, f)
        os.replace(temp, CAMERA_STATUS_FILE)
    except OSError:
        pass     # not a big deal, we try again in a second


def notify(kind, title, detail):
    # kind = "camera", "radio" or "bluetooth". Text only, never pictures.
    with lock:
        notifications.append({"kind": kind, "clock": time.strftime("%H:%M:%S"),
                              "title": title, "detail": detail})
        while len(notifications) > 100:
            notifications.pop(0)


# ---------- running the radio program ----------

def baseline_exists(practice):
    if practice:
        return os.path.exists(os.path.join(RF_FOLDER, "rf_baseline_sim.json"))
    return os.path.exists(os.path.join(RF_FOLDER, "rf_baseline.json"))


def rf_running():
    return rf_process is not None and rf_process.poll() is None


def start_rf(mode, practice):
    # Starts one radio step in the background. Gives back an error text, or "" if it started.
    global rf_process, rf_mode, rf_stopped, rf_status, active_bands_before, headphones_seen
    if mode not in RF_STEPS:
        return "Unknown step: " + mode
    if rf_running():
        return "The radio is already running (" + rf_mode + "). Press Stop first."
    if mode == "monitor" and not practice and not baseline_exists(practice):
        return "Do step 2 (Baseline) first, in the empty room with phones off."

    python = [sys.executable] if practice else RF_PYTHON   # the fake radio works on this Python too
    command = python + ["-u", RF_SCRIPT, mode, "--no-web"]
    if practice:
        command.append("--simulate")
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    if mode == "monitor":
        try:
            os.remove(RF_STATUS_FILE)     # don't show old results from last time
        except OSError:
            pass
    try:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace",
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except OSError as e:
        return "Could not start Python 3.12 (" + str(e) + "). Is it installed? Try: py -3.12 --version"

    with lock:
        rf_process = process
        rf_mode = mode
        rf_stopped = False
        rf_status = {}
        rf_output.clear()
        rf_output.append("> " + " ".join(python) + " rf_detect\\rf_monitor.py " + " ".join(command[len(python) + 2:]))
        rf_steps_done.pop(mode, None)
        if mode == "monitor":
            active_bands_before = set()
            headphones_seen = set()
    threading.Thread(target=read_rf_output, args=(process, mode), daemon=True).start()
    return ""


def read_rf_output(process, mode):
    # Collects what the radio program prints, and notes how the step ended.
    worked = False
    for line in process.stdout:
        line = line.rstrip()
        if mode == "scan" and line.startswith("==== CELL BANDS"):
            worked = True       # one sweep finished
        if mode == "baseline" and line.startswith("Saved"):
            worked = True       # the baseline file was written
        with lock:
            rf_output.append(line)
            while len(rf_output) > 80:
                rf_output.pop(0)
    process.wait()
    with lock:
        if rf_stopped:
            rf_steps_done[mode] = "stopped"
        elif worked or mode == "monitor":
            rf_steps_done[mode] = "done"
        else:
            rf_steps_done[mode] = "problem"


def stop_rf():
    global rf_stopped
    if rf_running():
        with lock:
            rf_stopped = True
        rf_process.terminate()


def watch_rf_status():
    # Read rf_status.json every second while it is fresh (the radio monitor writes it every sweep,
    # whether it was started from the panel or by hand). A phone band that just became ACTIVE,
    # or Bluetooth headphones close by, give a notification AND an RF alert that says whether
    # the camera saw a device at the same time.
    while True:
        time.sleep(1)
        try:
            if time.time() - os.path.getmtime(RF_STATUS_FILE) > RF_FRESH_SECONDS:
                continue     # old file: the radio monitor is not running
            with open(RF_STATUS_FILE, encoding="utf-8") as f:
                status = json.load(f)
        except (OSError, ValueError):
            continue     # no sweep finished yet
        try:
            check_rf_status(status)
        except Exception as e:
            print("[RF] problem with the radio status:", e)


def check_rf_status(status):
    global rf_status, active_bands_before
    active_now = set()
    for b in status.get("active_bands", []):
        active_now.add(b["band"])
        if b["band"] not in active_bands_before:
            detail = "carriers: " + str(b.get("carriers", ""))
            if status.get("camera") is True:
                detail = detail + " (camera also sees a phone)"
            notify("radio", "Phone signal ACTIVE: " + b["band"], detail)
            make_rf_alert({"type": "phone", "text": "Phone signal in band " + b["band"]})
    here_now = set()
    for h in status.get("headphones", []):
        here_now.add(h["address"])
        if h["address"] not in headphones_seen:
            headphones_seen.add(h["address"])
            notify("bluetooth", "Bluetooth headphones: " + h["name"],
                   str(h.get("distance", "")) + " (" + str(h["rssi"]) + " dBm)")
        if h["rssi"] >= CLOSE_RSSI and h["address"] not in close_headphones:
            close_headphones.add(h["address"])
            make_rf_alert({"type": "bluetooth",
                           "text": "Bluetooth " + h["name"] + " close by (" + str(h["rssi"]) + " dBm)"})
    close_headphones.intersection_update(here_now)   # gone from the list = may alert again later
    with lock:
        active_bands_before = active_now
        rf_status = status


def rf_info():
    with lock:
        return {
            "running": rf_running(),
            "mode": rf_mode,
            "steps_done": dict(rf_steps_done),
            "output": list(rf_output[-40:]),
            "baseline_real": baseline_exists(False),
            "baseline_practice": baseline_exists(True),
            "sweep": rf_status.get("sweep"),
            "active_bands": [b["band"] for b in rf_status.get("active_bands", [])],
            "headphones": rf_status.get("headphones", []),
            "bt_problem": rf_status.get("bt_problem"),
            "camera": rf_status.get("camera"),
        }


PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Exam Watch</title>
<style>
  :root {
    --bg:#0c0c0e; --panel:#151518; --panel2:#1c1c20; --line:#2a2a30;
    --text:#ececef; --muted:#9a9aa3; --accent:#e5484d; --accent-soft:#3b1517;
    --radius:8px;
    --mono: ui-monospace, "Cascadia Mono", Consolas, monospace;
  }
  * { box-sizing:border-box; }
  body { background:var(--bg); color:var(--text); margin:0; padding:16px;
         font-family:"Segoe UI Variable", "Segoe UI", system-ui, sans-serif; }
  header { display:flex; align-items:baseline; gap:16px; margin-bottom:14px; }
  h1 { margin:0; font-size:20px; font-weight:600; letter-spacing:-0.01em; }
  #pending { font-size:14px; color:var(--muted); }
  #pending.has { color:var(--accent); font-weight:600; }
  #rf { font-size:13px; color:var(--muted); margin-left:auto; text-align:right; }
  #rf.hot { color:var(--accent); font-weight:600; }
  .badge { display:inline-block; font-size:11px; font-weight:600; margin-top:6px; padding:2px 8px;
           border-radius:4px; border:1px solid var(--accent); color:var(--accent); }
  .layout { display:grid; grid-template-columns: minmax(0,3fr) minmax(300px,1.15fr); gap:16px; }
  @media (max-width: 900px) { .layout { grid-template-columns: 1fr; } }
  #video { width:100%; display:block; border:1px solid var(--line); border-radius:var(--radius); background:#000; }
  .people { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
  .person { padding:8px 12px; border-radius:var(--radius); font-size:13px; min-width:96px; cursor:pointer;
            border:1px solid transparent; background:var(--panel2); }
  .person b { display:block; font-size:14px; }
  .person.green { border-color:#1f5f3a; } .person.yellow { border-color:#7a5a12; }
  .person.red { border-color:var(--accent); background:var(--accent-soft); }
  .person.empty { color:var(--muted); }
  .person.selected { outline:2px solid var(--text); }
  aside { background:var(--panel); border:1px solid var(--line); border-radius:var(--radius);
          display:flex; flex-direction:column; max-height:calc(100dvh - 70px); min-height:300px; }
  .filters { display:flex; flex-wrap:wrap; gap:6px; padding:12px; border-bottom:1px solid var(--line); }
  .chip { font:inherit; font-size:12px; padding:5px 10px; border-radius:var(--radius); cursor:pointer;
          background:transparent; color:var(--muted); border:1px solid var(--line); }
  .chip.on { color:var(--text); border-color:var(--text); }
  #alerts { overflow-y:auto; padding:12px; display:flex; flex-direction:column; gap:10px; }
  .empty-state { color:var(--muted); font-size:14px; padding:24px 8px; text-align:center; }
  .card { background:var(--panel2); border:1px solid var(--line); border-radius:var(--radius);
          padding:10px; display:grid; grid-template-columns:96px 1fr; gap:10px; }
  .card.device { border-left:3px solid var(--accent); }
  .card.reviewed { opacity:0.5; }
  .thumb { position:relative; width:96px; height:72px; border-radius:var(--radius); overflow:hidden;
           background:#000; cursor:pointer; border:0; padding:0; }
  .thumb img { width:100%; height:100%; object-fit:cover; display:block; }
  .play { position:absolute; right:6px; bottom:6px; font-size:11px; background:rgba(0,0,0,.7);
          color:#fff; padding:2px 6px; border-radius:4px; }
  .msg { font-size:14px; font-weight:600; line-height:1.3; }
  .meta { font-family:var(--mono); font-size:12px; color:var(--muted); margin-top:3px; }
  .actions { display:flex; gap:6px; margin-top:8px; }
  .btn { font:inherit; font-size:13px; padding:6px 12px; border-radius:var(--radius); cursor:pointer; border:1px solid var(--line); }
  .btn:active { transform:translateY(1px); }
  .btn.confirm { background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600; }
  .btn.dismiss { background:transparent; color:var(--text); }
  .verdict { font-size:12px; margin-top:8px; font-weight:600; }
  .verdict.confirmed { color:var(--accent); } .verdict.dismissed { color:var(--muted); }
  #modal { position:fixed; inset:0; background:rgba(5,5,7,.85); display:none; align-items:center;
           justify-content:center; padding:16px; z-index:10; }
  #modal .box { background:var(--panel); border:1px solid var(--line); border-radius:var(--radius);
                padding:12px; width:min(960px,100%); }
  #modal video, #modal img { width:100%; border-radius:var(--radius); background:#000; display:block; }
  #modal .bar { display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; gap:12px; }
  #modal .note { color:var(--muted); font-size:13px; margin-top:8px; }
  .radio { background:var(--panel); border:1px solid var(--line); border-radius:var(--radius);
           padding:12px; margin-top:12px; font-size:14px; }
  .radio .step { display:flex; align-items:center; gap:8px; margin:6px 0; }
  .radio button { font:inherit; font-size:13px; width:110px; padding:6px 10px; border-radius:var(--radius);
                  cursor:pointer; background:var(--panel2); color:var(--text); border:1px solid var(--line); }
  .radio button:disabled { color:var(--muted); cursor:default; }
  .radio button.stop { border-color:var(--accent); color:var(--accent); }
  .hint { color:var(--muted); font-size:12px; }
  .ok { color:#5c5; } .bad { color:var(--accent); }
  #rferror { color:var(--accent); font-size:13px; margin-top:6px; }
  .radio pre { background:#000; color:#ccc; font-size:11px; padding:8px; max-height:220px; overflow:auto; white-space:pre-wrap; }
  .notes { margin-top:12px; }
  .nt { background:var(--panel2); padding:8px; margin-bottom:8px; border-radius:var(--radius); font-size:13px;
        border-left:3px solid var(--muted); }
  .nt.camera { border-left-color:#f90; } .nt.radio { border-left-color:var(--accent); } .nt.bluetooth { border-left-color:#39f; }
  .nt .time { color:var(--muted); font-size:12px; }
</style>
</head>
<body>
<header>
  <h1>Exam Watch</h1>
  <span id="pending">No devices to review</span>
  <span id="rf">RF monitor not running</span>
</header>
<div class="layout">
  <main>
    <img id="video" src="/video" alt="Live classroom camera">
    <div class="people" id="people"></div>
    <div class="radio">
      <b>Radio detector</b> <span class="hint">(phones + Bluetooth headphones)</span>
      <div class="step"><label><input type="checkbox" id="practice"> Practice mode (fake radio, no hardware)</label></div>
      <div class="step"><button id="btn-scan" onclick="startStep('scan')">1. Scan</button>
        <span>Tests the radio works (about 15 s) <span id="done-scan"></span></span></div>
      <div class="step"><button id="btn-baseline" onclick="startStep('baseline')">2. Baseline</button>
        <span>Empty room, phones OFF (about 1-2 min) <span id="done-baseline"></span></span></div>
      <div class="step"><button id="btn-monitor" onclick="startStep('monitor')">3. Monitor</button>
        <span>Watch for phones + Bluetooth <span id="done-monitor"></span></span></div>
      <div class="step"><button id="btn-stop" class="stop" onclick="stopRadio()">Stop</button>
        <span id="rfstate">Radio is off</span></div>
      <div id="rferror"></div>
      <div id="rflive" class="hint"></div>
      <details><summary class="hint">Radio output</summary><pre id="rfout"></pre></details>
    </div>
    <div class="notes"><b>Notifications</b><div id="notes" class="hint">Nothing yet</div></div>
  </main>
  <aside>
    <div class="filters" id="filters"></div>
    <div id="alerts"><div class="empty-state" id="empty">No alerts yet. They appear here as they happen.</div></div>
  </aside>
</div>

<div id="modal" role="dialog" aria-modal="true">
  <div class="box">
    <div class="bar"><b id="modal-title"></b><button class="btn dismiss" id="modal-close">Close</button></div>
    <div id="modal-body"></div>
    <div class="note" id="modal-note"></div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
<script>
const KIND_NAMES = {device: "Device", rf: "RF signal", looking: "Looking", long_look: "Long look"};
const cards = new Map();     // alert id -> {el, alert}
let lastId = 0;
let kindFilter = "all";
let personFilter = null;
let firstLoad = true;
let lastPeople = [];

// motion only when the user hasn't asked for less of it (and the page still works without GSAP)
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const hasGsap = typeof gsap !== "undefined" && !reduceMotion;

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function beep() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const o = ctx.createOscillator(), g = ctx.createGain();
    o.frequency.value = 880; g.gain.value = 0.08;
    o.connect(g); g.connect(ctx.destination);
    o.start(); o.stop(ctx.currentTime + 0.18);
  } catch (e) {}
}

function buildFilters() {
  const box = document.getElementById("filters");
  box.textContent = "";
  for (const k of ["all", "device", "rf", "looking", "long_look"]) {
    const c = el("button", "chip" + (kindFilter === k ? " on" : ""), k === "all" ? "All" : KIND_NAMES[k]);
    c.onclick = () => { kindFilter = k; buildFilters(); applyFilters(); };
    box.appendChild(c);
  }
  if (personFilter) {
    const c = el("button", "chip on", personFilter + "  x");
    c.onclick = () => { personFilter = null; buildFilters(); applyFilters(); renderPeople(lastPeople); };
    box.appendChild(c);
  }
}

function visible(a) {
  if (kindFilter !== "all" && a.kind !== kindFilter) return false;
  if (personFilter && a.person !== personFilter) return false;
  return true;
}

function applyFilters() {
  let shown = 0;
  for (const {el: c, alert} of cards.values()) {
    const v = visible(alert);
    c.style.display = v ? "" : "none";
    if (v) shown++;
  }
  const empty = document.getElementById("empty");
  empty.style.display = shown ? "none" : "";
  empty.textContent = cards.size ? "No alerts match this filter." : "No alerts yet. They appear here as they happen.";
}

function updatePending() {
  let n = 0;
  for (const {alert} of cards.values()) if (alert.status === "pending") n++;
  const p = document.getElementById("pending");
  p.textContent = n ? n + (n === 1 ? " device to review" : " devices to review") : "No devices to review";
  p.className = n ? "has" : "";
}

function setVerdict(card, alert) {
  const actions = card.querySelector(".actions");
  if (actions) actions.remove();
  let v = card.querySelector(".verdict");
  if (!v) { v = el("div", "verdict"); card.querySelector(".info").appendChild(v); }
  v.className = "verdict " + alert.status;
  v.textContent = alert.status === "confirmed" ? "Confirmed by proctor" : "Dismissed";
  card.classList.add("reviewed");
}

function review(alert, card, decision) {
  fetch("/review/" + alert.id, {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({status: decision})
  }).then(r => r.json()).then(res => {
    if (!res.ok) return;
    alert.status = decision;
    setVerdict(card, alert);
    updatePending();
    if (hasGsap) gsap.fromTo(card, {scale: 0.98}, {scale: 1, duration: 0.25, ease: "power2.out"});
  }).catch(e => {});
}

function makeCard(a) {
  const card = el("div", "card" + (a.kind === "device" || a.kind === "rf" ? " device" : ""));
  const thumb = el("button", "thumb");
  thumb.setAttribute("aria-label", "Play clip");
  if (a.image) { const img = el("img"); img.src = "/evidence/" + encodeURIComponent(a.image); img.alt = ""; thumb.appendChild(img); }
  thumb.appendChild(el("span", "play", "Play"));
  thumb.onclick = () => openClip(a);
  const info = el("div", "info");
  info.appendChild(el("div", "msg", a.message));
  info.appendChild(el("div", "meta", a.clock + (a.exam_time ? "   exam " + a.exam_time : "")));
  if (a.kind === "device" && a.rf) info.appendChild(el("div", "badge", "RF match: " + a.rf));
  if (a.status === "pending") {
    const actions = el("div", "actions");
    const yes = el("button", "btn confirm", "Confirm");
    const no = el("button", "btn dismiss", "Dismiss");
    yes.onclick = () => review(a, card, "confirmed");
    no.onclick = () => review(a, card, "dismissed");
    actions.appendChild(yes); actions.appendChild(no);
    info.appendChild(actions);
  }
  card.appendChild(thumb); card.appendChild(info);
  if (a.status === "confirmed" || a.status === "dismissed") setVerdict(card, a);
  return card;
}

function openClip(a) {
  const modal = document.getElementById("modal");
  const body = document.getElementById("modal-body");
  const note = document.getElementById("modal-note");
  body.textContent = "";
  note.textContent = "";
  document.getElementById("modal-title").textContent = a.message;
  // ask the server whether the clip is saved yet (it records a few seconds after the alert)
  fetch("/clip_ready/" + a.id).then(r => r.json()).then(res => {
    if (res.ready) {
      const v = el("video");
      v.src = "/evidence/" + encodeURIComponent(a.clip);
      v.controls = true; v.autoplay = true; v.muted = true; v.playsInline = true;
      if (a.image) v.poster = "/evidence/" + encodeURIComponent(a.image);
      body.appendChild(v);
      note.textContent = "Clip: 4 seconds before the alert and 3 seconds after.";
    } else {
      if (a.image) { const img = el("img"); img.src = "/evidence/" + encodeURIComponent(a.image); body.appendChild(img); }
      note.textContent = a.clip ? "The clip is still being saved. Try again in a few seconds." : "No clip for this alert.";
    }
  }).catch(e => {});
  modal.style.display = "flex";
  if (hasGsap) gsap.fromTo(modal.querySelector(".box"), {y: 12, autoAlpha: 0}, {y: 0, autoAlpha: 1, duration: 0.25, ease: "power2.out"});
}

function closeClip() {
  const modal = document.getElementById("modal");
  const v = modal.querySelector("video");
  if (v) v.pause();
  modal.style.display = "none";
}
document.getElementById("modal-close").onclick = closeClip;
document.getElementById("modal").onclick = (e) => { if (e.target.id === "modal") closeClip(); };
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeClip(); });

function renderPeople(people) {
  lastPeople = people;
  const box = document.getElementById("people");
  box.textContent = "";
  for (const d of people) {
    const p = el("div", "person " + d.status + (personFilter === d.name ? " selected" : ""));
    p.appendChild(el("b", "", d.name));
    p.appendChild(el("div", "", "alerts " + d.alert_count));
    p.appendChild(el("div", "", "looks " + d.looks + "  devices " + d.phones));
    p.onclick = () => { personFilter = personFilter === d.name ? null : d.name; buildFilters(); applyFilters(); renderPeople(lastPeople); };
    box.appendChild(p);
  }
}

function showRf(rf) {
  const box = document.getElementById("rf");
  if (!rf || !rf.sweep) { box.textContent = rf && rf.running ? "Radio: " + rf.mode : "RF monitor not running"; box.className = ""; return; }
  const close = (rf.headphones || []).filter(h => h.rssi >= -60).map(h => h.name);
  const parts = [rf.active_bands.length ? "Phone signal: " + rf.active_bands.join(", ") : "Phone bands quiet",
                 close.length ? "Bluetooth close: " + close.join(", ") : "No Bluetooth close"];
  box.textContent = parts.join(" | ");
  box.className = (rf.active_bands.length || close.length) ? "hot" : "";
}

function esc(text) {
  // device names come from other people's headphones, so never trust them as HTML
  return String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function startStep(step) {
  const practice = document.getElementById("practice").checked;
  fetch("/rf/start", {method: "POST", headers: {"Content-Type": "application/json"},
                      body: JSON.stringify({step: step, practice: practice})})
    .then(r => r.json()).then(d => { document.getElementById("rferror").textContent = d.error || ""; refresh(); });
}

function stopRadio() {
  fetch("/rf/stop", {method: "POST"}).then(() => refresh());
}

function showRadio(rf) {
  const practice = document.getElementById("practice").checked;
  const hasBaseline = practice ? rf.baseline_practice : rf.baseline_real;
  for (const step of ["scan", "baseline", "monitor"]) {
    document.getElementById("btn-" + step).disabled = rf.running;
    let mark = "";
    if (rf.running && rf.mode == step) { mark = "<b>running...</b>"; }
    else if (rf.steps_done[step] == "done") { mark = '<span class="ok">done</span>'; }
    else if (rf.steps_done[step] == "problem") { mark = '<span class="bad">problem - open Radio output</span>'; }
    else if (rf.steps_done[step] == "stopped") { mark = "stopped"; }
    document.getElementById("done-" + step).innerHTML = mark;
  }
  // monitor needs a baseline (practice mode makes its own)
  if (!hasBaseline && !practice) { document.getElementById("btn-monitor").disabled = true; }
  if (!hasBaseline && !practice && !rf.running) {
    document.getElementById("done-monitor").innerHTML = '<span class="hint">(needs step 2 first)</span>';
  }
  document.getElementById("btn-stop").disabled = !rf.running;
  document.getElementById("rfstate").textContent = rf.running ? "Running: " + rf.mode : "Radio is off";

  let live = "";
  if (rf.running && rf.mode == "monitor") {
    if (rf.sweep) {
      live += "Sweep " + rf.sweep + " | Active phone bands: " + (rf.active_bands.length ? esc(rf.active_bands.join(", ")) : "none");
      live += "<br>Headphones nearby: " + rf.headphones.length;
      let cam = "camera not running";
      if (rf.camera === true) { cam = "camera sees a phone"; }
      if (rf.camera === false) { cam = "camera sees no phone"; }
      live += "<br>Camera link: " + cam;
      if (rf.bt_problem) { live += '<br><span class="bad">Bluetooth: ' + esc(rf.bt_problem) + "</span>"; }
    } else {
      live = "Waiting for the first sweep (about 15 s)...";
    }
  }
  document.getElementById("rflive").innerHTML = live;

  const out = document.getElementById("rfout");
  const atBottom = out.scrollTop + out.clientHeight >= out.scrollHeight - 5;
  out.textContent = rf.output.join("\\n");
  if (atBottom) { out.scrollTop = out.scrollHeight; }
}

function showNotes(notes) {
  let n = "";
  for (const note of notes || []) {
    n += '<div class="nt ' + esc(note.kind) + '"><b>' + esc(note.title) + '</b><br>' + esc(note.detail) +
         '<div class="time">' + esc(note.clock) + '</div></div>';
  }
  document.getElementById("notes").innerHTML = n || "Nothing yet";
}

function refresh() {
  fetch("/data?since=" + lastId).then(r => r.json()).then(data => {
    renderPeople(data.people);
    const list = document.getElementById("alerts");
    let newDevice = false;
    for (const a of data.alerts) {          // oldest first, each new one goes on top
      if (cards.has(a.id)) continue;
      const card = makeCard(a);
      cards.set(a.id, {el: card, alert: a});
      list.insertBefore(card, list.firstChild);
      card.style.display = visible(a) ? "" : "none";
      if (!firstLoad && (a.kind === "device" || a.kind === "rf")) newDevice = true;
      if (!firstLoad && hasGsap) gsap.from(card, {y: -12, autoAlpha: 0, duration: 0.3, ease: "power2.out"});
      lastId = Math.max(lastId, a.id);
    }
    // decisions made in another tab, and RF matches found after the card was made
    for (const s of data.statuses) {
      const c = cards.get(s.id);
      if (!c) continue;
      if (s.status && c.alert.status !== s.status && s.status !== "pending") { c.alert.status = s.status; setVerdict(c.el, c.alert); }
      if (s.rf && !c.alert.rf && c.alert.kind === "device") {
        c.alert.rf = s.rf;
        const b = el("div", "badge", "RF match: " + s.rf);
        const info = c.el.querySelector(".info");
        info.insertBefore(b, info.querySelector(".actions") || info.querySelector(".verdict"));
      }
    }
    showRf(data.rf);
    showRadio(data.rf);
    showNotes(data.notifications);
    if (newDevice) beep();
    firstLoad = false;
    applyFilters();
    updatePending();
  }).catch(e => {});
}

buildFilters();
document.getElementById("practice").onchange = refresh;
setInterval(refresh, 1000);
refresh();
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return PAGE


def video_stream():
    while True:
        with lock:
            frame = latest_jpeg
        if frame is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.1)


@app.route("/video")
def video():
    return Response(video_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/data")
def data():
    since = request.args.get("since", "0")
    since = int(since) if since.isdigit() else 0
    with lock:
        new_alerts = [a for a in alerts_list if a["id"] > since][-50:]   # oldest first
        statuses = [{"id": a["id"], "status": a["status"], "rf": a.get("rf", "")}
                    for a in alerts_list if a["status"] or a.get("rf")]
        people = list(people_info)
        notes = list(reversed(notifications[-50:]))
    return jsonify({"alerts": new_alerts, "statuses": statuses, "people": people,
                    "notifications": notes, "rf": rf_info()})


def find_alert(alert_id):
    for a in alerts_list:
        if a["id"] == alert_id:
            return a
    return None


@app.route("/review/<int:alert_id>", methods=["POST"])
def review(alert_id):
    body = request.get_json(silent=True) or {}
    decision = body.get("status")
    if decision not in REVIEW_CHOICES:
        return jsonify({"ok": False, "error": "status must be confirmed or dismissed"}), 400
    with lock:
        alert = find_alert(alert_id)
        if alert is None:
            return jsonify({"ok": False, "error": "no such alert"}), 404
        if alert.get("kind") not in REVIEWED_KINDS:
            return jsonify({"ok": False, "error": "only device alerts are reviewed"}), 400
        alert["status"] = decision
        saved = dict(alert)
    save_review(saved)
    return jsonify({"ok": True})


@app.route("/clip_ready/<int:alert_id>")
def clip_ready_route(alert_id):
    with lock:
        alert = find_alert(alert_id)
    if alert is None:
        return jsonify({"ready": False}), 404
    return jsonify({"ready": clip_ready(alert.get("clip", ""))})


@app.route("/rf/start", methods=["POST"])
def rf_start():
    body = request.get_json(silent=True) or {}
    error = start_rf(body.get("step", ""), bool(body.get("practice")))
    return jsonify({"error": error})


@app.route("/rf/stop", methods=["POST"])
def rf_stop():
    stop_rf()
    return jsonify({"error": ""})


@app.route("/evidence/<path:filename>")
def evidence(filename):
    return send_from_directory(os.path.abspath(config.EVIDENCE_FOLDER), filename)


def start():
    def run():
        app.run(host="0.0.0.0", port=config.DASHBOARD_PORT, debug=False, use_reloader=False, threaded=True)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    threading.Thread(target=watch_rf_status, daemon=True).start()
    atexit.register(stop_rf)     # don't leave the radio program running after the app quits
