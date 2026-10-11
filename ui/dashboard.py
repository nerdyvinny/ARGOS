# Tiny web page for the proctor.  Open http://localhost:5000
# It shows the live video, every person's color, and the list of alerts with pictures.
#
# It also has a Radio panel: buttons that run the radio detector (rf_detect/rf_monitor.py)
# in the background, step by step: 1. Scan  2. Baseline  3. Monitor.
# When the camera sees a phone, or the radio hears a phone band / Bluetooth headphones,
# a text notification shows up on the side (no pictures).

import os
import json
import atexit
import time
import threading
import subprocess
from flask import Flask, Response, jsonify, request, send_from_directory
from settings import config

app = Flask(__name__)
lock = threading.Lock()

latest_jpeg = None
alerts_list = []
people_info = []

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

notifications = []      # text only, newest last
last_camera_write = 0


def set_frame(jpeg_bytes):
    global latest_jpeg
    with lock:
        latest_jpeg = jpeg_bytes


def add_alerts(new_alerts):
    with lock:
        for a in new_alerts:
            alerts_list.append(a)
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

    command = RF_PYTHON + ["-u", RF_SCRIPT, mode, "--no-web"]
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
        rf_output.append("> " + " ".join(command[:2]) + " rf_detect\\rf_monitor.py " + " ".join(command[4:]))
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
    # While "monitor" runs, read rf_status.json every second and send notifications
    # for phone bands that just became ACTIVE and for headphones we haven't seen yet.
    global rf_status, active_bands_before
    while True:
        time.sleep(1)
        if rf_mode != "monitor" or not rf_running():
            continue
        try:
            with open(RF_STATUS_FILE, encoding="utf-8") as f:
                status = json.load(f)
        except (OSError, ValueError):
            continue     # no sweep finished yet

        active_now = set()
        for b in status.get("active_bands", []):
            active_now.add(b["band"])
            if b["band"] not in active_bands_before:
                detail = "carriers: " + b["carriers"]
                if status.get("camera") is True:
                    detail = detail + " (camera also sees a phone)"
                notify("radio", "Phone signal ACTIVE: " + b["band"], detail)
        for h in status.get("headphones", []):
            if h["address"] not in headphones_seen:
                headphones_seen.add(h["address"])
                notify("bluetooth", "Bluetooth headphones: " + h["name"],
                       h["distance"] + " (" + str(h["rssi"]) + " dBm)")
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
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Exam Watch</title>
<style>
  body { background:#111; color:#eee; font-family:Arial, sans-serif; margin:0; padding:16px; }
  h1 { margin:0 0 12px 0; font-size:22px; }
  h3 { margin:16px 0 8px 0; }
  #video { width:100%; max-width:1280px; border:2px solid #333; border-radius:6px; }
  .row { display:flex; gap:16px; flex-wrap:wrap; }
  .left { flex:3; min-width:300px; }
  .right { flex:1; min-width:300px; }
  .people { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
  .person { padding:8px 12px; border-radius:6px; font-size:14px; min-width:90px; }
  .green { background:#1d5e2b; } .yellow { background:#8a6d00; }
  .red { background:#9b1c1c; } .empty { background:#333; color:#999; }
  .alert { background:#222; border-left:4px solid #e33; padding:8px; margin-bottom:8px; border-radius:4px; font-size:14px; }
  .alert img { width:100%; max-width:320px; margin-top:6px; border-radius:4px; }
  .time { color:#aaa; font-size:12px; }
  .box { background:#1b1b1b; border:1px solid #333; border-radius:6px; padding:12px; margin-bottom:12px; }
  .step { display:flex; align-items:center; gap:8px; margin:6px 0; font-size:14px; }
  .step button { width:110px; }
  button { background:#2a5db0; color:#fff; border:0; border-radius:4px; padding:8px 10px; font-size:14px; cursor:pointer; }
  button:disabled { background:#444; color:#888; cursor:default; }
  button.stop { background:#9b1c1c; }
  .hint { color:#aaa; font-size:12px; }
  .ok { color:#5c5; } .bad { color:#e55; }
  #rferror { color:#e55; font-size:13px; margin-top:6px; }
  pre { background:#000; color:#ccc; font-size:11px; padding:8px; max-height:220px; overflow:auto; white-space:pre-wrap; }
  .note { background:#222; padding:8px; margin-bottom:8px; border-radius:4px; font-size:14px; border-left:4px solid #888; }
  .note.camera { border-left-color:#f90; } .note.radio { border-left-color:#e33; } .note.bluetooth { border-left-color:#39f; }
</style>
</head>
<body>
<h1>Exam Watch - live</h1>
<div class="row">
  <div class="left">
    <img id="video" src="/video">
    <div class="people" id="people"></div>
    <h3>Camera alerts</h3>
    <div id="alerts">No alerts yet</div>
  </div>
  <div class="right">
    <div class="box">
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
    <h3>Notifications</h3>
    <div id="notes">Nothing yet</div>
  </div>
</div>
<script>
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

function refresh() {
  fetch("/data").then(r => r.json()).then(data => {
    let html = "";
    for (const d of data.people) {
      html += '<div class="person ' + d.status + '"><b>' + d.name + '</b><br>alerts ' + d.alert_count +
              '<br>looks ' + d.looks + ' | phone ' + d.phones + '</div>';
    }
    document.getElementById("people").innerHTML = html;

    let a = "";
    for (const al of data.alerts) {
      a += '<div class="alert"><b>' + al.message + '</b><div class="time">' + al.clock +
           ' (exam time ' + al.exam_time + ')</div>';
      if (al.image) { a += '<img src="/evidence/' + al.image + '">'; }
      a += '</div>';
    }
    if (a == "") { a = "No alerts yet"; }
    document.getElementById("alerts").innerHTML = a;

    let n = "";
    for (const note of data.notifications) {
      n += '<div class="note ' + note.kind + '"><b>' + esc(note.title) + '</b><br>' + esc(note.detail) +
           '<div class="time">' + note.clock + '</div></div>';
    }
    if (n == "") { n = "Nothing yet"; }
    document.getElementById("notes").innerHTML = n;

    showRadio(data.rf);
  }).catch(e => {});
}
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
    with lock:
        newest_first = list(reversed(alerts_list[-50:]))
        people = list(people_info)
        notes = list(reversed(notifications[-50:]))
    return jsonify({"alerts": newest_first, "people": people, "notifications": notes, "rf": rf_info()})


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
