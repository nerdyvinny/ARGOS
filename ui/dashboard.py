# Tiny web page for the proctor.  Open http://localhost:5000
# It shows the live video, every person's color, and the list of alerts.
# Each alert has a short video clip to play back. Device alerts (phone, watch...)
# also get Confirm / Dismiss buttons, because a person has to decide whose device it is.
# If the RF monitor (rf_detect/rf_monitor.py) is running, a close phone or Bluetooth signal
# makes an "RF" alert that says whether the camera saw a device at the same time.

import os
import csv
import time
import threading
from flask import Flask, Response, jsonify, request, send_from_directory
from settings import config
from app import clips
from app import rf_link

app = Flask(__name__)
lock = threading.Lock()

latest_jpeg = None
alerts_list = []
people_info = []
next_id = [1]
REVIEW_CHOICES = ("confirmed", "dismissed")
REVIEWED_KINDS = ("device", "rf")   # only alerts about devices get Confirm / Dismiss
MATCH_SECONDS = 15                  # RF and camera count as "the same moment" if this close in time
CAMERA_STATUS_SECONDS = 10          # a device alert counts as "camera sees a device" for this long
rf_info = rf_link.summary(None)
rf_times = []                       # (time, text) of recent close RF signals
last_camera_status = [0.0]


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
    # about once a second, tell the RF monitor who the camera sees with a device
    if time.time() - last_camera_status[0] >= 1.0:
        last_camera_status[0] = time.time()
        rf_link.write_camera_status(device_people(CAMERA_STATUS_SECONDS))


def device_people(seconds):
    # names of people with a device alert in the last few seconds
    now = time.time()
    with lock:
        names = [a["person"] for a in alerts_list if a.get("kind") == "device" and now - a["t"] <= seconds]
    return sorted(set(names))


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


def watch_rf():
    # check the RF monitor's file once a second
    global rf_info
    while True:
        try:
            status = rf_link.read_status()
            with lock:
                rf_info = rf_link.summary(status)
            for event in rf_link.new_close_signals(status):
                make_rf_alert(event)
        except Exception as e:
            print("[RF] problem reading the RF monitor:", e)
        time.sleep(1)


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
  if (!rf || !rf.connected) { box.textContent = "RF monitor not running"; box.className = ""; return; }
  const parts = [];
  if (rf.radio) parts.push(rf.active_bands.length ? "Phone signal: " + rf.active_bands.join(", ") : "Phone bands quiet");
  parts.push(rf.near.length ? "Bluetooth close: " + rf.near.join(", ") : "No Bluetooth close");
  box.textContent = "RF: " + parts.join("  |  ");
  box.className = (rf.active_bands.length || rf.near.length) ? "hot" : "";
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
    if (newDevice) beep();
    firstLoad = false;
    applyFilters();
    updatePending();
  }).catch(e => {});
}

buildFilters();
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
        rf = dict(rf_info)
    return jsonify({"alerts": new_alerts, "statuses": statuses, "people": people, "rf": rf})


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


@app.route("/evidence/<path:filename>")
def evidence(filename):
    return send_from_directory(os.path.abspath(config.EVIDENCE_FOLDER), filename)


def start():
    def run():
        app.run(host="0.0.0.0", port=config.DASHBOARD_PORT, debug=False, use_reloader=False, threaded=True)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    threading.Thread(target=watch_rf, daemon=True).start()
