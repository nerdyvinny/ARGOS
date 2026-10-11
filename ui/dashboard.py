# Tiny web page for the proctor.  Open http://localhost:5000
# It shows the live video, every person's color, and the list of alerts with pictures.

import os
import time
import threading
from flask import Flask, Response, jsonify, send_from_directory
from settings import config

app = Flask(__name__)
lock = threading.Lock()

latest_jpeg = None
alerts_list = []
people_info = []


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
  #video { width:100%; max-width:1280px; border:2px solid #333; border-radius:6px; }
  .row { display:flex; gap:16px; flex-wrap:wrap; }
  .left { flex:3; min-width:300px; }
  .right { flex:1; min-width:260px; }
  .people { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
  .person { padding:8px 12px; border-radius:6px; font-size:14px; min-width:90px; }
  .green { background:#1d5e2b; } .yellow { background:#8a6d00; }
  .red { background:#9b1c1c; } .empty { background:#333; color:#999; }
  .alert { background:#222; border-left:4px solid #e33; padding:8px; margin-bottom:8px; border-radius:4px; font-size:14px; }
  .alert img { width:100%; margin-top:6px; border-radius:4px; }
  .time { color:#aaa; font-size:12px; }
</style>
</head>
<body>
<h1>Exam Watch - live</h1>
<div class="row">
  <div class="left">
    <img id="video" src="/video">
    <div class="people" id="people"></div>
  </div>
  <div class="right">
    <h3>Alerts</h3>
    <div id="alerts">No alerts yet</div>
  </div>
</div>
<script>
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
  }).catch(e => {});
}
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
    return jsonify({"alerts": newest_first, "people": people})


@app.route("/evidence/<path:filename>")
def evidence(filename):
    return send_from_directory(os.path.abspath(config.EVIDENCE_FOLDER), filename)


def start():
    def run():
        app.run(host="0.0.0.0", port=config.DASHBOARD_PORT, debug=False, use_reloader=False, threaded=True)
    t = threading.Thread(target=run, daemon=True)
    t.start()
