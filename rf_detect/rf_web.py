# The web page for the phone + Bluetooth monitor, at http://localhost:5050
# No extra installs needed: we just write HTML text.
#
# 1. Waterfall: one row per sweep, one column per frequency window.
#    Color = how far above the empty-room baseline it is. Shows WHICH band and WHEN.
# 2. Bluetooth: a list of nearby headphones/earbuds (from bt_scan.py).

import os
import time
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def heat_color(db_above, max_db):
    # 0 dB above normal = blue, max_db or more = red, in between = green/yellow
    if db_above is None:
        return "#555555"            # gray = no data
    t = db_above / max_db
    if t < 0:
        t = 0
    if t > 1:
        t = 1
    hue = int(240 * (1 - t))        # 240 is blue, 0 is red
    return "hsl(%d, 90%%, 45%%)" % hue


def page_start(title, refresh_seconds):
    refresh = ""
    if refresh_seconds:
        refresh = '<meta http-equiv="refresh" content="%d">' % refresh_seconds
    return """<!doctype html>
<html><head><meta charset="utf-8">%s<title>%s</title>
<style>
  body { background: #111; color: #eee; font-family: Arial, sans-serif; margin: 16px; }
  table { border-collapse: collapse; }
  td, th { padding: 0; font-size: 11px; }
  th { padding: 2px 4px; border-left: 1px solid #888; text-align: left; font-weight: normal; }
  .cell { width: 14px; height: 9px; }
  .time { padding-right: 6px; color: #aaa; }
  .legend span { display: inline-block; width: 40px; text-align: center; color: #000; font-size: 11px; }
  .bt { padding: 6px 10px; font-size: 15px; }
</style></head><body>
<h2>%s</h2>
""" % (refresh, title, title)


def legend(max_db):
    html = '<p class="legend">dB above empty room: '
    steps = 5
    for i in range(steps):
        db = max_db * i / (steps - 1)
        html += '<span style="background:%s">%.0f</span>' % (heat_color(db, max_db), db)
    html += '</p>\n'
    return html


def write_waterfall(file_name, steps, history, max_db, have_baseline):
    # steps:   list of {"band": name, "center": Hz}  (one column each)
    # history: list of (time_text, [value for each step]), newest LAST
    html = page_start("RF waterfall (newest at top)", 3)
    if not have_baseline:
        html += "<p>No baseline yet, so colors are the level above the quietest window, not above the empty room.</p>\n"
    html += legend(max_db)
    html += "<table>\n"

    # top row: band names, each spanning its windows
    html += '<tr><th style="border:none"></th>'
    i = 0
    while i < len(steps):
        band = steps[i]["band"]
        count = 0
        while i < len(steps) and steps[i]["band"] == band:
            count = count + 1
            i = i + 1
        html += '<th colspan="%d">%s</th>' % (count, band)
    html += "</tr>\n"

    for time_text, values in reversed(history):
        html += '<tr><td class="time">%s</td>' % time_text
        for s, v in zip(steps, values):
            tip = "%s  %.2f MHz  %+.1f dB" % (s["band"], s["center"] / 1e6, v)
            html += '<td class="cell" title="%s" style="background:%s"></td>' % (tip, heat_color(v, max_db))
        html += "</tr>\n"
    html += "</table>\n"
    html += "<p>Hover a square to see its frequency. Page refreshes every 3 seconds.</p>\n"
    html += "</body></html>\n"
    save_text(file_name, html)


def write_bluetooth(file_name, headphones, total, problem, too_far, min_rssi):
    # headphones: list from BluetoothWatcher.current(), closest first
    html = page_start("Bluetooth headphones nearby", 3)
    html += "<p>%d Bluetooth devices nearby, %d look like headphones/earbuds. " % (total, len(headphones))
    html += "This shows headphones that are switched on, not who they are connected to. Distance is very rough.</p>\n"
    if too_far:
        html += "<p>Ignored %d headphones weaker than %d dBm (probably another room).</p>\n" % (too_far, min_rssi)
    if problem:
        html += '<p style="color:#f66; font-size:18px;">%s</p>\n' % problem
    html += '<table style="font-size:15px;">\n'
    html += '<tr><th>name</th><th>signal</th><th>distance</th><th>why we think it is headphones</th><th>first seen</th></tr>\n'
    for d in headphones:
        # signal -40 dBm (right here) = red ... -90 dBm (far) = blue
        color = heat_color(d["rssi"] + 90, 50)
        first = time.strftime("%H:%M:%S", time.localtime(d["first_seen"]))
        html += '<tr><td class="bt">%s</td><td class="bt" style="background:%s">%d dBm</td><td class="bt">%s</td><td class="bt">%s</td><td class="bt">%s</td></tr>\n' % (
            d["name"], color, d["rssi"], d["distance"], d["why"], first)
    html += "</table>\n"
    html += "<p>Updated %s</p>\n" % time.strftime("%H:%M:%S")
    html += "</body></html>\n"
    save_text(file_name, html)


# ---------------- little web server so the pages show at http://localhost:PORT ----------------

# web address -> file it shows. Only these files are shared, nothing else.
PAGES = {}


class PageHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            html = page_start("Phone + Bluetooth monitor", 0)
            html += '<p><a style="color:#8cf" href="/waterfall">Phone bands</a> (scan / monitor): which cell band is busy, over time</p>\n'
            html += '<p><a style="color:#8cf" href="/bluetooth">Bluetooth headphones</a> (bluetooth / monitor): earbuds switched on nearby</p>\n'
            html += "</body></html>\n"
        elif path in PAGES and os.path.exists(PAGES[path]):
            with open(PAGES[path], encoding="utf-8") as f:
                html = f.read()
        elif path in PAGES:
            html = page_start("Waiting for the first sweep...", 2)
            html += '<p>Nothing measured yet. This page reloads by itself. <a style="color:#8cf" href="/">Back</a></p></body></html>\n'
        else:
            self.send_error(404)
            return
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")     # always show the newest version
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        pass    # don't print a line for every page load


def start_web_server(port, pages, open_page):
    # Runs in the background while the monitor keeps measuring.
    # pages: web address -> file, e.g. {"/bluetooth": "C:/.../rf_bluetooth.html"}
    # Gives back the address, or None if it could not start.
    PAGES.update(pages)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), PageHandler)
    except OSError:
        print("Could not start the web page: port", port, "is busy.")
        print("Is another copy of rf_monitor running? Or change WEB_PORT in rf_config.py")
        return None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    address = "http://localhost:%d%s" % (port, open_page)
    print("Web page:", address)
    webbrowser.open(address)
    return address


def save_text(file_name, text):
    # write to a temp file first, then swap, so the browser never sees half a page
    temp = file_name + ".tmp"
    with open(temp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(temp, file_name)
