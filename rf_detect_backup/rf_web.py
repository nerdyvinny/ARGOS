# The web page for the phone + Bluetooth monitor, at http://localhost:5050
# No extra installs needed: we just write HTML text.
#
# Everything shows on ONE page (/). The two writers below each save a piece of it.
# 1. Phones: a big banner ("possible phone" or not) + one card per cell band.
# 2. Bluetooth: a big banner ("possible headphones" or not) + a short list (from bt_scan.py).

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
  body { background: #111; color: #eee; font-family: Arial, sans-serif; margin: 16px; max-width: 760px; }
  .banner { padding: 24px; border-radius: 10px; font-size: 30px; font-weight: bold; margin: 16px 0; }
  .banner small { display: block; font-size: 16px; font-weight: normal; margin-top: 6px; }
  .red { background: #b3261e; }
  .yellow { background: #b58900; }
  .green { background: #1e6b34; }
  .gray { background: #444; }
  .card { display: flex; justify-content: space-between; padding: 12px 16px; margin: 6px 0;
          border-radius: 8px; font-size: 18px; background: #222; border-left: 10px solid #444; }
  .card.red { border-left-color: #e5534b; }
  .card.yellow { border-left-color: #e3b341; }
  .card.green { border-left-color: #3fb950; }
  .card small { color: #aaa; font-size: 14px; }
  .note { color: #aaa; font-size: 14px; }
</style></head><body>
<h1>%s</h1>
""" % (refresh, title, title)


def write_phone_status(file_name, bands, have_baseline):
    # bands: list of (band name, carriers, state), state is "active", "maybe", "ok" or "nodata"
    active = [b for b in bands if b[2] == "active"]
    maybe = [b for b in bands if b[2] == "maybe"]
    html = "<h2>Phones</h2>" + "\n"
    if not have_baseline:
        html += '<div class="banner gray">No baseline yet<small>Run "baseline" in the empty room first.</small></div>\n'
    elif active:
        names = ", ".join(b[0] for b in active)
        html += '<div class="banner red">Possible phone in use<small>Extra signal on: %s</small></div>\n' % names
    elif maybe:
        html += '<div class="banner yellow">Maybe a phone<small>A short blip. Not confirmed yet.</small></div>\n'
    else:
        html += '<div class="banner green">No phone detected</div>\n'

    words = {"active": "Possible phone", "maybe": "Maybe", "ok": "Quiet", "nodata": "No data"}
    colors = {"active": "red", "maybe": "yellow", "ok": "green", "nodata": ""}
    for name, carriers, state in bands:
        html += '<div class="card %s"><span>%s<br><small>%s</small></span><span>%s</span></div>\n' % (
            colors[state], name, carriers, words[state])
    save_text(file_name, html)


def write_bluetooth(file_name, headphones, total, problem, too_far, min_rssi):
    # headphones: list from BluetoothWatcher.current(), closest first
    html = "<h2>Bluetooth</h2>" + "\n"
    if problem:
        html += '<div class="banner gray">Bluetooth problem<small>%s</small></div>\n' % problem
    elif headphones:
        html += '<div class="banner red">Possible Bluetooth headphones<small>%d switched on nearby</small></div>\n' % len(headphones)
    else:
        html += '<div class="banner green">No Bluetooth headphones detected</div>\n'

    for d in headphones:
        html += '<div class="card red"><span>%s<br><small>%s</small></span><span>%s</span></div>\n' % (
            d["name"], d["why"], d["distance"])
    if too_far:
        html += '<p class="note">Ignored %d weaker than %d dBm (probably another room).</p>\n' % (too_far, min_rssi)
    html += '<p class="note">Shows headphones that are switched on, not who they are connected to. Distance is very rough.</p>\n'
    save_text(file_name, html)


# ---------------- little web server so the pages show at http://localhost:PORT ----------------

# web address -> file with that part of the page. Only these files are read, nothing else.
PAGES = {}


class PageHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0]
        if path not in ("/", "/waterfall", "/bluetooth"):
            self.send_error(404)
            return
        # one page with both pieces, whichever address you use
        html = page_start("Phone + Bluetooth monitor", 3)
        for address in ("/waterfall", "/bluetooth"):
            file_name = PAGES.get(address)
            if file_name and os.path.exists(file_name):
                with open(file_name, encoding="utf-8") as f:
                    html += f.read()
            elif address == "/waterfall":
                html += "<h2>Phones</h2><p class=" + '"note"' + ">Not measured yet (starts with scan or monitor).</p>" + chr(10)
            else:
                html += "<h2>Bluetooth</h2><p class=" + '"note"' + ">Not measured yet (starts with bluetooth or monitor).</p>" + chr(10)
        html += '<p class="note">Updated %s. Refreshes every 3 seconds. It only flags, a person decides.</p>' % time.strftime("%H:%M:%S")
        html += "</body></html>"
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
