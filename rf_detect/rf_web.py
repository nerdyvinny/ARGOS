# Draws the RF heatmaps as simple web pages (.html). Open them in any browser.
# No extra installs needed: we just write HTML text.
#
# 1. Waterfall: one row per sweep, one column per frequency window.
#    Color = how far above the empty-room baseline it is. Shows WHICH band and WHEN.
# 2. Survey: a map of the desks. You carry the antenna to each desk, and each
#    desk gets colored by how strong the phone bands were right there.
#    With only 1 antenna this is the best way to find WHERE: the signal is
#    strongest close to the phone.

import os
import json
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
  .desk { position: absolute; border: 2px solid #000; border-radius: 6px; color: #fff;
          font-size: 13px; text-align: center; box-sizing: border-box; padding: 4px;
          text-shadow: 0 0 3px #000; overflow: hidden; }
  .top { border: 4px solid #fff; }
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


def load_desk_layout(desks_file, grid_rows, grid_cols):
    # Gives a list of desks: {"name", "x", "y", "w", "h"} with x/y/w/h from 0 to 1
    # (a fraction of the map size), plus the map's height/width ratio.
    if os.path.exists(desks_file):
        with open(desks_file) as f:
            data = json.load(f)
        fw = data.get("frame_width")
        fh = data.get("frame_height")
        if not fw or not fh:
            # old file without the picture size: use the biggest box corner instead
            fw = max(d["box"][2] for d in data["desks"])
            fh = max(d["box"][3] for d in data["desks"])
        desks = []
        for d in data["desks"]:
            x1, y1, x2, y2 = d["box"]
            desks.append({"name": d["name"], "x": x1 / fw, "y": y1 / fh,
                          "w": (x2 - x1) / fw, "h": (y2 - y1) / fh})
        return desks, fh / fw

    # no desks.json: make a simple grid (front of the room at the top)
    desks = []
    number = 1
    for r in range(grid_rows):
        for c in range(grid_cols):
            desks.append({"name": "Desk " + str(number),
                          "x": (c + 0.1) / grid_cols, "y": (r + 0.1) / grid_rows,
                          "w": 0.8 / grid_cols, "h": 0.8 / grid_rows})
            number = number + 1
    return desks, 0.6


def write_survey(file_name, desks, ratio, results, max_db):
    # results: desk name -> {"db": dB above normal, "band": loudest band}
    width = 900
    height = int(width * ratio)
    html = page_start("RF desk survey", 3)
    html += "<p>Each desk is colored by how strong the phone bands were when the antenna was AT that desk. "
    html += "Strongest = most likely phone (white border). Gray = not measured yet.</p>\n"
    html += legend(max_db)

    top_name = None
    for name in results:
        if top_name is None or results[name]["db"] > results[top_name]["db"]:
            top_name = name

    html += '<div style="position:relative; width:%dpx; height:%dpx; background:#222;">\n' % (width, height)
    for d in desks:
        r = results.get(d["name"])
        if r is None:
            color = heat_color(None, max_db)
            text = d["name"]
        else:
            color = heat_color(r["db"], max_db)
            text = "%s<br><b>%+.1f dB</b><br>%s" % (d["name"], r["db"], r["band"])
        css_class = "desk top" if d["name"] == top_name else "desk"
        html += '<div class="%s" style="left:%dpx; top:%dpx; width:%dpx; height:%dpx; background:%s;">%s</div>\n' % (
            css_class, int(d["x"] * width), int(d["y"] * height),
            int(d["w"] * width), int(d["h"] * height), color, text)
    html += "</div>\n"
    html += "<p>Updated %s</p>\n" % time.strftime("%H:%M:%S")
    html += "</body></html>\n"
    save_text(file_name, html)


# ---------------- little web server so the heatmaps show at http://localhost:PORT ----------------

# web address -> file it shows. Only these files are shared, nothing else.
PAGES = {}


class HeatmapHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            html = page_start("RF heatmaps", 0)
            html += '<p><a style="color:#8cf" href="/waterfall">Waterfall</a> (scan / monitor): which band is busy, over time</p>\n'
            html += '<p><a style="color:#8cf" href="/survey">Desk survey</a> (survey): which desk is the loudest</p>\n'
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


def start_web_server(port, waterfall_file, survey_file, open_page):
    # Runs in the background while the monitor keeps measuring.
    # Gives back the address, or None if it could not start.
    PAGES["/waterfall"] = waterfall_file
    PAGES["/survey"] = survey_file
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), HeatmapHandler)
    except OSError:
        print("Could not start the heatmap page: port", port, "is busy.")
        print("Is another copy of rf_monitor running? Or change WEB_PORT in rf_config.py")
        return None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    address = "http://localhost:%d%s" % (port, open_page)
    print("Heatmap page:", address)
    webbrowser.open(address)
    return address


def save_text(file_name, text):
    # write to a temp file first, then swap, so the browser never sees half a page
    temp = file_name + ".tmp"
    with open(temp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(temp, file_name)
