"""A page on this PC that shows what Lunes sees, hears, says and does, live: http://127.0.0.1:8765

The microscope live, the Kinect, the last picture the brain looked at (a mosaic or a close-up), the conversation
with the tools Lunes used, what it is doing now, whether it is talking, and how long each part of an answer took.
Nothing leaves the PC; the page only listens on 127.0.0.1.
"""
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

PORT = 8765

PAGE = r"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lunes en vivo</title>
<style>
:root { --bg:#f6f6f4; --panel:#ffffff; --ink:#1d1d1b; --muted:#6b6b66; --line:#e2e2dc; --you:#e8f0fe; --lunes:#eef7ee;
        --tool:#fff6e5; --accent:#2f7d4a; }
@media (prefers-color-scheme: dark) { :root { --bg:#141413; --panel:#1e1e1c; --ink:#ecece8; --muted:#9a9a93;
        --line:#33332f; --you:#1f2a3d; --lunes:#1d2e22; --tool:#33291a; --accent:#5fbf7f; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.45 system-ui, sans-serif; }
header { display:flex; flex-wrap:wrap; gap:12px; align-items:center; padding:12px 16px; border-bottom:1px solid var(--line); }
header h1 { font-size:18px; margin:0 12px 0 0; }
.pill { padding:3px 10px; border-radius:999px; border:1px solid var(--line); color:var(--muted); font-size:13px; }
.pill.on { color:#fff; background:var(--accent); border-color:var(--accent); }
main { display:grid; grid-template-columns:minmax(0,1.1fr) minmax(0,1fr); gap:16px; padding:16px; }
@media (max-width: 860px) { main { grid-template-columns:1fr; } }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px; }
.panel h2 { font-size:14px; margin:0 0 8px; color:var(--muted); font-weight:600; }
.views { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
.views .wide { grid-column:1 / -1; }
img { width:100%; border-radius:6px; background:#000; display:block; }
#status { font-weight:600; }
#log { height:70vh; overflow-y:auto; display:flex; flex-direction:column; gap:6px; }
.e { padding:6px 10px; border-radius:8px; max-width:100%; overflow-wrap:anywhere; }
.e .t { color:var(--muted); font-size:12px; margin-right:6px; }
.tu { background:var(--you); align-self:flex-end; }
.lunes, .progress { background:var(--lunes); }
.progress { font-style:italic; }
.tool, .job { background:var(--tool); font-family:ui-monospace, monospace; font-size:13px; }
.ear, .think, .info, .timing, .result { color:var(--muted); font-size:13px; padding:2px 10px; }
.timing { font-family:ui-monospace, monospace; }
</style></head><body>
<header><h1>Lunes en vivo</h1><span id="speaking" class="pill">callado</span><span id="busy" class="pill">esperando</span>
<span id="status"></span></header>
<main>
<section class="panel"><div class="views">
  <div><h2>Microscopio</h2><img id="mic" alt="microscopio"></div>
  <div><h2>Kinect</h2><img id="kin" alt="Kinect"></div>
  <div class="wide"><h2 id="seenh">Lo último que analizó la IA</h2><img id="seen" alt="última imagen analizada"></div>
</div></section>
<section class="panel"><h2>Conversación</h2><div id="log"></div></section>
</main>
<script>
let next = 0, seenVersion = -1;
const log = document.getElementById("log");
async function poll() {
  try {
    const r = await (await fetch("/events?since=" + next)).json();
    for (const e of r.events) {
      const d = document.createElement("div");
      d.className = "e " + e.kind;
      const t = document.createElement("span"); t.className = "t"; t.textContent = e.at;
      d.appendChild(t); d.appendChild(document.createTextNode(e.text));
      const atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 30;
      log.appendChild(d);
      if (atEnd) log.scrollTop = log.scrollHeight;
    }
    next = r.next;
    document.getElementById("status").textContent = r.status;
    const sp = document.getElementById("speaking"); sp.textContent = r.speaking ? "hablando" : "callado";
    sp.classList.toggle("on", r.speaking);
    const b = document.getElementById("busy"); b.textContent = r.busy ? "trabajando" : "esperando";
    b.classList.toggle("on", r.busy);
    if (r.seen_version !== seenVersion) {
      seenVersion = r.seen_version;
      const seen = document.getElementById("seen");
      seen.style.display = seenVersion > 0 ? "block" : "none";
      if (seenVersion > 0) seen.src = "/img/seen.jpg?v=" + seenVersion;
      document.getElementById("seenh").textContent = "Lo último que analizó la IA" + (r.seen_label ? " (" + r.seen_label + ")" : "");
    }
  } catch (err) {}
  setTimeout(poll, 500);
}
function refresh(id, path, ms) {
  const img = document.getElementById(id);
  const tick = () => { img.src = path + "?t=" + Date.now(); };
  img.onload = img.onerror = () => setTimeout(tick, ms);
  tick();
}
poll(); refresh("mic", "/img/microscope.jpg", 250); refresh("kin", "/img/kinect.jpg", 1500);
</script></body></html>"""


def _jpeg(image, max_side=960, quality=80):
    h, w = image.shape[:2]
    k = min(1.0, max_side / max(h, w))
    if k < 1:
        image = cv2.resize(image, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


class Dashboard:
    def __init__(self, camera=None, kinect_factory=None, speaking=lambda: False, port=PORT):
        """camera: the microscope (its frame() is safe from any thread); kinect_factory: makes a Kinect client of
        the page's own; speaking: tells whether Lunes is talking now."""
        self.camera, self.kinect_factory, self.speaking = camera, kinect_factory, speaking
        self.events = deque(maxlen=500)
        self.count = 0
        self.status, self.busy = "", False
        self.seen, self.seen_label, self.seen_version = None, "", 0
        self._kinect, self._kinect_jpg, self._kinect_at = None, b"", 0.0
        self._lock = threading.Lock()
        handler = self._handler()
        self.server = ThreadingHTTPServer(("127.0.0.1", port), handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{port}"

    # what Lunes reports ------------------------------------------------------------------------------------------
    def add(self, kind, text):
        with self._lock:
            self.events.append({"n": self.count, "at": time.strftime("%H:%M:%S"), "kind": kind, "text": text})
            self.count += 1

    def set_seen(self, image, label=""):
        with self._lock:
            self.seen, self.seen_label = _jpeg(image, 1600, 85), label
            self.seen_version += 1

    # what the page asks for ------------------------------------------------------------------------------------
    def _kinect_frame(self):
        if self.kinect_factory is None:
            return b""
        if time.time() - self._kinect_at < 1.0:
            return self._kinect_jpg
        try:
            if self._kinect is None:
                self._kinect = self.kinect_factory()
            self._kinect_jpg = _jpeg(self._kinect.color(), 640, 75)
        except Exception:
            self._kinect = None
        self._kinect_at = time.time()
        return self._kinect_jpg

    def _handler(self):
        board = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep the console for the conversation
                pass

            def _send(self, body, kind):
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path, _, query = self.path.partition("?")
                if path == "/":
                    return self._send(PAGE.encode(), "text/html; charset=utf-8")
                if path == "/events":
                    since = int(query.split("=")[1]) if query.startswith("since=") else 0
                    with board._lock:
                        events = [e for e in board.events if e["n"] >= since]
                        body = {"events": events, "next": board.count, "status": board.status, "busy": board.busy,
                                "speaking": bool(board.speaking()), "seen_version": board.seen_version,
                                "seen_label": board.seen_label}
                    return self._send(json.dumps(body, ensure_ascii=False).encode(), "application/json")
                if path == "/img/microscope.jpg" and board.camera is not None:
                    try:
                        return self._send(_jpeg(board.camera.frame(), 640, 75), "image/jpeg")
                    except Exception:
                        pass
                if path == "/img/kinect.jpg":
                    jpg = board._kinect_frame()
                    if jpg:
                        return self._send(jpg, "image/jpeg")
                if path == "/img/seen.jpg" and board.seen:
                    return self._send(board.seen, "image/jpeg")
                self.send_response(404)
                self.end_headers()

        return Handler
