"""The web lab: a page on this PC that shows what Lunes sees, hears, says, measures and does, live:
http://127.0.0.1:8765

The microscope, the bench camera, the Kinect and the last picture the brain looked at; the scope's traces and readings;
where the machine is on its table and whether the Kinect agrees; the circuit, what the engine thinks is wrong with it and
where to probe next; the Arduino's readings; and the conversation with the tools Lunes used. The page is lunes/web/lab.html
and what it shows of the bench comes from lunes/lab.py. Nothing leaves the PC; it only listens on 127.0.0.1.
"""
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

PORT = 8765

PAGE = (Path(__file__).resolve().parent / "web" / "lab.html").read_bytes()


def _jpeg(image, max_side=960, quality=80):
    h, w = image.shape[:2]
    k = min(1.0, max_side / max(h, w))
    if k < 1:
        image = cv2.resize(image, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


class Dashboard:
    def __init__(self, camera=None, kinect=None, speaking=lambda: False, port=PORT, bench=None, lab=None):
        """camera: the microscope and bench: the bench camera; kinect: the Kinect Hume uses too (all their frames are
        safe from any thread); speaking: tells whether Lunes is talking now; lab: gives the bench's state for the page
        (lunes.lab.state)."""
        self.camera, self.bench, self.kinect, self.speaking, self.lab = camera, bench, kinect, speaking, lab
        self.on_tap = None  # on_tap(x, y) for a tap on the microscope's picture, fractions of it
        self.events = deque(maxlen=500)
        self.count = 0
        self.status, self.busy = "", False
        self.seen, self.seen_label, self.seen_version = None, "", 0
        self._kinect_jpg, self._kinect_at = b"", 0.0
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

    def tapped(self, x, y):
        """Erlan tapped a point of the microscope's picture on the page (fractions); Hume reads it with confirm_pin1."""
        if self.on_tap:
            self.on_tap(float(x), float(y))

    def set_seen(self, image, label=""):
        with self._lock:
            self.seen, self.seen_label = _jpeg(image, 1600, 85), label
            self.seen_version += 1

    # what the page asks for ------------------------------------------------------------------------------------
    def _kinect_frame(self):
        if self.kinect is None:
            return b""
        if time.time() - self._kinect_at < 1.0:
            return self._kinect_jpg
        try:
            self._kinect_jpg = _jpeg(self.kinect.color(), 640, 75)
        except RuntimeError:  # unplugged: the tile keeps its last picture
            pass
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
                    return self._send(PAGE, "text/html; charset=utf-8")
                if path == "/tap":
                    try:
                        q = dict(part.split("=") for part in query.split("&") if "=" in part)
                        board.tapped(q["x"], q["y"])
                        board.add("info", f"toque en la foto del microscopio: {float(q['x']):.2f}, {float(q['y']):.2f}")
                        return self._send(b'{"ok": true}', "application/json")
                    except (KeyError, ValueError):
                        return self._send(b'{"ok": false}', "application/json")
                if path == "/events":
                    since = int(query.split("=")[1]) if query.startswith("since=") else 0
                    with board._lock:
                        events = [e for e in board.events if e["n"] >= since]
                        body = {"events": events, "next": board.count, "status": board.status, "busy": board.busy,
                                "speaking": bool(board.speaking()), "seen_version": board.seen_version,
                                "seen_label": board.seen_label}
                    if board.lab is not None:
                        try:
                            body["lab"] = board.lab()
                        except Exception:  # the worker changed something meanwhile: the next poll reads it
                            pass
                    return self._send(json.dumps(body, ensure_ascii=False, default=float).encode(), "application/json")
                for name, camera in (("/img/microscope.jpg", board.camera), ("/img/bench.jpg", board.bench)):
                    if path == name and camera is not None:
                        try:
                            return self._send(_jpeg(camera.frame(), 640, 75), "image/jpeg")
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
