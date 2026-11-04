"""The web lab: a page on this PC that shows what Lunes sees, hears, says, measures and does, live:
http://127.0.0.1:8765

The microscope, the bench camera, the Kinect and the last picture the brain looked at; the scope's traces and readings;
where the machine is on its table and whether the Kinect agrees; the circuit, what the engine thinks is wrong with it and
where to probe next; the Arduino's readings; and the conversation with the tools Lunes used. The page is lunes/web/lab.html
and what it shows of the bench comes from lunes/lab.py. The report page, lunes/web/report.html at /report, shows the
Markdown Lunes writes in runs/reports/, with its pictures from under runs/. With the hooks talk.py gives (on_text, on_mic,
mic_is_on) the page also types a line to Lunes and mutes the microphone. Nothing leaves the PC; it only listens on
127.0.0.1.
"""
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import cv2

PORT = 8765

WEB = Path(__file__).resolve().parent / "web"
RUNS = WEB.parent.parent / "runs"  # what Lunes writes on the bench (reports, pictures, JSON): the report page reads it
PAGE = (WEB / "lab.html").read_bytes()  # the tests look at it; the server reads the file again at every request
FILES = {".md": "text/markdown; charset=utf-8", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
         ".json": "application/json"}  # the kinds of file the page may read under runs/


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
        self.on_scope_live = None  # on_scope_live(on) for the scope's "live" button; returns what happened
        self.on_text = self.on_mic = self.mic_is_on = None  # talk.py: a line typed for Lunes, the mic's switch, its state
        self.circuits = self.on_circuit_load = None  # talk.py: the saved circuits (runs/circuits/) and loading one on the bench
        self.on_layout_edit = None  # talk.py: the breadboard as Erlan edited it by hand; rebuilds and keeps the circuit
        self.on_project_new = None  # talk.py: a new, empty project with the name Erlan typed
        # the project open now: every event is marked with it, so the page shows each project's own conversation
        self.project = {"id": None, "name": None}
        self.mosaic = None  # gives the last mosaic's picture, for the machine's map
        self.runs = RUNS  # the tests point it at a folder of their own
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
    def add(self, kind, text, at=None):
        with self._lock:
            self.events.append({"n": self.count, "at": at or time.strftime("%H:%M:%S"), "kind": kind, "text": text,
                                "project": self.project.get("id")})
            self.count += 1

    def tapped(self, x, y):
        """Erlan tapped a point of the microscope's picture on the page (fractions); Hume reads it with confirm_pin1."""
        if self.on_tap:
            self.on_tap(float(x), float(y))

    def set_seen(self, image, label=""):
        with self._lock:
            self.seen, self.seen_label = _jpeg(image, 1600, 85), label
            self.seen_version += 1

    def clear_seen(self):
        """No analysed picture: a new project starts with the AI analysis view blank. Version 0 tells the page so."""
        with self._lock:
            self.seen, self.seen_label, self.seen_version = None, "", 0

    # what the page asks for ------------------------------------------------------------------------------------
    def _kinect_frame(self):
        if self.kinect is None:
            return b""
        if time.time() - self._kinect_at < 1.0:
            return self._kinect_jpg
        self._kinect_at = time.time()  # the next requests take the last picture while this one waits for the Kinect
        try:
            self._kinect_jpg = _jpeg(self.kinect.color(), 640, 75)
        except (RuntimeError, OSError):  # unplugged or hung: the tile keeps its last picture
            pass
        return self._kinect_jpg

    def _handler(self):
        board = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep the console for the conversation
                pass

            def _send(self, body, kind, status=200):
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _json(self, data, status=200):
                return self._send(json.dumps(data, default=float).encode(), "application/json", status)

            def do_POST(self):
                try:
                    self._post()
                except ConnectionError:
                    self.close_connection = True

            def _post(self):
                """What the page sends: a line typed for Lunes, or the microphone's switch; both need talk.py's hooks."""
                path = self.path.partition("?")[0]
                try:
                    size = min(int(self.headers.get("Content-Length") or 0), 1 << 16)
                    data = json.loads(self.rfile.read(size) or b"{}")
                    if not isinstance(data, dict):
                        raise ValueError("not an object")
                except (ValueError, TypeError) as e:
                    return self._json({"ok": False, "error": f"bad request: {e}"}, 400)
                if path == "/say":
                    text = str(data.get("text", "")).strip()
                    if not text:
                        return self._json({"ok": False, "error": "nothing to say"}, 400)
                    if board.on_text is None:
                        return self._json({"ok": False, "error": "no conversation to type into"}, 503)
                    board.on_text(text)
                    return self._json({"ok": True})
                if path == "/mic":
                    if board.on_mic is None:
                        return self._json({"ok": False, "error": "no voice session"}, 503)
                    return self._json({"on": bool(board.on_mic(bool(data.get("on"))))})
                if path == "/layout/edit":
                    layout = data.get("layout")
                    if not isinstance(layout, dict):
                        return self._json({"ok": False, "error": "no layout"}, 400)
                    if board.on_layout_edit is None:
                        return self._json({"ok": False, "error": "no bench to edit"}, 503)
                    return self._json(board.on_layout_edit(layout))
                if path == "/circuit/load":
                    file = str(data.get("file", "")).strip()
                    if not file:
                        return self._json({"ok": False, "error": "no file"}, 400)
                    if board.on_circuit_load is None:
                        return self._json({"ok": False, "error": "no bench to load it on"}, 503)
                    return self._json(board.on_circuit_load(file))
                if path == "/project/new":
                    name = str(data.get("name", "")).strip()
                    if not name:
                        return self._json({"ok": False, "error": "no name"}, 400)
                    if board.on_project_new is None:
                        return self._json({"ok": False, "error": "no bench to keep a project on"}, 503)
                    return self._json(board.on_project_new(name))
                self._nothing()

            def do_GET(self):
                try:
                    self._get()
                except ConnectionError:  # the page let go of a picture or a poll it no longer needs: nothing to say
                    self.close_connection = True

            def _get(self):
                path, _, query = self.path.partition("?")
                if path == "/":  # read again each time: a change to the page shows with a reload while Lunes runs
                    return self._send((WEB / "lab.html").read_bytes(), "text/html; charset=utf-8")
                if path == "/report":  # ?file=<name> is for the page, which then asks for /runs/reports/<name>
                    return self._send((WEB / "report.html").read_bytes(), "text/html; charset=utf-8")
                if path == "/reports":
                    folder = board.runs / "reports"
                    names = sorted((p for p in folder.glob("*.md") if p.name != "latest.md"),
                                   key=lambda p: (p.stat().st_mtime, p.name), reverse=True) if folder.is_dir() else []
                    body = {"latest": (folder / "latest.md").is_file(), "reports": [p.name for p in names]}
                    return self._send(json.dumps(body).encode(), "application/json")
                if path.startswith("/runs/"):
                    return self._file(path[len("/runs/"):])
                if path == "/circuits":  # each one's file is readable at /runs/circuits/<file>, its plan in "layout"
                    return self._json(board.circuits() if board.circuits else [])
                if path == "/tap":
                    try:
                        q = dict(part.split("=") for part in query.split("&") if "=" in part)
                        board.tapped(q["x"], q["y"])
                        board.add("info", f"toque en la foto del microscopio: {float(q['x']):.2f}, {float(q['y']):.2f}")
                        return self._send(b'{"ok": true}', "application/json")
                    except (KeyError, ValueError):
                        return self._send(b'{"ok": false}', "application/json")
                if path == "/scope_live":
                    on = query == "on=1"
                    done = board.on_scope_live(on) if board.on_scope_live else {"ok": False, "error": "no scope"}
                    board.add("info", "osciloscopio en vivo: " + ("encendido" if on and done.get("ok") else
                                                                  "apagado" if done.get("ok") else done.get("error", "")))
                    return self._send(json.dumps(done, default=float).encode(), "application/json")
                if path == "/events":
                    since = int(query.split("=")[1]) if query.startswith("since=") else 0
                    with board._lock:
                        events = [e for e in board.events if e["n"] >= since]
                        body = {"events": events, "next": board.count, "status": board.status, "busy": board.busy,
                                "speaking": bool(board.speaking()), "seen_version": board.seen_version,
                                "seen_label": board.seen_label, "project": dict(board.project)}
                    body["mic"] = bool(board.mic_is_on()) if board.mic_is_on else None  # None: no voice session
                    if board.lab is not None:
                        try:
                            body["lab"] = board.lab()
                        except Exception:  # the worker changed something meanwhile: the next poll reads it
                            pass
                    return self._send(json.dumps(body, ensure_ascii=False, default=float).encode(), "application/json")
                for name, camera in (("/img/microscope.jpg", board.camera), ("/img/bench.jpg", board.bench)):
                    if path == name and camera is not None:
                        # ?w=1920 for a recording of the camera (the video's B-roll); the page asks for the small one
                        q = dict(part.split("=", 1) for part in query.split("&") if "=" in part)
                        try:
                            side = min(max(int(q.get("w", 640)), 160), 3840)
                        except ValueError:
                            side = 640
                        try:
                            return self._send(_jpeg(camera.frame(), side, 75 if side <= 640 else 88), "image/jpeg")
                        except Exception:
                            pass
                if path == "/img/kinect.jpg":
                    jpg = board._kinect_frame()
                    if jpg:
                        return self._send(jpg, "image/jpeg")
                if path == "/img/mosaic.jpg" and board.mosaic is not None:
                    image = board.mosaic()
                    if image is not None:
                        return self._send(_jpeg(image, 1600, 80), "image/jpeg")
                if path == "/img/seen.jpg" and board.seen:
                    return self._send(board.seen, "image/jpeg")
                self._nothing()

            def _file(self, rel):
                """A file under runs/, read-only: reports, pictures and JSON only, never a path that leaves runs/."""
                parts = unquote(rel).split("/")
                root = board.runs.resolve()
                if any(part in ("", ".", "..") or "\\" in part or ":" in part for part in parts):
                    return self._nothing()
                try:
                    target = root.joinpath(*parts).resolve(strict=True)
                except OSError:
                    return self._nothing()
                if not (target.is_relative_to(root) and target != root and target.is_file()
                        and target.suffix.lower() in FILES):
                    return self._nothing()
                return self._send(target.read_bytes(), FILES[target.suffix.lower()])

            def _nothing(self):
                self.send_response(404)
                self.end_headers()

        return Handler
