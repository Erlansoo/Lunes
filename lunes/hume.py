"""Lunes' local brain and its memory, Hume.

Muse Glimmer (Meta's open weights) runs on the bench PC through Ollama. Lunes plans with tools that move the CNC and
look through the microscope and the Kinect, and answers in short spoken Spanish.

Its memory is called Hume, after David Hume: Lunes only believes what it observed. Every fact it remembers keeps the
observation behind it, and it says so when it is only guessing.

Lunes works as an agent with whole skills rather than single steps: measure_board walks the board's outline,
survey_board photographs exactly that area, inspect_component photographs one part entirely and close up to read it.

Safety lives in the tools, not in the prompt: moves stay inside the bench's allowed area (the travel Erlan measured,
in config/machine.json, minus a margin), X/Y moves lift Z first, and Z never goes below the board's focus height (or
below the bench's Z origin before the first autofocus).
"""
import base64
import json
import math
import re
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from microscope import autofocus, sharpness  # noqa: E402
from survey import (Interrupted, closeup_for_reading, component_box, find_perimeter, grid_for,  # noqa: E402
                    mosaic_point_to_mm, survey)

OLLAMA = "http://127.0.0.1:11434/api/chat"
MODEL = "muse-glimmer:30b"
LOG = ROOT / "runs" / "hume_calls.jsonl"
MEMORY = ROOT / "runs" / "hume_memory.jsonl"
SURVEYS = ROOT / "runs" / "hume_surveys.json"  # the board mosaics taken so far, kept across restarts
CONFIG = ROOT / "config" / "machine.json"
MARGIN_MM = {"X": 10.0, "Y": 10.0, "Z": 3.0}  # kept clear of the travel ends Erlan measured


def _workspace():
    """Allowed area in bench mm: the travel Erlan measured (config/machine.json) minus a margin."""
    travel = json.loads(CONFIG.read_text(encoding="utf-8")).get("workspace_mm")
    if not travel:
        return {"X": (-80.0, 80.0), "Y": (-80.0, 80.0), "Z": (-3.0, 12.0)}
    return {a: (travel[a][0] + MARGIN_MM[a], travel[a][1] - MARGIN_MM[a]) for a in "XYZ"}


WORKSPACE = _workspace()
LIFT_MM = 3.0
MAX_ROUNDS = 10
MAX_TILES = 80  # one board survey: about 9 x 9 pictures, some 75 x 95 mm
MAX_PART_TILES = 12
BOARD_MARGIN_MM = 2.0
PART_MARGIN_MM = 1.5
PART_RUNS_OFF_MM = 8.0  # how much further to look where a part runs off the edge of the mosaic
SENTENCE_END = re.compile(r"[.!?¡¿]\s|[.!?]$")
PROMISE = re.compile(r"\b(voy a|vamos a|d[ée]jame|en un (momento|segundo|instante)|ahora mismo|intentar[ée]|usar[ée])\b",
                     re.IGNORECASE)
BOOKKEEPING = ("origin_px", "size_px", "shift_px_per_mm", "home_mm", "fov_mm", "step_mm")

SYSTEM = """Eres Lunes, el asistente del banco de electrónica de Erlan, un ingeniero electrónico. Te llamas Lunes
como el asistente de Iron Man se llama Viernes. Tu memoria se llama Hume, por el filósofo David Hume: solo crees lo que
observaste, y cada cosa que recuerdas guarda la evidencia que la respalda.
Tienes una CNC con un microscopio montado en el eje Z y un Kinect que mira la máquina. El microscopio ve unos
12 x 15 mm, así que moverte menos de 5 mm casi no cambia lo que ves. Coordenadas en milímetros sobre el origen del
banco; Z positivo sube.

Cómo trabajas: eres un agente, haces el trabajo completo con tus herramientas, no pasitos.
- Para ver o fotografiar una placa entera: measure_board encuentra su perímetro desde donde estás (si no ve placa
  debajo, la busca alrededor) y te dice cuánto mide; después survey_board fotografía exactamente esa área. Dile a
  Erlan cuánto mide la placa.
- Si Erlan puso otra placa, la movió o le dio la vuelta, llama measure_board con new_board=true: olvida la placa y
  la altura de foco anteriores y busca la superficie nueva desde más arriba. Si la placa no cambió, no la vuelvas a
  medir ni a recorrer: usa lo que ya tienes.
- Para volver a ver un mosaico que ya tomaste usa look_survey; no saques las fotos de nuevo. Cada survey_board te
  dice su número y su cara. look_survey con number 0 te muestra juntas la última foto de cada cara (frente y
  atrás), para explicar de qué trata la placa.
- Si Erlan te pide explicar o describir, explica con lo que ya viste; no repitas el trabajo.
- Si measure_board no encuentra la placa, mira con look_kinect o pregúntale a Erlan dónde está. Si Erlan te dice
  dónde está o cuánto mide, úsalo: survey_board con width_mm y height_mm.
- Para identificar o leer un integrado u otro componente: inspect_component con un punto sobre él en el mosaico. Lo
  fotografía entero, de cerca y enfocado en su parte de arriba, y te devuelve la imagen para leer su marca. No
  intentes leerlo moviéndote de a poco.
- Si te falta algo que Erlan sabe (dónde está la placa, qué es, qué quiere), pregúntale en una frase.

Reglas:
- Responde en español, corto y natural, para decirlo en voz alta: una a tres frases, sin listas, sin markdown.
- Antes de afirmar algo sobre la placa o el banco, míralo con una herramienta.
- Nunca anuncies una acción ("voy a mirar", "en un momento"): hazla llamando la herramienta en ese mismo turno.
- Solo crees lo que observaste. Distingue "vi" de "supongo", y di cuando la imagen no alcanza para saberlo. Al leer
  una marca, di qué caracteres no se ven claros en vez de inventarlos.
- Tu memoria viene de sesiones anteriores y puede ser de otra placa: dila como recuerdo ("recuerdo que...") y
  compruébala mirando antes de usarla. Si lo que ves ahora la contradice, cree lo que ves.
- Para moverte en X o Y no tienes que subir Z: la herramienta lo hace sola. No intentes bajar Z más allá del foco.
- Nunca inventes coordenadas. Para ir a algo que viste en el mosaico usa go_to_mosaic_point con su posición en la
  imagen (fracciones de 0 a 1).
- Solo atiendes a Erlan. Si lo que oyes no parece dirigido a ti (alguien más que habla en la habitación), no hagas
  nada y responde solo "(no era para mí)".
- Si Erlan pide algo peligroso o fuera del área permitida, dilo y no lo hagas.
- Guarda con remember los hechos importantes que descubras, con la evidencia que los respalda.
"""

TOOLS = [
    {"type": "function", "function": {
        "name": "where", "description": "Current machine position (mm), the focus height and the allowed area.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "move_to", "description": "Move to absolute machine coordinates in mm; leave out an axis to keep it.",
        "parameters": {"type": "object", "properties": {
            "x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}}}}},
    {"type": "function", "function": {
        "name": "move_by", "description": "Move relative to the current position, in mm.",
        "parameters": {"type": "object", "properties": {
            "dx": {"type": "number"}, "dy": {"type": "number"}, "dz": {"type": "number"}}}}},
    {"type": "function", "function": {
        "name": "look_microscope", "description": "Take a picture through the microscope at the current position.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "look_kinect", "description": "Take a picture of the whole machine and bench with the Kinect.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "autofocus",
        "description": "Focus the microscope. The board's surface is in focus near the focus height; the top of a "
                       "tall part is higher, so give up_mm to also search above (about 3 to 4 mm for a DIP chip). "
                       "Z never goes below the board's focus height.",
        "parameters": {"type": "object", "properties": {
            "up_mm": {"type": "number", "description": "how far above the current height to search, 0.3 to 5"}}}}},
    {"type": "function", "function": {
        "name": "measure_board",
        "description": "Find the outline of the circuit board from the current position: walk to each side until "
                       "its edges show up (searching around first if there is no board under the microscope). "
                       "Returns the board's size and position in mm. Takes some 20 to 90 seconds.",
        "parameters": {"type": "object", "properties": {
            "new_board": {"type": "boolean",
                          "description": "true when Erlan changed, moved or flipped the board since the last one"}}}}},
    {"type": "function", "function": {
        "name": "survey_board",
        "description": "Photograph the whole board that measure_board found (it measures first if needed) and "
                       "stitch one mosaic. With width_mm and height_mm, photograph an area of that size centred on "
                       "the current position instead.",
        "parameters": {"type": "object", "properties": {
            "width_mm": {"type": "number", "description": "size along X, only when Erlan gave it"},
            "height_mm": {"type": "number", "description": "size along Y, only when Erlan gave it"}}}}},
    {"type": "function", "function": {
        "name": "look_survey",
        "description": "See again a mosaic taken earlier in this session, without moving: number 1 is the first "
                       "survey_board, -1 the last one, -2 the one before it. Lists them all when no number is given.",
        "parameters": {"type": "object", "properties": {
            "number": {"type": "integer", "description": "0 shows the last mosaic of each face side by side"}}}}},
    {"type": "function", "function": {
        "name": "inspect_component",
        "description": "Photograph one component (a chip, a connector, a module) entirely, close up and focused on "
                       "its top, to read its marking. Give a point on it in the last survey_board mosaic, as "
                       "fractions of that image: x from the left edge, y from the top edge (0 to 1).",
        "parameters": {"type": "object", "properties": {
            "x": {"type": "number"}, "y": {"type": "number"}}, "required": ["x", "y"]}}},
    {"type": "function", "function": {
        "name": "go_to_mosaic_point",
        "description": "Move the microscope over a point of the last survey_board mosaic, given as fractions of "
                       "that mosaic image: x from the left edge (0 to 1), y from the top edge (0 to 1).",
        "parameters": {"type": "object", "properties": {
            "x": {"type": "number"}, "y": {"type": "number"}}, "required": ["x", "y"]}}},
    {"type": "function", "function": {
        "name": "remember", "description": "Save a fact for later, with the observation that supports it.",
        "parameters": {"type": "object", "properties": {
            "fact": {"type": "string"}, "evidence": {"type": "string"}}, "required": ["fact", "evidence"]}}},
]


def jpeg_b64(image, max_side=1280, quality=88):
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


def _mm(d):
    return {k: round(float(v), 3) for k, v in d.items()}


class Hume:
    def __init__(self, machine, microscope, kinect=None, show=None, model=MODEL):
        self.m, self.cam, self.kinect = machine, microscope, kinect
        self.show = show or (lambda frame, text: None)
        self.model = model
        self.focus_z = None
        self.board = None  # the last measure_board result
        self.last_survey, self.last_mosaic = None, None
        self.surveys = []  # every board mosaic: number, face, size, file
        self.side = 1  # which face of the board is up; a new_board measure after a flip counts the next one
        if SURVEYS.exists():
            saved = json.loads(SURVEYS.read_text(encoding="utf-8"))
            self.surveys, self.side = saved["surveys"], saved["side"]
        self.stop = threading.Event()  # set by the ear when Erlan says "para"
        self.messages = [{"role": "system", "content": SYSTEM + self._memory_text()}]

    # Focus ------------------------------------------------------------------------------------------------
    def _scan_focus(self, top, bottom):
        """Focus search from top down to bottom: coarse steps, then fine; ends at the sharpest height."""
        def scan(hi, lo, step):
            best = (-1.0, hi)
            z = hi
            while z >= lo - 1e-9:  # top to bottom, so backlash is always taken the same way
                self.m.move_to(Z=z)
                s = sharpness(self.cam.fresh_gray(2))
                self.show(self.cam.frame(), f"enfocando  Z {z:+.2f}  nitidez {s:.0f}")
                best = max(best, (s, z))
                z -= step
            return best

        s1, z1 = scan(top, bottom, 0.25)
        s2, z2 = scan(min(z1 + 0.25, top), max(bottom, z1 - 0.25), 0.05)
        self.m.move_to(Z=min(z2 + 0.3, top))
        self.m.move_to(Z=z2)
        return z2, s2

    def _focus_up(self, up_mm):
        """Focus on something up to up_mm above the current height (the top of a tall part), down to the floor."""
        here = self.m.position()["Z"]
        return self._scan_focus(min(here + up_mm, WORKSPACE["Z"][1]), max(self._z_floor(), here - 0.3))

    def _focus_board(self, above_mm=0.5):
        """Puts the board's surface in focus: at the known focus height, or found by searching from above_mm over
        the current height down to the floor."""
        if self.focus_z is not None:
            self.m.move_to(Z=self.focus_z)
            return self.focus_z
        # The bench's Z origin was set by eye: the board may sit a little lower, so look 1.5 mm below it too.
        here = self.m.position()["Z"]
        z, _ = self._scan_focus(min(here + above_mm, WORKSPACE["Z"][1]), max(self._z_floor() - 1.5, WORKSPACE["Z"][0]))
        self.focus_z = z
        return z

    # Memory ------------------------------------------------------------------------------------------------
    def _memory_text(self, last=15):
        if not MEMORY.exists():
            return ""
        notes = [json.loads(line) for line in MEMORY.read_text(encoding="utf-8").splitlines() if line.strip()][-last:]
        if not notes:
            return ""
        lines = "\n".join(f"- {n['fact']} (evidencia: {n['evidence']}; {n['at']})" for n in notes)
        return ("\nLo que recuerdas de sesiones anteriores. Puede ser de otra placa que ya no está en la máquina: "
                "compruébalo mirando antes de decirlo como cierto.\n" + lines + "\n")

    # Moves -------------------------------------------------------------------------------------------------
    def _z_floor(self):
        return self.focus_z if self.focus_z is not None else 0.0

    def _check(self, target):
        for axis, v in target.items():
            lo, hi = WORKSPACE[axis]
            if axis == "Z":
                lo = max(lo, self._z_floor())
            if not lo - 0.05 <= v <= hi + 0.05:  # positions come back rounded to whole steps (Y: 0.025 mm)
                return f"{axis}={v:.2f} mm está fuera del área permitida ({lo:.2f} a {hi:.2f} mm)"
        return None

    def _move(self, x=None, y=None, z=None):
        here = self.m.position()
        target = {"X": here["X"] if x is None else float(x), "Y": here["Y"] if y is None else float(y),
                  "Z": here["Z"] if z is None else float(z)}
        problem = self._check(target)
        if problem:
            return {"ok": False, "error": problem}
        if abs(target["X"] - here["X"]) > 1e-3 or abs(target["Y"] - here["Y"]) > 1e-3:
            safe_z = max(here["Z"], target["Z"], self._z_floor() + LIFT_MM)
            self.m.move_to(Z=min(safe_z, WORKSPACE["Z"][1]))
            self.m.move_to(X=target["X"], Y=target["Y"])
        self.m.move_to(Z=target["Z"])
        return {"ok": True, "position_mm": _mm(self.m.position())}

    # Skills ------------------------------------------------------------------------------------------------
    def _measure(self, new_board=False):
        above = 0.5
        if not new_board and self.board:
            return dict(self.board, note="already measured and the board has not changed; if Erlan changed, moved "
                                         "or flipped it, call measure_board with new_board=true")
        if new_board:  # another board, or this one flipped: its surface may be higher or lower than the last one
            if self.surveys:
                self.side += 1
            self.focus_z, self.board, self.last_survey, self.last_mosaic = None, None, None, None
            above = 8.0
            self.m.move_to(Z=min(self.m.position()["Z"] + above, WORKSPACE["Z"][1]))
            above = 0.0
        z = self._focus_board(above)
        found = find_perimeter(self.m, self.cam, z, self.m.config["microscope"], WORKSPACE, self.show,
                               stop=self.stop.is_set)
        if found["found"]:
            self.board = found
            found["note"] = "the microscope is now over the board's centre; tell Erlan the size"
        else:
            found["note"] = "no board under the microscope nor around it: look with the Kinect or ask Erlan where it is"
        return found

    def _survey_board(self, width_mm=None, height_mm=None):
        cal = self.m.config["microscope"]
        if width_mm and height_mm:
            here = self.m.position()
            center, size = (here["X"], here["Y"]), (float(width_mm), float(height_mm))
        else:
            if not self.board:
                found = self._measure()
                if not found["found"]:
                    return {"ok": False, "error": "no board found under or around the microscope", "measure": found}, None
            center = tuple(self.board["center_mm"])
            size = tuple(s + 2 * BOARD_MARGIN_MM for s in self.board["size_mm"])
        cols, rows = grid_for(*size, cal)
        note = None
        if cols * rows > MAX_TILES:
            k = math.sqrt(MAX_TILES / (cols * rows))
            cols, rows = max(1, int(cols * k)), max(1, int(rows * k))
            note = "the board is bigger than one survey can cover: only its central part was photographed"
        moved = self._move(*center)
        if not moved["ok"]:
            return moved, None
        self._focus_board()
        mosaic, _, summary = survey(self.m, self.cam, span=0.3, show=self.show, recalibrate=False, cols=cols,
                                    rows=rows, stop=self.stop.is_set)
        self.focus_z = summary["focus_z_mm"] if self.focus_z is None else min(self.focus_z, summary["focus_z_mm"])
        self.last_survey, self.last_mosaic = summary, mosaic
        result = {k: v for k, v in summary.items() if k not in BOOKKEEPING}
        result["pictures"] = cols * rows
        result["survey_number"] = len(self.surveys) + 1
        result["face"] = f"face {self.side} of this board (face 2 is the other side after Erlan flipped it)"
        self.surveys.append({"number": result["survey_number"], "face": self.side, "summary": summary,
                             "board_size_mm": self.board["size_mm"] if self.board else None,
                             "at": datetime.now().strftime("%Y-%m-%d %H:%M")})
        SURVEYS.write_text(json.dumps({"surveys": self.surveys, "side": self.side}, default=float), encoding="utf-8")
        if self.board and not (width_mm and height_mm):
            result["board_size_mm"] = self.board["size_mm"]
        if note:
            result["note"] = note
        return result, mosaic

    def _inspect(self, x, y):
        if self.last_mosaic is None:
            return {"ok": False, "error": "no mosaic yet: call survey_board first"}, None
        s, mosaic = self.last_survey, self.last_mosaic
        H, W = mosaic.shape[:2]
        ppm = s["px_per_mm"]
        ppm = float(np.mean(list(ppm.values()) if isinstance(ppm, dict) else ppm))
        box = component_box(mosaic, x, y)
        if box is None:  # no part body found there: take 12 mm around the point
            half = 6.0 * ppm
            x0, y0, x1, y1 = x - half / W, y - half / H, x + half / W, y + half / H
        else:
            x0, y0, x1, y1 = box
            run_x, run_y = PART_RUNS_OFF_MM * ppm / W, PART_RUNS_OFF_MM * ppm / H
            x0, y0 = x0 - (run_x if x0 <= 0.01 else 0), y0 - (run_y if y0 <= 0.01 else 0)
            x1, y1 = x1 + (run_x if x1 >= 0.99 else 0), y1 + (run_y if y1 >= 0.99 else 0)
        mx, my = PART_MARGIN_MM * ppm / W, PART_MARGIN_MM * ppm / H
        corners = [mosaic_point_to_mm(s, a, b) for a in (x0 - mx, x1 + mx) for b in (y0 - my, y1 + my)]
        xs, ys = [c[0] for c in corners], [c[1] for c in corners]
        center = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
        size = (max(xs) - min(xs), max(ys) - min(ys))
        cols, rows = grid_for(*size, self.m.config["microscope"])
        while cols * rows > MAX_PART_TILES:
            cols, rows = (cols - 1, rows) if cols >= rows else (cols, rows - 1)
        moved = self._move(*center, z=self.focus_z)
        if not moved["ok"]:
            return moved, None
        top_z, _ = self._focus_up(5.0)
        close, folder, _ = survey(self.m, self.cam, span=0.15, show=self.show, recalibrate=False, cols=cols,
                                  rows=rows, stop=self.stop.is_set, register=True)
        image = closeup_for_reading(close)
        cv2.imwrite(str(folder / "closeup.png"), image)
        self.m.move_to(Z=max(top_z, self._z_floor()) + LIFT_MM)  # stay clear of the part afterwards
        return {"part_found_in_mosaic": box is not None,
                "part_size_mm": [round(size[0] - 2 * PART_MARGIN_MM, 1), round(size[1] - 2 * PART_MARGIN_MM, 1)],
                "center_mm": [round(center[0], 2), round(center[1], 2)],
                "top_height_above_board_mm": round(top_z - self._z_floor(), 1), "pictures": cols * rows,
                "note": "two copies of the part, the second turned upside down: read the marking in whichever is "
                        "upright, character by character, and say which characters are not clear"}, image

    def _faces_side_by_side(self, listing):
        """The last mosaic of each face of the board, next to each other at the same height."""
        last = {}
        for v in self.surveys:
            last[v["face"]] = v
        pictures = []
        for face in sorted(last):
            img = cv2.imread(last[face]["summary"]["mosaic"])
            img = cv2.resize(img, None, fx=900 / img.shape[0], fy=900 / img.shape[0], interpolation=cv2.INTER_AREA)
            label = np.full((50, img.shape[1], 3), 255, np.uint8)
            cv2.putText(label, f"cara {face}", (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
            pictures.append(np.vstack([label, img]))
            pictures.append(np.full((950, 20, 3), 255, np.uint8))
        return {"faces": [{"face": f, "survey": last[f]["number"], "board_size_mm": last[f]["board_size_mm"]}
                          for f in sorted(last)], "face_up_now": self.side, "surveys": listing}, np.hstack(pictures[:-1])

    # Tools -------------------------------------------------------------------------------------------------
    def run_tool(self, name, args):
        """Runs one tool; returns (result for the model, image to show the model or None)."""
        if name == "where":
            return {"position_mm": _mm(self.m.position()), "focus_z_mm": self.focus_z, "area_mm": WORKSPACE,
                    "board": self.board and {k: self.board[k] for k in ("size_mm", "center_mm")}}, None
        if name == "move_to":
            return self._move(args.get("x"), args.get("y"), args.get("z")), None
        if name == "move_by":
            here = self.m.position()
            return self._move(here["X"] + float(args.get("dx", 0)), here["Y"] + float(args.get("dy", 0)),
                              here["Z"] + float(args.get("dz", 0))), None
        if name == "look_microscope":
            self.cam.fresh_gray(1)
            frame = self.cam.frame()
            s = sharpness(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
            self.show(frame, f"microscopio  nitidez {s:.0f}")
            return {"position_mm": _mm(self.m.position()), "sharpness": round(s),
                    "note": "below about 300 the image is blurry; it shows about 12 x 15 mm"}, frame
        if name == "look_kinect":
            if self.kinect is None:
                return {"ok": False, "error": "the Kinect is not connected"}, None
            frame = self.kinect.color()
            return {"note": "the Kinect sees the whole machine; the microscope is the black tube with a lit ring at "
                            "its tip"}, frame
        if name == "autofocus":
            up = float(min(max(args.get("up_mm", 0.3), 0.3), 5.0))
            if up <= 0.3:
                z, s = autofocus(self.m, self.cam, 0.3, lambda t: self.show(self.cam.frame(), t))
            else:
                z, s = self._focus_up(up)
            # The floor stays at the board's focus height: focusing on the top of a tall part never lowers it.
            self.focus_z = z if self.focus_z is None else min(self.focus_z, z)
            return {"focused_z_mm": round(z, 3), "sharpness": round(s), "z_floor_mm": round(self.focus_z, 3)}, None
        if name == "measure_board":
            return self._measure(bool(args.get("new_board"))), None
        if name == "survey_board":
            return self._survey_board(args.get("width_mm"), args.get("height_mm"))
        if name == "inspect_component":
            return self._inspect(float(args["x"]), float(args["y"]))
        if name == "look_survey":
            listing = [{"number": v["number"], "face": v["face"], "board_size_mm": v["board_size_mm"], "at": v["at"]}
                       for v in self.surveys]
            if "number" not in args or not self.surveys:
                return {"surveys": listing}, None
            n = int(args["number"])
            if n == 0:
                return self._faces_side_by_side(listing)
            try:
                v = self.surveys[n - 1 if n > 0 else n]
            except IndexError:
                return {"ok": False, "error": f"there is no survey {n}", "surveys": listing}, None
            mosaic = cv2.imread(v["summary"]["mosaic"])
            # pointing at it (go_to_mosaic_point, inspect_component) only makes sense while that face is up
            if v["face"] == self.side:
                self.last_survey, self.last_mosaic = v["summary"], mosaic
            return {"number": v["number"], "face": v["face"], "board_size_mm": v["board_size_mm"],
                    "face_up_now": v["face"] == self.side}, mosaic
        if name == "go_to_mosaic_point":
            if not self.last_survey:
                return {"ok": False, "error": "no mosaic yet: call survey_board first"}, None
            x, y = mosaic_point_to_mm(self.last_survey, float(args["x"]), float(args["y"]))
            return self._move(x, y), None
        if name == "remember":
            MEMORY.parent.mkdir(exist_ok=True)
            note = {"at": datetime.now().isoformat(timespec="seconds"), "fact": args["fact"],
                    "evidence": args["evidence"]}
            with MEMORY.open("a", encoding="utf-8") as f:
                f.write(json.dumps(note, ensure_ascii=False) + "\n")
            return {"ok": True}, None
        return {"ok": False, "error": f"unknown tool {name}"}, None

    # Brain -------------------------------------------------------------------------------------------------
    def _chat(self, on_text=None):
        """One model turn, streamed. on_text gets each finished sentence as soon as it exists, so the voice can
        start talking while the model is still writing. Stops reading when Erlan interrupts."""
        # think=False: a spoken assistant has to answer in seconds, not reason at length
        body = json.dumps({"model": self.model, "messages": self.messages, "tools": TOOLS, "stream": True,
                           "think": False, "options": {"num_ctx": 8192, "temperature": 0.3}}).encode()
        req = urllib.request.Request(OLLAMA, body, {"Content-Type": "application/json"})
        t0 = time.time()
        content, calls, pending, last = "", [], "", {}
        with urllib.request.urlopen(req, timeout=300) as r:
            for line in r:
                if self.stop.is_set():
                    break
                if not line.strip():
                    continue
                last = json.loads(line)
                msg = last.get("message", {})
                calls += msg.get("tool_calls") or []
                piece = msg.get("content") or ""
                content += piece
                pending += piece
                while on_text:
                    m = SENTENCE_END.search(pending)
                    if not m:
                        break
                    on_text(pending[:m.end()].strip())
                    pending = pending[m.end():]
        if on_text and pending.strip() and not self.stop.is_set():
            on_text(pending.strip())
        LOG.parent.mkdir(exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:  # provenance: which model ran where, and how much
            f.write(json.dumps({"at": datetime.now().isoformat(timespec="seconds"), "model": last.get("model"),
                                "where": "local RTX 3090 through Ollama", "seconds": round(time.time() - t0, 2),
                                "prompt_tokens": last.get("prompt_eval_count"),
                                "output_tokens": last.get("eval_count")}) + "\n")
        out = {"role": "assistant", "content": content}
        if calls:
            out["tool_calls"] = calls
        return out

    def _trim(self, keep=24):
        """Keeps the conversation small: the system prompt, the last messages, and only the newest image."""
        head, tail = self.messages[:1], self.messages[1:][-keep:]
        while tail and tail[0].get("role") == "tool":  # never start with an orphan tool result
            tail = tail[1:]
        seen_image = False
        for msg in reversed(tail):
            if msg.get("images"):
                if seen_image:
                    msg.pop("images")
                    msg["content"] += " (imagen ya no disponible)"
                seen_image = True
        self.messages = head + tail

    def handle(self, text, on_tool=None, on_text=None):
        """One request from Erlan; runs tools until there is an answer, and returns the whole answer text.
        Setting self.stop (Erlan said "para") ends it at the next step."""
        self.stop.clear()
        self.messages.append({"role": "user", "content": text})
        used_tools, nudged, tiny_moves, said = False, set(), 0, []
        speak = (lambda t: (said.append(t), on_text(t))) if on_text else None
        for _ in range(MAX_ROUNDS):
            if self.stop.is_set():
                break
            self._trim()
            msg = self._chat(speak)
            self.messages.append(msg)
            if self.stop.is_set():
                break
            calls = msg.get("tool_calls") or []
            if not calls:
                answer = (msg.get("content") or "").strip()
                # Models like to announce an action ("voy a tomar una foto") and stop there. That is a promise,
                # not an observation: make it act now, once.
                if not used_tools and "promise" not in nudged and PROMISE.search(answer):
                    nudged.add("promise")
                    self.messages.append({"role": "user", "content": "(Sistema: no anuncies lo que vas a hacer. "
                                          "Llama ahora la herramienta que corresponde y responde con lo que veas.)"})
                    continue
                return " ".join(said) if on_text else answer
            used_tools = True
            for call in calls:
                name = call["function"]["name"]
                args = call["function"].get("arguments") or {}
                if isinstance(args, str):
                    args = json.loads(args or "{}")
                if name == "move_by" and math.hypot(float(args.get("dx", 0)), float(args.get("dy", 0))) < 3:
                    tiny_moves += 1
                if on_tool:
                    on_tool(name, args)
                try:
                    result, image = self.run_tool(name, args)
                except Interrupted:
                    self.stop.set()
                    result, image = {"ok": False, "error": "Erlan stopped it"}, None
                except Exception as e:  # a failed tool is information for the model, not a crash
                    result, image = {"ok": False, "error": str(e)}, None
                self.messages.append({"role": "tool", "tool_name": name,
                                      "content": json.dumps(result, ensure_ascii=False, default=float)})
                if image is not None:
                    self.messages.append({"role": "user", "content": f"(imagen tomada por {name})",
                                          "images": [jpeg_b64(image)]})
                if self.stop.is_set():
                    break
            # Inching along a few tenths of a millimetre never shows a whole part: point to the real skills.
            if tiny_moves >= 2 and "tiny" not in nudged:
                nudged.add("tiny")
                self.messages.append({"role": "user", "content": "(Sistema: te estás moviendo de a poco y el "
                                      "microscopio ve unos 12 x 15 mm. Para ver un componente entero usa "
                                      "inspect_component; para toda la placa, measure_board y survey_board.)"})
        if self.stop.is_set():
            self.messages.append({"role": "user", "content": "(Sistema: Erlan te interrumpió; quedó a medias.)"})
            return "Listo, me detengo."
        return "No alcancé a terminar eso con los pasos que tengo. Pídemelo de otra forma."
