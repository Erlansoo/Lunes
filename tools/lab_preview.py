"""The web lab on a made-up bench, no hardware needed: python tools/lab_preview.py, then open http://127.0.0.1:8765

A 555 blinker with a resistor ten times too big sits on a simulated bench. ngspice works out what every fault would read,
the simulated scope measures the nets the diagnosis asks for, and the page shows it all as on the real bench. No model is
called and the words of the conversation are written here. With --hantek the scope is the real one, on its calibration
output, and the rest stays made up.

    python tools/lab_preview.py
    python tools/lab_preview.py --hantek
"""
import argparse
import json
import sys
import threading
import time
import types
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lunes.circuit import with_fault  # noqa: E402
from lunes.dashboard import Dashboard  # noqa: E402
from lunes.hantek import Scope, SimulatedScope, square  # noqa: E402
from lunes.hume import Hume  # noqa: E402
from lunes.lab import state  # noqa: E402
from lunes.simulator import expected, periodic_signal, waveform  # noqa: E402

FAULT = "R2:x10"
BOARD = {"center_mm": [118.0, -152.0], "size_mm": [72.0, 54.0]}


def drawn_board(w, h):
    """A plain drawing of a board, for a PC with no pictures of a real one."""
    img = np.full((h, w, 3), (52, 92, 38), np.uint8)
    rng = np.random.default_rng(5)
    for _ in range(90):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.line(img, (x, y), (x + int(rng.integers(-180, 180)), y), (70, 124, 54), 2)
    for _ in range(26):
        x, y = int(rng.integers(0, w - 60)), int(rng.integers(0, h - 40))
        cv2.rectangle(img, (x, y), (x + 34, y + 18), (28, 28, 30), -1)
        cv2.rectangle(img, (x - 4, y + 3), (x, y + 15), (170, 170, 175), -1)
        cv2.rectangle(img, (x + 34, y + 3), (x + 38, y + 15), (170, 170, 175), -1)
    return img


def picture_of_the_board():
    surveys = sorted((ROOT / "captures").glob("survey_*/mosaic.png"))
    image = cv2.imread(str(surveys[-1])) if surveys else None
    return image if image is not None else drawn_board(1280, 960)


class Camera:
    """A window that wanders over a picture, like a head moving over a board."""

    def __init__(self, picture, size=(640, 480), step=0.0):
        self.picture, self.size, self.step, self.t0 = picture, size, step, time.time()

    def frame(self):
        h, w = self.picture.shape[:2]
        cw, ch = min(self.size[0], w), min(self.size[1], h)
        phase = (time.time() - self.t0) * self.step
        x = int((np.sin(phase) * 0.5 + 0.5) * (w - cw))
        y = int((np.sin(phase * 0.6) * 0.5 + 0.5) * (h - ch))
        return self.picture[y:y + ch, x:x + cw]

    color = frame


def real_scope_session(hume, board):
    board.add("tu", "Lunes, mide la salida de calibración con las dos puntas")
    hume.scope.calibration_output(1000)
    hume.run_tool("scope_measure", {"channel": "both"})
    board.add("tool", "scope_measure both")
    result, _ = hume.run_tool("scope_selftest", {"channel": "CH1"})
    board.add("tool", "scope_selftest CH1")
    hume.run_tool("scope_measure", {"channel": "both"})
    board.add("lunes", "Los dos canales leen 2 voltios a un kilohertz, tal como debe ser." if result.get("ok") else
              "El auto-test encontró un problema: " + "; ".join(result.get("problems", [])))


def diagnosis_session(hume, board):
    say, tool = (lambda t: board.add("lunes", t)), (lambda t: board.add("tool", t))
    board.add("tu", "Lunes, armé el blinker con el 555 pero parpadea muy lento, revísalo")
    board.busy = True
    board.status = "Diseñando el circuito y simulándolo"
    tool("circuit_design astable_555")
    hume.run_tool("circuit_design", {"template": "astable_555", "params": {"frequency_hz": 1000, "duty": 0.55,
                                                                         "supply_v": 5, "led_color": "red"}})
    tool("diagnose_start")
    board.status = "Calculando lo que leería cada falla"
    current, _ = hume.run_tool("diagnose_start", {})
    bench = with_fault(hume.circuit, FAULT)
    truth = expected(bench)

    def signal(net):  # what the probe finds on the bench: the fault can make the circuit far slower than designed
        rate, v = waveform(bench, net, truth["simulated_s"])
        return periodic_signal(rate, v, truth["nets"][net].get("frequency_hz"))

    while "next" in current and not current.get("clear") and len(current.get("readings", [])) < 6:
        net = current["next"]["net"]
        say(f"Pon la punta en {net}, que toca {', '.join(current['next']['touches'][:3])}.")
        board.status = f"Midiendo {net}"
        time.sleep(2.5)
        hume.scope = SimulatedScope({"CH1": signal(net), "CH2": lambda t: np.zeros_like(t)})
        tool(f"diagnose_measure {net}")
        current, _ = hume.run_tool("diagnose_measure", {"net": net})
        hume.m.at_mm = {"X": hume.m.at_mm["X"] + 6.0, "Y": hume.m.at_mm["Y"] - 4.0, "Z": hume.m.at_mm["Z"]}
        time.sleep(1.5)
    tool("scope_measure both")
    hume.scope = SimulatedScope({"CH1": signal("OUT"), "CH2": signal("TH")})
    hume.run_tool("scope_measure", {"channel": "both"})
    board.busy, board.status = False, ""
    say("Encontré la falla: R2 tiene diez veces su valor, y el circuito oscila diez veces más lento."
        if current.get("clear") and FAULT in current["likely"][0]["faults"] else
        "Midiendo no puedo separar estas explicaciones: hay que mirar las piezas.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hantek", action="store_true", help="use the real scope, its probes on the calibration output")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    config_path = ROOT / "config" / "machine.json"
    machine = types.SimpleNamespace(s=types.SimpleNamespace(port="COM8"), config=json.loads(config_path.read_text("utf-8")),
                                    config_path=config_path, at_mm={"X": 96.0, "Y": -138.0, "Z": 7.0})
    picture = picture_of_the_board()
    scope = Scope(probe={"CH1": 10, "CH2": 10}) if args.hantek else SimulatedScope(
        {"CH1": square(1000, 0, 2), "CH2": square(1000, 0, 2)})
    hume = Hume(machine, Camera(picture, step=0.25), scope=scope, bench=Camera(picture, (1280, 720)),
                kinect=Camera(picture))
    hume.board = dict(BOARD)
    hume.focus_z = 7.0
    hume.last_kinect_check = {"at": time.time(), "agrees": True, "off_mm": 0.6}
    board = Dashboard(camera=hume.cam, bench=hume.bench, kinect=hume.kinect, port=args.port, lab=lambda: state(hume))
    board.set_seen(picture, "mosaico de la placa")
    print(f"web lab: {board.url}  (Ctrl+C to stop)")
    session = real_scope_session if args.hantek else diagnosis_session
    threading.Thread(target=session, args=(hume, board), daemon=True).start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        if args.hantek:
            scope.close()


if __name__ == "__main__":
    main()
