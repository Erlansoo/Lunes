"""Measures the backlash of the X and Y axes with the microscope, and saves it for the machine to make up for.

A home-made CNC loses a little motion each time an axis turns round (slack in the lead screw nut, a belt that
gives): the carriage stops short of where the steps say. Here the microscope looks at the same spot reached once
coming from below and once coming from above; the shift between the two pictures, turned into millimetres with the
microscope calibration, is the backlash. It is saved as backlash_mm in config/machine.json, and lunes.machine adds
that much travel whenever an axis reverses.

First the machine finds where it really is by looking for the current picture inside the last mosaic of the board
(runs/hume_surveys.json), in case the saved position went wrong. Put the microscope over a part of the board with
detail (pads, parts), not over bare solder mask.

    python tools/backlash.py COM8
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from lunes.machine import Machine  # noqa: E402
from microscope import Camera  # noqa: E402
from survey import LIFT_MM, geometry, locate_in_mosaic  # noqa: E402

SURVEYS = ROOT / "runs" / "hume_surveys.json"
APPROACH_MM = 2.0
REPEATS = 3


def show(frame, text):
    f = frame.copy()
    for ink, width in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.putText(f, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ink, width)
    cv2.imshow("Lunes - juego de los ejes", f)
    cv2.waitKey(1)


def picture(cam, text):
    cam.fresh_gray(2)
    frame = cam.frame()
    show(frame, text)
    return frame


def relocate(m, cam):
    """Sets the machine's position from the last mosaic of the board face that is up; returns the correction."""
    if not SURVEYS.exists():
        return None
    saved = json.loads(SURVEYS.read_text(encoding="utf-8"))
    face_up = [v for v in saved["surveys"] if v["face"] == saved["side"]]
    if not face_up:
        return None
    summary = face_up[-1]["summary"]
    mosaic = cv2.imread(summary["mosaic"])
    found = locate_in_mosaic(picture(cam, "buscandome en el mosaico..."), mosaic, summary)
    if not found:
        return None
    (x, y), score, second = found
    here = m.position()
    error = float(np.hypot(x - here["X"], y - here["Y"]))
    correct = score >= 0.5 and score - second >= 0.08 and 1.0 < error <= 15.0  # as in Hume.relocate
    if correct:
        m.set_position(X=x, Y=y)
    return {"was_mm": [round(here["X"], 2), round(here["Y"], 2)], "seen_mm": [round(x, 2), round(y, 2)],
            "score": round(score, 2), "next_best": round(second, 2), "corrected": correct}


def measure(m, cam, axis, vector, z):
    """Backlash of one axis in mm: the same spot approached from below and from above."""
    m.axes[axis].backlash = 0.0  # measure the bare mechanics, not what is already made up for
    here = m.position()[axis]
    values = []
    for n in range(REPEATS):
        for direction in (-1, 1):  # first approach coming up (start below), then coming down (start above)
            m.move_to(Z=z + LIFT_MM)
            m.move_to(**{axis: here + direction * APPROACH_MM})
            m.move_to(**{axis: here})
            m.move_to(Z=z)
            frame = picture(cam, f"juego en {axis}: llegando {'subiendo' if direction < 0 else 'bajando'} ({n + 1}/{REPEATS})")
            gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32), (5, 5), 1.0)
            if direction < 0:
                from_below = gray
            else:
                (sx, sy), response = cv2.phaseCorrelate(from_below, gray, cv2.createHanningWindow((640, 480), cv2.CV_32F))
                # content shifts by vector * mm when the carriage moves; project the shift on this axis
                values.append(float(np.dot([sx, sy], vector) / np.dot(vector, vector)))
    return float(np.median(values)), values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", default="COM8")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    m, cam = Machine(args.port), Camera(args.camera)
    try:
        moved = relocate(m, cam)
        print("ubicación por el mosaico:", moved or "no encontré la imagen en el mosaico; sigo con la posición guardada")
        vx, vy, _, _ = geometry(m.config["microscope"])
        z = m.position()["Z"]
        result = {}
        for axis, vector in (("X", vx), ("Y", vy)):
            b, values = measure(m, cam, axis, vector, z)
            result[axis] = round(b, 3)
            print(f"juego en {axis}: {b:.3f} mm  (medidas {[round(v, 3) for v in values]})")
        if not args.no_save:
            for axis, b in result.items():
                m.config["axes"][axis]["backlash_mm"] = max(0.0, b)
            m.save_config()
            print("guardado en config/machine.json")
    finally:
        m.close()
        cam.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
