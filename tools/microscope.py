"""Live microscope view on the bench CNC: nudge the axes, autofocus with Z and calibrate millimetres per pixel.

Keys in the image window: arrows move X and Y by the current step, Page Up / Page Down move Z, + and - change the
step, f autofocuses, c calibrates (moves X and then Y by 3 mm and measures how far the image shifts), s saves a
snapshot, q or Esc quits.

    python tools/microscope.py COM8 --camera 0
"""
import argparse
import math
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lunes.machine import Machine  # noqa: E402

CAPTURES = Path(__file__).resolve().parent.parent / "captures"
STEPS_MM = (0.01, 0.1, 1.0, 5.0)
CAL_MOVE_MM = 3.0  # about 140 px: long enough for a precise ratio, short enough to keep most of the view
KEY_LEFT, KEY_UP, KEY_RIGHT, KEY_DOWN = 2424832, 2490368, 2555904, 2621440  # cv2.waitKeyEx on Windows
KEY_PGUP, KEY_PGDN = 2162688, 2228224


class Camera:
    """The microscope, read continuously on its own thread. frame() is always the newest picture, so anyone (the
    dashboard too) can look at any moment; fresh_gray() waits for pictures taken after it was called."""

    def __init__(self, index):
        self._cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        # 640x480 is the MS5's real resolution; its larger modes are the same image stretched.
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self._cap.isOpened():
            raise RuntimeError(f"no camera at index {index}")
        self.cap = self  # older callers release the camera with cam.cap.release()
        self._new = threading.Condition()
        self._latest, self._count, self._alive = None, 0, True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        self._wait_for(1, timeout=5)

    def _reader(self):
        while self._alive:
            ok, f = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._new:
                self._latest, self._count = f, self._count + 1
                self._new.notify_all()

    def _wait_for(self, count, timeout=3.0):
        with self._new:
            if not self._new.wait_for(lambda: self._count >= count, timeout):
                raise RuntimeError("the camera stopped sending frames")
            return self._latest

    def frame(self):
        with self._new:
            f = self._latest
        if f is None:
            raise RuntimeError("the camera stopped sending frames")
        return f.copy()

    def fresh_gray(self, n=3):
        """Grey frame taken after the machine stops: skips the frame being exposed during the call (it may have
        started while moving), then averages n new ones to cut noise."""
        start = self._count
        frames = [cv2.cvtColor(self._wait_for(start + 2 + k), cv2.COLOR_BGR2GRAY).astype(np.float32) for k in range(n)]
        return sum(frames) / n

    def release(self):
        self._alive = False
        self._thread.join(timeout=1)
        self._cap.release()


def sharpness(gray):
    h, w = gray.shape
    center = gray[h // 4:3 * h // 4, w // 4:3 * w // 4]
    return float(cv2.Laplacian(center, cv2.CV_32F).var())


def autofocus(m, cam, span_mm, show):
    """Scans Z around the current height, coarse then fine, and parks at the sharpest image."""
    z0 = m.position()["Z"]

    def scan(center, half, step):
        best = (-1.0, center)
        z = center + half
        while z >= center - half - 1e-9:  # always top to bottom, so backlash is taken the same way
            m.move_to(Z=z)
            s = sharpness(cam.fresh_gray(2))
            show(f"enfocando  Z {z:+.3f}  nitidez {s:.0f}")
            best = max(best, (s, z))
            z -= step
        return best

    s1, z1 = scan(z0, span_mm, 0.25)
    s2, z2 = scan(z1, 0.25, 0.05)
    m.move_to(Z=z2 + 0.3)  # approach the final height from above as well
    m.move_to(Z=z2)
    return z2, s2


def calibrate(m, cam, move_by=None):
    """Measures how many pixels the image moves per millimetre of X and of Y, and the camera's rotation.

    move_by(axis, mm) does the moves; the default is a plain move, a caller can lift Z around it instead."""
    move_by = move_by or (lambda axis, mm: m.move_by(**{axis: mm}))
    ref = cam.fresh_gray()
    window = cv2.createHanningWindow(ref.shape[::-1], cv2.CV_32F)
    vectors = {}
    for axis in ("X", "Y"):
        move_by(axis, CAL_MOVE_MM)
        img = cam.fresh_gray()
        (dx, dy), response = cv2.phaseCorrelate(ref, img, window)
        move_by(axis, -CAL_MOVE_MM)
        if response < 0.05:
            raise RuntimeError(f"the image shift for {axis} is not clear (response {response:.2f}); "
                               "focus on a textured area first")
        vectors[axis] = (dx / CAL_MOVE_MM, dy / CAL_MOVE_MM)
    vx, vy = vectors["X"], vectors["Y"]
    between = math.degrees(math.acos((vx[0] * vy[0] + vx[1] * vy[1]) / (math.hypot(*vx) * math.hypot(*vy))))
    result = {
        "resolution": [ref.shape[1], ref.shape[0]],
        "px_per_mm": {"X": round(math.hypot(*vx), 3), "Y": round(math.hypot(*vy), 3)},
        "image_shift_px_per_mm": {"X": [round(v, 3) for v in vx], "Y": [round(v, 3) for v in vy]},
        "x_axis_angle_deg": round(math.degrees(math.atan2(vx[1], vx[0])), 2),
        "angle_between_axes_deg": round(between, 2),
        "calibrated_at": datetime.now().isoformat(timespec="seconds"),
    }
    m.config["microscope"] = result
    m.save_config()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", default="COM8")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--focus-span", type=float, default=1.5, help="autofocus searches this many mm up and down")
    args = ap.parse_args()

    cam = Camera(args.camera)
    m = Machine(args.port)
    step_i = 1
    message = "flechas X/Y, RePag/AvPag Z, +/- paso, f enfocar, c calibrar, s foto, q salir"
    win = "Lunes - microscopio"
    cv2.namedWindow(win)

    def draw(frame, text, pump=False):
        h, w = frame.shape[:2]
        cv2.line(frame, (w // 2 - 20, h // 2), (w // 2 + 20, h // 2), (0, 255, 255), 1)
        cv2.line(frame, (w // 2, h // 2 - 20), (w // 2, h // 2 + 20), (0, 255, 255), 1)
        pos = m.position()
        line1 = "  ".join(f"{n} {v:8.3f}" for n, v in pos.items()) + f"   paso {STEPS_MM[step_i]} mm"
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        for y, t in ((22, line1), (44, f"nitidez {sharpness(gray):.0f}"), (h - 12, text)):
            cv2.putText(frame, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(frame, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(win, frame)
        if pump:  # refresh the window during long actions (autofocus)
            cv2.waitKey(1)

    try:
        while True:
            frame = cam.frame()
            draw(frame, message)
            key = cv2.waitKeyEx(15)
            if key in (27, ord("q")) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            step = STEPS_MM[step_i]
            moves = {KEY_LEFT: {"X": -step}, KEY_RIGHT: {"X": step}, KEY_UP: {"Y": step}, KEY_DOWN: {"Y": -step},
                     KEY_PGUP: {"Z": step}, KEY_PGDN: {"Z": -step}}
            try:
                if key in moves and all(a in m.axes for a in moves[key]):
                    m.move_by(**moves[key])
                elif key in (ord("+"), ord("=")):
                    step_i = min(step_i + 1, len(STEPS_MM) - 1)
                elif key == ord("-"):
                    step_i = max(step_i - 1, 0)
                elif key == ord("f"):
                    z, s = autofocus(m, cam, args.focus_span, lambda t: draw(cam.frame(), t, pump=True))
                    message = f"enfocado en Z {z:.3f} mm (nitidez {s:.0f})"
                elif key == ord("c"):
                    draw(cam.frame(), f"calibrando: mueve X y Y {CAL_MOVE_MM:g} mm y vuelve...", pump=True)
                    r = calibrate(m, cam)
                    message = (f"{r['px_per_mm']['X']:.1f} px/mm en X, {r['px_per_mm']['Y']:.1f} px/mm en Y, "
                               f"giro {r['x_axis_angle_deg']:.1f} grados (guardado)")
                    print("calibración:", r)
                elif key == ord("s"):
                    CAPTURES.mkdir(exist_ok=True)
                    path = CAPTURES / f"micro_{datetime.now():%Y%m%d_%H%M%S}.png"
                    cv2.imwrite(str(path), cam.frame())
                    message = f"foto guardada: {path.name}"
            except (RuntimeError, TimeoutError) as e:
                message = f"error: {e}"
            m.release_if_idle()
    finally:
        m.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
