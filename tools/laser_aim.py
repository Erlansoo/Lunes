"""Live microscope view with a cross at its centre and the laser on, to aim the laser by hand onto the cross.

The dot is found against a picture with the laser off, and its distance from the cross is written in mm. Nothing
moves: the motors stay disabled. Keys: + and - change the brightness, o takes a new picture with the laser off (after
the lighting changed), q or Esc quits and turns the laser off.

    python tools/laser_aim.py
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from lunes.cameras import find_camera  # noqa: E402
from lunes.laser import find_dot, too_bright  # noqa: E402
from lunes.machine import Machine  # noqa: E402
from microscope import Camera  # noqa: E402

MICROSCOPE = "vid_1224&pid_2825"


def mean_frame(cam, n=3):
    shots = []
    for _ in range(n):
        cam.fresh_gray(1)
        shots.append(cam.frame().astype(np.float32))
    return np.mean(shots, axis=0).astype(np.uint8)


def main(port="COM8"):
    index = find_camera(MICROSCOPE)
    if index is None:
        sys.exit("the microscope is not plugged in")
    cam, m = Camera(index), Machine(port)
    ppm = float(np.mean(list(m.config["microscope"]["px_per_mm"].values())))
    level, renewed = 30, 0.0  # brighter glows red over the board
    try:
        m.enable(False)
        m.laser(0)
        time.sleep(0.4)
        off = mean_frame(cam)
        while True:
            if time.time() - renewed > 20:  # the firmware puts the laser out a minute after the last command
                m.laser(level)
                renewed = time.time()
            frame = cam.frame()
            view = frame.copy()
            h, w = view.shape[:2]
            for colour, thick in (((0, 0, 0), 3), ((0, 255, 255), 1)):
                cv2.line(view, (w // 2 - 60, h // 2), (w // 2 + 60, h // 2), colour, thick, cv2.LINE_AA)
                cv2.line(view, (w // 2, h // 2 - 60), (w // 2, h // 2 + 60), colour, thick, cv2.LINE_AA)
                cv2.circle(view, (w // 2, h // 2), 12, colour, thick, cv2.LINE_AA)
            dot = find_dot(off, frame, ppm)
            if dot:
                cv2.circle(view, (int(dot["x"]), int(dot["y"])), 8, (0, 255, 0), 1, cv2.LINE_AA)
                dx, dy = (dot["x"] - w / 2) / ppm, (dot["y"] - h / 2) / ppm
                text = f"dot {np.hypot(dx, dy):.2f} mm from the cross (right {dx:+.2f}, down {dy:+.2f}), " \
                       f"{dot['size_mm']:.2f} mm wide"
            elif too_bright(off, frame, ppm):
                text = "too bright: it glows red, lower it with -"
            else:
                text = "no dot seen: aim the laser into the view"
            text += f"   laser {level}   + / - brightness, o new dark picture, q quit"
            for colour, thick in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                cv2.putText(view, text, (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, thick, cv2.LINE_AA)
            cv2.imshow("Lunes - apuntar el laser", cv2.resize(view, (w * 3 // 2, h * 3 // 2)))
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord("+"), ord("=")):
                level = min(255, level + (1 if level < 40 else 16))
                m.laser(level)
            elif key == ord("-"):
                level = max(0, level - (1 if level <= 40 else 16))
                m.laser(level)
            elif key == ord("o"):
                m.laser(0)
                time.sleep(0.4)
                off = mean_frame(cam)
                m.laser(level)
                renewed = time.time()
    finally:
        m.laser(0)
        m.close()
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(*sys.argv[1:])
