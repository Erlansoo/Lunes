"""One live window with everything Lunes sees: Kinect colour, Kinect depth and the microscope.

It only looks, it never moves the machine, so it can run next to the jog window. Starts the Kinect bridge if
needed. Keys: s saves the three images to captures/, q or Esc quits.

    python tools/vision_view.py --camera 0
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from kinect_view import BRIDGE, bridge_running  # noqa: E402
from lunes.kinect import Kinect, depth_colormap  # noqa: E402
from microscope import Camera, sharpness  # noqa: E402

CAPTURES = ROOT / "captures"


def label(img, text, y=22):
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="microscope camera index")
    args = ap.parse_args()

    proc = None
    if not bridge_running():
        proc = subprocess.Popen([str(BRIDGE)])
        for _ in range(50):
            if bridge_running():
                break
            time.sleep(0.2)
        time.sleep(1.5)
    kinect = Kinect()
    micro = Camera(args.camera)
    win = "Lunes - lo que veo"
    try:
        while True:
            color, depth, scope = kinect.color(), kinect.depth(), micro.frame()
            h, w = depth.shape
            center = depth[h // 2 - 2:h // 2 + 3, w // 2 - 2:w // 2 + 3]
            known = center[center > 0]
            dview = depth_colormap(depth, 500, 3000)
            gray = cv2.cvtColor(scope, cv2.COLOR_BGR2GRAY).astype(np.float32)

            label(color, "Kinect: color")
            label(dview, f"Kinect: profundidad (centro {int(np.median(known)) if known.size else '-'} mm)")
            cv2.drawMarker(dview, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
            sv = scope.copy()
            cv2.drawMarker(sv, (sv.shape[1] // 2, sv.shape[0] // 2), (0, 255, 255), cv2.MARKER_CROSS, 30, 1)
            label(sv, f"Microscopio: nitidez {sharpness(gray):.0f}  (s = guardar, q = salir)")
            view = np.vstack([np.hstack([color, dview]), np.hstack([sv, np.zeros_like(sv)])])
            view = cv2.resize(view, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
            cv2.imshow(win, view)
            key = cv2.waitKey(30)
            if key in (27, ord("q")) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("s"):
                CAPTURES.mkdir(exist_ok=True)
                stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
                cv2.imwrite(str(CAPTURES / f"vista_kinect_{stamp}.png"), color)
                cv2.imwrite(str(CAPTURES / f"vista_profundidad_{stamp}.png"), depth)
                cv2.imwrite(str(CAPTURES / f"vista_micro_{stamp}.png"), scope)
    finally:
        kinect.close()
        micro.cap.release()
        cv2.destroyAllWindows()
        if proc:
            proc.terminate()


if __name__ == "__main__":
    main()
