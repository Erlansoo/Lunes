"""Shows the Kinect's colour and depth side by side, to aim it at the bench.

Keys: s saves both frames, q or Esc quits. The text shows the depth in millimetres at the centre of the image. The
Kinect is aimed by hand: Lunes' driver does not move the tilt motor.

    python tools/kinect_view.py
"""
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lunes.kinect import Kinect, depth_colormap  # noqa: E402

CAPTURES = ROOT / "captures"


def main():
    try:
        k = Kinect()
    except RuntimeError as e:
        sys.exit(f"Kinect: {e}")
    try:
        while True:
            color, depth = k.color(), k.depth()
            h, w = depth.shape
            center = depth[h // 2 - 2:h // 2 + 3, w // 2 - 2:w // 2 + 3]
            known = center[center > 0]
            mm = f"{int(np.median(known))} mm" if known.size else "sin dato"
            view = np.hstack([color, depth_colormap(depth)])
            for x in (w // 2, w + w // 2):
                cv2.drawMarker(view, (x, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 24, 1)
            text = f"centro: {mm}   s = foto, q = salir"
            cv2.putText(view, text, (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(view, text, (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("Lunes - Kinect", view)
            key = cv2.waitKeyEx(30)
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                CAPTURES.mkdir(exist_ok=True)
                stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
                cv2.imwrite(str(CAPTURES / f"kinect_color_{stamp}.png"), color)
                cv2.imwrite(str(CAPTURES / f"kinect_depth_{stamp}.png"), depth)  # 16-bit PNG, millimetres
    finally:
        k.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
