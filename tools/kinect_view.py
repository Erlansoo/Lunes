"""Shows the Kinect's colour and depth side by side, to aim it at the bench.

Starts the bridge if it is not running. Keys: up / down arrows tilt the Kinect 2 degrees, s saves both frames,
q or Esc quits. The text shows the depth in millimetres at the centre of the image.

    python tools/kinect_view.py
"""
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lunes.kinect import PORT, Kinect, depth_colormap  # noqa: E402

BRIDGE = ROOT / "kinect" / "bin" / "kinect_bridge.exe"
CAPTURES = ROOT / "captures"
KEY_UP, KEY_DOWN = 2490368, 2621440


def bridge_running():
    try:
        socket.create_connection(("127.0.0.1", PORT), timeout=0.3).close()
        return True
    except OSError:
        return False


def main():
    proc = None
    if not bridge_running():
        if not BRIDGE.exists():
            sys.exit(f"build the bridge first: powershell -File {ROOT / 'kinect' / 'build.ps1'}")
        proc = subprocess.Popen([str(BRIDGE)])
        for _ in range(50):
            if bridge_running():
                break
            if proc.poll() is not None:
                sys.exit("the Kinect bridge stopped: is the Kinect plugged in (12 V adapter and USB)?")
            time.sleep(0.2)
        time.sleep(1.5)  # first frames
    k = Kinect()
    tilt = k.info()["tilt"]
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
            text = f"centro: {mm}   inclinacion {tilt} grados   flechas = inclinar, s = foto, q = salir"
            cv2.putText(view, text, (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(view, text, (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("Lunes - Kinect", view)
            key = cv2.waitKeyEx(30)
            if key in (27, ord("q")):
                break
            if key in (KEY_UP, KEY_DOWN):
                tilt = k.tilt(tilt + (2 if key == KEY_UP else -2))["deg"]
            elif key == ord("s"):
                CAPTURES.mkdir(exist_ok=True)
                stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
                cv2.imwrite(str(CAPTURES / f"kinect_color_{stamp}.png"), color)
                cv2.imwrite(str(CAPTURES / f"kinect_depth_{stamp}.png"), depth)  # 16-bit PNG, millimetres
    finally:
        k.close()
        cv2.destroyAllWindows()
        if proc:
            proc.terminate()


if __name__ == "__main__":
    main()
