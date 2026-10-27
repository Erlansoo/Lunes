"""Asks Muse Spark about what the microscope sees (or about an image file).

    python tools/ask_muse.py "What components do you see? Read any markings."
    python tools/ask_muse.py "Is this solder joint good?" --image captures/micro_x.png
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lunes.meta_api import LOG, MetaModels  # noqa: E402

SYSTEM = ("You are Lunes, an assistant on an electronics repair bench. You see a circuit board through a USB "
          "microscope whose image covers about 14 x 10 mm. Describe only what you can actually see, and say when "
          "something is too blurry or too small to read.")


def grab(camera):
    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for _ in range(10):  # let the auto exposure settle
        ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"no frame from camera {camera}")
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--image", help="an image file instead of a live microscope frame")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()
    image = cv2.imread(args.image) if args.image else grab(args.camera)
    print(MetaModels().ask(args.question, [image], system=SYSTEM, purpose="ask_muse"))
    print(f"\n(registro de la llamada en {LOG})")


if __name__ == "__main__":
    main()
