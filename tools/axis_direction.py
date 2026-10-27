"""Checks which way each axis really moves by watching the machine with the Kinect.

For each axis: average a few Kinect frames, move the axis, average again, find the part of the image that moved and
measure its shift (phase correlation) and its change in depth, then move back. The Kinect is taken to stand at the
front of the machine, so the expected directions are: X+ to the right in the image, Y+ away from the Kinect (depth
grows), Z+ up in the image. With --apply, axes that go the other way get inverted in config/machine.json; a dual
motor axis flips its second motor too, so the two keep pulling the same way.

    python tools/axis_direction.py COM8 --apply
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lunes.kinect import Kinect  # noqa: E402
from lunes.machine import CONFIG, Machine  # noqa: E402

MOVE_MM = {"X": 20.0, "Y": 20.0, "Z": 5.0}


def snapshot(k, n=6):
    grays, depths = [], []
    for _ in range(n):
        grays.append(cv2.cvtColor(k.color(), cv2.COLOR_BGR2GRAY).astype(np.float32))
        depths.append(k.depth().astype(np.float32))
        time.sleep(0.04)
    d = np.stack(depths)
    d[d == 0] = np.nan
    return sum(grays) / n, np.nanmedian(d, axis=0)


def measure(before, after):
    """Image shift (dx, dy) of the region that moved, and its median depth change in mm."""
    (g0, d0), (g1, d1) = before, after
    mask = (cv2.absdiff(g0, g1) > 10).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if mask.sum() < 40:
        return None
    # Keep only the biggest moving blob: the box around everything that changed is mostly still background,
    # which pulls a small shift (Z moves only a pixel or two at this distance) toward zero.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(mask, np.ones((9, 9), np.uint8)))
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = mask * (labels == biggest)
    ys, xs = np.nonzero(mask)
    x0, x1 = max(xs.min() - 12, 0), min(xs.max() + 12, g0.shape[1])
    y0, y1 = max(ys.min() - 12, 0), min(ys.max() + 12, g0.shape[0])
    a, b = g0[y0:y1, x0:x1], g1[y0:y1, x0:x1]
    (dx, dy), resp = cv2.phaseCorrelate(a, b, cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F))
    dz = np.nanmedian((d1 - d0)[mask > 0])
    return {"dx_px": round(float(dx), 2), "dy_px": round(float(dy), 2), "depth_mm": round(float(dz), 1),
            "response": round(float(resp), 2), "moved_px": int(mask.sum()), "box": [int(x0), int(y0), int(x1), int(y1)]}


def verdict(axis, r):
    """True if the axis moves the expected way, False if reversed, None if the evidence is too weak."""
    if r is None:
        return None
    if axis == "X":
        return r["dx_px"] > 0 if abs(r["dx_px"]) >= 1.0 else None
    if axis == "Y":
        return r["depth_mm"] > 0 if abs(r["depth_mm"]) >= 5 else (r["dy_px"] < 0 if abs(r["dy_px"]) >= 1 else None)
    return r["dy_px"] < 0 if abs(r["dy_px"]) >= 0.6 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", default="COM8")
    ap.add_argument("--axes", default="XYZ")
    ap.add_argument("--apply", action="store_true", help="invert the reversed axes in config/machine.json")
    args = ap.parse_args()

    k = Kinect()
    m = Machine(args.port)
    results = {}
    try:
        for axis in args.axes:
            if axis not in m.axes:
                continue
            d = MOVE_MM[axis]
            before = snapshot(k)
            m.move_by(**{axis: d})
            time.sleep(0.4)
            after = snapshot(k)
            m.move_by(**{axis: -d})
            time.sleep(0.4)
            back = snapshot(k)
            r = measure(before, after)
            ret = measure(before, back)
            results[axis] = (verdict(axis, r), r)
            print(f"{axis} +{d:g} mm: {json.dumps(r)}")
            print(f"   de vuelta al inicio: {'cambio residual ' + json.dumps(ret) if ret else 'imagen igual que antes'}")
    finally:
        m.close()
        k.close()

    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    changed = []
    for axis, (ok, _) in results.items():
        state = {True: "va en el sentido esperado", False: "va AL REVÉS", None: "no se pudo decidir"}[ok]
        print(f"{axis}: {state}")
        if ok is False and args.apply:
            ax = cfg["axes"][axis]
            ax["invert"] = not ax.get("invert", False)
            if "slave" in ax:  # the second motor must flip with the first or they fight
                ax["slave"]["invert"] = not ax["slave"].get("invert", False)
            changed.append(axis)
    if changed:
        CONFIG.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        print("invertidos en la configuración:", ", ".join(changed))


if __name__ == "__main__":
    main()
