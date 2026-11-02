"""Reaching a point on the bench quickly, with every camera that can see it: the bench camera's wide view, the Kinect's
and the microscope's fine one, tied together on the machine's own frame. An approximation by design: the machine is
hand-built, with its squareness and belt tension, so the cameras take the head to a few mm of a thing and the
microscope (autofocus, centring) settles the rest.

The bench camera (the 4K webcam) sees the whole table but knows no millimetres. It is calibrated like the Kinect is:
the microscope head's ring light is found in its picture while the machine stands at known places over the whole
table and at two heights far apart, and a 3 x 4 camera projection is fitted to those pairs (lunes.head.fit_camera).
Two heights matter: a camera cannot be fitted from points on one plane, and the further apart they are the better the
projection reaches below them, down to the table.

The ring is not the surface: it hangs some 90 mm over whatever the microscope is focused on (its working distance),
and the machine's Z for a surface is the head's Z in focus on it. So the projection, fitted to the ring at the
machine's Z, places a surface with focus Z h at physical height h - ring_above_mm. That height is taught with one
thing the machine knows and the camera sees: the microscope centred on it (in focus, so its X, Y, Z are known) and the
bench camera's picture of it (where the model, or Erlan, says it is). The one height that puts that thing on the
camera's ray to it is kept; a correspondence that fits no height is refused. The laser's dot was tried for this and
dropped: the webcam loses it on a light table and in a breadboard's holes.

From then on, anything the webcam sees at a known focus height has machine millimetres (locate), and the laser is
checked where it is seen well, under the microscope: Hume puts the dot on the point and where it lands tells the
height of the surface there (lunes.laser), which must be the one expected.

Nothing here moves the machine; Hume does the moving and feeds the pairs in.
"""
import math

import cv2
import numpy as np

from lunes.head import fit_camera, held_out, locate

MIN_PAIRS = 8
RING = dict(bright=235, min_area=100, max_area=4000, max_w=120, max_h=80, tint=8)  # the ring light, at 960 px wide
ABOVE = dict(low=20.0, high=200.0, step=0.5, fine=0.02)  # where the ring's height over the surface in focus may lie, mm
TEACH_MAX_OFF_MM = 6.0  # a taught thing further than this from the camera's ray to it, at any height, is not that thing


def ring_in_picture(color):
    """(column, row) of the microscope's ring light in the webcam's picture, or None. The picture is scaled to 960
    wide; there the lit ring is a white patch some 30 x 20 px with the black microscope tube right above it, and the
    patch's centre is given back at the picture's own size. Other white things are told apart: a breadboard under the
    lamp or a cable has no black tube over it; the supply's red digits burn out to a warm white in the webcam, small
    patches with a tint, where the ring is grey-white through and through. Of what is left, the biggest is the ring."""
    k = 960 / color.shape[1]
    small = cv2.resize(color, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else color
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    white = (small.min(axis=2) > RING["bright"] - 30) & (gray > RING["bright"])
    n, labels, stats, centres = cv2.connectedComponentsWithStats(white.astype(np.uint8))
    best = None
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (RING["min_area"] <= area <= RING["max_area"] and w <= RING["max_w"] and h <= RING["max_h"]):
            continue
        cx, cy = centres[i]
        above = gray[max(0, y - int(1.5 * h) - 4):max(1, y - 2), max(0, int(cx) - w // 4):int(cx) + w // 4]
        darkness = float(above.mean()) if above.size else 255.0
        mean = small[labels == i].mean(axis=0)
        if darkness < 90 and np.ptp(mean) < RING["tint"] and (best is None or area > best[0]):
            best = (area, cx, cy, i)
    if best is None:
        return None
    _, cx, cy, i = best
    ys, xs = np.nonzero((labels == i) & (gray >= 250))
    if xs.size >= 4:
        cx, cy = float(xs.mean()), float(ys.mean())
    return float(cx) / k, float(cy) / k


class BenchCamera:
    """The bench camera's projection from ring pairs (where the ring light was seen and where the head was), and the
    ring's height over the surface in focus, taught from one thing seen by both the microscope and the camera."""

    def __init__(self, pairs=None, P=None, ring_above_mm=None, errors=None, taught=None):
        self.pairs = list(pairs or [])
        self.P, self.ring_above_mm, self.errors, self.taught = P, ring_above_mm, errors, taught

    def add(self, seen_uv, at_mm):
        """One ring pair: the ring light at column u, row v of the webcam, with the head at machine X, Y, Z."""
        self.pairs.append(((float(seen_uv[0]), float(seen_uv[1]), None), {"X": float(at_mm["X"]), "Y": float(at_mm["Y"]),
                                                                           "Z": float(at_mm["Z"])}))

    def fit(self):
        """Fits the projection to the ring pairs; returns {"ok", "pairs", "typical_error_mm", "worst_error_mm",
        "held_out_mm", "ring_above_mm"} or why not. A height taught before is kept and checked again on the new
        projection; without one the camera is fitted but locate() refuses until teach() is called."""
        if len(self.pairs) < MIN_PAIRS:
            return {"ok": False, "error": f"{len(self.pairs)} pairs; at least {MIN_PAIRS} spread over the table are needed"}
        spread = np.ptp([[p["X"], p["Y"]] for _, p in self.pairs], axis=0)
        if min(spread) < 60:
            return {"ok": False, "error": f"the pairs span only {spread[0]:.0f} x {spread[1]:.0f} mm: spread them over the table"}
        self.P, self.errors = fit_camera(self.pairs)
        unseen = np.array(held_out(self.pairs))
        out = {"ok": True, "pairs": len(self.pairs),
               "typical_error_mm": round(float(np.median(self.errors)), 2),
               "worst_error_mm": round(float(max(self.errors)), 2),
               "held_out_mm": round(float(np.median(np.hypot(unseen[:, 0], unseen[:, 1]))), 2)}
        if self.taught:  # the same thing, on the new projection
            t = self.taught
            again = self.teach(t["u"], t["v"], t["X"], t["Y"], t["Z"])
            if not again["ok"]:
                self.ring_above_mm, self.taught = None, None
        out["ring_above_mm"] = None if self.ring_above_mm is None else round(self.ring_above_mm, 2)
        if self.ring_above_mm is None:
            out["warning"] = ("the ring's height over the surface in focus is unknown, so nothing on the table can be "
                              "located yet: centre the microscope on a thing the bench camera also sees and "
                              "teach_bench_height with where it is in the bench picture")
        return out

    def _nearest_height(self, u, v, x, y):
        """(frame Z, picture distance in px) where the vertical through machine X, Y passes nearest the camera's ray
        through column u, row v: the projection of that vertical is a line, scanned then refined."""
        p = np.asarray(self.P, float)

        def off(z):
            pu, pv, w = p @ np.array([x, y, z, 1.0])
            return math.hypot(pu / w - u, pv / w - v)

        zs = np.arange(-ABOVE["high"] - 60.0, ABOVE["high"] + 60.0, ABOVE["step"])
        z = float(min(zs, key=off))
        fine = np.arange(z - ABOVE["step"], z + ABOVE["step"] + 1e-9, ABOVE["fine"])
        z = float(min(fine, key=off))
        return z, off(z)

    def teach(self, u, v, x, y, focus_z):
        """The ring's height over the surface in focus, from one thing the camera sees at column u, row v that the
        machine knows at X, Y on a surface of focus height focus_z (the microscope centred on it, in focus). Returns
        {"ok", "ring_above_mm", "off_mm"} or why not: the thing must lie on the camera's ray to it at some height in
        reason, else it is not the same thing (or the projection is off)."""
        if self.P is None:
            return {"ok": False, "error": "the bench camera is not calibrated: calibrate_bench first"}
        z, _ = self._nearest_height(u, v, x, y)
        gx, gy = locate(self.P, u, v, z)
        off_mm = math.hypot(gx - x, gy - y)
        above = focus_z - z
        if off_mm > TEACH_MAX_OFF_MM:
            return {"ok": False, "error": f"no height puts that point of the picture on X {x:.1f} Y {y:.1f}: the nearest "
                                          f"is {off_mm:.1f} mm off; it is not the thing under the microscope, or the "
                                          f"picture is not fresh"}
        if not (ABOVE["low"] <= above <= ABOVE["high"]):
            return {"ok": False, "error": f"that would hang the ring {above:.0f} mm over the surface, which it does not: "
                                          f"is the microscope in focus on the thing (its Z), and is that point it?"}
        self.ring_above_mm = float(above)
        self.taught = {"u": float(u), "v": float(v), "X": float(x), "Y": float(y), "Z": float(focus_z)}
        return {"ok": True, "ring_above_mm": round(above, 2), "off_mm": round(off_mm, 2)}

    def locate(self, u, v, z):
        """Machine X, Y of what the webcam sees at column u, row v, lying on a surface of focus height z (the head's Z
        in focus on it)."""
        if self.P is None:
            raise ValueError("the bench camera is not calibrated")
        if self.ring_above_mm is None:
            raise ValueError("the bench camera knows no heights yet: teach_bench_height after calibrate_bench")
        return locate(self.P, u, v, z - self.ring_above_mm)

    def project(self, x, y, z):
        """Column, row where the webcam sees machine X, Y on a surface of focus height z."""
        if self.P is None or self.ring_above_mm is None:
            raise ValueError("the bench camera is not calibrated with heights")
        u, v, w = np.asarray(self.P, float) @ np.array([x, y, z - self.ring_above_mm, 1.0])
        return float(u / w), float(v / w)

    def to_config(self):
        return {"P": self.P, "model": "camera", "pairs": len(self.pairs),
                "ring_above_mm": None if self.ring_above_mm is None else round(self.ring_above_mm, 2),
                "taught": self.taught,
                "errors_mm": [round(e, 2) for e in (self.errors or [])],
                "points": [[u, v, p["X"], p["Y"], p["Z"]] for (u, v, _), p in self.pairs]}

    @classmethod
    def from_config(cls, cal):
        if not cal or cal.get("model") != "camera":
            return None
        pairs = [((u, v, None), {"X": x, "Y": y, "Z": z}) for u, v, x, y, z in cal.get("points", [])]
        return cls(pairs, cal["P"], cal.get("ring_above_mm"), cal.get("errors_mm"), cal.get("taught"))


def agree(a, b, tolerance_mm=8.0):
    """Whether two estimates of the same point (X, Y) agree: (True/False, the distance)."""
    d = math.hypot(a[0] - b[0], a[1] - b[1])
    return d <= tolerance_mm, round(d, 1)


def plan_route(workspace, z_low, z_high, n_x=4, n_y=3, margin=0.12):
    """Where the head goes for the webcam's calibration: a grid over the work area at z_low, as a snake, then the four
    corners at z_high. Two heights, or a projection cannot be fitted."""
    xs = np.linspace(workspace["X"][0] + margin * np.ptp(workspace["X"]), workspace["X"][1] - margin * np.ptp(workspace["X"]), n_x)
    ys = np.linspace(workspace["Y"][0] + margin * np.ptp(workspace["Y"]), workspace["Y"][1] - margin * np.ptp(workspace["Y"]), n_y)
    route = []
    for i, y in enumerate(ys):
        row = [(float(x), float(y), float(z_low)) for x in xs]
        route += row if i % 2 == 0 else row[::-1]
    route += [(float(xs[-1]), float(ys[-1]), float(z_high)), (float(xs[0]), float(ys[-1]), float(z_high)),
              (float(xs[0]), float(ys[0]), float(z_high)), (float(xs[-1]), float(ys[0]), float(z_high))]
    return route
