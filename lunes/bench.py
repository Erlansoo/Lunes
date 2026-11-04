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
CHIP = dict(white_s=70, white_v=150, min_board=20000, dark=0.55, min_area=150, max_area=6000, max_side=160, min_fill=0.25,
            max_aspect=4.0)  # a chip on a breadboard in the 4K picture
MOVED_PX = 40  # the ring this far from where the projection puts it, at 4K: the camera was moved, not just noise


def ring_in_picture(color, near=None, reach=350):
    """(column, row) of the microscope's ring light in the webcam's picture, or None. The picture is scaled to 960
    wide; there the lit ring is a white patch some 30 x 20 px with the black microscope tube right above it, and the
    patch's centre is given back at the picture's own size. Other white things are told apart: a breadboard under the
    lamp or a cable has no black tube over it; the supply's red digits burn out to a warm white in the webcam, small
    patches with a tint, where the ring is grey-white through and through. Of what is left, the biggest is the ring.
    near (column, row at the picture's size) keeps to what lies within reach px of it: a lamp or a lit strip elsewhere
    in the picture is then not the ring, and no ring there means it is not seen, not that the camera moved."""
    k = 960 / color.shape[1]
    small = cv2.resize(color, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else color
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    best = None
    # In daylight the webcam shortens its exposure and the lit ring comes out at 230-236, not saturated: then the
    # usual threshold breaks it into specks. A second pass 20 lower keeps the same shape, tint and dark-tube checks.
    for bright in (RING["bright"], RING["bright"] - 20):
        white = (small.min(axis=2) > bright - 30) & (gray > bright)
        n, labels, stats, centres = cv2.connectedComponentsWithStats(white.astype(np.uint8))
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if not (RING["min_area"] <= area <= RING["max_area"] and w <= RING["max_w"] and h <= RING["max_h"]):
                continue
            cx, cy = centres[i]
            if near is not None and math.hypot(cx / k - near[0], cy / k - near[1]) > reach:
                continue
            above = gray[max(0, y - int(1.5 * h) - 4):max(1, y - 2), max(0, int(cx) - w // 4):int(cx) + w // 4]
            darkness = float(above.mean()) if above.size else 255.0
            mean = small[labels == i].mean(axis=0)
            if darkness < 90 and np.ptp(mean) < RING["tint"] and (best is None or area > best[0]):
                best = (area, cx, cy, i)
        if best is not None:
            break
    if best is None:
        return None
    _, cx, cy, i = best
    ys, xs = np.nonzero((labels == i) & (gray >= 250))
    if xs.size >= 4:
        cx, cy = float(xs.mean()), float(ys.mean())
    return float(cx) / k, float(cy) / k


def chip_near(color, u, v, reach=260):
    """Column and row of the dark thing nearest column u, row v of the bench picture, or None: the chip under the
    microscope, when the camera's geometry says about where it must be. Anything darker elsewhere is not looked at."""
    x0, y0 = max(0, int(u) - reach), max(0, int(v) - reach)
    crop = color[y0:max(y0 + 1, int(v) + reach), x0:max(x0 + 1, int(u) + reach)]
    if crop.size == 0:
        return None
    gray = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    n, _, stats, centres = cv2.connectedComponentsWithStats((gray < 0.55 * np.median(gray)).astype(np.uint8))
    spots = [(float(centres[i][0]) + x0, float(centres[i][1]) + y0) for i in range(1, n) if 250 <= stats[i, cv2.CC_STAT_AREA] <= 6000]
    return min(spots, key=lambda s: math.hypot(s[0] - u, s[1] - v)) if spots else None


def chip_in_picture(color, area=None):
    """[(column, row, size in px)] of the dark compact things on the breadboard in the bench picture, biggest first:
    SAM does not find a chip this small from so far, but a black chip on a white board is plain to see. The board is
    the biggest white patch inside area (corners of the table in the picture, so the wall and the bench top do not
    count); its holes are too small to pass, and what lies off the board does not count. Nothing found: []."""
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 1] < CHIP["white_s"]) & (hsv[..., 2] > CHIP["white_v"])).astype(np.uint8)
    if area is not None:
        inside = np.zeros(white.shape, np.uint8)
        cv2.fillConvexPoly(inside, np.array(area, np.int32), 1)
        white &= inside
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8)))
    if n < 2:
        return []
    board = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[board, cv2.CC_STAT_AREA] < CHIP["min_board"]:
        return []
    on = np.zeros(white.shape, np.uint8)
    cv2.fillConvexPoly(on, cv2.convexHull(cv2.findNonZero((labels == board).astype(np.uint8))), 1)
    on = cv2.erode(on, np.ones((21, 21), np.uint8))
    gray = cv2.GaussianBlur(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    lit = (on > 0) & (white > 0)
    if not lit.any():
        return []
    dark = ((gray < CHIP["dark"] * float(np.median(gray[lit]))) & (on > 0)).astype(np.uint8)
    n, _, stats, centres = cv2.connectedComponentsWithStats(dark)
    found = []
    for i in range(1, n):
        x, y, w, h, size = stats[i]
        if (CHIP["min_area"] <= size <= CHIP["max_area"] and max(w, h) <= CHIP["max_side"] and size >= CHIP["min_fill"] * w * h
                and max(w, h) <= CHIP["max_aspect"] * min(w, h)):
            found.append((float(centres[i][0]), float(centres[i][1]), int(size)))
    return sorted(found, key=lambda f: -f[2])


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

    def ring_at(self, at_mm):
        """Where the projection puts the ring with the head at at_mm, or None if not calibrated."""
        if self.P is None:
            return None
        u, v, w = np.asarray(self.P, float) @ np.array([at_mm["X"], at_mm["Y"], at_mm["Z"], 1.0])
        return float(u / w), float(v / w)

    def ring_off_px(self, seen_uv, at_mm):
        """How far the ring light is, in pixels, from where the projection puts it with the head at at_mm: over
        MOVED_PX, someone moved the camera and its millimetres are not to be trusted. None if not calibrated."""
        at = self.ring_at(at_mm)
        return None if at is None else math.hypot(seen_uv[0] - at[0], seen_uv[1] - at[1])

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
