"""The laser pointer on the head: its dot found with the microscope, its brightness chosen, and where it lands in
machine millimetres, so Lunes can put the dot on a point of the bench (a breadboard hole, a pad to probe).

The dot: how much the red channel rose between a picture with the laser off and one with it on, blurred a little
against the camera's noise (two laser-off pictures differ by up to about 30). A fine dot lights well under a square
millimetre; a laser too bright for the microscope lights much more: on shiny metal it blooms into a glow over the
whole part (3 mm² and up), and brighter still the light scattered off the board floods the view red. Those count as
too bright, not as a dot. The dot's centre is weighted by brightness, to a fraction of a pixel; its size is the
diameter at half its peak, in mm.

The brightness: the motion board's PWM on D6 (LASER 0..255, at 31 kHz). The dot is finest just above where the diode
starts to lase (on the bench between 27 and 30, and it moves a little as the diode warms). calibrate steps the level
down, past the levels that are too bright, to where the dot goes out, and keeps a level a little above the faintest
that showed it, so it still lases when the threshold moves.

Where it lands: the laser is fixed to the carriage, not to Z, so moving the head up or down does not move the dot.
The beam is slanted (about 33 degrees on the bench), so where it lands depends on how high the surface is: the dot's
distance from the microscope's centre shifts about 0.65 mm for each mm the surface rises. The microscope focuses at
a fixed distance, so the head's Z in focus is the surface's height. calibrate measures the distance (the dot's
centre in mm minus where the microscope is) on the surface in focus and keeps it with that Z, for up to three
surfaces (the bed, the board, a breadboard's top). With two or more at different heights the slant is fitted, and
it works both ways: the dot can be put on a surface at any height, and where the dot lands tells how high the
surface under it is (laser triangulation), with no focusing and no fourth axis.
"""
import math

import cv2
import numpy as np

# PWM steps, bright to faint, one by one where the diode starts to lase
LEVELS = (255, 128, 64, 48, 40, 36, 34, 32, 31, 30, 29, 28, 27, 26, 25, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2)
NOISE = 30  # the red rise between two laser-off pictures peaks near here
LIT = 40  # a red rise above this is light from the laser
GLOW_MM2 = 1.0  # lighting more than this is a glow or a flood, not a dot
MARGIN = 2  # PWM above the faintest level that showed the dot
SAME_SURFACE_MM = 1.0  # surfaces closer in height than this are taken as the same one


def lit(off, on):
    """How much the laser raised the red channel, pixel by pixel."""
    rise = np.clip(on[..., 2].astype(np.int16) - off[..., 2].astype(np.int16), 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(rise, (5, 5), 0)


def glow_mm2(off, on, px_per_mm):
    """How much of the view the laser lights, in mm²."""
    return float((lit(off, on) > LIT).sum()) / px_per_mm ** 2


def too_bright(off, on, px_per_mm):
    """Whether the laser lights more than a dot: a glow over a shiny part, or the whole view flooded red."""
    return glow_mm2(off, on, px_per_mm) > GLOW_MM2


def find_dot(off, on, px_per_mm, noise=NOISE):
    """The laser dot in a picture taken with the laser on, against one with it off; None if there is none, or if the
    laser is too bright to make one."""
    rise = lit(off, on)
    if float((rise > LIT).sum()) / px_per_mm ** 2 > GLOW_MM2:
        return None
    peak = int(rise.max())
    if peak < max(55, 1.8 * noise):
        return None
    y, x = np.unravel_index(int(np.argmax(rise)), rise.shape)
    _, labels = cv2.connectedComponents((rise >= peak / 2).astype(np.uint8))
    blob = labels == labels[y, x]
    ys, xs = np.nonzero(blob)
    w = rise[blob].astype(float)
    area = int(blob.sum())
    return {"x": float((xs * w).sum() / w.sum()), "y": float((ys * w).sum() / w.sum()), "peak": peak,
            "size_mm": round(2 * math.sqrt(area / math.pi) / px_per_mm, 3), "area_px": area}


def choose(rows):
    """The level to point with: among the clear dots, the one nearest MARGIN above the faintest level that showed a
    dot (as fine, nearly, and it still lases when the diode warms)."""
    seen = [r for r in rows if r.get("dot")]
    clear = [r for r in seen if r["dot"]["peak"] >= 80] or seen
    if not clear:
        return None
    target = min(r["level"] for r in seen) + MARGIN
    return min(clear, key=lambda r: (abs(r["level"] - target), r["level"]))


def remember(cal, surface, fresh=False, keep=3):
    """The surfaces a calibration keeps after measuring one more (a dict with z, the head's Z in focus on it): this
    one first, then the latest earlier ones at other heights. fresh forgets the earlier ones (the laser was moved, so
    they no longer hold)."""
    earlier = [] if fresh else [s for s in (cal or {}).get("surfaces", [])
                                if abs(s["z"] - surface["z"]) >= SAME_SURFACE_MM]
    return [surface] + earlier[:keep - 1]


def _fit(cal):
    """The calibrated surfaces' line: the dot's distance from the microscope's centre is o0 + k (z - z0), with z0 and
    o0 their mean height and distance; k is None with a single height."""
    s = cal["surfaces"]
    z = np.array([v["z"] for v in s], float)
    o = np.array([[v["offset_mm"]["X"], v["offset_mm"]["Y"]] for v in s], float)
    z0, o0 = float(z.mean()), o.mean(axis=0)
    if np.ptp(z) < SAME_SURFACE_MM:
        return z0, o0, None
    return z0, o0, ((z - z0)[:, None] * (o - o0)).sum(axis=0) / ((z - z0) ** 2).sum()


def slant(cal):
    """mm the dot moves along X and Y for each mm the surface rises; None until two heights are measured."""
    k = _fit(cal)[2]
    return None if k is None else (float(k[0]), float(k[1]))


def residuals(cal):
    """How far each calibrated surface's distance is from the fitted line, mm (large: the laser moved in between)."""
    z0, o0, k = _fit(cal)
    if k is None:
        return [0.0 for _ in cal["surfaces"]]
    return [float(np.hypot(*(np.array([s["offset_mm"]["X"], s["offset_mm"]["Y"]]) - o0 - k * (s["z"] - z0))))
            for s in cal["surfaces"]]


def offset_at(cal, z):
    """Where the dot lands relative to the microscope's centre (X, Y mm) on a surface whose height is z (the head's Z
    in focus on it)."""
    z0, o0, k = _fit(cal)
    o = o0 if k is None else o0 + k * (z - z0)
    return float(o[0]), float(o[1])


def height_from(cal, offset_mm):
    """(z, off_line_mm): the height of the surface where the dot lands (as the head's Z in focus on it), from where it
    lands relative to the microscope's centre; off_line_mm is how far the dot is from where any height would put it
    (more than a few tenths: a wrong dot, or the laser moved). None until the slant is known."""
    z0, o0, k = _fit(cal)
    if k is None:
        return None
    d = np.asarray(offset_mm, float) - o0
    t = float(d @ k / (k @ k))
    return z0 + t, float(np.hypot(*(d - t * k)))


def mark(frame, dot, text=""):
    """The picture with the dot circled and its size written, for the model and the dashboard."""
    view = frame.copy()
    if dot:
        r = max(6, int(math.sqrt(dot["area_px"] / math.pi)) + 4)
        cv2.circle(view, (int(round(dot["x"])), int(round(dot["y"]))), r, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.drawMarker(view, (320, 240), (255, 255, 0), cv2.MARKER_CROSS, 16, 1)
    if text:
        cv2.putText(view, text, (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, text, (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return view
