"""Surveys the board under the microscope: fine autofocus, millimetre calibration, then a grid of images stitched
into one mosaic using the machine positions. Shows every frame live while it works.

    python tools/survey.py COM8 --grid 3 --span 0.3
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from lunes.machine import Machine  # noqa: E402
from microscope import Camera, autofocus, calibrate, sharpness  # noqa: E402

OVERLAP = 0.3
LIFT_MM = 3.0  # Z goes up this much for every XY move, so tall parts (connectors) never meet the lens


class Interrupted(Exception):
    """Erlan asked to stop in the middle of a survey or a walk."""
WIN = "Lunes - recorriendo la placa"


def show_window(frame, text):
    f = frame.copy()
    for color, width in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.putText(f, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, width)
    cv2.imshow(WIN, f)
    cv2.waitKey(1)


def hop(m, z_focus, **xy):
    """XY move with Z lifted, then back down to the focus height (the lead screw repeats it exactly)."""
    m.move_to(Z=z_focus + LIFT_MM)
    m.move_to(**xy)
    m.move_to(Z=z_focus)


def geometry(cal):
    """Image shift per mm of each machine axis, and the field of view and grid step (mm) along X and Y."""
    vx = np.array(cal["image_shift_px_per_mm"]["X"])
    vy = np.array(cal["image_shift_px_per_mm"]["Y"])
    # Field of view along each machine axis: the image extent in the direction that axis moves the image
    # (the microscope may be mounted turned, e.g. by 90 degrees).
    fov_x, fov_y = (float((abs(v[0]) * 640 + abs(v[1]) * 480) / np.dot(v, v)) for v in (vx, vy))
    return vx, vy, (fov_x, fov_y), (fov_x * (1 - OVERLAP), fov_y * (1 - OVERLAP))


def grid_for(width_mm, height_mm, cal):
    """Columns (along X) and rows (along Y) of pictures that cover width x height mm."""
    _, _, (fov_x, fov_y), (step_x, step_y) = geometry(cal)
    cols = int(np.ceil(max(width_mm - fov_x, 0) / step_x)) + 1
    rows = int(np.ceil(max(height_mm - fov_y, 0) / step_y)) + 1
    return cols, rows


def match_pair(a, b, r, margin=30, min_score=0.5):
    """Where picture b really sits relative to picture a, given the expected offset r (b(u, v) shows a(u + r)):
    the overlap of b is searched for in a within margin pixels. Returns (offset, match score), or None when the
    overlap is too small or too flat to match (bare solder mask, the inside of a chip)."""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if a.ndim == 3 else a
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) if b.ndim == 3 else b
    rx, ry = int(round(r[0])), int(round(r[1]))
    u0, u1 = max(0, -rx) + margin, min(640, 640 - rx) - margin
    v0, v1 = max(0, -ry) + margin, min(480, 480 - ry) - margin
    if u1 - u0 < 40 or v1 - v0 < 40:
        return None
    templ = gb[v0:v1, u0:u1]
    if templ.std() < 6:  # nothing to lock onto
        return None
    su0, sv0 = u0 + rx - margin, v0 + ry - margin
    search = ga[sv0:v1 + ry + margin, su0:u1 + rx + margin]
    if search.shape[0] <= templ.shape[0] or search.shape[1] <= templ.shape[1]:
        return None
    _, score, _, loc = cv2.minMaxLoc(cv2.matchTemplate(search, templ, cv2.TM_CCOEFF_NORMED))
    if score < min_score:
        return None
    return np.array([su0 + loc[0] - u0, sv0 + loc[1] - v0], float), float(score)


def align_tiles(frames, cells, predicted, prior=0.02, max_jump=60.0):
    """Places every picture of a grid by its content as well as by the machine's positions.

    A home-made CNC does not come back to exactly where it was (backlash when an axis turns round, a belt that
    gives a little), so the machine's positions leave seams a few pixels off. Each picture is matched against its
    neighbours on the right and below; then all positions are solved together by least squares, every match
    pulling its two pictures into place and a weak pull towards the machine's position keeping featureless
    pictures where the machine put them. frames: the pictures; cells: their (column, row); predicted: their
    offsets in the mosaic from the machine's positions. Returns (offsets, matches used, worst residual px)."""
    n = len(frames)
    where = {cell: k for k, cell in enumerate(cells)}
    predicted = [np.asarray(p, float) for p in predicted]
    pairs = []
    for k, (i, j) in enumerate(cells):
        for nb in ((i + 1, j), (i, j + 1)):
            if nb not in where:
                continue
            q = where[nb]
            r = predicted[q] - predicted[k]
            found = match_pair(frames[k], frames[q], r)
            if found and np.linalg.norm(found[0] - r) <= max_jump:
                pairs.append((k, q, found[0], found[1]))

    def solve(pairs):
        rows, rhs, weights = [], [], []
        for k, q, r, score in pairs:
            row = np.zeros(n)
            row[q], row[k] = 1, -1
            rows.append(row)
            rhs.append(r)
            weights.append(score)
        for k in range(n):
            row = np.zeros(n)
            row[k] = 1
            rows.append(row)
            rhs.append(predicted[k])
            weights.append(prior)
        w = np.sqrt(np.array(weights))[:, None]
        sol, *_ = np.linalg.lstsq(np.array(rows) * w, np.array(rhs) * w, rcond=None)
        return sol

    sol = solve(pairs)
    residual = [np.linalg.norm((sol[q] - sol[k]) - r) for k, q, r, _ in pairs]
    if residual and max(residual) > 6:  # a few wrong matches (repeated pads look alike): drop them and solve again
        pairs = [p for p, e in zip(pairs, residual) if e <= 6]
        sol = solve(pairs)
        residual = [np.linalg.norm((sol[q] - sol[k]) - r) for k, q, r, _ in pairs]
    return list(sol), len(pairs), (max(residual) if residual else 0.0)


def blend(frames, offsets):
    """Pastes the pictures at their offsets, feathering every overlap so no seam line shows (each picture weighs
    less towards its own edges)."""
    ox = min(o[0] for o in offsets)
    oy = min(o[1] for o in offsets)
    w = int(np.ceil(max(o[0] for o in offsets) - ox)) + 641
    h = int(np.ceil(max(o[1] for o in offsets) - oy)) + 481
    acc = np.zeros((h, w, 3), np.float32)
    weight = np.zeros((h, w), np.float32)
    u = np.minimum(np.arange(640) + 1, 640 - np.arange(640))
    v = np.minimum(np.arange(480) + 1, 480 - np.arange(480))
    feather = np.minimum(np.minimum.outer(v, u), 60).astype(np.float32) / 60
    for frame, o in zip(frames, offsets):
        x, y = int(round(o[0] - ox)), int(round(o[1] - oy))
        acc[y:y + 480, x:x + 640] += frame.astype(np.float32) * feather[..., None]
        weight[y:y + 480, x:x + 640] += feather
    mosaic = (acc / np.maximum(weight, 1e-6)[..., None]).clip(0, 255).astype(np.uint8)
    mosaic[weight == 0] = 0
    return mosaic, (float(ox), float(oy)), (w, h)


def survey(m, cam, grid=3, span=0.3, show=show_window, recalibrate=True, cols=None, rows=None, stop=None,
           register=True):
    """Autofocus, (re)calibrate, photograph a grid centred on the current position and stitch it.

    The grid is grid x grid pictures, or cols (along X) x rows (along Y). Returns (mosaic image, folder with the
    tiles, summary dict). With recalibrate=False the microscope calibration in config/machine.json is used.
    register=True also lines the pictures up by their content (see align_tiles); False keeps the machine's
    positions alone."""
    cols, rows = cols or grid, rows or grid
    out = ROOT / "captures" / f"survey_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True)
    show(cam.frame(), "afinando el foco...")
    z, s = autofocus(m, cam, span, lambda t: show(cam.frame(), t))
    cal = m.config.get("microscope")
    if recalibrate or not cal:
        show(cam.frame(), "calibrando milimetros por pixel...")

        def lifted(axis, mm):
            hop(m, z, **{axis: m.position()[axis] + mm})

        cal = calibrate(m, cam, lifted)
    vx, vy, (fov_x, fov_y), (step_x, step_y) = geometry(cal)

    home = m.position()
    half_x, half_y = (cols - 1) / 2, (rows - 1) / 2
    tiles, cells = [], []
    for j in range(rows):
        for i in (range(cols) if j % 2 == 0 else reversed(range(cols))):  # back and forth: shorter moves
            dx, dy = (i - half_x) * step_x, (half_y - j) * step_y
            if stop and stop():
                hop(m, z, X=home["X"], Y=home["Y"])
                raise Interrupted()
            hop(m, z, X=home["X"] + dx, Y=home["Y"] + dy)
            time.sleep(0.25)
            cam.fresh_gray(1)  # drop frames taken while moving
            frame = cam.frame()
            cv2.imwrite(str(out / f"tile_r{j}_c{i}.png"), frame)
            tiles.append((dx, dy, frame))
            cells.append((i, j))
            show(frame, f"foto {len(tiles)}/{cols * rows}  X{dx:+.1f} Y{dy:+.1f} mm  nitidez "
                        f"{sharpness(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)):.0f}")
    hop(m, z, X=home["X"], Y=home["Y"])

    # The image content shifts by (vx*dx + vy*dy) pixels when the camera moves (dx, dy) mm, so each tile goes
    # at minus that offset in the mosaic.
    offsets = [-(vx * dx + vy * dy) for dx, dy, _ in tiles]
    frames = [f for _, _, f in tiles]
    alignment = None
    if register and len(frames) > 1:
        offsets, used, worst = align_tiles(frames, cells, offsets)
        alignment = {"matches_used": used, "worst_residual_px": round(worst, 1)}
    mosaic, (ox, oy), (w, h) = blend(frames, offsets)
    cv2.imwrite(str(out / "mosaic.png"), mosaic)
    summary = {"focus_z_mm": round(z, 3), "sharpness": round(s), "px_per_mm": cal["px_per_mm"],
               "area_mm": [round(fov_x + (cols - 1) * step_x, 1), round(fov_y + (rows - 1) * step_y, 1)],
               "fov_mm": [fov_x, fov_y], "step_mm": [step_x, step_y],
               "mosaic": str(out / "mosaic.png"),
               "board_in_mosaic": find_board(mosaic) or "no circuit board found, only the bed",
               # what it takes to turn a mosaic pixel back into machine millimetres (see mosaic_point_to_mm)
               "home_mm": {"X": home["X"], "Y": home["Y"]}, "origin_px": [float(ox), float(oy)],
               "size_px": [w, h], "shift_px_per_mm": {"X": vx.tolist(), "Y": vy.tolist()},
               "alignment": alignment}
    (out / "survey.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    return mosaic, out, summary


def bed_mask(image):
    """Pixels of the wooden bed: under the microscope's LEDs a strongly saturated orange (hue about 17, saturation
    above 100)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return (hsv[..., 0] >= 5) & (hsv[..., 0] <= 30) & (hsv[..., 1] >= 70)


def board_mask(image):
    """Pixels of an image (a picture or a mosaic) that belong to a circuit board rather than the wooden bed.

    Boards come out almost grey under the microscope's LEDs (saturation about 30), pads and metal cans included, so
    the board is whatever is neither bed, nor dark, nor a mosaic's empty black margin."""
    v = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 2]
    bed = bed_mask(image)
    empty = image.sum(axis=2) == 0
    mask = (~bed & ~empty & (v >= 30)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))


def find_board(mosaic):
    """Where the circuit board is in a mosaic: its share of the image and its bounding box and centre as
    fractions of the mosaic, or None."""
    mask = board_mask(mosaic)
    fraction = float(mask.mean())
    if fraction < 0.02:
        return None
    ys, xs = np.nonzero(mask)
    H, W = mask.shape
    lo_x, hi_x = np.percentile(xs, [2, 98]) / W
    lo_y, hi_y = np.percentile(ys, [2, 98]) / H
    return {"share_of_image": round(fraction, 2), "center": [round(float(xs.mean() / W), 2), round(float(ys.mean() / H), 2)],
            "box": [round(float(lo_x), 2), round(float(lo_y), 2), round(float(hi_x), 2), round(float(hi_y), 2)]}


def pixels_to_mm(at, vx, vy, xs, ys):
    """Machine X, Y (mm) that put each pixel (xs, ys) of a picture taken at `at` at the centre of the image."""
    shift = np.vstack([320.0 - np.asarray(xs, float), 240.0 - np.asarray(ys, float)])
    d = np.linalg.solve(np.column_stack([vx, vy]), shift)
    return at["X"] + d[0], at["Y"] + d[1]


def board_in_picture(frame, at, vx, vy):
    """The board's share of one picture, and its extent there in machine mm (x0, x1, y0, y1), or None."""
    mask = board_mask(frame)
    share = float(mask.mean())
    if share < 0.03:
        return share, None
    ys, xs = np.nonzero(mask[::4, ::4])
    bx, by = pixels_to_mm(at, vx, vy, xs * 4, ys * 4)
    return share, (*np.percentile(bx, [1, 99]), *np.percentile(by, [1, 99]))


def board_corner_in_picture(frame, at, vx, vy, sx, sy, bin_mm=0.1, skip_mm=0.4, reach_mm=8.0):
    """One corner of the circuit board, in machine mm, from one microscope picture taken at `at`.

    sx, sy (+1 or -1) say which corner, by the board's side along machine X and along Y: sx=-1, sy=+1 is the corner
    where the board has its smallest X and its largest Y. The board's edges are long straight lines, so each one is
    fitted to the board's outermost pixels on its side, slice by slice, away from the corner's rounding; the corner
    is where the two lines cross. Returns (corner, None) or (None, why it cannot be seen); corner has "X" and "Y" (mm)
    and how straight each edge came out."""
    mask = board_mask(frame)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        return None, "there is no board in the picture"
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))  # bits of grey on the bed are not the board
    ys, xs = np.nonzero(labels[::2, ::2] == biggest)
    xs, ys = xs * 2, ys * 2
    if len(xs) < 500:
        return None, "there is almost no board in the picture"
    h, w = mask.shape
    on_border = (xs < 6) | (xs > w - 7) | (ys < 6) | (ys > h - 7)  # the board goes on past the picture there
    bx, by = pixels_to_mm(at, vx, vy, xs, ys)
    guess = {"X": np.percentile(sx * bx, 99.5) * sx, "Y": np.percentile(sy * by, 99.5) * sy}

    def edge(axis):
        """(slope, offset) of the edge on the `axis` side, as axis = slope * other + offset, and its spread."""
        along, across = (bx, by) if axis == "X" else (by, bx)
        sign, other_sign = (sx, sy) if axis == "X" else (sy, sx)
        other = "Y" if axis == "X" else "X"
        away = other_sign * (guess[other] - across)  # how far from the corner along this edge
        keep = (away > skip_mm) & (away < reach_mm)
        slices = np.round(across[keep] / bin_mm).astype(int)
        points = {}
        for s, a, b in zip(slices, along[keep], on_border[keep]):
            if s not in points or sign * a > sign * points[s][0]:
                points[s] = (a, b)
        pts = np.array([(s * bin_mm, a) for s, (a, b) in points.items() if not b])
        if len(pts) < 15:
            return None
        fit = np.ones(len(pts), bool)
        for _ in range(3):  # a part sticking out of the edge is not the edge
            slope, offset = np.polyfit(pts[fit, 0], pts[fit, 1], 1)
            residual = np.abs(pts[:, 1] - (slope * pts[:, 0] + offset))
            fit = residual < max(0.08, 2.5 * np.median(residual[fit]))
        if fit.sum() < 15:
            return None
        return slope, offset, float(residual[fit].std()), float(np.ptp(pts[fit, 0]))

    ex, ey = edge("X"), edge("Y")
    missing = [name for name, e in (("X", ex), ("Y", ey)) if e is None]
    if missing:
        return None, f"the board's {' and '.join(missing)} edge of that corner is not in the picture"
    (a, b, spread_x, length_x), (c, d, spread_y, length_y) = ex, ey
    x = (a * d + b) / (1 - a * c)  # x = a*y + b crossed with y = c*x + d
    y = c * x + d
    angle = abs(np.degrees(np.arctan(a) + np.arctan(c)))  # 0 when the edges are square to each other
    corner = {"X": float(x), "Y": float(y), "edges_seen_mm": [round(length_x, 1), round(length_y, 1)],
              "edge_spread_mm": [round(spread_x, 3), round(spread_y, 3)], "out_of_square_deg": round(float(angle), 1)}
    # The outline of a part inside the board can look like a corner too; the board's own corner is crisp, square,
    # and has no board beyond it.
    ys, xs = np.nonzero(mask[::2, ::2])
    ax, ay = pixels_to_mm(at, vx, vy, xs * 2, ys * 2)
    beyond = (sx * (ax - x) > 0.3) & (sx * (ax - x) < 2.5) & (sy * (ay - y) > 0.3) & (sy * (ay - y) < 2.5)
    if max(spread_x, spread_y) > 0.12 or angle > 12 or min(length_x, length_y) < 1.5 or beyond.sum() > 40:
        return None, (f"what looks like that corner is not crisp or square enough to be the board's own (edges "
                      f"spread {spread_x:.2f} and {spread_y:.2f} mm, {angle:.0f} degrees out of square)")
    return corner, None


def distance_in_picture(at, vx, vy, p1, p2):
    """Distance in mm between two pixels of a microscope picture, and its X and Y parts (machine axes)."""
    x, y = pixels_to_mm(at, vx, vy, [p1[0], p2[0]], [p1[1], p2[1]])
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    return float(np.hypot(dx, dy)), dx, dy


def _fit_circle(x, y):
    """(centre x, centre y, radius) of the circle closest to the points (least squares)."""
    a = np.column_stack([x, y, np.ones_like(x)])
    (d, e, f), *_ = np.linalg.lstsq(a, -(x ** 2 + y ** 2), rcond=None)
    cx, cy = -d / 2, -e / 2
    return float(cx), float(cy), float(np.sqrt(cx ** 2 + cy ** 2 - f))


def _edge_points(gray, cx, cy, r0, sign, band=0.25, rays=180):
    """Where each ray from (cx, cy) crosses the sharpest edge near radius r0 (pixels, to a fraction of one): from
    dark to bright going out (sign +1) or from bright to dark (-1). Rays that leave the picture are skipped."""
    h, w = gray.shape
    rs = np.arange(r0 * (1 - band), r0 * (1 + band), 0.5)
    points = []
    for a in np.linspace(0, 2 * np.pi, rays, endpoint=False):
        xs, ys = cx + rs * np.cos(a), cy + rs * np.sin(a)
        if xs.min() < 1 or ys.min() < 1 or xs.max() > w - 2 or ys.max() > h - 2:
            continue
        values = cv2.remap(gray, xs.astype(np.float32)[None], ys.astype(np.float32)[None], cv2.INTER_LINEAR)[0]
        g = np.gradient(values) * sign
        k = int(np.argmax(g))
        if g[k] < 3 or k in (0, len(g) - 1):
            continue
        bend = g[k - 1] - 2 * g[k] + g[k + 1]
        r = rs[k] + (0.5 * (g[k - 1] - g[k + 1]) / bend * 0.5 if bend else 0.0)  # the peak, between samples
        points.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return np.array(points).reshape(-1, 2), rays


def _circle_in_mm(points, at, vx, vy):
    """Circle fitted to edge pixels in machine mm, dropping the points that are not on it (a via in the ring, a
    scratch): (centre x, centre y, radius, the points kept, how far they stray, mm)."""
    for _ in range(3):
        mx, my = pixels_to_mm(at, vx, vy, points[:, 0], points[:, 1])
        cx, cy, r = _fit_circle(mx, my)
        off = np.abs(np.hypot(mx - cx, my - cy) - r)
        points = points[off < max(0.02, 3 * np.median(off))]
    mx, my = pixels_to_mm(at, vx, vy, points[:, 0], points[:, 1])
    cx, cy, r = _fit_circle(mx, my)
    return cx, cy, r, points, float((np.hypot(mx - cx, my - cy) - r).std())


def hole_in_picture(frame, at, vx, vy, px, py):
    """The round hole nearest a pixel (px, py) of a microscope picture taken at `at`: a mounting hole, a via, a
    through-hole pad. Returns (hole, None) or (None, why not); hole has the hole's diameter and, if it has one, its
    copper ring's outer diameter (mm), with how round each edge came out, and the edge pixels found.

    Circles are found first (Hough), then each edge is measured where every ray from the centre crosses it, turned
    into mm with the calibration (so X and Y scale each by their own) and fitted with a circle."""
    u8 = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    circles = cv2.HoughCircles(u8, cv2.HOUGH_GRADIENT_ALT, 1.5, 8, param1=120, param2=0.6, minRadius=6,
                               maxRadius=min(u8.shape) // 2)
    if circles is None:
        return None, "there is no round edge in the picture"
    circles = circles[0]
    holding = [c for c in circles if np.hypot(c[0] - px, c[1] - py) < c[2]]  # circles with the point inside
    if not holding:
        return None, "no round hole around that point; give a point inside the hole"
    seed = min(holding, key=lambda c: np.hypot(c[0] - px, c[1] - py))
    family = sorted((c for c in circles if np.hypot(c[0] - seed[0], c[1] - seed[1]) < max(4.0, 0.15 * c[2])),
                    key=lambda c: c[2])  # the hole and the rings around it share a centre
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32)
    hole = family[0]
    points, rays = _edge_points(gray, hole[0], hole[1], hole[2], +1)
    if len(points) < 0.6 * rays:
        return None, ("the hole is cut by the picture's border or its edge is not clear: put it in the middle "
                      "(center_on_point) and measure again")
    cx, cy, r, kept, stray = _circle_in_mm(points, at, vx, vy)
    found = {"hole_diameter_mm": round(2 * r, 3), "hole_roundness_mm": round(stray, 3),
             "centre_mm": {"X": round(cx, 3), "Y": round(cy, 3)}, "edges_px": [kept]}
    ring = next((c for c in family if c[2] > 1.25 * hole[2]), None)
    if ring is not None:
        points, _ = _edge_points(gray, hole[0], hole[1], ring[2], -1)
        if len(points) >= 0.4 * rays:
            _, _, rr, kept, stray = _circle_in_mm(points, at, vx, vy)
            found.update(ring_diameter_mm=round(2 * rr, 3), ring_roundness_mm=round(stray, 3))
            found["edges_px"].append(kept)
    return found, None


def find_perimeter(m, cam, z, cal, limits, show=show_window, max_mm=180.0, seen=0.15, stop=None):
    """Finds the board's outline by walking from the current position.

    With no board under the microscope it first looks around in two rings of pictures. Then it walks to each side
    (+X, -X, +Y, -Y) until the board's edge shows up inside a picture. limits: allowed (lo, hi) mm per axis.
    Returns a dict with the board's box in machine mm, and leaves the microscope over the board's centre."""
    vx, vy, (fov_x, fov_y), _ = geometry(cal)
    step = {"X": fov_x * 0.8, "Y": fov_y * 0.8}
    corners = (np.array([0, 639, 0, 639]), np.array([0, 0, 479, 479]))
    count = [0]

    def inside(x, y):
        return limits["X"][0] <= x <= limits["X"][1] and limits["Y"][0] <= y <= limits["Y"][1]

    def shoot(x, y, what):
        if stop and stop():
            raise Interrupted()
        hop(m, z, X=x, Y=y)
        time.sleep(0.2)
        cam.fresh_gray(1)
        frame = cam.frame()
        share, ext = board_in_picture(frame, {"X": x, "Y": y}, vx, vy)
        fx, fy = pixels_to_mm({"X": x, "Y": y}, vx, vy, *corners)
        count[0] += 1
        show(frame, f"{what}  X{x:+.1f} Y{y:+.1f}  placa {share:.0%}")
        return share, ext, (fx.min(), fx.max(), fy.min(), fy.max())

    start = m.position()
    share, ext, frame_box = shoot(start["X"], start["Y"], "buscando la placa")
    if share < seen:  # nothing under the lens: look around, nearest pictures first
        found = False
        rings = [(i, j) for ring in (1, 2) for i in range(-ring, ring + 1) for j in range(-ring, ring + 1)
                 if max(abs(i), abs(j)) == ring]
        for i, j in rings:
            x, y = start["X"] + i * fov_x * 0.9, start["Y"] + j * fov_y * 0.9
            if inside(x, y):
                share, ext, frame_box = shoot(x, y, "buscando la placa")
                if share >= seen:
                    found = True
                    break
        if not found:
            hop(m, z, X=start["X"], Y=start["Y"])
            return {"found": False, "pictures": count[0],
                    "searched_mm": [round(4 * fov_x * 0.9 + fov_x), round(4 * fov_y * 0.9 + fov_y)]}
    # walk from the middle of the board seen in that picture
    cx, cy = (ext[0] + ext[1]) / 2, (ext[2] + ext[3]) / 2
    if not inside(cx, cy):
        here = m.position()
        cx, cy = here["X"], here["Y"]
    share, e, fb = shoot(cx, cy, "midiendo la placa")
    if e is None:
        e, fb = ext, frame_box
    center_ext, center_frame = e, fb
    box = list(e)
    edges = {}
    for axis, sign in (("X", 1), ("X", -1), ("Y", 1), ("Y", -1)):
        k = (0 if axis == "X" else 2) + (1 if sign > 0 else 0)  # this side's index in (x0, x1, y0, y1)
        pos = {"X": cx, "Y": cy}
        last_ext, last_frame = center_ext, center_frame
        edge = None
        while True:
            # the edge is inside the last picture when the board stops short of the picture's border on this side
            if sign * (last_frame[k] - last_ext[k]) > 0.6:
                edge = last_ext[k]
                break
            pos[axis] += sign * step[axis]
            if not inside(pos["X"], pos["Y"]) or abs(pos[axis] - (cx if axis == "X" else cy)) > max_mm:
                break
            share, e, fb = shoot(pos["X"], pos["Y"], f"siguiendo el borde {'+' if sign > 0 else '-'}{axis}")
            if e is None:  # walked off the board: its edge was in the previous picture
                edge = last_ext[k]
                break
            last_ext, last_frame = e, fb
            box = [min(box[0], e[0]), max(box[1], e[1]), min(box[2], e[2]), max(box[3], e[3])]
        box[k] = edge if edge is not None else last_ext[k]
        edges[f"{'+' if sign > 0 else '-'}{axis}"] = edge is not None
    hop(m, z, X=(box[0] + box[1]) / 2, Y=(box[2] + box[3]) / 2)
    return {"found": True,
            "box_mm": {"X": [round(box[0], 1), round(box[1], 1)], "Y": [round(box[2], 1), round(box[3], 1)]},
            "size_mm": [round(box[1] - box[0], 1), round(box[3] - box[2], 1)],
            "center_mm": [round((box[0] + box[1]) / 2, 1), round((box[2] + box[3]) / 2, 1)],
            "edges_found": edges, "pictures": count[0]}


def component_box(mosaic, x_frac, y_frac, max_frac=0.6):
    """Bounding box (fractions of the mosaic: x0, y0, x1, y1) of the component body (a chip's grey epoxy) under or
    right next to a point of the mosaic, or None when there is no such body there.

    Under the microscope's LEDs a chip's epoxy comes out a flat grey (saturation below 40, value 40 to 125); the
    median filter first wipes out its printed marking and the pin-1 dot."""
    hsv = cv2.cvtColor(cv2.medianBlur(mosaic, 15), cv2.COLOR_BGR2HSV)
    body = ((hsv[..., 1] < 40) & (hsv[..., 2] > 40) & (hsv[..., 2] < 125)).astype(np.uint8)
    body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    body = cv2.morphologyEx(body, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(body)
    H, W = body.shape
    px, py = int(x_frac * W), int(y_frac * H)
    r = 40
    near = labels[max(0, py - r):py + r, max(0, px - r):px + r]
    ids, counts = np.unique(near[near > 0], return_counts=True)
    if not len(ids):
        return None
    x, y, w, h, _ = stats[ids[np.argmax(counts)]]
    if w > max_frac * W and h > max_frac * H:  # that is the whole board's shadow, not one part
        return None
    return x / W, y / H, (x + w) / W, (y + h) / H


def closeup_for_reading(mosaic):
    """A component's mosaic made easy to read: long side horizontal, contrast evened out, and a second copy turned
    upside down below it, so the marking is upright in one of the two."""
    if mosaic.shape[0] > mosaic.shape[1]:
        mosaic = cv2.rotate(mosaic, cv2.ROTATE_90_CLOCKWISE)
    lab = cv2.cvtColor(mosaic, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(lab[..., 0])
    even = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gap = np.full((12, even.shape[1], 3), 255, np.uint8)
    return np.vstack([even, gap, cv2.rotate(even, cv2.ROTATE_180)])


def locate_in_mosaic(frame, mosaic, summary, min_score=0.35):
    """Where the microscope really is, found by looking for what it sees now inside a mosaic of the same board.
    Returns ((X, Y) in machine mm, match score, score of the best match elsewhere), or None when the picture is not
    in the mosaic. A board repeats itself (rows of pads, twin parts), so a match is only trustworthy when it clearly
    beats every other place."""
    k = 0.5
    small_mosaic = cv2.cvtColor(cv2.resize(mosaic, None, fx=k, fy=k, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    best = (-1.0, None, None)
    # The picture's scale changes a few per cent with the height of the lens over the board, so try a few.
    for scale in (0.9, 0.94, 0.97, 1.0, 1.03, 1.06, 1.1):
        small_frame = cv2.resize(gray, None, fx=k * scale, fy=k * scale, interpolation=cv2.INTER_AREA)
        if small_mosaic.shape[0] < small_frame.shape[0] or small_mosaic.shape[1] < small_frame.shape[1]:
            continue
        result = cv2.matchTemplate(small_mosaic, small_frame, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(result)
        if score > best[0]:
            best = (score, loc, small_frame.shape, result)
    score, loc, shape, result = best
    if score < min_score:
        return None
    # the best match anywhere else (away from this one by more than half a picture)
    away = result.copy()
    x0, y0 = max(0, loc[0] - shape[1] // 2), max(0, loc[1] - shape[0] // 2)
    away[y0:loc[1] + shape[0] // 2 + 1, x0:loc[0] + shape[1] // 2 + 1] = -1
    second = float(away.max())
    h, w = mosaic.shape[:2]
    cx, cy = (loc[0] + shape[1] / 2) / k, (loc[1] + shape[0] / 2) / k  # the picture's centre, in mosaic pixels
    return mosaic_point_to_mm(summary, cx / w, cy / h), float(score), second


def mosaic_point_to_mm(summary, x_frac, y_frac):
    """Machine position (X, Y in mm) that puts a point of the mosaic at the centre of the microscope image.

    The centre tile was taken at home; a mosaic pixel p sits at p + origin in that tile's pixel frame. Moving the
    machine by (dX, dY) shifts the image content by vx*dX + vy*dY pixels, so solve for the shift that brings the
    point to the image centre."""
    w, h = summary["size_px"]
    q = np.array([x_frac * w, y_frac * h]) + np.array(summary["origin_px"])
    shift = np.array([320.0, 240.0]) - q
    m = np.column_stack([summary["shift_px_per_mm"]["X"], summary["shift_px_per_mm"]["Y"]])
    d_x, d_y = np.linalg.solve(m, shift)
    return summary["home_mm"]["X"] + float(d_x), summary["home_mm"]["Y"] + float(d_y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", default="COM8")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--grid", type=int, default=3, help="tiles per side, centred on the current position")
    ap.add_argument("--span", type=float, default=0.3, help="autofocus searches this many mm up and down")
    args = ap.parse_args()

    cam = Camera(args.camera)
    m = Machine(args.port)
    try:
        mosaic, out, summary = survey(m, cam, args.grid, args.span)
        print(summary)
        h, w = mosaic.shape[:2]
        show_window(cv2.resize(mosaic, None, fx=min(1.0, 1200 / w), fy=min(1.0, 1200 / w)), "mosaico listo")
        cv2.waitKey(1500)
    finally:
        m.close()
        cam.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
