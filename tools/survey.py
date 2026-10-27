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


def register_offsets(tiles, offsets, margin=30, min_score=0.5):
    """Refines where each picture goes in the mosaic by matching it against the picture taken just before it, its
    neighbour since the grid is shot back and forth. The machine's positions alone leave seams a few pixels off,
    enough to split a character of a chip's marking. Flat areas that match nothing keep the machine's position."""
    fixed = [np.asarray(offsets[0], float)]
    for k in range(1, len(tiles)):
        prev = cv2.cvtColor(tiles[k - 1][2], cv2.COLOR_BGR2GRAY)
        cur = cv2.cvtColor(tiles[k][2], cv2.COLOR_BGR2GRAY)
        r = np.asarray(offsets[k], float) - np.asarray(offsets[k - 1], float)  # cur(u, v) shows prev(u + r)
        rx, ry = int(round(r[0])), int(round(r[1]))
        u0, u1 = max(0, -rx) + margin, min(640, 640 - rx) - margin
        v0, v1 = max(0, -ry) + margin, min(480, 480 - ry) - margin
        if u1 - u0 >= 40 and v1 - v0 >= 40:
            su0, sv0 = u0 + rx - margin, v0 + ry - margin
            search = prev[sv0:v1 + ry + margin, su0:u1 + rx + margin]
            score, loc = -1.0, None
            if search.shape[0] > v1 - v0 and search.shape[1] > u1 - u0:
                _, score, _, loc = cv2.minMaxLoc(cv2.matchTemplate(search, cur[v0:v1, u0:u1], cv2.TM_CCOEFF_NORMED))
            if score >= min_score:
                r = np.array([su0 + loc[0] - u0, sv0 + loc[1] - v0], float)
        fixed.append(fixed[-1] + r)
    return fixed


def survey(m, cam, grid=3, span=0.3, show=show_window, recalibrate=True, cols=None, rows=None, stop=None,
           register=False):
    """Autofocus, (re)calibrate, photograph a grid centred on the current position and stitch it.

    The grid is grid x grid pictures, or cols (along X) x rows (along Y). Returns (mosaic image, folder with the
    tiles, summary dict). With recalibrate=False the microscope calibration in config/machine.json is used.
    register=True lines the pictures up by their content too (for close-ups; see register_offsets)."""
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
    tiles = []
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
            show(frame, f"foto {len(tiles)}/{cols * rows}  X{dx:+.1f} Y{dy:+.1f} mm  nitidez "
                        f"{sharpness(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)):.0f}")
    hop(m, z, X=home["X"], Y=home["Y"])

    # The image content shifts by (vx*dx + vy*dy) pixels when the camera moves (dx, dy) mm, so each tile goes
    # at minus that offset in the mosaic.
    offsets = [-(vx * dx + vy * dy) for dx, dy, _ in tiles]
    if register:
        offsets = register_offsets(tiles, offsets)
    ox = min(o[0] for o in offsets)
    oy = min(o[1] for o in offsets)
    w = int(np.ceil(max(o[0] for o in offsets) - ox)) + 641
    h = int(np.ceil(max(o[1] for o in offsets) - oy)) + 481
    mosaic = np.zeros((h, w, 3), np.uint8)
    for (dx, dy, frame), o in zip(tiles, offsets):
        x, y = int(round(o[0] - ox)), int(round(o[1] - oy))
        mosaic[y:y + 480, x:x + 640] = frame
    cv2.imwrite(str(out / "mosaic.png"), mosaic)
    summary = {"focus_z_mm": round(z, 3), "sharpness": round(s), "px_per_mm": cal["px_per_mm"],
               "area_mm": [round(fov_x + (cols - 1) * step_x, 1), round(fov_y + (rows - 1) * step_y, 1)],
               "fov_mm": [fov_x, fov_y], "step_mm": [step_x, step_y],
               "mosaic": str(out / "mosaic.png"),
               "board_in_mosaic": find_board(mosaic) or "no circuit board found, only the bed",
               # what it takes to turn a mosaic pixel back into machine millimetres (see mosaic_point_to_mm)
               "home_mm": {"X": home["X"], "Y": home["Y"]}, "origin_px": [float(ox), float(oy)],
               "size_px": [w, h], "shift_px_per_mm": {"X": vx.tolist(), "Y": vy.tolist()}}
    (out / "survey.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    return mosaic, out, summary


def board_mask(image):
    """Pixels of an image (a picture or a mosaic) that belong to a circuit board rather than the wooden bed.

    Under the microscope's LEDs the bed is a strongly saturated orange (hue about 17, saturation above 100) while
    boards come out almost grey (saturation about 30), pads and metal cans included. So the board is whatever is
    neither bed, nor dark, nor a mosaic's empty black margin."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    bed = (h >= 5) & (h <= 30) & (s >= 70)
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
