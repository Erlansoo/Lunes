"""A DIP chip on the breadboard, seen through the microscope: where its body is, which way it stands, and where its
pin 1 is. Classical image processing only, no model, so a chip that is not seen is reported as not seen, never guessed.

How a DIP looks from above through the microscope: a dark epoxy body (a rectangle, longer along the pins' rows), a row
of pins on each long side, a notch at one short end and sometimes a small round dent at pin 1's corner. Seen from above
with the notch to the left, pin 1 is the lower left pin and the pins count anticlockwise: down the near row, then back
along the far row.

find_chip(frame, px_per_mm) gives the body's box and its angle; notch_end(frame, body) which short end has the notch or
the dent, with how sure it is. pin1(body, end) gives pin 1's place from the body's geometry. orientation(body, end)
names the way the chip stands: the direction the notch faces in the picture.

What the code cannot see, Erlan can say, three ways: a point of the microscope's picture (tapped on the web lab or given
as fractions), the laser's dot moved onto the pin, or a pointer (a wooden stick) laid on the breadboard with its tip at
pin 1, which pointer_tip finds in the picture. All three end as the same thing: pin 1's place in machine mm.
"""
import math

import cv2
import numpy as np

MIN_BODY_MM = (5.0, 4.0)  # a DIP-8's body is about 9.3 x 6.3 mm; smaller things are not chips
MAX_BODY_MM = 60.0  # longer than this is not a chip in one picture
MAX_SHARE = 0.75  # a body filling more of the picture than this is the whole board or the bench, not a chip
LEGS_MM = 2.0  # thinner than this is a leg, a lead or a shadow line, never a chip's body


BESIDE_PIN_MM = 1.4  # from the long axis: well inside the top of a body 6.35 mm wide, since the slanted beam misses a
# chip's edge by millimetres, and which edge depends on how the chip is turned


def body_mask(frame):
    """The dark, flat grey of a chip's epoxy: low saturation and darker than most of the picture. The darkness is judged
    against the picture itself, since under the microscope's light a white breadboard can come out as dark as 70 of 255
    and a chip's body at 38: a fixed line between them would not hold from one light to another."""
    hsv = cv2.cvtColor(cv2.medianBlur(frame, 9), cv2.COLOR_BGR2HSV)
    value = hsv[..., 2]
    most = float(np.percentile(value, 60))
    # darker than most of the picture, by a margin; a wider one under a side lamp, which lifts the chip's own top too
    lamp = most >= 150
    ceiling = min(160, most * 0.55) if lamp else min(110, most * 0.8)
    # a lamp from the side tints the whole picture, the chip too: how coloured a thing is counts against the picture's own
    tint = max(60.0, 1.6 * float(np.median(hsv[..., 1])))
    body = ((hsv[..., 1] < tint) & (value > 12) & (value < ceiling)).astype(np.uint8)
    body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    size = 21 if lamp else 11  # under a side lamp the shadow by the legs joins the body: a wider opening parts them
    return cv2.morphologyEx(body, cv2.MORPH_OPEN, np.ones((size, size), np.uint8))


def find_chip(frame, px_per_mm):
    """{"center", "size_mm", "angle_deg", "corners", "fill", "share"} of the biggest chip body in the picture, or None.
    The angle is of the body's long axis, from the picture's x axis, in [-90, 90)."""
    mask = body_mask(frame)
    found = _biggest_body(mask, px_per_mm)
    if found is None:
        # parts over the chip (a resistor's legs crossing it, Nov 4) join the body to their dark shadows, and the glare
        # on the marking leaves holes in it, so it is no solid rectangle. Its holes filled and everything thinner than
        # about 2 mm opened away, the legs go and the body stays: a DIP's body is 6 mm wide and more
        solid = np.zeros_like(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(solid, contours, -1, 1, cv2.FILLED)
        k = int(round(LEGS_MM * px_per_mm)) | 1
        found = _biggest_body(cv2.morphologyEx(solid, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))),
                              px_per_mm)
    return found


def _biggest_body(mask, px_per_mm):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    best = None
    for i in range(1, n):
        area = stats[i][cv2.CC_STAT_AREA]
        if area < MIN_BODY_MM[0] * MIN_BODY_MM[1] * px_per_mm ** 2 or area > MAX_SHARE * mask.size:
            continue
        points = np.column_stack(np.nonzero(labels == i))[:, ::-1].astype(np.float32)
        (cx, cy), (w, h), angle = cv2.minAreaRect(points)
        fill = area / max(w * h, 1)
        if fill < 0.65:  # a chip's body is a solid rectangle (its pins may cling to it); a shadow or a cable is not
            continue
        if w < h:
            w, h, angle = h, w, angle + 90
        angle = (angle + 90) % 180 - 90
        w_mm, h_mm = w / px_per_mm, h / px_per_mm
        if w_mm < MIN_BODY_MM[0] or h_mm < MIN_BODY_MM[1] or w_mm > MAX_BODY_MM or w / h > 3.5:
            continue
        if best is None or area > best["area"]:
            corners = cv2.boxPoints(((cx, cy), (w, h), angle))
            x0, y0, bw, bh = stats[i][cv2.CC_STAT_LEFT], stats[i][cv2.CC_STAT_TOP], stats[i][cv2.CC_STAT_WIDTH], stats[i][cv2.CC_STAT_HEIGHT]
            cut = x0 <= 2 or y0 <= 2 or x0 + bw >= mask.shape[1] - 2 or y0 + bh >= mask.shape[0] - 2
            best = {"area": int(area), "center": (float(cx), float(cy)), "size_mm": (round(w_mm, 2), round(h_mm, 2)),
                    "angle_deg": round(float(angle), 1), "corners": corners.tolist(), "fill": round(float(fill), 2),
                    "share": round(float(area / mask.size), 3), "cut_by_edge": bool(cut)}
    return best


def _upright(frame, chip, mask=None):
    """The chip's body cut out and turned so its long axis is horizontal: the cut, its box in the turned picture, and
    the body's mask cut the same way (the mask of the whole picture, so its threshold is the picture's)."""
    (cx, cy), angle = chip["center"], chip["angle_deg"]
    corners = np.array(chip["corners"], np.float32)
    sides = np.linalg.norm(corners[1] - corners[0]), np.linalg.norm(corners[2] - corners[1])
    w, h = int(round(max(sides))), int(round(min(sides)))
    m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    turned = cv2.warpAffine(frame, m, (frame.shape[1], frame.shape[0]))
    mask = body_mask(frame) if mask is None else mask
    turned_mask = cv2.warpAffine(mask, m, (frame.shape[1], frame.shape[0]))
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    box = (slice(max(0, y0), y0 + h), slice(max(0, x0), x0 + w))
    return turned[box], (x0, y0, w, h), turned_mask[box]


def notch_end(frame, chip):
    """Which short end of the body has the notch, and the pin-1 dent near it: ("left" | "right", how sure 0..1,
    why), or (None, 0, why not). The notch is a half-round bite out of the body at one end, so the body's mask is
    missing there. The dent is a small dark dot in one corner. Both are judged on the straightened body."""
    cut, (x0, y0, w, h), inner = _upright(frame, chip)
    if cut.size == 0 or w < 20 or h < 20:
        return None, 0.0, "the body is too small in the picture"
    gray = cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)
    # the box may hold the pins too, which lean to one side: the body alone is where the mask is solid from top to
    # bottom of the middle half, so the ends are measured on that span, not the box's
    # bottom of the middle half, so the body's span is where the top and bottom quarters are both masked: the notch
    # bites only the middle, so it stays inside that span and shows as the missing middle at one end
    rows_top, rows_bottom = inner[int(h * 0.1):int(h * 0.3)], inner[int(h * 0.7):int(h * 0.9)]
    solid = np.flatnonzero((rows_top.mean(axis=0) > 0.6) & (rows_bottom.mean(axis=0) > 0.6))
    if solid.size < 20:
        return None, 0.0, "the body is not solid enough in the picture"
    x_left, x_right = int(solid[0]), int(solid[-1]) + 1
    inner, gray, cut = inner[:, x_left:x_right], gray[:, x_left:x_right], cut[:, x_left:x_right]
    w = x_right - x_left
    band = max(2, int(h * 0.25))
    edge = max(4, int(w * 0.12))
    body = inner[h // 2 - band:h // 2 + band]
    left, right = float(body[:, :edge].mean()), float(body[:, w - edge:].mean())
    notch = left - right  # negative: the left end is missing body, so the notch is on the left
    r = max(3, int(min(w, h) * 0.12))
    corners = {("left", "low"): gray[h - 2 * r:h, 0:2 * r], ("left", "high"): gray[0:2 * r, 0:2 * r],
               ("right", "low"): gray[h - 2 * r:h, w - 2 * r:w], ("right", "high"): gray[0:2 * r, w - 2 * r:w]}
    base = float(np.median(gray[r:h - r, r:w - r]))
    darkest = {k: base - float(np.percentile(v, 5)) for k, v in corners.items() if v.size}
    (dent_side, dent_row), dent_depth = max(darkest, key=darkest.get), max(darkest.values())
    dent_contrast = dent_depth - sorted(darkest.values())[-2]
    votes = []
    if abs(notch) > 0.08:
        votes.append(("left" if notch < 0 else "right", min(1.0, abs(notch) / 0.3), "the notch"))
    # a notch that is only moulded in relief shows as a drawn arc inside one end: edges where a plain body has none
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 25, 70)[h // 2 - int(h * 0.3):h // 2 + int(h * 0.3)]
    inset, wide = max(3, int(0.06 * w)), max(6, int(0.18 * w))
    drawn_left = float(edges[:, inset:inset + wide].mean()) / 255
    drawn_right = float(edges[:, w - inset - wide:w - inset].mean()) / 255
    drawn = drawn_right - drawn_left
    if abs(drawn) > 0.002 and max(drawn_left, drawn_right) > 2.5 * min(drawn_left, drawn_right):
        votes.append(("right" if drawn > 0 else "left", min(1.0, abs(drawn) / 0.008), "the notch's outline"))
    # a notch moulded in relief on a speckled body (the real 555): the body goes on behind it and the speckle
    # litters the edge counts, but after a strong blur the speckle is gone and the notch's arc stays, a half circle
    # sitting on the middle of one end, which Hough finds there and nowhere else
    big = cv2.GaussianBlur(gray, (9, 9), 0)
    arcs = {}
    for side in ("left", "right"):
        band_w = max(8, int(0.3 * w))
        band = big[:, :band_w] if side == "left" else big[:, w - band_w:]
        circles = cv2.HoughCircles(band, cv2.HOUGH_GRADIENT, dp=1, minDist=h, param1=60, param2=12,
                                   minRadius=max(3, int(0.08 * h)), maxRadius=max(4, int(0.22 * h)))
        for cx, cy, rr in ([] if circles is None else circles[0]):
            x_abs = cx if side == "left" else w - band_w + cx
            on_end = x_abs <= 0.08 * w if side == "left" else x_abs >= 0.92 * w
            if on_end and abs(cy - h / 2) <= 0.2 * h:
                arcs[side] = float(rr)
                break
    if len(arcs) == 1:
        # the half circle itself, seen at one end: it outweighs the body's mask missing at an end, which a light,
        # speckled top or a lead lying over it fakes (Nov 4: the mask said left on a 555 whose arc stood plainly on
        # the right, and every pin was counted from the wrong end, pin 1 taken for pin 5)
        votes.append((next(iter(arcs)), 1.5, "the notch's arc"))
    dent = dent_contrast > 18
    if dent:
        votes.append((dent_side, min(1.0, dent_contrast / 45), "the pin-1 dent"))
    if not votes:
        return None, 0.0, "no notch and no dent seen on this body"
    sides = {s: sum(c for side, c, _ in votes if side == s) for s in ("left", "right")}
    side = max(sides, key=sides.get)
    agree = [why for s, _, why in votes if s == side]
    disagree = [why for s, _, why in votes if s != side]
    sure = min(1.0, sides[side]) * (0.5 if disagree else 1.0)
    why = " and ".join(agree) + (f" (but {', '.join(disagree)} said the other end)" if disagree else "")
    chip["dent_row"] = dent_row if dent and dent_side == side else None  # which long side pin 1 is on, if the dent says
    return side, round(sure, 2), why


def _pin_place(chip, end, number=1, pins=None, across_mm=None):
    """A point at a pin's place along the body, across_mm from its long axis (the leg's row if None), in pixels. Seen
    from above with the notch to the left, pin 1 is on the lower side and the numbers run away from the notch along it
    and back along the upper side, so the last pin stands right across from pin 1; with the notch to the right, the chip
    is that one turned half a turn. "Lower" is in the straightened body's frame, which the dent confirms when it was
    seen. A body's corner is not its pin: the pins stand on the 2.54 mm grid, so pin 1 of a DIP-8 is 3.81 mm from the
    centre along the body and 3.81 mm across it (rows 7.62 mm apart). pins is how many the chip has: its body does not
    tell a DIP-14 from a DIP-16, so a pin other than 1 of a long chip needs it."""
    corners = np.array(chip["corners"], np.float32)
    angle = math.radians(chip["angle_deg"])
    along = np.array([math.cos(angle), math.sin(angle)])
    across = np.array([-math.sin(angle), math.cos(angle)])  # the straightened body's "down"
    center = np.array(chip["center"])
    length_px = float(np.ptp((corners - center) @ along))
    length_mm, width_mm = chip["size_mm"]
    px_per_mm = length_px / length_mm
    if pins is None:
        pins = 8 if length_mm < 12 else 0
    half = pins // 2
    first = (half - 1) * 1.27 if pins else length_mm / 2 - 1.27  # the longer DIPs: the first pin near the end
    down_body = first - 2.54 * (number - 1 if number <= half or not pins else pins - number)
    if across_mm is None:
        across_mm = 3.81 if width_mm < 10 else width_mm / 2 + 0.6  # a wide DIP has its rows 15.24 mm apart
    row = chip.get("dent_row") or ("low" if end == "left" else "high")
    side = -1.0 if end == "left" else 1.0
    far_row = pins and number > half  # the numbers come back along the other long side
    lower = (row == "low") != bool(far_row)
    point = center + side * down_body * px_per_mm * along + (1.0 if lower else -1.0) * across_mm * px_per_mm * across
    return float(point[0]), float(point[1])


def notch_at(chip, end):
    """The middle of the body's end with the notch, in pixels."""
    angle = math.radians(chip["angle_deg"])
    along = np.array([math.cos(angle), math.sin(angle)])
    center = np.array(chip["center"])
    length_px = float(np.ptp((np.array(chip["corners"], np.float32) - center) @ along))
    point = center + (-1.0 if end == "left" else 1.0) * length_px / 2 * along
    return float(point[0]), float(point[1])


def pin_at(chip, end, number=1, pins=None):
    """A pin's leg in the picture, in pixels."""
    return _pin_place(chip, end, number, pins)


def beside_pin(chip, end, number=1, pins=None):
    """The point on the body's flat top right beside a pin's leg, in pixels: where a laser's dot can be seen and
    checked. The leg itself is a few mm lower and shiny, and the slanted beam misses it or the camera loses it."""
    return _pin_place(chip, end, number, pins, across_mm=BESIDE_PIN_MM)


def pin1(chip, end):
    """Pin 1's leg in the picture, in pixels."""
    return pin_at(chip, end)


def beside_pin1(chip, end):
    """The point on the body's flat top beside pin 1's leg, in pixels."""
    return beside_pin(chip, end)


def on_body(frame, x, y):
    """Whether a point of a picture lies on the chip: on the biggest patch of the body's mask, since a breadboard's
    holes show as small dark patches too and a dot lost in one must not pass for a dot on the chip."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(body_mask(frame))
    if n < 2:
        return False
    return int(labels[int(y), int(x)]) == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def orientation(chip, end):
    """The direction the notch faces in the picture, in degrees from the picture's +x axis (0 right, 90 down, 180
    left, -90 up), and in words."""
    angle = chip["angle_deg"] + (180 if end == "left" else 0)
    angle = (angle + 180) % 360 - 180
    words = {0: "notch to the right", 90: "notch down", 180: "notch to the left", -180: "notch to the left",
             -90: "notch up"}
    nearest = min(words, key=lambda a: abs((angle - a + 180) % 360 - 180))
    return round(angle, 1), words[nearest]


def breadboard_grid(frame, px_per_mm, chip_px=None, chip_row=None):
    """Where a breadboard's holes stand in a mosaic of it, row 1 at the left and column a at the bottom, from the holes
    themselves: a comb at 2.54 mm finds the block, then each hole is looked for near its place and the whole grid is
    fitted to them, turned as the board lies. A chip's top stands higher than the board and shows shifted, so the holes
    are the way to know them. chip_px, where a DIP on the board is in the mosaic, keeps the comb to the places that put
    the middle channel under it: the rails and the parts beside them match the comb too (Nov 3: five columns off).
    chip_row, the row its centre is on (from 1), keeps the rows too: a mosaic wider than the board slid them 15 rows.
    {"a1", "a63", "j1": (x, y) in pixels, "holes": how many, "off_px": their typical miss}, or None if it is not a
    breadboard's grid."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    dark = np.clip(cv2.blur(gray, (121, 121)) - gray, 0, None)
    pitch = 2.54 * px_per_mm
    columns = np.array(list(range(5)) + list(range(7, 12)), float)
    rows = np.arange(63, dtype=float)
    if w < 63 * pitch or h < 12 * pitch:
        return None
    best = None
    span = np.arange(int(w * 0.1), int(w * 0.9), 40)
    for tilt in np.arange(-0.015, 0.0151, 0.001):  # across the block's whole length the tilt shifts a row by a whole pitch
        ys = np.arange(h)[None, :] + (tilt * span)[:, None]
        profile = np.mean([np.interp(np.arange(h), ys[i], dark[:, span[i]]) for i in range(len(span))], axis=0)
        for y0 in range(int(11 * pitch) + 1, h - 1):
            if chip_px is not None and abs(y0 - tilt * chip_px[0] - 5.5 * pitch - chip_px[1]) > 1.5 * pitch:
                continue  # column c runs at y0 - tilt x - c pitch, and the channel, 5.5, under the chip
            value = profile[(y0 - columns * pitch).astype(int)].mean()
            if best is None or value > best[0]:
                best = (float(value), float(tilt), float(y0))
    if best is None:
        return None
    _, tilt, y0 = best
    x_best = None
    for x0 in range(0, int(w - 62 * pitch) - 1, 2):
        if chip_px is not None and chip_row is not None and abs((chip_px[0] - x0) / pitch - (chip_row - 1)) > 2.5:
            continue
        xs = (x0 + rows * pitch).astype(int)
        yy = (y0 - tilt * xs).astype(int)
        band = sum(dark[np.clip((yy - c * pitch).astype(int), 0, h - 1), xs].sum() for c in range(5))
        if x_best is None or band > x_best[0]:
            x_best = (float(band), float(x0))
    if x_best is None:
        return None
    x0 = x_best[1]
    gx, gy = pitch, 0.0  # one row along, then one column up, in pixels
    fit = np.array([[x0, gx, tilt * pitch], [y0 - tilt * x0, -tilt * pitch, -pitch]])  # x, y from 1, row, column
    reach = int(0.32 * pitch)
    for _ in range(3):
        found, where = [], []
        for r in rows:
            for c in columns:
                px, py = fit @ np.array([1.0, r, c])
                x, y = int(round(px)), int(round(py))
                if not (reach <= x < w - reach and reach <= y < h - reach):
                    continue
                patch = dark[y - reach:y + reach + 1, x - reach:x + reach + 1]
                mask = patch > 0.5 * patch.max()
                if patch.max() < 12 or not (0.04 * patch.size < mask.sum() < 0.5 * patch.size):
                    continue
                vy, vx = np.nonzero(mask)
                found.append((x - reach + vx.mean(), y - reach + vy.mean()))
                where.append((1.0, r, c))
        if len(found) < 200:
            return None
        A, B = np.array(where), np.array(found)
        keep = np.ones(len(A), bool)
        for _ in range(3):
            fit = np.linalg.lstsq(A[keep], B[keep], rcond=None)[0].T
            miss = np.linalg.norm(A @ fit.T - B, axis=1)
            keep = miss < max(6.0, 2.5 * np.median(miss[keep]))
    if abs(math.hypot(fit[0, 1], fit[1, 1]) / pitch - 1) > 0.03:
        return None
    at = lambda r, c: tuple(float(v) for v in fit @ np.array([1.0, r, c]))  # noqa: E731
    return {"a1": at(0, 0), "a63": at(62, 0), "j1": at(0, 11), "holes": int(keep.sum()), "off_px": round(float(np.median(miss[keep])), 1)}


def pointer_tip(frame, px_per_mm):
    """The tip of a pointer laid over the board (a wooden stick, a toothpick): a long thin thing that comes in from the
    picture's edge; its tip is the end farthest from where it enters. Returns ((x, y), width_mm) or None."""
    hsv = cv2.cvtColor(cv2.medianBlur(frame, 7), cv2.COLOR_BGR2HSV)
    # wood: a warm hue (OpenCV's hue 0..30 is red to yellow), more saturated than a breadboard's white and a chip's grey
    wood = ((hsv[..., 0] <= 30) & (hsv[..., 1] > 80) & (hsv[..., 2] > 70)).astype(np.uint8)
    wood = cv2.morphologyEx(wood, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(wood)
    h, w = wood.shape
    best = None
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 1.0 * px_per_mm ** 2:
            continue
        touches = x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
        if not touches:
            continue
        points = np.column_stack(np.nonzero(labels == i))[:, ::-1].astype(np.float32)
        (cx, cy), (rw, rh), angle = cv2.minAreaRect(points)
        long, short = max(rw, rh), min(rw, rh)
        if long < 2.5 * short or short / px_per_mm > 6:
            continue
        if best is None or area > best[0]:
            best = (area, points, (cx, cy), long, short, angle if rw >= rh else angle + 90)
    if best is None:
        return None
    _, points, (cx, cy), long, short, angle = best
    axis = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
    along = (points - [cx, cy]) @ axis
    ends = points[int(np.argmin(along))], points[int(np.argmax(along))]
    # the tip is the end farther from the picture's edges: the stick comes in from an edge
    def edge_distance(p):
        return min(p[0], p[1], w - 1 - p[0], h - 1 - p[1])
    tip = max(ends, key=edge_distance)
    return (float(tip[0]), float(tip[1])), round(short / px_per_mm, 2)
