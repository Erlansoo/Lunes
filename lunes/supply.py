"""Reads the bench supply's front panel with the bench camera: volts, amps and its lights.

The KORAD KA3010D has no USB, so Lunes reads it as Erlan does, by looking at it, and with no model: the numbers come from
this code, segment by segment, and a reading it is not sure of is no reading. The digits are multiplexed LEDs, so one
picture can catch some of them dark: several pictures are stacked, keeping each pixel's brightest moment.

The two red windows (volts above, amps below) are found by their colour, straightened, and the slant of the digits is
taken out. Each window holds four seven-segment digits. With the output off the panel shows what is set; with it on, what
comes out, and C.V or C.C says which limit is acting. So the limit set is only known from a reading with OUT off.
"""
import time

import cv2
import numpy as np

WIDTH, HEIGHT = 340, 150  # a window straightened
DIGIT = (64, 110)
MARGIN_X, MARGIN_Y = 4, 2
# where each of the seven segments is sampled in a digit's box, as fractions of it: top, top right, bottom right, bottom,
# bottom left, top left, middle
SEGMENTS = {"a": (0.5, 0.08), "b": (0.88, 0.28), "c": (0.88, 0.72), "d": (0.5, 0.92), "e": (0.12, 0.72),
            "f": (0.12, 0.28), "g": (0.5, 0.5)}
DIGITS = {"abcdef": 0, "bc": 1, "abdeg": 2, "abcdg": 3, "bcfg": 4, "acdfg": 5, "acdefg": 6, "abc": 7, "abcdefg": 8,
          "abcdfg": 9}
LIT = 215  # the red of a light that is on; an off one still shows a dull red
FIT, MARGIN = 5.0, 0.3  # of 7 segments: how well a digit must fit, and how much better than the next best
MAX_VOLTS, MAX_AMPS = 31.0, 10.3  # past these a reading is a misread: a 30 V, 10 A supply
REDS = dict(red=150, over=70)


def stack(frames):
    """Each pixel's brightest moment, over pictures of the same scene: the multiplexed digits all lit."""
    return np.max(np.array(frames), axis=0)


def _red(image):
    b, g, r = cv2.split(image.astype(np.int16))
    return ((r > REDS["red"]) & (r - g > REDS["over"]) & (r - b > REDS["over"])).astype(np.uint8)


def _windows(image):
    """The quadrilaterals of the two windows, volts first, or None: the two widest red patches, one above the other."""
    closed = cv2.morphologyEx(_red(image), cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, labels, stats, centres = cv2.connectedComponentsWithStats(closed)
    found = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w > 1.5 * h and h >= 30 and area > 0.6 * w * h:
            found.append((area, i))
    found = sorted(sorted(found, reverse=True)[:2], key=lambda f: centres[f[1]][1])
    if len(found) != 2:
        return None
    a, b = found[0][1], found[1][1]
    if abs(stats[a][cv2.CC_STAT_WIDTH] - stats[b][cv2.CC_STAT_WIDTH]) > 0.3 * stats[a][cv2.CC_STAT_WIDTH] or \
            abs(centres[a][0] - centres[b][0]) > 0.3 * stats[a][cv2.CC_STAT_WIDTH]:
        return None
    return [_corners(labels == k) for k in (a, b)]


def _corners(mask):
    """A patch's corners, top left first, clockwise."""
    points = cv2.boxPoints(cv2.minAreaRect(np.column_stack(np.nonzero(mask))[:, ::-1].astype(np.float32)))
    order = np.argsort(points[:, 1])
    top, bottom = points[order[:2]], points[order[2:]]
    top, bottom = top[np.argsort(top[:, 0])], bottom[np.argsort(bottom[:, 0])]
    return np.array([top[0], top[1], bottom[1], bottom[0]], np.float32)


def _core(image):
    """The lit segments: the brightest part of the digits, with the glow around them left out."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return (gray > 140).astype(np.uint8)


def _straight(image, corners):
    box = np.array([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]], np.float32)
    return cv2.warpPerspective(_core(image) * 255, cv2.getPerspectiveTransform(corners, box), (WIDTH, HEIGHT))


def _unslanted(window):
    """The window with the digits upright: the shear that makes their vertical strokes line up best."""
    best = (-1.0, window)
    for shear in np.arange(-0.30, 0.31, 0.02):
        m = np.array([[1, shear, -shear * HEIGHT / 2], [0, 1, 0]], np.float32)
        sheared = cv2.warpAffine(window, m, (WIDTH, HEIGHT))
        score = float(np.var(sheared.sum(axis=0)))
        if score > best[0]:
            best = (score, sheared)
    return best[1]


def _cells(window):
    """Four digit boxes (x0, x1) and the rows (y0, y1) of the digits, from what is lit."""
    rows = np.flatnonzero(window.sum(axis=1) > 255 * 6)
    cols = np.flatnonzero(window.sum(axis=0) > 255 * 3)
    if rows.size < 30 or cols.size < 40:
        return None
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    height = y1 - y0
    width = height * 0.56  # the digits of this panel, as tall as they are wide by this much
    right = int(cols[-1]) + 1
    upper = window[y0:y0 + int(height * 0.72)].sum(axis=0) > 255 * 2  # above the decimal point, which joins two digits
    edges = np.flatnonzero(np.diff(np.r_[0, upper.astype(np.int8), 0]))
    runs = [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2]) if b - a >= 4]
    if len(runs) == 4:  # each digit as wide as it really is; a 1 is narrow, on the right of a box as wide as the others
        wide = max(max(b - a for a, b in runs), 0.5 * height)  # if all are 1s, their height says
        return [(a if b - a >= 0.7 * wide else b - wide, b) for a, b in runs], (y0, y1)
    pitch = (right - width - (WIDTH * 0.03)) / 3
    return [(right - width - pitch * (3 - k), right - pitch * (3 - k)) for k in range(4)], (y0, y1)


def _digit(window, box, rows):
    """(digit or None, how clear it was) from one digit's box: the digit whose segments best fit how lit each one is, if
    it fits well and clearly better than the next best. A segment caught at its edge reads half lit, so no fixed line
    between lit and dark would do."""
    (x0, x1), (y0, y1) = box, rows
    levels = {}
    for name, (fx, fy) in SEGMENTS.items():
        x, y = int(x0 + fx * (x1 - x0)), int(y0 + fy * (y1 - y0))
        r = max(3, int((x1 - x0) * 0.10))
        levels[name] = float(window[max(0, y - r):y + r, max(0, x - r):x + r].mean()) / 255
    fits = sorted(((sum(levels[n] if n in lit else 1 - levels[n] for n in levels), digit)
                   for lit, digit in DIGITS.items()), reverse=True)
    (best, digit), (second, _) = fits[0], fits[1]
    return (digit if best >= FIT and best - second >= MARGIN else None), round(best - second, 2)


def _point(window, boxes, rows):
    """Which digit the decimal point follows: the gap at the foot of the row with the most light."""
    y0, y1 = rows
    foot = window[int(y1 - (y1 - y0) * 0.16):min(y1 + 3, HEIGHT)]
    scores = []
    for k in range(3):
        a, b = int(boxes[k][1] - 2), int(boxes[k + 1][0] + 2)
        scores.append(float(foot[:, max(a, 0):max(b, a + 1)].mean()))
    return int(np.argmax(scores)) + 1


def _row(image, corners):
    window = _unslanted(_straight(image, corners))
    cells = _cells(window)
    if cells is None:
        return None
    boxes, rows = cells
    digits, clear = [], 7.0
    for box in boxes:
        d, c = _digit(window, box, rows)
        digits.append(d)
        clear = min(clear, c)
    return {"digits": digits, "point": _point(window, boxes, rows), "clear": round(clear, 2)}


# where the panel's lights are, in widths and heights of the volts window from its top right corner
LIGHTS = {"ovp": (0.19, 0.21), "ocp": (0.17, 0.68), "cc": (0.157, 1.15), "cv": (0.14, 1.6), "out": (0.127, 2.07)}


def _light(image, corners, right_of, below):
    """How bright the panel is at a spot given in window widths and heights from the volts window's top right."""
    across = (corners[1] - corners[0]) / np.linalg.norm(corners[1] - corners[0])
    down = (corners[3] - corners[0]) / np.linalg.norm(corners[3] - corners[0])
    width, height = np.linalg.norm(corners[1] - corners[0]), np.linalg.norm(corners[3] - corners[0])
    x, y = corners[1] + across * right_of * width + down * below * height
    r = max(4, int(height * 0.12))
    x, y = int(x), int(y)
    patch = image[max(0, y - r):y + r, max(0, x - r):x + r]
    return float(patch[..., 2].max()) if patch.size else 0.0


def read(image):
    """{"ok", "volts", "amps", "out", "cc", "cv"} from a picture of the panel; {"ok": False, "why"} if it cannot be sure."""
    corners = _windows(image)
    if corners is None:
        return {"ok": False, "why": "no veo los dos números rojos de la fuente en la imagen"}
    volts, amps = _row(image, corners[0]), _row(image, corners[1])
    if volts is None or amps is None:
        return {"ok": False, "why": "veo la pantalla de la fuente pero no separo sus dígitos"}
    if None in volts["digits"] or None in amps["digits"]:
        return {"ok": False, "why": "hay un dígito de la fuente que no alcanzo a leer", "digits": [volts, amps]}

    def number(row):
        text = "".join(str(d) for d in row["digits"])
        return float(text[:row["point"]] + "." + text[row["point"]:])
    v, a = number(volts), number(amps)
    if v > MAX_VOLTS or a > MAX_AMPS:
        return {"ok": False, "why": f"leí {v:g} V y {a:g} A, imposible para esta fuente", "digits": [volts, amps]}
    lights = {name: round(_light(image, corners[0], *at)) for name, at in LIGHTS.items()}
    return {"ok": True, "volts": v, "amps": a, "clear": min(volts["clear"], amps["clear"]), "lights": lights,
            "out": lights["out"] > LIT, "cc": lights["cc"] > LIT, "cv": lights["cv"] > LIT}


def look(camera, pictures=12, gap_s=0.03):
    """The panel read from a few pictures of a camera with a frame() method."""
    frames = []
    for _ in range(pictures):
        frame = camera.frame()
        if frame is not None:
            frames.append(frame.copy())
        time.sleep(gap_s)
    if not frames:
        return {"ok": False, "why": "la cámara del banco no da imagen"}
    return read(stack(frames))


def needed_amps(circuit, sim):
    """About what the circuit draws from VCC at most: the peak of every part that touches it."""
    total = 0.0
    for part in circuit.get("parts", []):
        pins = part.get("pins") or []
        nets = pins.values() if isinstance(pins, dict) else pins
        if "VCC" in nets and part.get("ref") in sim.get("currents", {}):
            total += sim["currents"][part["ref"]]["peak_a"]
    return total


def check(reading, circuit, sim):
    """Whether the circuit can be powered from what the supply shows: {"safe", "problems", "advice", "limit_a"}. The volts
    must match; a current limit far above what it draws would let a fault burn parts instead of stopping it."""
    want, problems, advice = float(circuit.get("supply_v", 5)), [], []
    if not reading.get("out") and abs(reading["volts"] - want) > 0.03 * want + 0.02:
        problems.append(f"la fuente está en {reading['volts']:.2f} V y el circuito pide {want:g} V")
    if reading.get("out"):
        advice.append("la salida está encendida: lo que marca es lo que sale, no lo que está configurado")
    limit = max(0.1, float(np.ceil(3 * needed_amps(circuit, sim) / 0.05)) * 0.05)  # chips draw some too
    if not reading.get("out") and reading["amps"] > 2 * limit:
        advice.append(f"el límite está en {reading['amps']:.3f} A y este circuito pide poco: con {limit:.2f} A una "
                      f"falla no quema nada")
    return {"safe": not problems, "problems": problems, "advice": advice, "limit_a": round(limit, 2), "want_v": want}
