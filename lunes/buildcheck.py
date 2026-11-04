"""Checking a build against its plan, hole by hole: the way a technician checks a breadboard, not a guess at a whole
picture. For every lead and wire end of the plan the mosaic is cut around its hole, the cuts are laid out as one
labelled sheet, and the vision model answers one plain question per tile: is there a lead or a wire end in the ringed
hole? The code then marks every part and wire as there, missing or unclear, and asks the simulator what the missing
ones do, so the finding can be checked against what the instruments measured.

Before any of that the plan is checked to fall on the real board: a breadboard's grid of holes looks the same turned
half a turn, and twice on Nov 3 Lunes read the board that way round (pin 1 taken for f60). The chip is the one thing
that breaks the symmetry: the plan's chip area must be dark in the mosaic (a black DIP body). If it is not, but is
with the grid turned round, the grid is turned and Lunes says so.
"""
import json
import math
import re

import cv2
import numpy as np

from lunes.breadboard import BoardGrid, PITCH, is_chip, pin_key

TILE = 170  # px of each cut on the sheet
COLUMNS = 6
HALF_PITCHES = 1.7  # the cut reaches this many pitches round the hole: its neighbours show, so a lead one off is seen


def turned(grid):
    """The same holes named the other way round (row r is 64 - r, column a is j): a1, a63, j1 of the turned grid."""
    a1, a63, j1 = grid["a1"], grid["a63"], grid["j1"]
    j63 = [a63[0] + j1[0] - a1[0], a63[1] + j1[1] - a1[1]]
    return {"a1": [round(v, 2) for v in j63], "a63": list(j1), "j1": list(a63)}


def _px(summary, grid, hole):
    from survey import mm_to_mosaic_px
    x, y = BoardGrid(grid["a1"], grid["a63"], grid["j1"]).mm(hole)
    return mm_to_mosaic_px(summary, x, y)


def _ppm(summary):
    p = summary.get("px_per_mm")
    return float(np.mean(list(p.values()))) if isinstance(p, dict) else float(p or 38.0)


def chip_darkness(mosaic, summary, grid, layout):
    """How dark the plan's chips are in the mosaic under this grid (0 black to 255 white), or None without a chip:
    the mean brightness over the middle of each chip's body, between its two rows of pins."""
    values = []
    for ref, chip in (layout.get("chips") or {}).items():
        pins = list((layout["holes"].get(ref) or {}).values())
        if len(pins) < 4:
            continue
        pts = np.array([_px(summary, grid, h) for h in pins])
        cx, cy = pts.mean(axis=0)
        r = max(4, int(0.6 * PITCH * _ppm(summary)))
        h, w = mosaic.shape[:2]
        if not (r <= cx < w - r and r <= cy < h - r):
            values.append(255.0)
            continue
        patch = cv2.cvtColor(mosaic[int(cy) - r:int(cy) + r, int(cx) - r:int(cx) + r], cv2.COLOR_BGR2GRAY)
        values.append(float(patch.mean()))
    return float(np.mean(values)) if values else None


def shifted(grid, columns):
    """The same grid slid along the board by a number of columns: hole N of the result is hole N + columns of grid."""
    a1, a63, j1 = (np.array(grid[k], float) for k in ("a1", "a63", "j1"))
    step = (a63 - a1) / 62
    return {"a1": [round(v, 2) for v in a1 + columns * step], "a63": [round(v, 2) for v in a63 + columns * step],
            "j1": [round(v, 2) for v in j1 + columns * step]}


def orient(mosaic, summary, grid, layout):
    """(grid, note): the grid as read, or the one the holes also fit that puts the plan's chip on a dark body.

    A breadboard's holes fit a grid in more than one way: turned half round (row r is 64 - r, column a is j), and slid
    by a group of five columns, since the rails' holes come in fives (Nov 3, 23:13: both at once put the 555, standing
    in e3-f6, at column 53 and sent the laser to the empty end). Of those, the one whose chip area is darkest wins, if
    it is clearly darker than the grid as read."""
    here = chip_darkness(mosaic, summary, grid, layout)
    if here is None:
        return grid, "no chip in the plan to check the board's orientation with"
    best, best_dark, how = grid, here, None
    for turn in (False, True):
        base = turned(grid) if turn else grid
        for slide in (-10, -5, 0, 5, 10):
            if not turn and slide == 0:
                continue
            candidate = shifted(base, slide) if slide else base
            dark = chip_darkness(mosaic, summary, candidate, layout)
            if dark is not None and dark < best_dark:
                best, best_dark, how = candidate, dark, (turn, slide)
    if how is not None and best_dark < 0.6 * here and best_dark < 110:
        words = " and ".join(w for w in ("turned half round" if how[0] else "",
                                          f"slid {abs(how[1])} columns" if how[1] else "") if w)
        return dict(grid, **best), (f"the grid was read {words} (the plan's chip area was bright, {here:.0f}, and dark, "
                                    f"{best_dark:.0f}, on the other fit): turned back")
    return grid, f"the plan's chip falls on a dark body ({here:.0f} of 255): the board is read the right way round"


def items(circuit, layout):
    """[(what, kind, [holes])] for every part's leads (chips are seen as a body, not hole by hole) and every wire."""
    out = []
    for part in circuit["parts"]:
        holes = layout["holes"].get(part["ref"]) or {}
        if not holes or is_chip(part):
            continue
        label = f"{part['ref']} {part.get('value') or part.get('color') or part.get('model') or part['kind']}"
        out.append((label, "part", [str(h) for h in holes.values()]))
    for w in layout.get("wires", []):
        out.append((f"cable {w.get('color', '')} {w['from']}-{w['to']} ({w.get('net', '')})", "wire", [w["from"], w["to"]]))
    return out


def sheet(mosaic, summary, grid, holes):
    """One picture of every hole's cut, each labelled with its hole and its target ringed: (image, holes in order)."""
    ppm = _ppm(summary)
    half = int(HALF_PITCHES * PITCH * ppm)
    rows = math.ceil(len(holes) / COLUMNS)
    label_h = 28
    out = np.full((rows * (TILE + label_h), COLUMNS * TILE, 3), 255, np.uint8)
    H, W = mosaic.shape[:2]
    for i, hole in enumerate(holes):
        cx, cy = (int(round(v)) for v in _px(summary, grid, hole))
        x0, y0 = max(0, cx - half), max(0, cy - half)
        cut = mosaic[y0:min(H, cy + half), x0:min(W, cx + half)]
        tile = np.full((2 * half, 2 * half, 3), 200, np.uint8)
        if cut.size:
            tile[y0 - (cy - half):y0 - (cy - half) + cut.shape[0], x0 - (cx - half):x0 - (cx - half) + cut.shape[1]] = cut
        tile = cv2.resize(tile, (TILE, TILE), interpolation=cv2.INTER_AREA)
        ring = max(6, int(0.55 * PITCH * ppm * TILE / (2 * half)))
        cv2.circle(tile, (TILE // 2, TILE // 2), ring, (255, 255, 0), 2)
        r, c = divmod(i, COLUMNS)
        y, x = r * (TILE + label_h), c * TILE
        out[y + label_h:y + label_h + TILE, x:x + TILE] = tile
        cv2.putText(out, hole, (x + 6, y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.rectangle(out, (x, y), (x + TILE - 1, y + label_h + TILE - 1), (120, 120, 120), 1)
    return out


SYSTEM = ("You inspect close-up photographs of a breadboard for an electronics bench assistant. Look at each tile on its "
          "own, decide only what the question asks, and answer with the JSON object alone: no prose, no markdown.")
QUESTION = ("Each tile is a close-up of a solderless breadboard around ONE hole, ringed in cyan, with its name above "
            "the tile. The holes in the same vertical line as the ring (directly above and below it) are joined inside "
            "the board, so a lead there counts the same. For every tile answer: \"yes\" if a component lead or the "
            "bare end of a jumper wire goes into the ringed hole or the hole directly above or below it; \"covered\" "
            "if a part's body, a big component or a wire hides that spot so you cannot see the holes; \"no\" only if "
            "you can see those holes and they are empty (a wire passing over without going in is not in). Answer only "
            "with JSON, one key per tile name: {\"b5\": \"yes\", \"h6\": \"no\", ...}")


def parse(answer, holes):
    m = re.search(r"\{.*\}", answer or "", re.S)
    try:
        got = json.loads(m.group(0)) if m else {}
    except ValueError:
        got = {}
    out = {}
    for h in holes:
        v = str(got.get(h, "unclear")).lower()
        out[h] = v if v in ("yes", "no") else "unclear"  # covered or anything else: not to be called wrong
    return out


SECOND = ("Second, careful look. Each row is ONE connection that a first quick pass thought was missing: its two ends, "
          "each a close-up of a breadboard hole ringed in cyan (the holes directly above and below the ring are joined "
          "inside the board and count the same). Look hard: a lead or wire end can be thin, shiny, half hidden by a "
          "part or a wire passing over it. For each connection say \"present\" if both ends have a lead or wire end "
          "going in, \"missing\" if at least one end is clearly visible and empty, \"unsure\" if you cannot tell. Add "
          "how sure you are from 0 to 1. Each row starts with its number. Only JSON, keyed by the row number: "
          "{\"1\": {\"verdict\": \"missing\", \"sure\": 0.8}, \"2\": {...}, ...}")


def second_sheet(mosaic, summary, grid, suspects):
    """One row per suspect connection, its ends side by side, bigger than the first pass, named on the left."""
    rows = []
    for n, (name, holes) in enumerate(suspects, 1):
        tiles = sheet(mosaic, summary, grid, holes)  # its ends, at the first pass' size
        tiles = cv2.resize(tiles[:, :len(holes) * TILE], None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
        band = np.full((tiles.shape[0], 380, 3), 255, np.uint8)
        cv2.putText(band, f"{n}.", (8, tiles.shape[0] // 2 - 14), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(band, name[:30], (8, tiles.shape[0] // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        rows.append(np.hstack([band, tiles]))
    width = max(r.shape[1] for r in rows)
    return np.vstack([np.pad(r, ((0, 6), (0, width - r.shape[1]), (0, 0)), constant_values=255) for r in rows])


def parse_second(answer, names):
    m = re.search(r"\{.*\}", answer or "", re.S)
    try:
        got = json.loads(m.group(0)) if m else {}
    except ValueError:
        got = {}
    out = {}
    for i, n in enumerate(names, 1):
        v = got.get(str(i)) or got.get(n) or got.get(n[:30]) or {}
        verdict = str(v.get("verdict", "unsure")).lower() if isinstance(v, dict) else "unsure"
        try:
            sure = float(v.get("sure", 0)) if isinstance(v, dict) else 0.0
        except (TypeError, ValueError):
            sure = 0.0
        out[n] = {"verdict": verdict if verdict in ("present", "missing") else "unsure", "sure": round(sure, 2)}
    return out


COUNT = ("This is a whole solderless breadboard seen from above. Count the jumper wires on it: insulated wires with "
         "both ends plugged into the board's holes. Do NOT count the two thicker supply leads that come in from "
         "outside the board, an oscilloscope probe or its clip, or the legs of components (resistors, LED, chip, "
         "potentiometer, capacitor). Only JSON: {\"jumpers\": <number>, \"supply_leads\": <number>, \"sure\": <0 to 1>}")


def parse_count(answer):
    m = re.search(r"\{.*\}", answer or "", re.S)
    try:
        got = json.loads(m.group(0)) if m else {}
        return int(got.get("jumpers")), float(got.get("sure", 0))
    except (TypeError, ValueError):
        return None, 0.0


def explains(predicted, measured, tolerance=0.4):
    """Whether a simulated fault explains what the instruments read: every measured net, as the fault's simulation
    predicts it, within tolerance. predicted: {net: volts}; measured: {net: volts}. None with nothing to compare."""
    common = [n for n in measured if n in predicted]
    if not common:
        return None
    return all(abs(predicted[n] - measured[n]) <= tolerance for n in common)
