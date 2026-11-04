"""Seeing a DIP chip and its pin 1, no hardware: python tests/test_chip.py

Chips are drawn as the microscope sees them on a breadboard: the white board with its holes, a dark body with its pins, a
notch at one end and a dent at pin 1, turned any way. The body must be found with its size and angle, the notch end and
pin 1's corner must be right, a body with no notch must say so, and a wooden pointer's tip must be found.
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lunes import chip  # noqa: E402

FAILED = []
PPM = 42.4  # the microscope's pixels per mm


def ok(condition, what):
    print(("  ok   " if condition else "  FAIL ") + what)
    if not condition:
        FAILED.append(what)


def breadboard_picture():
    img = np.full((480, 640, 3), (232, 236, 240), np.uint8)
    for y in range(20, 480, int(2.54 * PPM)):
        for x in range(10, 640, int(2.54 * PPM)):
            cv2.rectangle(img, (x, y), (x + 8, y + 8), (60, 60, 60), -1)
    return img


def dip(angle_deg=0.0, notch="left", dent=True, center=(320, 240), pins=8):
    """A DIP-8 body 9.3 x 6.3 mm with the notch at one short end, drawn turned by angle_deg."""
    img = breadboard_picture()
    w, h = 9.3 * PPM, 6.3 * PPM
    body = np.full((int(h) + 60, int(w) + 60, 3), (232, 236, 240), np.uint8)
    cx, cy = body.shape[1] / 2, body.shape[0] / 2
    cv2.rectangle(body, (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2)), (52, 50, 48), -1)
    for k in range(pins // 2):  # pins out of both long sides
        x = int(cx - w / 2 + (k + 0.5) * w / (pins // 2))
        cv2.rectangle(body, (x - 5, int(cy - h / 2) - 22), (x + 5, int(cy - h / 2)), (170, 175, 180), -1)
        cv2.rectangle(body, (x - 5, int(cy + h / 2)), (x + 5, int(cy + h / 2) + 22), (170, 175, 180), -1)
    nx = int(cx - w / 2) if notch == "left" else int(cx + w / 2)
    cv2.circle(body, (nx, int(cy)), int(0.9 * PPM), (232, 236, 240), -1)
    if dent:
        dx = int(cx - w / 2 + 1.0 * PPM) if notch == "left" else int(cx + w / 2 - 1.0 * PPM)
        cv2.circle(body, (dx, int(cy + h / 2 - 1.0 * PPM)), int(0.35 * PPM), (22, 22, 22), -1)
    m = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)
    m[:, 2] += (center[0] - cx, center[1] - cy)
    turned = cv2.warpAffine(body, m, (640, 480), borderValue=(0, 0, 0))
    mask = cv2.warpAffine(np.full(body.shape[:2], 255, np.uint8), m, (640, 480))
    img[mask > 0] = turned[mask > 0]
    # where pin 1's leg lands: 3.81 mm from the centre each way. The notch left: lower left. The notch right: the chip
    # turned half a turn, upper right, unless the dent drawn (lower right here) says otherwise, and the dent wins
    low = notch == "left" or dent
    leg = np.array([cx + (-3.81 if notch == "left" else 3.81) * PPM, cy + (3.81 if low else -3.81) * PPM])
    pin1 = m @ np.array([leg[0], leg[1], 1.0])
    return cv2.GaussianBlur(img, (3, 3), 0), (float(pin1[0]), float(pin1[1]))


def bodies():
    print("The body, its size and its angle")
    for angle in (0.0, 15.0, 90.0, -40.0, 180.0):
        img, _ = dip(angle)
        found = chip.find_chip(img, PPM)
        ok(found is not None and abs(found["size_mm"][0] - 9.3) < 0.6 and abs(found["size_mm"][1] - 6.3) < 0.6
           and abs((found["angle_deg"] - angle + 90) % 180 - 90) < 3,
           f"turned {angle:g}: " + (f"{found['size_mm']} mm at {found['angle_deg']} deg" if found else "not found"))
    ok(chip.find_chip(breadboard_picture(), PPM) is None, "an empty breadboard has no chip")


def pin_one():
    print("The notch end and pin 1")
    for angle, notch in ((0.0, "left"), (0.0, "right"), (90.0, "left"), (-30.0, "right"), (180.0, "left")):
        img, truth = dip(angle, notch)
        found = chip.find_chip(img, PPM)
        end, sure, why = chip.notch_end(img, found)
        p = chip.pin1(found, end) if end else None
        off = math.dist(p, truth) / PPM if p else 99
        ok(end is not None and off < 0.8, f"turned {angle:g}, notch {notch}: pin 1 {off:.2f} mm off ({why}, sure {sure})")
        if end:
            b = chip.beside_pin1(found, end)
            bx, by = int(round(b[0])), int(round(b[1]))
            ok(chip.on_body(img, bx, by), f"turned {angle:g}, notch {notch}: the point beside pin 1 is on the body, "
                                          f"{math.dist(b, p) / PPM:.1f} mm from the leg")
            ok(not chip.on_body(img, int(p[0] + (p[0] - b[0])), int(p[1] + (p[1] - b[1]))) if 0 <= p[0] + (p[0] - b[0]) < 640
               and 0 <= p[1] + (p[1] - b[1]) < 480 else True, f"turned {angle:g}, notch {notch}: past the leg is not the body")
    print("Every other pin, from pin 1")
    for angle, notch in ((0.0, "left"), (0.0, "right"), (90.0, "left"), (-30.0, "right")):
        img, truth = dip(angle, notch)
        found = chip.find_chip(img, PPM)
        end, _, _ = chip.notch_end(img, found)
        cx, cy = found["center"]
        one = chip.pin_at(found, end, 1, 8)
        turned = (2 * cx - one[0], 2 * cy - one[1])  # half a turn about the chip's centre takes pin 1 to pin 5
        ok(math.dist(chip.pin_at(found, end, 5, 8), turned) / PPM < 0.05, f"turned {angle:g}, notch {notch}: pin 5 faces pin 1 "
                                                                          f"through the centre")
        ok(abs(math.dist(chip.pin_at(found, end, 8, 8), one) / PPM - 7.62) < 0.3, f"turned {angle:g}, notch {notch}: pin 8 is "
                                                                                 f"right across from pin 1, a row away")
        ok(abs(math.dist(chip.pin_at(found, end, 4, 8), one) / PPM - 7.62) < 0.3, f"turned {angle:g}, notch {notch}: pin 4 is "
                                                                                 f"at the far end of pin 1's side")
        ok(chip.on_body(img, *(int(round(v)) for v in chip.beside_pin(found, end, 8, 8))), f"turned {angle:g}, notch {notch}: "
                                                                                          f"beside pin 8 is on the body")
    long_chip = {"corners": [[-95.25, -31.75], [95.25, -31.75], [95.25, 31.75], [-95.25, 31.75]], "center": (0.0, 0.0),
                 "angle_deg": 0.0, "size_mm": (19.05, 6.35)}
    want = {1: (-76.2, 38.1), 7: (76.2, 38.1), 8: (76.2, -38.1), 14: (-76.2, -38.1)}
    ok(all(math.dist(chip.pin_at(long_chip, "left", n, 14), at) < 0.01 for n, at in want.items()),
       "a DIP-14 with the notch left: 1 and 14 by the notch, 7 and 8 at the far end, at 10 px per mm")
    img, _ = dip(0.0, "left", dent=False)
    found = chip.find_chip(img, PPM)
    end, sure, why = chip.notch_end(img, found)
    ok(end == "left", f"with the notch alone: {end} ({why})")
    img, truth = dip(0.0, "right", dent=False)  # the 555 on the bench: a notch, no dent
    found = chip.find_chip(img, PPM)
    end, sure, why = chip.notch_end(img, found)
    off = math.dist(chip.pin1(found, end), truth) / PPM if end else 99
    ok(end == "right" and off < 0.8, f"notch right and no dent: pin 1 upper right, {off:.2f} mm off the leg")
    img, truth = dip(0.0, "left")
    warm = np.clip(img.astype(np.float32) * (0.72, 0.8, 1.25), 0, 255).astype(np.uint8)  # a lamp's warm light from the side
    found = chip.find_chip(warm, PPM)
    ok(found is not None, "under a warm lamp the whole picture is tinted, and the chip is still found")
    plain = breadboard_picture()
    cv2.rectangle(plain, (160, 120), (480, 330), (52, 50, 48), -1)
    found = chip.find_chip(plain, PPM)
    end, sure, why = chip.notch_end(plain, found)
    ok(end is None, f"a body with no notch and no dent is not guessed: {why}")


def pointer():
    print("A wooden pointer's tip")
    img, truth = dip(0.0, "left")
    tip = (int(truth[0]) + 6, int(truth[1]) + 6)
    cv2.line(img, (0, 470), tip, (60, 130, 200), 11)  # a wooden stick (BGR) from the lower left edge to the pin
    img = cv2.GaussianBlur(img, (3, 3), 0)
    found = chip.pointer_tip(img, PPM)
    off = math.dist(found[0], tip) / PPM if found else 99
    ok(found is not None and off < 0.6, f"the tip is found {off:.2f} mm from where it is")
    ok(chip.pointer_tip(dip(0.0)[0], PPM) is None, "with no stick, no tip")


def holes_of_a_board():
    print("The holes of a breadboard, in a mosaic of it")
    ppm, pitch = 37.25, 2.54 * 37.25
    board = np.full((2600, 6600, 3), (150, 155, 150), np.uint8)
    tilt = 0.012
    truth = {}
    for row in range(63):
        for column in list(range(5)) + list(range(7, 12)):
            x = 370 + row * pitch + column * pitch * tilt
            y = 1790 - column * pitch + row * pitch * tilt
            cv2.rectangle(board, (int(x - 14), int(y - 14)), (int(x + 14), int(y + 14)), (70, 70, 70), -1)
            truth[(row, column)] = (x, y)
    grid = chip.breadboard_grid(board, ppm)
    ok(grid is not None and grid["holes"] > 300, f"the grid is found by its holes: {grid and grid['holes']}")
    if grid:
        off = max(math.dist(grid["a1"], truth[(0, 0)]), math.dist(grid["a63"], truth[(62, 0)]), math.dist(grid["j1"], truth[(0, 11)]))
        ok(off < 4, f"a1, a63 and j1 within {off:.1f} px, on a board turned 0.7 degrees")
    ok(chip.breadboard_grid(np.full((2600, 6600, 3), 150, np.uint8), ppm) is None, "no holes, no grid")


def real_pictures():
    """Pictures of the real 555 on the bench (Nov 2, 2026): the body must be found whole, and the notch end, if
    given, must be the right one (it is on the right in both)."""
    print("The real 555 on the breadboard")
    here = Path(__file__).resolve().parent / "pictures"
    # the second: two resistors' legs over it, taken with the Nov 4 calibration
    for name, ppm in (("dip8_555_mosaic.jpg", PPM), ("dip8_555_legs_over.jpg", 37.9), ("dip8_555_arc_right.jpg", 34.0)):
        img = cv2.imread(str(here / name))
        if img is None:
            ok(False, f"{name} is missing")
            continue
        found = chip.find_chip(img, ppm)
        ok(found is not None and not found["cut_by_edge"] and 8.5 < found["size_mm"][0] < 11 and 6 < found["size_mm"][1] < 9.5,
           f"{name}: body " + (f"{found['size_mm']} mm, cut {found['cut_by_edge']}" if found else "not found"))
        if found:
            end, sure, why = chip.notch_end(img, found)
            ok(end in (None, "right"), f"{name}: notch {end} ({why}, sure {sure}); never the wrong end")


if __name__ == "__main__":
    bodies()
    pin_one()
    pointer()
    holes_of_a_board()
    real_pictures()
    print(f"\n{'all passed' if not FAILED else f'{len(FAILED)} FAILED'}")
    sys.exit(1 if FAILED else 0)
