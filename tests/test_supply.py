"""Reading the bench supply's panel, no hardware: python tests/test_supply.py

Panels are drawn as the camera sees the real one: two red windows with bright seven-segment digits, slanted, turned a
little and blurred. Every digit is read at both windows, with the decimal point in place. A panel that cannot be read must
say so, and the check must stop a circuit from being powered at the wrong voltage.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lunes import supply  # noqa: E402

FAILED = []
SEGMENTS = {0: "abcdef", 1: "bc", 2: "abdeg", 3: "abcdg", 4: "bcfg", 5: "acdfg", 6: "acdefg", 7: "abc", 8: "abcdefg",
            9: "abcdfg"}
STROKES = {"a": ((0.12, 0), (0.88, 0)), "b": ((1, 0.05), (1, 0.45)), "c": ((1, 0.55), (1, 0.95)), "d": ((0.12, 1), (0.88, 1)),
           "e": ((0, 0.55), (0, 0.95)), "f": ((0, 0.05), (0, 0.45)), "g": ((0.12, 0.5), (0.88, 0.5))}


def ok(condition, what):
    print(("  ok   " if condition else "  FAIL ") + what)
    if not condition:
        FAILED.append(what)


def window(image, x, y, text, point):
    w, h, pitch = 52, 96, 80
    cv2.rectangle(image, (x - 20, y - 16), (x + 4 * pitch - 8, y + h + 16), (30, 16, 170), -1)
    for k, digit in enumerate(text):
        left = x + k * pitch
        for segment in SEGMENTS[int(digit)]:
            (x0, y0), (x1, y1) = STROKES[segment]
            cv2.line(image, (int(left + x0 * w), int(y + y0 * h)), (int(left + x1 * w), int(y + y1 * h)), (225, 225, 255), 15)
        if k + 1 == point:
            cv2.circle(image, (left + w + 16, y + h), 8, (225, 225, 255), -1)


def panel(volts, amps, out=False, turn=-6.0, slant=0.12):
    image = np.full((900, 1300, 3), (200, 205, 205), np.uint8)
    cv2.rectangle(image, (60, 120), (1000, 700), (18, 12, 14), -1)
    window(image, 170, 200, volts, 2)
    window(image, 170, 370, amps, 1)
    for k in range(5):  # the five lights on the right of the windows, dull when off
        cv2.circle(image, (560, 214 + 57 * k), 9, (60, 40, 200 if out and k == 4 else 90), -1)
        if out and k == 4:
            cv2.circle(image, (560, 214 + 57 * k), 12, (240, 240, 255), -1)
    m = np.array([[1, -slant, slant * 400], [0, 1, 0]], np.float32)
    image = cv2.warpAffine(image, m, (1300, 900), borderValue=(200, 205, 205))
    r = cv2.getRotationMatrix2D((650, 450), -turn, 1.0)
    image = cv2.warpAffine(image, r, (1300, 900), borderValue=(200, 205, 205))
    return cv2.GaussianBlur(image, (7, 7), 0)


def digits():
    print("Every digit, in both windows")
    for d in range(10):
        volts, amps = f"0{d}{d}{d}", str(d) * 4
        got = supply.read(panel(volts, amps))
        want_v, want_a = float(volts[:2] + "." + volts[2:]), float(amps[0] + "." + amps[1:])
        ok(got["ok"] and got["volts"] == want_v and got["amps"] == want_a,
           f"{d}: {got.get('volts')} V, {got.get('amps')} A {got.get('why', '')}")


def readings():
    print("Readings as on the bench")
    for volts, amps, wv, wa in (("2400", "2000", 24.0, 2.0), ("0500", "0100", 5.0, 0.1), ("1205", "0750", 12.05, 0.75),
                                ("0330", "1234", 3.3, 1.234), ("3000", "0050", 30.0, 0.05), ("1800", "9876", 18.0, 9.876)):
        got = supply.read(panel(volts, amps))
        ok(got["ok"] and got["volts"] == wv and got["amps"] == wa, f"{wv} V, {wa} A: read {got.get('volts')}, {got.get('amps')}")
    ok(supply.read(panel("0500", "0100", out=True, turn=-3))["ok"], "with another turn of the camera it still reads")


def blind():
    print("When it cannot read")
    wall = np.full((900, 1300, 3), (210, 215, 215), np.uint8)
    ok(not supply.read(wall)["ok"], "a picture with no panel is no reading")
    covered = panel("2400", "2000")
    covered[300:460, 300:520] = (18, 12, 14)
    ok(not supply.read(covered)["ok"], "a panel half hidden is no reading")


def check():
    print("Before powering a circuit")
    circuit = {"supply_v": 5, "parts": [{"ref": "R1", "kind": "R", "pins": ["VCC", "A"]}]}
    sim = {"currents": {"R1": {"peak_a": 0.02}}}
    high = supply.check({"volts": 24.0, "amps": 2.0, "out": False}, circuit, sim)
    ok(not high["safe"], f"24 V for a 5 V circuit is stopped: {high['problems'][0][:60]}")
    right = supply.check({"volts": 5.0, "amps": 0.1, "out": False}, circuit, sim)
    ok(right["safe"] and not right["advice"], "5 V with a sensible limit is fine")
    loose = supply.check({"volts": 5.0, "amps": 2.0, "out": False}, circuit, sim)
    ok(loose["safe"] and "límite" in loose["advice"][0], "a limit far above the need gets advice")


if __name__ == "__main__":
    digits()
    readings()
    blind()
    check()
    print(f"\n{'all passed' if not FAILED else f'{len(FAILED)} FAILED'}")
    sys.exit(1 if FAILED else 0)
