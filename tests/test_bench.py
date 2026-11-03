"""The bench camera tied to the machine by the ring light and one taught thing, no hardware: python tests/test_bench.py

A made-up webcam looks at the table from the side. The microscope's ring light hangs 90 mm over whatever is in focus,
so the camera is fitted to the ring at the machine's Z over two heights far apart, and it must learn those 90 mm from
one thing seen by both (a chip under the microscope, found in the webcam's picture), refuse a point that is not that
thing, and then locate points on the table and on a chip's top to within a few mm. Too few or too narrow pairs must
be refused, and two estimates that differ must be told apart from ones that agree.
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lunes import bench  # noqa: E402

FAILED = []
WORKSPACE = {"X": [10.0, 240.0], "Y": [-235.0, -10.0], "Z": [-6.7, 47.3]}
RING_ABOVE = 90.0  # the ring light's height over the surface in focus, in the made-up machine
TABLE = -5.7  # the table's focus height
CHIP = (100.0, -100.0, 4.0)  # a chip's top, 10 mm over the table: its own focus height
rng = np.random.default_rng(3)


def ok(condition, what):
    print(("  ok   " if condition else "  FAIL ") + what)
    if not condition:
        FAILED.append(what)


def camera(centre=(125.0, 420.0, 380.0), target=(125.0, -120.0, -60.0), focal=1400.0, size=(1920, 1080)):
    centre, target = np.array(centre), np.array(target)
    forward = (target - centre) / np.linalg.norm(target - centre)
    right = np.cross(forward, [0, 0, 1.0]); right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    k = np.array([[focal, 0, size[0] / 2], [0, focal, size[1] / 2], [0, 0, 1]])
    r = np.vstack([right, down, forward])
    return k @ np.hstack([r, -(r @ centre)[:, None]])


def project(p, xyz):
    u, v, w = p @ np.r_[xyz, 1.0]
    return u / w, v / w


def surface(p, x, y, focus_z):
    """Where the camera sees a point on a surface of focus height focus_z: physically RING_ABOVE lower."""
    return project(p, np.array([x, y, focus_z - RING_ABOVE]))


def picture(size=(1920, 1080)):
    img = np.full((size[1], size[0], 3), (150, 160, 170), np.uint8)
    img[400:700, 500:1400] = (190, 200, 210)  # a light board
    img += rng.integers(0, 6, img.shape, dtype=np.uint8)
    return img


def ring():
    print("The microscope's ring light in the webcam's picture")
    img = picture()
    cv2.rectangle(img, (890, 300), (950, 520), (20, 20, 20), -1)  # the black tube
    cv2.ellipse(img, (920, 540), (30, 20), 0, 0, 360, (255, 255, 255), -1)  # the ring light at its tip
    for x, y in ((100, 900), (160, 900), (100, 980), (160, 980)):  # a supply's digits, burnt out to a warm white
        cv2.rectangle(img, (x - 60, y - 70), (x + 60, y + 50), (20, 20, 20), -1)
        cv2.ellipse(img, (x, y), (14, 19), 0, 0, 360, (238, 252, 253), -1)
    found = bench.ring_in_picture(img)
    ok(found is not None and math.dist(found, (920, 540)) < 3, f"found at {found and (round(found[0]), round(found[1]))}")
    ok(bench.ring_in_picture(picture()) is None, "with no ring, none")
    board = np.full((800, 1200, 3), 225, np.uint8)
    cv2.rectangle(board, (600, 400), (650, 430), (25, 25, 25), -1)
    cv2.rectangle(board, (100, 100), (260, 260), (10, 10, 10), -1)  # darker and bigger, and far from where it must be
    near = bench.chip_near(board, 640, 380)
    ok(near is not None and math.dist(near, (625, 415)) < 2, "the chip nearest where the geometry says, not the big dark thing")
    ok(bench.chip_near(np.full((800, 1200, 3), 225, np.uint8), 640, 380) is None, "nothing there, nothing found")
    ok(bench.ring_in_picture(img, near=(300, 200), reach=100) is None, "near another place, it is not seen")
    far = bench.ring_in_picture(img, near=(900, 500), reach=100)
    ok(far is not None and math.dist(far, (920, 540)) < 3, "near where it should be, it is")
    unlit = img.copy()
    cv2.ellipse(unlit, (920, 540), (30, 20), 0, 0, 360, (60, 60, 60), -1)
    ok(bench.ring_in_picture(unlit) is None, "with the ring off, the supply's digits are not taken for it")


def chip_on_board():
    print("A chip on a breadboard, in the bench picture")
    img = np.full((2160, 3840, 3), (70, 120, 190), np.uint8)  # a wooden table
    img[0:900, 2800:3840] = (245, 245, 245)  # a white wall
    cv2.fillConvexPoly(img, np.array([(1500, 1300), (2300, 1300), (2300, 1700), (1500, 1700)]), (235, 235, 235))  # the board
    for x in range(1520, 2290, 12):  # its holes
        for y in range(1320, 1690, 12):
            cv2.circle(img, (x, y), 1, (90, 90, 90), -1)
    cv2.rectangle(img, (1880, 1490), (1915, 1512), (30, 30, 30), -1)  # the chip
    cv2.rectangle(img, (1200, 1450), (1240, 1490), (30, 30, 30), -1)  # a screw on the table, off the board
    table = [(1000, 1100), (2700, 1100), (2700, 1950), (1000, 1950)]
    found = bench.chip_in_picture(img, table)
    ok(len(found) == 1 and math.dist(found[0][:2], (1897, 1501)) < 3, f"the chip found, and only it: {found}")
    cv2.rectangle(img, (1880, 1490), (1915, 1512), (235, 235, 235), -1)
    ok(bench.chip_in_picture(img, table) == [], "a bare board: nothing, the holes do not pass for chips")
    ok(bench.chip_in_picture(np.full((2160, 3840, 3), (70, 120, 190), np.uint8), table) == [], "no board: nothing")


def fitted(p, noise=0.7):
    cam = bench.BenchCamera()
    for x, y, z in bench.plan_route(WORKSPACE, 2.3, 47.3):  # the head's ring at two heights far apart
        u, v = project(p, np.array([x, y, z]))
        cam.add((u + rng.normal(0, noise), v + rng.normal(0, noise)), {"X": x, "Y": y, "Z": z})
    return cam, cam.fit()


def fit():
    print("Fitting the camera from the ring's pairs, teaching the height from one thing")
    p = camera()
    cam, result = fitted(p)
    ok(result["ok"] and result["typical_error_mm"] < 1.5, f"fit: {result}")
    ok(result["ok"] and "warning" in result and result["ring_above_mm"] is None, "fitted but warned: no height yet")
    try:
        cam.locate(500, 500, TABLE)
        ok(False, "locate without a height must refuse")
    except ValueError:
        ok(True, "locate without a height refuses")
    ring_at = {"X": 100.0, "Y": -100.0, "Z": 30.0}
    u, v = project(p, np.array([100.0, -100.0, 30.0]))
    ok(cam.ring_off_px((u, v), ring_at) < 5, "the ring where the projection puts it: the camera is where it was")
    ok(cam.ring_off_px((u + 237, v + 35), ring_at) > bench.MOVED_PX, "the ring 240 px off: the camera was moved")
    u, v = surface(p, *CHIP)  # the chip under the microscope, seen by the webcam too
    taught = cam.teach(u + 1.0, v - 1.0, CHIP[0], CHIP[1], CHIP[2])
    ok(taught["ok"] and abs(taught["ring_above_mm"] - RING_ABOVE) < 3.0,
       f"the ring's height over the surface learnt from one thing: {taught} (truly {RING_ABOVE})")
    wrong = cam.teach(*surface(p, CHIP[0] + 30.0, CHIP[1], CHIP[2]), CHIP[0], CHIP[1], CHIP[2])
    ok(not wrong["ok"], f"a point 30 mm from the thing is refused: {wrong.get('error', '')[:60]}")
    worst = 0.0
    for x, y in ((60.0, -200.0), (200.0, -40.0), (125.0, -120.0), (30.0, -60.0)):
        gx, gy = cam.locate(*surface(p, x, y, TABLE), TABLE)
        worst = max(worst, math.dist((gx, gy), (x, y)))
    ok(worst < 3.0, f"points on the table never seen located within {worst:.2f} mm")
    u, v = surface(p, 150.0, -60.0, 4.0)  # another chip's top, at its own focus height
    gx, gy = cam.locate(u, v, 4.0)
    ok(math.dist((gx, gy), (150.0, -60.0)) < 2.5, "a point at another height, located at that height")
    pu, pv = cam.project(150.0, -60.0, 4.0)
    ok(math.dist((pu, pv), (u, v)) < 3, "and projected back to where the camera sees it")
    again = bench.BenchCamera.from_config(cam.to_config())
    ok(again is not None and np.allclose(again.P, cam.P) and again.ring_above_mm == round(cam.ring_above_mm, 2)
       and again.taught == cam.taught, "kept in the config and read back, the taught thing too")
    refit = again.fit()  # calibrated again: the taught thing is checked on the new projection and the height kept
    ok(refit["ok"] and refit["ring_above_mm"] is not None and abs(refit["ring_above_mm"] - RING_ABOVE) < 3.0,
       f"fitted again, the height is kept: {refit['ring_above_mm']}")


def refusals():
    print("What is refused")
    cam = bench.BenchCamera()
    for k in range(5):
        cam.add((100 + k, 200 + k), {"X": 50.0 + k, "Y": -100.0, "Z": 0.0})
    ok(not cam.fit()["ok"], "five pairs are too few")
    cam = bench.BenchCamera()
    p = camera()
    for k in range(10):
        x, y = 100.0 + k * 3, -100.0 + k * 2
        cam.add(project(p, np.array([x, y, 0.0])), {"X": x, "Y": y, "Z": 0.0})
    ok(not cam.fit()["ok"], "pairs in one small patch are too narrow")
    ok(not bench.BenchCamera().teach(100, 100, 50.0, -50.0, 0.0)["ok"], "teaching an uncalibrated camera is refused")
    ok(bench.agree((100.0, -100.0), (103.0, -98.0))[0] and not bench.agree((100.0, -100.0), (120.0, -100.0))[0],
       "3.6 mm apart agree, 20 mm apart do not")


if __name__ == "__main__":
    ring()
    chip_on_board()
    fit()
    refusals()
    print(f"\n{'all passed' if not FAILED else f'{len(FAILED)} FAILED'}")
    sys.exit(1 if FAILED else 0)
