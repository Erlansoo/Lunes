"""The bench's geometry on made-up data, no hardware: python tests/test_geometry.py

- The laser dot: found to a fraction of a pixel in a picture with the laser on against one with it off, and a laser
  too bright for a dot (a glow over a shiny part) is refused rather than read as one.
- Laser triangulation: from three surfaces at known heights, the slant is fitted, the height of a surface is read
  back from where the dot lands, and a dot off every height's line is flagged.
- The Kinect as a camera: a 3 x 4 projection fitted to the head seen at the calibration's positions over the whole
  table puts the head back within the check's tolerance, at points it never saw too, and a point on the table seen
  with its depth lands where it is. The Kinect sees the back of the table at a grazing angle, where one pixel is
  about 6 mm of Y, so the noise here is the ring core's, a fifth of a pixel (the real calibration: 0.3 mm typical).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lunes import head, laser  # noqa: E402

FAILED = []
WORKSPACE = {"X": [0.0, 250.0], "Y": [-300.0, 0.0], "Z": [-9.7, 50.3]}  # the bench's, as in config/machine.json
rng = np.random.default_rng(7)


def ok(condition, what):
    print(("  ok   " if condition else "  FAIL ") + what)
    if not condition:
        FAILED.append(what)


def dot_pictures(x, y, sigma_px, peak, glow_px=0):
    """A textured board picture with the laser off, and the same with a red dot (and a glow around it) on."""
    off = rng.integers(40, 160, (480, 640, 3)).astype(np.uint8)
    yy, xx = np.mgrid[0:480, 0:640]
    rise = peak * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma_px ** 2))
    if glow_px:
        rise += 90 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * glow_px ** 2))
    on = off.copy()
    on[..., 2] = np.clip(off[..., 2] + rise + rng.normal(0, 4, rise.shape), 0, 255).astype(np.uint8)
    return off, on


def dots():
    print("Laser dot")
    px_per_mm = 42.0
    for x, y in ((320.0, 240.0), (101.3, 377.8), (512.6, 66.2)):
        found = laser.find_dot(*dot_pictures(x, y, 2.5, 120), px_per_mm)
        ok(found is not None and np.hypot(found["x"] - x, found["y"] - y) < 0.5,
           f"dot at ({x}, {y}) found at " + (f"({found['x']:.2f}, {found['y']:.2f}), {found['size_mm']} mm"
                                             if found else "nothing"))
    off, on = dot_pictures(300, 200, 2.5, 120, glow_px=40)
    ok(laser.find_dot(off, on, px_per_mm) is None and laser.too_bright(off, on, px_per_mm),
       f"a glow of {laser.glow_mm2(off, on, px_per_mm):.1f} mm² is too bright, not a dot")
    off, on = dot_pictures(300, 200, 2.5, 20)
    ok(laser.find_dot(off, on, px_per_mm) is None, "a rise within the camera's noise is no dot")


def triangulation():
    print("Laser triangulation")
    slant, at_zero = np.array([0.41, -0.30]), np.array([1.8, -0.4])  # mm per mm of height, and the offset at Z 0
    cal = {"surfaces": [{"z": z, "offset_mm": dict(zip("XY", at_zero + slant * z + rng.normal(0, 0.03, 2)))}
                        for z in (-5.68, -1.68, 7.0)]}
    k = laser.slant(cal)
    ok(np.allclose(k, slant, atol=0.01), f"slant fitted as {k[0]:.3f}, {k[1]:.3f} mm per mm")
    for z in (-4.0, 0.0, 3.5, 12.0):
        got, off_line = laser.height_from(cal, at_zero + slant * z)
        ok(abs(got - z) < 0.15 and off_line < 0.1, f"a surface at Z {z:+.1f} read at {got:+.2f}")
    _, off_line = laser.height_from(cal, at_zero + np.array([0.30, 0.41]) * 4)  # across the line, not along it
    ok(off_line > 1.0, f"a dot {off_line:.1f} mm off every height's line is flagged")
    ok(max(laser.residuals(cal)) < 0.1, "the three surfaces sit on one line")


def camera_looking_at(centre, target, focal_px=550.0, size=(640, 480)):
    """A 3 x 4 projection of a camera at centre (machine mm) looking at target."""
    forward = (target - centre) / np.linalg.norm(target - centre)
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    k = np.array([[focal_px, 0, size[0] / 2], [0, focal_px, size[1] / 2], [0, 0, 1]])
    r = np.vstack([right, down, forward])
    return k @ np.hstack([r, -(r @ centre)[:, None]])


def project(p, xyz):
    u, v, w = p @ np.r_[xyz, 1.0]
    return u / w, v / w, w


def kinect():
    print("Kinect as a camera")
    truth = camera_looking_at(np.array([119.0, 474.0, 197.0]), np.array([125.0, -150.0, 0.0]))
    z_low, z_high = 10.0, 25.0
    plan = [(119 + dx, -150 + dy, z_low + dz) for dx, dy, dz in head.LOCAL_OFFSETS] + \
        head.area_points(WORKSPACE, z_low, z_high)
    points = []
    for x, y, z in plan:
        u, v, _ = project(truth, np.array([x, y, z]))
        points.append(((u + rng.normal(0, 0.2), v + rng.normal(0, 0.2), None), {"X": x, "Y": y, "Z": z}))
    p, errors = head.fit_camera(points)
    ok(np.median(errors) < 0.6 and max(errors) < 1.5,
       f"fitted over {len(points)} positions: typical {np.median(errors):.2f} mm, worst {max(errors):.2f} mm")
    worst = max(max(dx, dy) for dx, dy in head.held_out(points))
    ok(worst < 5.0, f"held out, each position without itself: worst {worst:.2f} mm, inside the check's 5 mm")
    for x, y, z in ((60.0, -80.0, 18.0), (200.0, -260.0, 30.0), (130.0, -20.0, 12.0)):
        u, v, _ = project(truth, np.array([x, y, z]))
        gx, gy = head.locate(p, u, v, z)
        ok(np.hypot(gx - x, gy - y) < 2.0, f"head at X {x:g}, Y {y:g}, Z {z:g}, never seen: located within "
                                           f"{np.hypot(gx - x, gy - y):.2f} mm")
    x, y = 150.0, -120.0  # a board on the table: the Kinect gives its depth along the ray
    u, v, depth = project(truth, np.array([x, y, -5.68]))
    gx, gy = head.on_ray(p, u, v, depth)
    ok(np.hypot(gx - x, gy - y) < 3.0, f"a board on the table seen with its depth: {np.hypot(gx - x, gy - y):.2f} mm "
                                       f"off")


if __name__ == "__main__":
    dots()
    triangulation()
    kinect()
    print(f"\n{'all passed' if not FAILED else f'{len(FAILED)} FAILED'}")
    sys.exit(1 if FAILED else 0)
