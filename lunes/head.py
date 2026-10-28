"""Where the microscope head is, seen by the Kinect, so Lunes checks its position instead of trusting the step count.

The Kinect looks at the CNC from the front, at an angle. The head is easy to find in its colour picture: the ring
light at the tip of the microscope is the brightest small spot that has the black tube right above it. Where it
appears (column and row) and how far it is from the Kinect change with the machine's X, Y and Z, so a calibration
that moves the machine to a few known positions gives a linear map from (column, row, distance) to machine
millimetres. With it, check() says where the Kinect sees the head and how far that is from where Lunes thinks it is.

Classical image processing only, no model. The Kinect sees about 2 mm per pixel at this distance, so the check is
good to a few millimetres: enough to catch a carriage moved by hand, lost steps or a move the wrong way, not to
replace the microscope for fine work.
"""
import numpy as np

# Offsets from the current position for the calibration, in mm. Kept small so they stay inside the workspace from
# wherever the machine is; Z only goes up, never towards the board.
CALIBRATION_OFFSETS = [(0, 0, 0), (30, 0, 0), (-30, 0, 0), (0, 30, 0), (0, -30, 0), (0, 0, 4), (30, 30, 4),
                       (-30, -30, 0), (0, 0, 0)]


def find_ring(color, depth):
    """(column, row, distance in mm) of the microscope's ring light in the Kinect's pictures, or None."""
    import cv2
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    n, _, stats, centres = cv2.connectedComponentsWithStats((gray > 235).astype(np.uint8))
    best = None
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (20 <= area <= 800 and w < 70 and h < 45):
            continue
        cx, cy = centres[i]
        above = gray[max(0, int(cy) - 30):max(1, int(cy) - 12), max(0, int(cx) - 8):int(cx) + 8]
        darkness = float(above.mean()) if above.size else 255.0  # the black tube right above the ring
        if darkness < 90 and (best is None or darkness < best[0]):
            best = (darkness, cx, cy)
    if best is None:
        return None
    _, cx, cy = best
    # The depth picture is not exactly aligned with the colour one; the nearest tenth of a window around the ring is
    # the tube in front of the bed behind it.
    window = depth[max(0, int(cy) - 30):int(cy) + 10, max(0, int(cx) - 20):int(cx) + 20]
    near = window[window > 0]
    if near.size < 20:
        return None
    return float(cx), float(cy), float(np.percentile(near, 10))


def fit(points):
    """The linear map from (column, row, distance) to machine (X, Y, Z) that best fits the calibration points
    [((column, row, distance), {"X":, "Y":, "Z":}), ...]; returns (map as a 4 x 3 list, error of each point in mm)."""
    seen = np.array([[u, v, d, 1.0] for (u, v, d), _ in points])
    machine = np.array([[p["X"], p["Y"], p["Z"]] for _, p in points])
    a, *_ = np.linalg.lstsq(seen, machine, rcond=None)
    errors = np.linalg.norm(seen @ a - machine, axis=1)
    return a.tolist(), [round(float(e), 1) for e in errors]


def predict(a, seen):
    u, v, d = seen
    x, y, z = np.array([u, v, d, 1.0]) @ np.array(a)
    return {"X": float(x), "Y": float(y), "Z": float(z)}
