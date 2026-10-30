"""Where the microscope head is, seen by the Kinect, so Lunes checks its position instead of trusting the step count.

The Kinect looks at the CNC from the front, at an angle. The head is easy to find in its colour picture: the ring
light at the tip of the microscope is the brightest small spot that has the black tube right above it. Where it
appears (column and row) changes with the machine's X, Y and Z. The Kinect measures no depth on the black tube (it
swallows the infrared pattern), so the depth is not used: the colour camera is calibrated as a camera, a 3 x 4
projection fitted to known positions over the whole work area (about 30 x 40 cm), and since the machine knows its
height Z, the column and row of the ring give X and Y. The older maps (linear in column, row and depth) needed a
depth next to the ring, which the bed below gave only with the head low, and held only a few centimetres around
where they were made. With it, check() says where the Kinect sees the head and how far that is from where Lunes thinks it is.

Classical image processing only, no model. The Kinect sees about 2 mm per pixel at this distance, so the check is
good to a few millimetres: enough to catch a carriage moved by hand, lost steps or a move the wrong way, not to
replace the microscope for fine work.
"""
import numpy as np

# Offsets from the current position, in mm, for the part of the calibration at the working height: small, so the
# head stays over what it was already over; Z only goes up, never towards the board.
CALIBRATION_OFFSETS = [(0, 0, 0), (30, 0, 0), (-30, 0, 0), (0, 30, 0), (0, -30, 0), (0, 0, 4), (30, 30, 4),
                       (-30, -30, 0), (0, 0, 0)]
LOCAL_OFFSETS = [(0, 0, 0), (30, 0, 0), (-30, 0, 0), (0, 30, 0), (0, -30, 0)]
AREA_FRACTIONS = (0.1, 0.5, 0.9)  # where the grid over the whole work area goes, as a share of each axis' range


def area_points(workspace, z_low, z_high):
    """The calibration's positions over the whole work area: a 3 x 3 grid at z_low, walked as a snake so no move
    crosses the table twice, then the four corners at z_high."""
    def at(axis, f):
        lo, hi = workspace[axis]
        return lo + f * (hi - lo)
    grid = []
    for i, fy in enumerate(AREA_FRACTIONS):
        row = [(at("X", fx), at("Y", fy), z_low) for fx in AREA_FRACTIONS]
        grid += row if i % 2 == 0 else row[::-1]
    corners = [(at("X", fx), at("Y", fy), z_high) for fx, fy in ((0.9, 0.9), (0.1, 0.9), (0.1, 0.1), (0.9, 0.1))]
    return grid + corners


def _features(u, v, d, model):
    return [u * d / 1000, v * d / 1000, d, 1.0] if model == "pinhole" else [u, v, d, 1.0]


def find_ring(color, depth):
    """(column, row, distance in mm) of the microscope's ring light in the Kinect's pictures, or None."""
    import cv2
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    n, labels, stats, centres = cv2.connectedComponentsWithStats((gray > 235).astype(np.uint8))
    best = None
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (20 <= area <= 800 and w < 70 and h < 45):
            continue
        cx, cy = centres[i]
        above = gray[max(0, int(cy) - 30):max(1, int(cy) - 12), max(0, int(cx) - 8):int(cx) + 8]
        darkness = float(above.mean()) if above.size else 255.0  # the black tube right above the ring
        if darkness < 90 and (best is None or darkness < best[0]):
            best = (darkness, cx, cy, i)
    if best is None:
        return None
    _, cx, cy, i = best
    # The ring itself is saturated; with the head high it stands in front of light parts of the gantry, whose glare
    # joins its spot and pulled the centre 8 pixels (10 to 15 mm) aside. The saturated core is the ring alone.
    ys, xs = np.nonzero((labels == i) & (gray >= 250))
    if xs.size >= 20:
        cx, cy = float(xs.mean()), float(ys.mean())
    # The depth picture is not exactly aligned with the colour one; the nearest tenth of a window around the ring is
    # the tube in front of the bed behind it.
    window = depth[max(0, int(cy) - 30):int(cy) + 10, max(0, int(cx) - 20):int(cx) + 20]
    near = window[window > 0]
    return float(cx), float(cy), float(np.percentile(near, 10)) if near.size >= 20 else None


def fit(points, model="pinhole"):
    """The map from (column, row, distance) to machine (X, Y, Z) that best fits the calibration points
    [((column, row, distance), {"X":, "Y":, "Z":}), ...]; returns (map as a list of rows, error of each point in mm).
    model "linear" is the plain linear map of older calibrations, good only over a few centimetres."""
    seen = np.array([_features(u, v, d, model) for (u, v, d), _ in points])
    machine = np.array([[p["X"], p["Y"], p["Z"]] for _, p in points])
    a, *_ = np.linalg.lstsq(seen, machine, rcond=None)
    errors = np.linalg.norm(seen @ a - machine, axis=1)
    return a.tolist(), [round(float(e), 1) for e in errors]


def predict(a, seen, model="linear"):
    u, v, d = seen
    x, y, z = np.array(_features(u, v, d, model)) @ np.array(a)
    return {"X": float(x), "Y": float(y), "Z": float(z)}


def fit_camera(points):
    """The Kinect colour camera's projection (3 x 4) that best fits the calibration points
    [((column, row, depth or None), {"X":, "Y":, "Z":}), ...], by the direct linear transform on normalised
    coordinates; returns (the matrix as a list, the error of each point in mm, located again at its own height)."""
    uv = np.array([[u, v] for (u, v, _), _ in points], float)
    xyz = np.array([[p["X"], p["Y"], p["Z"]] for _, p in points], float)

    def normaliser(a):  # centred, scaled to an average distance of sqrt(dimensions)
        c = a.mean(axis=0)
        s = np.sqrt(a.shape[1]) / np.mean(np.linalg.norm(a - c, axis=1))
        t = np.eye(a.shape[1] + 1)
        t[:-1, :-1] *= s
        t[:-1, -1] = -s * c
        return t

    tu, tx = normaliser(uv), normaliser(xyz)
    uvn = (tu @ np.c_[uv, np.ones(len(uv))].T).T
    xn = (tx @ np.c_[xyz, np.ones(len(xyz))].T).T
    rows = []
    for (u, v, _), x in zip(uvn, xn):
        rows.append(np.r_[x, np.zeros(4), -u * x])
        rows.append(np.r_[np.zeros(4), x, -v * x])
    _, _, vt = np.linalg.svd(np.array(rows))
    p = np.linalg.inv(tu) @ vt[-1].reshape(3, 4) @ tx
    p /= np.linalg.norm(p[2, :3])
    if p[2] @ np.r_[xyz.mean(axis=0), 1.0] < 0:  # the third row is then the depth in front of the camera, in mm
        p = -p
    errors = [float(np.hypot(*(np.array(locate(p, u, v, q["Z"])) - [q["X"], q["Y"]]))) for (u, v, _), q in points]
    return p.tolist(), [round(e, 1) for e in errors]


def held_out(points):
    """How far off each calibration point is when the camera is fitted without it (mm in X and in Y): an honest error
    for points the calibration never saw, from which the check's tolerance per axis comes."""
    out = []
    for i, ((u, v, _), q) in enumerate(points):
        p, _ = fit_camera(points[:i] + points[i + 1:])
        x, y = locate(p, u, v, q["Z"])
        out.append((abs(x - q["X"]), abs(y - q["Y"])))
    return out


def on_ray(p, u, v, depth):
    """Machine X, Y of the thing the camera p sees at column u, row v, depth mm in front of it: what the Kinect sees
    on the table (a board) has a depth, unlike the black microscope tube. Its Z is in the calibration's own frame (the
    head's height while its ring was seen), so only X and Y are given."""
    p = np.asarray(p, float)
    m = p[:, :3]
    centre = -np.linalg.solve(m, p[:, 3])
    direction = np.linalg.solve(m, np.array([u, v, 1.0]))
    x = centre + direction * depth / float(p[2, :3] @ direction)
    return float(x[0]), float(x[1])


def locate(p, u, v, z):
    """Machine X, Y of the point at height z that the camera p sees at column u, row v."""
    p = np.asarray(p, float)
    a = np.array([p[0] - u * p[2], p[1] - v * p[2]])  # each row . (X, Y, Z, 1) = 0
    x, y = np.linalg.solve(a[:, :2], -(a[:, 2] * z + a[:, 3]))
    return float(x), float(y)


def where(cal, seen, z):
    """Machine X, Y, Z where the Kinect sees the head, from its calibration and find_ring's result; z is the head's
    height, which the machine knows. None when an old calibration needs the depth and there is none."""
    u, v, d = seen
    if cal.get("model") == "camera":
        x, y = locate(cal["P"], u, v, z)
        return {"X": x, "Y": y, "Z": float(z)}
    return None if d is None else predict(cal["map"], seen, cal.get("model", "linear"))
