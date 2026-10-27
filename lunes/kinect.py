"""Python side of the Kinect bridge (kinect/KinectBridge.cs): colour and depth frames as NumPy arrays.

Start the bridge first (kinect\\bin\\kinect_bridge.exe, built by kinect\\build.ps1); this client talks to it on
127.0.0.1:9601.
"""
import json
import socket

import cv2
import numpy as np

PORT = 9601


class Kinect:
    def __init__(self, host="127.0.0.1", port=PORT, timeout=3.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.file = self.sock.makefile("rb")

    def _request(self, line):
        self.sock.sendall(line.encode() + b"\n")
        header = json.loads(self.file.readline())
        if "err" in header:
            raise RuntimeError(f"Kinect bridge: {header['err']}")
        return header

    def _frame(self, kind):
        h = self._request(kind)
        data = self.file.read(h["bytes"])
        if len(data) != h["bytes"]:
            raise IOError("the Kinect bridge closed the connection mid-frame")
        return h, data

    def color(self):
        """BGR image, 640x480."""
        h, data = self._frame("COLOR")
        bgra = np.frombuffer(data, np.uint8).reshape(h["h"], h["w"], 4)
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

    def depth(self):
        """Depth in millimetres, 640x480 uint16; 0 where the sensor could not measure."""
        h, data = self._frame("DEPTH")
        return np.frombuffer(data, "<u2").reshape(h["h"], h["w"]).copy()

    def tilt(self, degrees):
        """Moves the tilt motor; the SDK call blocks until the motor stops, which takes a few seconds."""
        old = self.sock.gettimeout()
        self.sock.settimeout(15)
        try:
            return self._request(f"TILT {int(degrees)}")
        finally:
            self.sock.settimeout(old)

    def info(self):
        return self._request("INFO")

    def close(self):
        self.sock.close()


class KinectSnapshot:
    """The Kinect switched on only for one picture (kinect_bridge.exe --snapshot), then released.

    While the sensor runs, Windows gets silence from the Kinect's microphone array, which is also Hume's ear, so
    Hume looks this way instead of keeping the bridge running. Each look takes two or three seconds."""

    def __init__(self, bridge=None):
        from pathlib import Path
        self.bridge = Path(bridge) if bridge else Path(__file__).resolve().parent.parent / "kinect" / "bin" / "kinect_bridge.exe"
        self.folder = self.bridge.parent.parent.parent / "runs" / "kinect_snapshot"
        self.last_depth = None

    def color(self):
        import subprocess
        r = subprocess.run([str(self.bridge), "--snapshot", str(self.folder)], capture_output=True, text=True,
                           timeout=20)
        if r.returncode != 0:
            raise RuntimeError(f"Kinect snapshot failed: {(r.stderr or r.stdout).strip()}")
        raw = np.fromfile(self.folder / "depth.raw", "<u2")
        self.last_depth = raw.reshape(480, 640)
        return cv2.imread(str(self.folder / "color.png"))

    def depth(self):
        """Depth (mm) of the last color() picture."""
        return self.last_depth

    def close(self):
        pass


class KinectAudio:
    """The microphone array as a 16 kHz mono stream, through the SDK (beamforming, gain control, noise suppression).

    Uses its own connection because the bridge streams audio on it until the connection closes."""

    rate = 16000

    def __init__(self, host="127.0.0.1", port=PORT):
        self.sock = socket.create_connection((host, port), timeout=5.0)
        self.file = self.sock.makefile("rb")
        self.sock.sendall(b"AUDIO\n")
        header = json.loads(self.file.readline())
        if "err" in header:
            raise RuntimeError(f"Kinect bridge: {header['err']}")

    def read(self, n):
        """Next n samples as float32 in [-1, 1]."""
        data = self.file.read(2 * n)
        if len(data) != 2 * n:
            raise IOError("the Kinect audio stream ended")
        return np.frombuffer(data, "<i2").astype(np.float32) / 32768.0

    def close(self):
        self.sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def depth_colormap(depth_mm, near=800, far=2000):
    """False-colour view of a depth image; unknown pixels are black."""
    d = np.clip((depth_mm.astype(np.float32) - near) / (far - near), 0, 1)
    img = cv2.applyColorMap((255 - d * 255).astype(np.uint8), cv2.COLORMAP_JET)
    img[depth_mm == 0] = 0
    return img


def start_bridge(timeout=10.0):
    """Starts kinect/bin/kinect_bridge.exe unless it is already serving; True once it answers."""
    import subprocess
    import time
    from pathlib import Path

    def up():
        try:
            socket.create_connection(("127.0.0.1", PORT), timeout=0.3).close()
            return True
        except OSError:
            return False

    if up():
        return True
    exe = Path(__file__).resolve().parent.parent / "kinect" / "bin" / "kinect_bridge.exe"
    proc = subprocess.Popen([str(exe)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if up():
            time.sleep(1.5)  # first frames
            return True
        if proc.poll() is not None:
            return False
        time.sleep(0.25)
    return False
