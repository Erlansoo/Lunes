"""Client for the Lunes Motion firmware: machine axes in millimetres on top of firmware axes in steps.

The firmware counts steps on its five driver sockets (X, Y, Z, A, B). config/machine.json says which socket drives
each machine axis, its steps per millimetre, speed, acceleration, direction and any slaved second motor.

The board forgets its position whenever the port opens (it resets), and there are no limit switches. So the last
position is kept in runs/machine_position.json after every move and taken back on the next start: positions stay
on the bench's own origin across sessions, as long as nobody moves the machine by hand in between.
"""
import json
import time
from pathlib import Path

import serial

CONFIG = Path(__file__).resolve().parent.parent / "config" / "machine.json"
POSITION = Path(__file__).resolve().parent.parent / "runs" / "machine_position.json"
FW_AXES = "XYZAB"
RATE_CAP = 18000  # steps/s, the firmware's ceiling


def load_position():
    """Machine position (mm per axis) saved by the last session, or None."""
    try:
        return json.loads(POSITION.read_text(encoding="utf-8"))["mm"]
    except (OSError, ValueError, KeyError):
        return None


def save_position(mm, by):
    POSITION.parent.mkdir(exist_ok=True)
    POSITION.write_text(json.dumps({"mm": {k: round(v, 4) for k, v in mm.items()}, "by": by,
                                    "at": time.strftime("%Y-%m-%d %H:%M:%S")}), encoding="utf-8")


class Axis:
    def __init__(self, name, cfg):
        self.name = name
        self.socket = cfg.get("socket", name)
        self.spm = cfg["steps_per_mm"]
        self.speed = cfg.get("speed_mm_s", 2.0)
        self.accel = cfg.get("accel_mm_s2", 10.0)
        self.invert = cfg.get("invert", False)
        self.slave = cfg.get("slave")


class Machine:
    def __init__(self, port="COM8", config_path=CONFIG, keep_position=True):
        """keep_position=False starts from zero instead of the position the last session left."""
        self.config_path = Path(config_path)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.axes = {n: Axis(n, c) for n, c in self.config["axes"].items() if "steps_per_mm" in c}
        # The board resets when the port opens (also with DTR held low, on this FT232). Send nothing until the
        # sketch says it booted: bytes arriving during the bootloader's wait keep it in the bootloader.
        self.s = serial.Serial(port, 115200, timeout=0.05)
        t0 = time.time()
        while time.time() - t0 < 4 and not any(m.get("ev") == "boot" for m in self._read()):
            pass
        for ax in self.axes.values():
            rate = min(ax.speed * ax.spm, RATE_CAP)
            self.send(f"CFG {ax.socket} R{rate:.0f} A{ax.accel * ax.spm:.0f} I{1 if ax.invert else 0}")
            if ax.slave:
                self.send(f"SLAVE {ax.slave['socket']} {ax.socket}")
                self.send(f"CFG {ax.slave['socket']} I{1 if ax.slave.get('invert') else 0}")
        self.enabled = False
        self.last_motion = time.time()
        # Firmware steps count from where the board last woke up; offset_mm says where that is on the bench.
        self.offset_mm = {n: 0.0 for n in self.axes}
        saved = load_position() if keep_position else None
        if saved:
            here = self.position()
            self.offset_mm = {n: float(saved.get(n, here[n])) - here[n] for n in self.axes}

    def _read(self):
        out = []
        while True:
            raw = self.s.readline()
            if not raw:
                return out
            try:
                out.append(json.loads(raw))
            except ValueError:
                pass

    def send(self, cmd):
        self.s.write(cmd if isinstance(cmd, bytes) else cmd.encode() + b"\n")
        time.sleep(0.03)
        return self._read()

    def status(self, timeout=1.0):
        self.s.write(b"?")
        t0 = time.time()
        while time.time() - t0 < timeout:
            for m in self._read():
                if "st" in m:
                    return m
        raise TimeoutError("the motion board did not answer")

    def enable(self, on=True):
        if on != self.enabled:
            self.send(f"EN {1 if on else 0}")
            self.enabled = on
        if on:
            self.last_motion = time.time()

    def release_if_idle(self, seconds=20):
        """Releases the motors after a while without moving, so small drivers stay cool."""
        if self.enabled and time.time() - self.last_motion > seconds:
            self.enable(False)

    def position(self):
        pos = self.status()["pos"]
        return {n: pos[FW_AXES.index(a.socket)] / a.spm + self.offset_mm[n] for n, a in self.axes.items()}

    def set_position(self, **mm):
        """Says where the machine is now (for the axes given), without moving it."""
        here = self.position()
        for n, v in mm.items():
            self.offset_mm[n] += float(v) - here[n]
        save_position(self.position(), "set_position")

    def move_to(self, timeout=120, **targets_mm):
        """Coordinated move to absolute millimetres; every axis arrives together, each within its own speed."""
        words = [f"{self.axes[n].socket}{round((v - self.offset_mm[n]) * self.axes[n].spm)}"
                 for n, v in targets_mm.items()]
        if not words:
            return
        self.enable(True)
        reply = self.send(f"MOVE {' '.join(words)} F{RATE_CAP}")
        errors = [m["err"] for m in reply if "err" in m]
        if errors:
            raise RuntimeError(f"the motion board refused the move: {errors[0]}")
        try:
            if not any(m.get("ev") == "done" for m in reply):
                self.wait_done(timeout)
        finally:  # also after a stopped move: wherever it ended is where the machine is
            save_position(self.position(), "move")
        self.last_motion = time.time()

    def soft_stop(self):
        """Brakes the current move (safe to call from another thread); the waiting move_to then raises."""
        self.s.write(b"!")

    def move_by(self, **deltas_mm):
        here = self.position()
        self.move_to(**{n: here[n] + d for n, d in deltas_mm.items()})

    def wait_done(self, timeout=120):
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.s.write(b"?")  # status polls keep the host watchdog quiet
            for m in self._read():
                if m.get("ev") == "done":
                    if m.get("stopped"):
                        raise RuntimeError(f"the move stopped early: {m}")
                    return m
            time.sleep(0.05)
        raise TimeoutError("the move did not finish")

    def save_config(self):
        self.config_path.write_text(json.dumps(self.config, indent=2) + "\n", encoding="utf-8")

    def close(self):
        try:
            self.send("EN 0")
        finally:
            self.s.close()
