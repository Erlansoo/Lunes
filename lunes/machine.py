"""Client for the Lunes Motion firmware: machine axes in millimetres on top of firmware axes in steps.

The firmware counts steps on its five driver sockets (X, Y, Z, A, B). config/machine.json says which socket drives
each machine axis, its steps per millimetre, speed, acceleration, direction and any slaved second motor.

The board forgets its position whenever the port opens (it resets). So the last position is kept in
runs/machine_position.json after every move and taken back on the next start: positions stay on the bench's own
origin across sessions, as long as nobody moves the machine by hand in between. An axis with an endstop can also be
homed (home), which puts it back on that origin whatever happened.

Backlash: an axis with backlash_mm in the config gets that much extra travel whenever it turns round, so the
carriage, not just the motor, ends where it was sent (measured by tools/backlash.py).
"""
import json
import time
from pathlib import Path

import serial

CONFIG = Path(__file__).resolve().parent.parent / "config" / "machine.json"
POSITION = Path(__file__).resolve().parent.parent / "runs" / "machine_position.json"
EVENTS = Path(__file__).resolve().parent.parent / "runs" / "machine_events.jsonl"
FW_AXES = "XYZAB"
RATE_CAP = 18000  # steps/s, the firmware's ceiling


def load_position(key="mm"):
    """Machine position (mm per axis) saved by the last session, or None; key="dir" gives the direction each
    axis last moved in."""
    try:
        return json.loads(POSITION.read_text(encoding="utf-8"))[key]
    except (OSError, ValueError, KeyError):
        return None


def log_event(event):
    """One line in runs/machine_events.jsonl: a move the firmware stopped, and how long the PC had been quiet."""
    try:
        EVENTS.parent.mkdir(exist_ok=True)
        with EVENTS.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"at": time.strftime("%Y-%m-%d %H:%M:%S"), **event}) + "\n")
    except OSError:
        pass


def save_position(mm, by, direction=None):
    POSITION.parent.mkdir(exist_ok=True)
    state = {"mm": {k: round(v, 4) for k, v in mm.items()}, "by": by, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if direction:
        state["dir"] = direction
    POSITION.write_text(json.dumps(state), encoding="utf-8")


class Axis:
    def __init__(self, name, cfg):
        self.name = name
        self.socket = cfg.get("socket", name)
        self.spm = cfg["steps_per_mm"]
        self.speed = cfg.get("speed_mm_s", 2.0)
        self.accel = cfg.get("accel_mm_s2", 10.0)
        self.invert = cfg.get("invert", False)
        self.slave = cfg.get("slave")
        self.backlash = float(cfg.get("backlash_mm", 0.0))


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
        # which way each axis last moved (+1 / -1, 0 unknown): a reversal takes up the backlash first
        self.direction = {n: int((load_position("dir") or {}).get(n, 0)) if keep_position else 0 for n in self.axes}
        self.poll_gap = 0.0  # of the last move: the longest the PC went without writing to the board
        self.at_mm = self.position()

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
        self.at_mm = self.position()
        save_position(self.at_mm, "set_position", self.direction)

    def move_to(self, timeout=120, _again=20, **targets_mm):
        """Coordinated move to absolute millimetres; every axis arrives together, each within its own speed. A move the
        firmware stopped with host_lost (it heard nothing from the PC for a second, now and then in a long X move even
        while the PC polled every 0.5 s: once three times in a row, every 27 mm) is made again while each try gets
        closer: the firmware stops cleanly and counts every step, so the position stays exact."""
        remaining = max((abs(v - self.at_mm[n]) for n, v in targets_mm.items()), default=0.0)
        for n, v in targets_mm.items():
            delta = v - self.at_mm[n]
            if abs(delta) < 1e-6:
                continue
            sign = 1 if delta > 0 else -1
            if self.axes[n].backlash and self.direction[n] and sign != self.direction[n]:
                # turning round: the motor goes the backlash further, which the carriage never sees
                self.offset_mm[n] -= sign * self.axes[n].backlash
            self.direction[n] = sign
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
            done = [m for m in reply if m.get("ev") == "done"]
            if not done:
                self.wait_done(timeout)
            elif done[0].get("stopped"):  # a short move can end, or be stopped, before send() returns
                raise RuntimeError(f"the move stopped early: {done[0]}")
        except RuntimeError:
            alarm, here = self.status().get("alarm"), self.position()
            closer = max(abs(v - here[n]) for n, v in targets_mm.items()) < remaining - 0.5
            _again = _again if alarm == "host_lost" and closer else 0
            log_event({"event": alarm or "stopped", "targets_mm": targets_mm, "longest_poll_gap_s": self.poll_gap,
                       "again": bool(_again), "tries_left": _again})
            if not _again:
                raise
        else:
            _again = False  # it arrived
        finally:  # also after a stopped move: wherever it ended is where the machine is
            self.at_mm = self.position()
            save_position(self.at_mm, "move", self.direction)
        self.last_motion = time.time()
        if _again:  # stopped by host_lost: clear it and finish the same move
            self.reset_alarm()
            self.move_to(timeout, _again=_again - 1, **targets_mm)

    def soft_stop(self):
        """Brakes the current move (safe to call from another thread); the waiting move_to then raises."""
        self.s.write(b"!")

    def move_by(self, **deltas_mm):
        here = self.position()
        self.move_to(**{n: here[n] + d for n, d in deltas_mm.items()})

    def wait_done(self, timeout=120):
        t0 = last = time.time()
        self.poll_gap = 0.0  # the longest the PC went without writing to the board during this move
        while time.time() - t0 < timeout:
            now = time.time()
            self.poll_gap, last = max(self.poll_gap, now - last), now
            self.s.write(b"?")  # status polls keep the host watchdog quiet
            for m in self._read():
                if m.get("ev") == "done":
                    if m.get("stopped"):
                        raise RuntimeError(f"the move stopped early: {m}")
                    return m
            time.sleep(0.05)
        raise TimeoutError("the move did not finish")

    def _ask(self, cmd, key, timeout=1.0):
        """Sends cmd and returns the firmware's reply that carries key."""
        self.s.write(cmd.encode() + b"\n")
        t0 = time.time()
        while time.time() - t0 < timeout:
            for m in self._read():
                if "err" in m:
                    raise RuntimeError(f"the motion board refused {cmd.split()[0]}: {m['err']}")
                if key in m:
                    return m[key]
        raise TimeoutError("the motion board did not answer")

    def laser(self, level=255):
        """Laser pointer brightness, 0 (off) to 255. The firmware puts it out a minute after the last call."""
        self._ask(f"LASER {int(level)}", "ok")

    def imu(self):
        """The MPU6050 on the machine: acceleration in g, rotation in degrees per second, temperature in degC."""
        m = self._ask("IMU", "imu")
        return {"accel_g": [round(v / 16384, 4) for v in m["a"]], "gyro_dps": [round(v / 131, 2) for v in m["g"]],
                "temp_c": m["t"] / 100}

    def switches(self):
        """Which switches are closed: {"probe": bool, and one per endstop by machine axis}."""
        sw = self._ask("SW", "sw")
        out = {"probe": bool(sw["probe"])}
        out.update({n: bool(sw[a.socket]) for n, a in self.axes.items() if a.socket in sw})
        return out

    def reset_alarm(self):
        """Clears a firmware alarm (host_lost, endstop, hard_stop) once the cause is understood."""
        self._ask("RESET", "ok")

    def home(self, axis, travel_mm=300.0, fast_mm_s=5.0, slow_mm_s=0.5):
        """Drives axis toward its minimum until its endstop closes, backs off 2 mm and comes back slowly. There the
        machine is at the axis's home_mm in config/machine.json; the first homing records it from where Lunes believes
        it is, so the bench's own origin stays where it was. Returns where Lunes believed it was and the correction."""
        ax = self.axes[axis]
        self.enable(True)

        def seek(mm, speed):
            reply = self.send(f"SEEK {ax.socket}{-round(mm * ax.spm)} F{speed * ax.spm:.0f}")
            errors = [m["err"] for m in reply if "err" in m]
            if errors:
                raise RuntimeError(f"the motion board refused to home {axis}: {errors[0]}")
            done = next((m for m in reply if m.get("ev") == "done"), None) or self.wait_done(travel_mm / speed + 10)
            if not done.get("touch"):
                raise RuntimeError(f"{axis} went {mm} mm toward its minimum without closing its endstop")

        try:
            seek(travel_mm, fast_mm_s)
            self.send(f"REL {ax.socket}{round(2.0 * ax.spm)} F{fast_mm_s * ax.spm:.0f}")
            self.wait_done(10)
            seek(3.0, slow_mm_s)
        finally:
            self.at_mm = self.position()
            self.direction[axis] = -1
            self.last_motion = time.time()
        believed = self.at_mm[axis]
        cfg = self.config["axes"][axis]
        if "home_mm" not in cfg:
            cfg["home_mm"] = round(believed, 3)
            self.save_config()
        self.set_position(**{axis: cfg["home_mm"]})
        return {"believed_mm": round(believed, 3), "home_mm": cfg["home_mm"],
                "correction_mm": round(cfg["home_mm"] - believed, 3)}

    def save_config(self):
        self.config_path.write_text(json.dumps(self.config, indent=2) + "\n", encoding="utf-8")

    def close(self):
        try:
            self.send("EN 0")
        finally:
            self.s.close()
