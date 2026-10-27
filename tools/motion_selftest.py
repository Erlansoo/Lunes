"""Self-test for the Lunes Motion firmware in dry-run mode: no step pulses leave the board, so nothing moves.

Checks the boot banner, the trapezoid timing against the expected duration, soft stop, hard stop, the host-silence
watchdog, soft limits and the refusal to move with motors disabled.

    python tools/motion_selftest.py COM8
"""
import json
import sys
import time

import serial


class Board:
    def __init__(self, port):
        self.s = serial.Serial(port, 115200, timeout=0.05)
        time.sleep(2.5)  # the board resets when the port opens
        self.banner = self.read_all()

    def read_all(self):
        out = []
        while True:
            raw = self.s.readline()
            if not raw:
                return out
            out.append(json.loads(raw))

    def send(self, text):
        self.s.write(text.encode() + b"\n")
        time.sleep(0.05)
        return self.read_all()

    def status(self):
        self.s.write(b"?")
        while True:
            for msg in self.read_all():
                if "st" in msg:
                    return msg

    def wait_done(self, timeout, poll=True):
        """Waits for the done event; polls status meanwhile so the watchdog stays quiet (unless poll is False)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if poll:
                self.s.write(b"?")
            for msg in self.read_all():
                if msg.get("ev") == "done":
                    return msg, time.time() - t0
            time.sleep(0.1)
        raise TimeoutError("no done event")


def expected_duration(n, rate, accel, start=200.0):
    """Trapezoid (or triangle) duration for n steps with the firmware's start rate."""
    d_acc = (rate**2 - start**2) / (2 * accel)
    if 2 * d_acc <= n:
        return 2 * (rate - start) / accel + (n - 2 * d_acc) / rate
    peak = (start**2 + accel * n) ** 0.5
    return 2 * (peak - start) / accel


def main():
    b = Board(sys.argv[1])
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")

    check("boot banner", any(m.get("ev") == "boot" for m in b.banner), str(b.banner))
    check("info", any(m.get("fw") == "lunes-motion" for m in b.send("INFO")))
    check("refuses MOVE with motors disabled", b.send("REL X100 F500") == [{"err": "disabled"}])

    b.send("DRY 1")
    b.send("CFG X R5000 A8000")
    b.send("ZERO")
    for n, rate in ((10000, 5000), (1000, 5000)):
        b.send(f"REL X{n} F{rate}")
        done, took = b.wait_done(10)
        want = expected_duration(n, min(rate, 5000), 8000)
        check(f"trapezoid {n} steps", abs(took - want) < 0.15 + 0.03 * want and done["stopped"] == 0,
              f"{took:.2f} s, expected {want:.2f} s")
    check("position count", b.status()["pos"][0] == 11000, str(b.status()["pos"]))

    b.send("REL X-20000 F5000")
    time.sleep(0.8)
    b.s.write(b"!")
    done, _ = b.wait_done(5)
    st = b.status()
    check("soft stop", done["stopped"] == 1 and st["alarm"] == "" and 0 < st["pos"][0] < 11000, str(st["pos"]))

    b.send("REL X5000 F2000")
    time.sleep(0.3)
    b.s.write(b"\x18")
    done, _ = b.wait_done(2)
    check("hard stop raises alarm", b.status()["alarm"] == "hard_stop")
    check("refuses MOVE in alarm", b.send("REL X10 F100") == [{"err": "alarm"}])
    b.send("RESET")

    b.send("REL X50000 F2000")
    done, took = b.wait_done(5, poll=False)
    check("host-silence watchdog", done["stopped"] == 1 and b.status()["alarm"] == "host_lost", f"{took:.2f} s")
    b.send("RESET")

    p = b.status()["pos"][0]
    b.send(f"LIM X {p - 100} {p + 100}")
    check("soft limit", b.send("REL X500 F500") == [{"err": "limit"}])
    b.send("LIM X OFF")

    b.send("SLAVE A Y")
    b.send("CFG A I1")
    b.send("ZERO")
    b.send("REL Y1000 F2000")
    b.wait_done(5)
    reply = b.send("REL A500 F2000")  # a slaved axis ignores its own target, so nothing moves
    if not any(m.get("ev") == "done" for m in reply):
        b.wait_done(5)
    pos = b.status()["pos"]
    check("slaved axis copies its master", pos[1] == 1000 and pos[3] == 1000, str(pos))
    check("refuses a slave chain", b.send("SLAVE Y A") == [{"err": "axis"}])
    b.send("SLAVE A OFF")
    b.send("CFG A I0")

    # Jog: velocities held by repeating JV, like a held key; stopping is the deadman's job.
    def jog(cmd, seconds):
        t0 = time.time()
        while time.time() - t0 < seconds:
            b.s.write(cmd.encode() + b"\n")
            time.sleep(0.1)

    def wait_idle(timeout=2.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            st = b.status()
            if st["st"] == "idle":
                return st, time.time() - t0
            time.sleep(0.05)
        return st, timeout

    for ax in "XY":
        b.send(f"CFG {ax} R5000 A20000")
    b.send("ZERO")
    jog("JV X4000 Y-3000", 1.0)
    st, took = wait_idle()
    x, y = st["pos"][0], st["pos"][1]
    check("jog moves two axes at once and the deadman stops them", st["st"] == "idle" and took < 1.0
          and 3000 < x < 7000 and -5500 < y < -2000, f"pos {st['pos'][:2]}, stopped {took:.2f} s after release")
    b.read_all()

    jog("JV X3000", 0.4)
    b.s.write(b"!")
    st, took = wait_idle()
    check("stop key ends a jog", st["st"] == "idle", f"{took:.2f} s")
    b.read_all()

    b.send("SLAVE A Y")
    b.send("ZERO")
    jog("JV Y2000", 0.5)
    st, _ = wait_idle()
    check("jog drives the slaved axis", st["pos"][1] == st["pos"][3] and st["pos"][1] > 0, str(st["pos"]))
    b.send("SLAVE A OFF")
    b.read_all()

    p = b.status()["pos"][0]
    b.send(f"LIM X {p - 10000} {p + 800}")
    jog("JV X5000", 1.0)
    st, _ = wait_idle()
    check("jog stops at the soft limit", p + 600 <= st["pos"][0] <= p + 800, f"{st['pos'][0] - p} of 800 steps")
    b.send("LIM X OFF")
    b.read_all()
    b.send("DRY 0")
    print(f"\n{sum(results)}/{len(results)} tests passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
