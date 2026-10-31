"""The oscilloscope's auto measurement on made-up signals, no scope needed: python tests/test_scope_auto.py

autoset picks the range and the sample rate the way a bench scope's Auto button does; measure reads the signal. Each
signal is checked with both probe settings (x10 and x1): square waves from 50 Hz to 1 MHz, PWM, logic at 3.3, 5 and
12 V, sines, steady levels, and serial data at 9600 and 115200 baud. The simulated scope quantises to 8 bits, clips
at its range and adds 10 mV of noise, so the swing reads a little high. Past the widest range the reading must say
it clips, not pretend. Then the self-test on the calibration output, with the probe right and wrong.
"""
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lunes.hantek import SimulatedScope, sine, square, zero_of  # noqa: E402
from lunes.hume import Hume  # noqa: E402
from lunes.signals import autoset  # noqa: E402

FAILED = []


def ok(condition, what):
    print(("  ok   " if condition else "  FAIL ") + what)
    if not condition:
        FAILED.append(what)


def uart(baud, text=b"Lunes 123\n", high=3.3):
    bits = []
    for byte in text:
        bits += [0] + [(byte >> i) & 1 for i in range(8)] + [1] * 4  # start, 8 data bits, stop and a gap
    bits = np.array([1] * 10 + bits, float)
    return lambda t: bits[(np.asarray(t) * baud).astype(np.int64) % len(bits)] * high


def steady(v):
    return lambda t: np.full_like(np.asarray(t, float), v)


CASES = [(f"square {hz:g} Hz, 0-5 V", square(hz, 0, 5), {"frequency_hz": hz, "duty": 0.5, "vpp_v": 5})
         for hz in (50, 1e3, 10e3, 100e3, 1e6)]
CASES += [(f"PWM 490 Hz at {d:.0%}", square(490, 0, 5, d), {"frequency_hz": 490, "duty": d}) for d in (0.1, 0.3, 0.75)]
CASES += [("logic 1 kHz, 0-3.3 V", square(1e3, 0, 3.3), {"frequency_hz": 1e3, "vpp_v": 3.3}),
          ("logic 2 kHz, 0-12 V", square(2e3, 0, 12), {"frequency_hz": 2e3, "vpp_v": 12}),
          ("sine 60 Hz, 1 V", sine(60, 1.0), {"frequency_hz": 60, "vpp_v": 2}),
          ("sine 1 kHz, 1 V on 2.5 V", sine(1e3, 1.0, 2.5), {"frequency_hz": 1e3, "vpp_v": 2, "mean_v": 2.5}),
          ("sine 50 kHz, 2 V", sine(50e3, 2.0), {"frequency_hz": 50e3, "vpp_v": 4})]
CASES += [(f"steady {v:g} V", steady(v), {"mean_v": v, "shape": "dc"}) for v in (0.0, 1.5, 3.3, 5.0, 12.0)]
CASES += [(f"serial at {b} baud", uart(b), {"baud_if_uart": b}) for b in (9600, 115200)]


def agrees(key, got, want):
    if key in ("baud_if_uart", "shape"):
        return got == want
    if key == "duty":
        return abs(got - want) <= 0.02
    if key == "frequency_hz":
        return abs(got / want - 1) <= 0.01
    return abs(got - want) <= 0.03 * abs(want) + 0.12  # the noise adds to the swing


def auto():
    for probe in (10, 1):
        print(f"Auto measurement, probe x{probe}")
        for name, signal, want in CASES:
            scope = SimulatedScope({"CH1": signal, "CH2": steady(0)}, probe={"CH1": probe, "CH2": probe})
            cap, m, _ = autoset(scope, "CH1", alone=True)
            if float(np.abs(signal(np.linspace(0, 0.01, 2000))).max()) > 5.12 * probe:
                # past the widest range: the capture must say it clips, at the widest range
                ok(cap.clipped["CH1"] > 0.001 and scope.range_v["CH1"] == 5 * probe,
                   f"{name}: past the range, reported as clipping")
                continue
            wrong = [f"{k} {m.get(k)} not {w}" for k, w in want.items() if m.get(k) is None or not agrees(k, m[k], w)]
            ok(not wrong, f"{name}: " + ("; ".join(wrong) if wrong else
                                         f"range ±{scope.range_v['CH1']:g} V, {scope.rate:g} S/s, {m['shape']}"))


def zero():
    print("A range's zero from readings in ADC counts")
    noise = np.random.default_rng(3).normal(0, 0.4, 100000)
    flat = np.round(1.2 + noise)  # whole counts, as the converter gives them: the noise lets the mean tell tenths
    ok(abs(zero_of(flat) - 1.2) < 0.02, f"tip on the ground clip, 1.2 counts off: read {zero_of(flat):.3f}")
    for high in (5, 50):
        square_wave = np.round(np.where(np.arange(100000) % 1000 < 500, 0.7, 0.7 + high) + noise)
        ok(abs(zero_of(square_wave) - 0.7) < 0.03,
           f"tip on the calibration output, {high} counts high: its low level read {zero_of(square_wave):.3f}")


def selftest():
    print("Self-test on the calibration output (2 V, 1 kHz)")
    machine = types.SimpleNamespace(s=types.SimpleNamespace(port="COM8"), config={})
    for believed, really, expect in ((10, 10, None), (1, 1, None), (1, 10, "ten times too small"),
                                     (10, 1, "ten times too big")):
        # what reaches the tip, as the scope reports it with the factor Lunes believes
        amplitude = 2.0 * believed / really
        scope = SimulatedScope({"CH1": square(1000, 0, amplitude), "CH2": steady(0)},
                               probe={"CH1": believed, "CH2": believed})
        hume = Hume(machine, None, scope=scope)
        result, _ = hume.run_tool("scope_selftest", {"channel": "CH1"})
        said = " ".join(result.get("problems", []))
        ok(result["ok"] if expect is None else expect in said,
           f"probe at x{really}, Lunes counting x{believed}: " + ("passes" if result["ok"] else said[:90]))


if __name__ == "__main__":
    auto()
    zero()
    selftest()
    print(f"\n{'all passed' if not FAILED else f'{len(FAILED)} FAILED'}")
    sys.exit(1 if FAILED else 0)
