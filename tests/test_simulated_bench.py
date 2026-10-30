"""Lunes' circuit work on a made-up bench, no hardware: python tests/test_simulated_bench.py

- The Arduino templates meet their goals in ngspice, and their sketches compile (when arduino-cli is installed).
- Through Hume's own tools: a circuit with an Arduino is designed, its sketch is put on a simulated board, the
  serial monitor shows the sketch's lines, mcu_read agrees with the simulation on a healthy circuit, and with a
  fault planted the diagnosis takes the board's readings first, then the simulated scope on each net the engine asks
  for. It may stay unsure, but it must never name a wrong fault.
- The 555 blinker's 24 faults with the scope alone: the engine's record (20 found, 4 left undecided, none wrong).

The first run simulates every fault dictionary (a minute or two); later runs read them from runs/.
"""
import random
import sys
import time
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lunes import mcu  # noqa: E402
from lunes.circuit import check, with_fault  # noqa: E402
from lunes.designs import TEMPLATES, verify  # noqa: E402
from lunes.diagnosis import Diagnosis  # noqa: E402
from lunes.hantek import SimulatedScope  # noqa: E402
from lunes.hume import Hume  # noqa: E402
from lunes.simulator import SELFTEST_555, expected, fault_dictionary, periodic_signal, waveform  # noqa: E402

BLINKER = {"name": "555 astable at 1 kHz with an LED", "supply_v": 5, "parts": SELFTEST_555["parts"] + [
    {"ref": "R3", "kind": "R", "value": "330", "pins": ["OUT", "LED"]},
    {"ref": "D1", "kind": "LED", "color": "red", "pins": {"A": "LED", "K": "GND"}}]}
FAILED = []


def ok(condition, what):
    print(("  ok   " if condition else "  FAIL ") + what)
    if not condition:
        FAILED.append(what)


def templates():
    print("Arduino templates")
    for name, params in [("arduino_blink", {}), ("arduino_pwm_rc", {"duty": 0.3}), ("arduino_transistor", {}),
                         ("arduino_read_divider", {"vin": 12}), ("arduino_blink", {"board": "BLUEPILL", "pin": "PC13"})]:
        circuit, spec, _ = TEMPLATES[name](**params)
        ok(not check(circuit) and all(g["ok"] for g in verify(expected(circuit), spec)), f"{name} {params} meets its goals")
        if mcu.CLI.exists():
            board = next(p for p in circuit["parts"] if p["kind"] in mcu.BOARDS)
            ok(mcu.compile_sketch(board, circuit["name"])["ok"], f"{name} {params} sketch compiles")


def bench(template, params, fault):
    """Hume on a simulated board with fault planted; returns the diagnosis' last state."""
    machine = types.SimpleNamespace(s=types.SimpleNamespace(port="COM8"), config={})
    hume = Hume(machine, None, mcu=mcu.SimulatedBoard(fault))
    hume.run_tool("circuit_design", {"template": template, "params": params})
    hume.run_tool("mcu_program", {})
    state, _ = hume.run_tool("diagnose_start", {})
    shown, _ = hume.run_tool("serial_monitor", {"seconds": 1})
    read, _ = hume.run_tool("mcu_read", {"seconds": 2})
    if fault is None:
        ok(shown["ok"] and shown["lines"] and shown["lines"][0].startswith("{"), "serial_monitor shows the sketch's lines")
        ok(read.get("all_as_simulated") is True, f"{template}: a healthy board reads as simulated")
    state = read.get("diagnosis") or state
    on_bench = with_fault(hume.circuit, fault) if fault else hume.circuit
    for _ in range(6):
        if state.get("clear") or "next" not in state:
            break
        net = state["next"]["net"]
        rate, v = waveform(on_bench, net, hume.expected["simulated_s"])
        hume.scope = SimulatedScope({"CH1": periodic_signal(rate, v), "CH2": lambda t: np.zeros_like(t)})
        state, _ = hume.run_tool("diagnose_measure", {"net": net})
    return state


def arduino_diagnosis():
    print("Arduino on the simulated bench, through Hume's tools")
    for template, params, fault in [("arduino_pwm_rc", {"duty": 0.3}, None), ("arduino_pwm_rc", {"duty": 0.3}, "R1:open"),
                                    ("arduino_pwm_rc", {"duty": 0.3}, "C1:short"),
                                    ("arduino_pwm_rc", {"duty": 0.3}, "U1:D9_floating"),
                                    ("arduino_pwm_rc", {"duty": 0.3}, "U1:not_running"),
                                    ("arduino_read_divider", {"vin": 12}, "R1:x10"),
                                    ("arduino_read_divider", {"vin": 12}, "R2:open")]:
        state = bench(template, params, fault)
        top = state["likely"][0]
        truth = fault or "healthy"
        named_wrong = state.get("clear") and truth not in top["faults"]
        ok(not named_wrong, f"{template} with {truth}: " +
           (f"found ({top['p']})" if state.get("clear") else f"unsure, leading {top['faults']} ({top['p']})"))


def diagnosis_555():
    print("555 blinker, every fault, scope only")
    random.seed(4)
    table = fault_dictionary(BLINKER)
    right = unsure = wrong = 0
    for truth, entry in table.items():
        if entry["nets"] is None:
            continue
        d = Diagnosis(BLINKER, table)
        for _ in range(8):
            s = d.state()
            if s["clear"] or "next" not in s:
                break
            pred = entry["nets"][s["next"]["net"]]
            reading = {"mean_v": pred["mean_v"] + random.gauss(0, 0.03),
                       "vpp_v": max(0, pred["vpp_v"] + random.gauss(0, 0.04)), "frequency_hz": None, "duty": None}
            if pred.get("frequency_hz") and pred["vpp_v"] >= 0.1:
                reading["frequency_hz"] = pred["frequency_hz"] * (1 + random.gauss(0, 0.06))
                reading["duty"] = pred.get("duty") and pred["duty"] + random.gauss(0, 0.01)
            d.update(s["next"]["net"], reading)
        s = d.state()
        found = truth in s["likely"][0]["faults"]
        right += s["clear"] and found
        unsure += not s["clear"]
        wrong += s["clear"] and not found
    ok(wrong == 0 and right >= 20, f"555: {right} found, {unsure} undecided, {wrong} wrong")


if __name__ == "__main__":
    t0 = time.time()
    templates()
    arduino_diagnosis()
    diagnosis_555()
    print(f"\n{'all passed' if not FAILED else f'{len(FAILED)} FAILED'} in {time.time() - t0:.0f} s")
    sys.exit(1 if FAILED else 0)
