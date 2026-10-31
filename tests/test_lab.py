"""The web lab with no hardware: python tests/test_lab.py

What the page gets from Hume must be plain JSON at every stage of a session, and the page must load nothing from outside
the PC. A real server on a free port answers as the browser would ask.
"""
import json
import sys
import types
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lunes import mcu  # noqa: E402
from lunes.dashboard import PAGE, Dashboard  # noqa: E402
from lunes.hantek import SimulatedScope, square  # noqa: E402
from lunes.hume import Hume  # noqa: E402
from lunes.lab import state  # noqa: E402

FAILED = []


def ok(condition, what):
    print(("  ok   " if condition else "  FAIL ") + what)
    if not condition:
        FAILED.append(what)


def plain(data):
    return json.loads(json.dumps(data))


def stages():
    print("What the page gets, stage by stage")
    config = json.loads((ROOT / "config" / "machine.json").read_text("utf-8"))
    machine = types.SimpleNamespace(s=types.SimpleNamespace(port="COM8"), config=config,
                                    at_mm={"X": 10.0, "Y": -20.0, "Z": 5.0})
    hume = Hume(machine, None, scope=SimulatedScope({"CH1": square(1000, 0, 2), "CH2": square(1000, 0, 2)}),
                mcu=mcu.SimulatedBoard("R1:open"))
    first = plain(state(hume))
    ok(first["circuit"] is None and first["diagnosis"] is None, "at the start there is no circuit and no diagnosis")
    ok(first["machine"]["at_mm"]["X"] == 10.0, "the machine's position comes through")
    ok(first["instruments"]["scope"] and first["scope"]["simulated"], "a simulated scope says it is simulated")

    hume.run_tool("scope_measure", {"channel": "both"})
    scope = plain(state(hume))["scope"]
    ok(set(scope["channels"]) == {"CH1", "CH2"}, "both channels' readings arrive")
    ok(len(scope["trace"]["channels"]["CH1"]["low"]) == 480, "the traces arrive for the page to draw")

    hume.run_tool("circuit_design", {"template": "arduino_pwm_rc", "params": {"duty": 0.3}})
    hume.run_tool("mcu_program", {})
    hume.run_tool("diagnose_start", {})
    hume.run_tool("serial_monitor", {"seconds": 1})
    hume.run_tool("mcu_read", {"seconds": 1})
    data = plain(state(hume))
    ok(data["circuit"]["nets"] and "GND" not in data["circuit"]["nets"], "the circuit's nets arrive, without GND")
    ok(data["diagnosis"]["likely"], "the diagnosis' explanations arrive")
    ok(data["arduino"]["pins"]["A0"]["ok"] is False, "the Arduino's reading against the simulation arrives")
    ok(data["arduino"]["serial"], "the serial monitor's lines arrive")
    readings = len(hume.diagnosis.readings)
    again = state(hume)["diagnosis"]
    ok(hume.lab_diagnosis[0][1] == readings and again is hume.lab_diagnosis[1], "the diagnosis is worked out once per reading")
    return hume


def page(hume):
    print("The page")
    outside = [w for w in PAGE.decode().replace("http://www.w3.org/2000/svg", "").split() if "http" in w or "//cdn" in w]
    ok(not outside, f"it loads nothing from outside the PC {outside[:3]}")
    board = Dashboard(port=0, lab=lambda: state(hume))
    base = f"http://127.0.0.1:{board.server.server_address[1]}"
    with urllib.request.urlopen(base + "/") as r:
        ok(b"<title>Lunes Lab</title>" in r.read(), "the server gives the page")
    board.add("tu", "hola")
    with urllib.request.urlopen(base + "/events?since=0") as r:
        data = json.loads(r.read())
    ok(data["events"][0]["text"] == "hola" and "lab" in data, "the events carry the conversation and the bench")
    board.server.shutdown()


if __name__ == "__main__":
    page(stages())
    print(f"\n{'all passed' if not FAILED else f'{len(FAILED)} FAILED'}")
    sys.exit(1 if FAILED else 0)
