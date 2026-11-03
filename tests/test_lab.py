"""The web lab with no hardware: python tests/test_lab.py

What the page gets from Hume must be plain JSON at every stage of a session, and the page must load nothing from outside
the PC. A real server on a free port answers as the browser would ask.
"""
import inspect
import json
import re
import sys
import types
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lunes import catalog, mcu  # noqa: E402
from lunes import hume as hume_module  # noqa: E402
from lunes.breadboard import plan  # noqa: E402
from lunes.dashboard import PAGE, Dashboard  # noqa: E402
from lunes.hantek import SimulatedScope, square  # noqa: E402
from lunes.hume import NO_TOKENS, Hume  # noqa: E402
from lunes.lab import _parts, state  # noqa: E402
from lunes.meta_api import OverBudget  # noqa: E402
from lunes.voice import latin  # noqa: E402

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
    drawn = plain(state(hume))["breadboard"]
    ok(drawn and drawn["parts"], "a circuit shows on the breadboard at once, with nobody asking for a plan")
    hume.run_tool("circuit_design", {"template": "pushbutton_led", "params": {}})
    ok(plain(state(hume))["breadboard"]["parts"] != drawn["parts"], "a new circuit never shows the plan of the last one")
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


def trimmed():
    print("A long conversation keeps no tool result without its call")
    hume = Hume(types.SimpleNamespace(s=types.SimpleNamespace(port="COM8"), config={}), None)
    call = {"role": "assistant", "content": "", "tool_calls": [{"id": "a", "function": {"name": "see"}},
                                                               {"id": "b", "function": {"name": "see"}}]}
    # a picture's message stands between the two results: the cut falls right on it
    hume.messages = [{"role": "system", "content": "s"}, call, {"role": "tool", "tool_call_id": "a", "content": "1"},
                     {"role": "user", "content": "(imagen)"}, {"role": "tool", "tool_call_id": "b", "content": "2"},
                     {"role": "assistant", "content": "listo"}]
    hume._trim(keep=3)
    ok(not [m for m in hume.messages if m.get("role") == "tool"], "the cut left the result of a call that is gone: dropped")
    hume.messages = [{"role": "system", "content": "s"}, call, {"role": "tool", "tool_call_id": "a", "content": "1"},
                     {"role": "tool", "tool_call_id": "b", "content": "2"}]
    hume._trim(keep=3)
    ok(len([m for m in hume.messages if m.get("role") == "tool"]) == 2, "results whose call is kept stay")
    print("The local fallback does not drive the machine")
    hume.on_fallback = True
    ok(not hume.run_tool("move_by", {"dx": 1})[0]["ok"], "a move is refused while Glimmer stands in for the main model")
    hume.on_fallback = False
    print("When the main model is out, Lunes says why and does not move")
    worker = Hume(types.SimpleNamespace(s=types.SimpleNamespace(port="COM8"), config={}), None)
    worker.brain = "spark"

    def spent(on_text=None):
        raise OverBudget("today's budget is spent")

    worker._chat_spark = spent
    worker._chat_glimmer = lambda on_text=None: {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "measure_board", "arguments": {}}}]}
    ok(worker.handle("recorre la protoboard") == NO_TOKENS, "the tokens: the job ends there, with the real reason in Spanish")
    print("Never lower than a chip can be")
    heights = []
    worker.m = types.SimpleNamespace(position=lambda: {"X": 0.0, "Y": 0.0, "Z": 20.0}, move_to=lambda Z: heights.append(Z),
                                     config={"laser": {"surfaces": [{"surface": "protoboard", "z": 12.85}]}})
    worker.cam = types.SimpleNamespace(fresh_gray=lambda n: np.zeros((48, 64), np.uint8), frame=lambda: np.zeros((48, 64, 3), np.uint8))
    worker.show = lambda *a: None
    worker._scan_focus = lambda top, low: (heights.append(low), (low, 0.0))[1]
    worker._focus_here(worker._chip_floor())
    ok(min(heights) >= 11.85, f"over nothing sharp it stops at Z {min(heights):.2f}, not on the table")
    heights.clear()
    head = {"X": 0.0, "Y": 0.0, "Z": 25.0}
    speckle = np.random.default_rng(1).integers(0, 255, (48, 64), np.uint8)
    worker.m = types.SimpleNamespace(position=lambda: dict(head), move_to=lambda Z: (head.update(Z=Z), heights.append(Z)),
                                     config={"laser": {"surfaces": [{"surface": "protoboard", "z": 12.85}]}})
    worker.cam = types.SimpleNamespace(fresh_gray=lambda n: speckle if abs(head["Z"] - 17) < 1.2 else np.zeros((48, 64), np.uint8),
                                       frame=lambda: np.zeros((48, 64, 3), np.uint8))
    worker.focus_z = None
    worker._focus_board()
    ok(min(heights) > 14, f"focusing on a board stops at the first sharp surface: lowest Z {min(heights):.2f}, not the table")
    source = (ROOT / "lunes" / "hume.py").read_text("utf-8")
    offered = re.findall(r'"name":\s*"([a-z_0-9]+)",\s*\n?\s*"description"', source)
    body = inspect.getsource(Hume.run_tool)
    ok(len(offered) > 50 and not [t for t in offered if f'"{t}"' not in body], "every tool offered to the model is connected to its code")
    print("What Lunes has in mind")
    chip_seen = {"center": (320.0, 240.0), "corners": [(120, 110), (520, 110), (520, 370), (120, 370)], "size_mm": (10.0, 7.0),
                 "angle_deg": 0.0}
    config = json.loads((ROOT / "config" / "machine.json").read_text("utf-8"))
    mapped = Hume(types.SimpleNamespace(s=types.SimpleNamespace(port="COM8"), config=config, at_mm={"X": 62.0, "Y": -74.0, "Z": 17.0},
                                        position=lambda: {"X": 62.0, "Y": -74.0, "Z": 17.0}), None)
    mapped._remember_chip(chip_seen, {"X": 62.0, "Y": -74.0, "Z": 17.0}, end="left")
    ok(mapped._across_chip() == "X", "the picture lies turned on the machine: across this chip is the X axis")
    ok(mapped.run_tool("note_on_map", {"label": "NE555P", "chip": True})[0]["ok"], "the model names the chip it read")
    mapped.run_tool("note_on_map", {"label": "riel +", "x": 40, "y": -74})
    drawn = plain(state(mapped))["machine"]["mind"]
    ok(drawn["chip"]["name"] == "NE555P" and len(drawn["chip"]["corners_mm"]) == 4 and drawn["notes"][0]["label"] == "riel +",
       "the page gets the chip's corners in mm, its name and the notes")
    mapped._remember_chip(chip_seen, {"X": 90.0, "Y": -74.0, "Z": 17.0})
    ok("name" not in mapped.world["chip"], "a chip somewhere else is another chip: the old name does not follow it")
    ok("NE555P" not in mapped.brief() and "sin leer" in mapped.brief(), "the conductor is told what is on the map")
    mapped.world["chip"]["center_mm"] = [62.65, -74.6]
    mapped.m.at_mm = {"X": 62.0, "Y": -74.0, "Z": 20.0}
    low, near = {"X": 62.0, "Y": -74.0, "Z": 8.0}, {"X": 62.0, "Y": -74.0, "Z": 15.0}
    ok(mapped._no_go(low) and not mapped._no_go(near), "the last guard: below the breadboard next to the chip, refused; above it, not")
    ok(not mapped._no_go(dict(low, X=-49.0, Y=-180.0)), "the table far from the chip stays allowed")
    mapped.m.at_mm = {"X": 62.0, "Y": -74.0, "Z": 6.0}
    ok(not mapped._no_go(dict(low, Z=8.0)), "coming up from below is always allowed")
    print("A calibration that cannot be true is not kept")
    good = {"px_per_mm": {"X": 37.43, "Y": 37.08}, "angle_between_axes_deg": 90.0}
    bad = {"px_per_mm": {"X": 38.16, "Y": 35.81}, "angle_between_axes_deg": 86.76}  # tonight's, over the breadboard's grid
    rig = types.SimpleNamespace(s=types.SimpleNamespace(port="COM8"), config={"microscope": good}, save_config=lambda: None,
                                move_to=lambda Z: None)
    careful = Hume(rig, None)

    def reading(value):
        def measured(machine, camera, lifted):
            machine.config["microscope"] = value
            return value
        return measured

    real, hume_module.calibrate = hume_module.calibrate, reading(bad)
    careful._calibrate_here(18.0)
    ok(rig.config["microscope"] is good, "axes at 86.8 degrees and pixels 6 % out of square: refused, the last one stays")
    fresh = {"px_per_mm": {"X": 37.3, "Y": 37.2}, "angle_between_axes_deg": 90.2}
    hume_module.calibrate = reading(fresh)
    careful._calibrate_here(18.0)
    ok(rig.config["microscope"] is fresh, "a sound one is kept")
    hume_module.calibrate = real
    print("A placa is not a protoboard")
    ok(mapped.bed_name() == "lo que hay en la mesa", "until the model says what it sees, Lunes does not call it a placa")
    mapped._say_bed("protoboard")
    ok(mapped.bed_name() == "la protoboard" and "la protoboard" in mapped.brief() + mapped.bed_name(), "then it uses that word")
    ok(plain(state(mapped))["machine"]["bed"] == "protoboard", "and the map says it")
    tools = {t: re.search(rf'"name": "{t}".*?"required": \[(.*?)\]', source, re.S).group(1) for t in ("measure_board", "survey_board")}
    ok(all("bed" in need for need in tools.values()), "measuring or scanning, the model must say what lies on the table")
    print("A chip that has not moved keeps its notch")
    view = {"center": (320.0, 240.0), "angle_deg": 1.0}
    hume = Hume(types.SimpleNamespace(s=types.SimpleNamespace(port="COM8"),
                                      config=json.loads((ROOT / "config" / "machine.json").read_text("utf-8"))), None)
    at = {"X": 60.0, "Y": -70.0, "Z": 17.0}
    hume._kept_notch(view, at, "left", 0.9, "seen")
    end, sure, why = hume._kept_notch(view, at, None, 0.0, "not seen")
    ok(end == "left" and sure == 0.9 and "last good look" in why, "a look that misses it takes the last good one, and says so")
    ok(hume._kept_notch(view, dict(at, X=65.0), None, 0.0, "not seen")[0] is None, "a chip somewhere else has nothing to keep")
    ok(hume._kept_notch(view, at, "right", 0.9, "seen")[0] == "right", "a good look wins: the chip was turned")
    print("The breadboard as the page draws it")
    data = types.SimpleNamespace(circuit=catalog.get("LM339")["test"]["circuit"])
    laid = plan(data.circuit)
    parts = {p["ref"]: p for p in _parts(data, laid)}
    chip = next(p for p in parts.values() if p["kind"] == "IC")
    ok(chip["label"] == "LM339" and chip["learnt"] and chip["chip"] == {"first": laid["chips"][chip["ref"]]["first"], "pins": 14},
       "a chip Lunes learnt arrives with its name, its pins and its origin, so it needs no drawing of its own")
    ok(all(p["holes"] for p in parts.values()) and not any(p["kind"] == "V" for p in parts.values()),
       "every part arrives with its holes, and the supply, off the board, does not")
    print("The ear's text in another alphabet")
    ok(latin("ok lunes, señala la pata número 1") and not latin("\u0c95\u0ccd\u0cb2\u0cbe\u0cb0\u0ccb"), "Spanish passes; Kannada does not")


if __name__ == "__main__":
    page(stages())
    trimmed()
    print(f"\n{'all passed' if not FAILED else f'{len(FAILED)} FAILED'}")
    sys.exit(1 if FAILED else 0)
