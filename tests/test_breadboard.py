"""The laser's guide on a breadboard, no hardware: python tests/test_breadboard.py

Every circuit Lunes can design is laid out, and the plan is rebuilt from its holes alone and compared with the circuit. A
hole's place in mm follows the board however it is turned, and a wrong reference hole is refused. Then the guide runs
through Hume's tools with the laser replaced by a list of where it was sent.
"""
import json
import math
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lunes import breadboard  # noqa: E402
from lunes.designs import TEMPLATES  # noqa: E402
from lunes.hume import Hume  # noqa: E402

FAILED = []
CASES = [("led_resistor", {}), ("astable_555", {"frequency_hz": 1000, "duty": 0.55, "led_color": "red"}), ("npn_switch", {}),
         ("divider", {"vout": 2.5}), ("rc_filter", {"cutoff_hz": 1000}), ("opamp_noninverting", {"gain": 3}),
         ("arduino_blink", {}), ("arduino_pwm_rc", {}), ("arduino_transistor", {}), ("arduino_read_divider", {}),
         ("astable_555_adjustable", {"f_min_hz": 10e3, "f_max_hz": 15e3}), ("mosfet_switch", {}),
         ("zener_regulator", {}), ("linear_regulator", {}), ("pot_divider", {}), ("light_sensor", {}),
         ("pushbutton_led", {})]


def ok(condition, what):
    print(("  ok   " if condition else "  FAIL ") + what)
    if not condition:
        FAILED.append(what)


def plans():
    print("Every design, laid out and rebuilt from its holes")
    for name, params in CASES:
        circuit = TEMPLATES[name](**params)[0]
        try:
            result = breadboard.plan(circuit)
            used = [h for pins in result["holes"].values() for h in pins.values()]
            ends = used + [h for w in result["wires"] for h in (w["from"], w["to"])]
            twice = len(ends) - len(set(ends))
            ok(not twice, f"{name}: {len(result['steps'])} steps, {len(result['wires'])} wires"
               + (f", {twice} holes used twice" if twice else ""))
        except ValueError as e:
            ok(False, f"{name}: {e}")


def faulty():
    print("A wrong plan is caught")
    circuit = TEMPLATES["divider"](vout=2.5)[0]
    result = breadboard.plan(circuit)
    first = result["holes"]["R1"]
    second = result["holes"]["R2"]
    broken = {**result, "holes": {**result["holes"], "R1": {"1": first["1"], "2": "a60"}, "R2": second}}
    ok(breadboard.verify(circuit, broken), "a lead moved to another row splits its net")


def placement():
    print("Where a hole is on the machine")
    for turn in (0.0, 12.0, 90.0, -35.0):
        c, s = math.cos(math.radians(turn)), math.sin(math.radians(turn))

        def machine(hole):
            x, y = breadboard.board_xy(hole)
            return 100 + c * x - s * y, -150 + s * x + c * y
        place = breadboard.Placement("a1", machine("a1"), "j63", machine("j63"))
        worst = max(math.dist(place.mm(h), machine(h)) for h in ("e12", "f40", "+30", "-55"))
        ok(worst < 0.02, f"board turned {turn:g} degrees: worst {worst:.3f} mm")
    try:
        breadboard.Placement("a1", (100, -150), "j63", (100, -300))
        ok(False, "a wrong hole is refused")
    except ValueError as e:
        ok("wrong" in str(e), "two holes that cannot be those are refused")


def guide():
    print("The guide, through Hume's tools")
    config = json.loads((ROOT / "config" / "machine.json").read_text("utf-8"))
    machine = types.SimpleNamespace(s=types.SimpleNamespace(port="COM8"), config=config, at_mm={"X": 0, "Y": 0, "Z": 0})
    hume = Hume(machine, None)
    sent = []
    hume._laser_point = lambda args: (sent.append((args["x"], args["y"])) or {"ok": True}, None)
    hume.run_tool("circuit_design", {"template": "astable_555", "params": {"frequency_hz": 1000, "duty": 0.55,
                                                                         "led_color": "red"}})
    planned, _ = hume.run_tool("breadboard_plan", {})
    ok(planned["ok"] and planned["steps"] > 5, f"the plan has {planned.get('steps')} steps")
    first, _ = hume.run_tool("guide_step", {})
    ok(first["ok"] and "pointed" not in first and "no sé dónde" in first["note"], "with the board unplaced it says so")
    placed, _ = hume.run_tool("breadboard_place", {"hole_a": "a1", "x_a": 60, "y_a": -200, "hole_b": "a63", "x_b": 217.48,
                                                   "y_b": -200})
    ok(placed["ok"], "the board is placed from two holes")
    step, _ = hume.run_tool("guide_step", {"step": 1})
    ok(step["ok"] and sent, f"step 1 points the laser: {step.get('says', '')[:60]}")
    other, _ = hume.run_tool("guide_step", {"step": "next"})
    ok(other["step"] == 2, "the next step follows")
    net, _ = hume.run_tool("guide_net", {"net": "OUT"})
    ok(net["ok"] and "OUT" in net["says"] and len(sent) >= 3, f"a net is shown: {net.get('says', '')[:70]}")
    ground, _ = hume.run_tool("guide_net", {"net": "GND"})
    ok("masa" in ground["says"], "ground is shown as where the clip goes")
    ok(not hume.run_tool("guide_net", {"net": "NOPE"})[0]["ok"], "a net that does not exist is refused")


if __name__ == "__main__":
    plans()
    faulty()
    placement()
    guide()
    print(f"\n{'all passed' if not FAILED else f'{len(FAILED)} FAILED'}")
    sys.exit(1 if FAILED else 0)
