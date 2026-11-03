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


def straight(a, b):
    """Two holes on one line of the board: along it, or across it."""
    (xa, ya), (xb, yb) = breadboard.board_xy(a), breadboard.board_xy(b)
    return xa == xb or ya == yb


def mirrored_board():
    print("A board seen from a left-handed machine")
    x, y = 57.83, -80.05  # pin 1 of the real chip, row 53; the board's rows run along +Y
    place = breadboard.Placement("e53", (x, y), "e13", (x - 1.04, y - 101.6), mirrored=True)
    ok(place.mm("f53")[0] > x + 7 and place.mm("a53")[0] < x - 9, "up the columns, a to j, goes towards +X, as the mosaic shows")
    ok(place.mm("-53")[0] < place.mm("a53")[0] and place.mm("+53")[0] < place.mm("-53")[0], "the rails beyond column a, the red one last")
    ok(all(breadboard.nearest_hole(place, *place.mm(h)) == h for h in ("a40", "j1", "c55", "+40", "-41")), "and each point goes back to its hole")


def grid_of_the_real_board():
    print("The real board, from three of its corners")
    grid = breadboard.BoardGrid((48.58, -212.2), (48.15, -53.88), (76.31, -212.16))
    ok(math.dist(grid.mm("a63"), (48.15, -53.88)) < 0.02 and math.dist(grid.mm("j1"), (76.31, -212.16)) < 0.02, "the corners stay where they were measured")
    ok(abs(grid.mm("a40")[1] + 112.6) < 0.1 and grid.mm("f53")[0] > grid.mm("e53")[0] + 7, "row 40 where the mosaic has it, and a to j towards +X")
    ok(all(breadboard.nearest_hole(grid, *grid.mm(h)) == h for h in ("a40", "+40", "-41", "j63")), "each point goes back to its hole")
    middle = [(a + b) / 2 for a, b in zip(grid.mm("e53"), grid.mm("f53"))]
    ok(grid.holds_chip(*middle) and not grid.holds_chip(*grid.mm("a53")), "a chip straddles the channel, never column a")
    # Nov 3: a mosaic's grid took the rails for holes and put the chip, at 62.45, -82.15, on its column a
    wrong = breadboard.BoardGrid((61.06, -219.61), (64.54, -61.02), (89.27, -219.52))
    ok(not wrong.holds_chip(62.45, -82.15), "a grid that puts the chip on column a is not the board's")


def around_the_chip():
    print("Around a chip already on the board")
    circuit = TEMPLATES["astable_555"](frequency_hz=2, duty=0.6, supply_v=5, led_color="green")[0]
    laid = breadboard.plan(circuit, first_row=53)
    ok(laid["chips"]["U1"]["first"] == "e53" and laid["holes"]["U1"][1] == "e53", "pin 1 where the chip really is, e53")
    rows = [int(h[1:]) for pins in laid["holes"].values() for h in pins.values()]
    ok(min(rows) >= 1 and max(rows) <= 63 and not breadboard.verify(circuit, laid), "the rest fits around it, and it is still the circuit")
    aslant = [ref for ref, pins in laid["holes"].items() if len(pins) == 2 and ref != "V1" and not straight(*pins.values())]
    ok(not aslant, f"every part still lies straight {aslant or ''}")


def rails():
    print("The rails stand as on the real board")
    ok([r for r in range(1, 64) if not breadboard.on_rail(r)][:6] == [1, 2, 8, 14, 20, 26] and not breadboard.on_rail(56)
       and breadboard.on_rail(61) and not breadboard.on_rail(62), "groups of five from row 3 to 61, as the mosaic shows")
    ok(breadboard.RAIL_COLUMN["-"] > breadboard.RAIL_COLUMN["+"], "next to column a comes the blue one, then the red one at the edge")
    ok(breadboard.strip_of("-5") != breadboard.strip_of("+5") and breadboard.board_xy("-5")[0] == breadboard.board_xy("+5")[0],
       "they are two different rails, side by side")


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
            outside = {item["ref"] for item in result["external"]}
            lost = [h for h in ends if h[0] in "+-" and not breadboard.on_rail(int(h[1:]))]
            ok(not lost, f"{name}: every rail hole exists {lost or ''}")
            aslant = [ref for ref, pins in result["holes"].items()
                      if len(pins) == 2 and ref not in outside and not straight(*pins.values())]
            ok(not aslant, f"{name}: every part lies straight {aslant or ''}")
            rails = [w["to"][1:] for w in result["wires"] if w["net"] in ("VCC", "GND")]
            ok(len(rails) == len(set(rails)), f"{name}: no two supply wires lie one over the other")
            colors = {w["net"]: w["color"] for w in result["wires"]}
            ok(colors.get("VCC", "rojo") == "rojo" and colors.get("GND", "negro") == "negro"
               and len(set(colors.values())) == len(colors), f"{name}: VCC red, GND black, every other net its own colour")
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
    rails()
    mirrored_board()
    grid_of_the_real_board()
    around_the_chip()
    plans()
    faulty()
    placement()
    guide()
    print(f"\n{'all passed' if not FAILED else f'{len(FAILED)} FAILED'}")
    sys.exit(1 if FAILED else 0)
