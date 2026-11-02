"""The parts Lunes knows beyond the basic ones, with no simulator: python tests/test_parts.py

A potentiometer, a zener, a Schottky diode, MOSFETs, a switch, light and heat sensors and a regulator must be
accepted when written right and refused when not; their SPICE lines and every typical fault must come out as the
bench breaks them; the review must catch a zener straight across the supply, a regulator fed too little, a floating
gate and a half-driven MOSFET; and every new design must lay out on the breadboard and rebuild as itself.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lunes import breadboard, review  # noqa: E402
from lunes.circuit import check, faults, netlist, with_fault  # noqa: E402
from lunes.designs import TEMPLATES  # noqa: E402

FAILED = []
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def ok(condition, what):
    print(("  ok   " if condition else "  FAIL ") + what)
    if not condition:
        FAILED.append(what)


def circuit(*parts, supply_v=12.0):
    return {"name": "test", "supply_v": supply_v, "parts": list(parts)}


POT = {"ref": "P1", "kind": "POT", "value": "10k", "position": 0.25, "pins": {"A": "GND", "W": "OUT", "B": "VCC"}}
ZENER = {"ref": "D1", "kind": "ZENER", "model": "BZX55C5V1", "pins": {"A": "GND", "K": "OUT"}}
RS = {"ref": "R1", "kind": "R", "value": "680", "pins": ["VCC", "OUT"]}
MOS = {"ref": "Q1", "kind": "NMOS", "model": "IRLZ44N", "pins": {"D": "D", "G": "G", "S": "GND"}}
SW = {"ref": "SW1", "kind": "SW", "state": "closed", "pins": ["VCC", "SWO"]}
LDR = {"ref": "R2", "kind": "LDR", "dark": "1M", "lit": "10k", "condition": "lit", "pins": ["VCC", "OUT"]}
NTC = {"ref": "R3", "kind": "NTC", "cold": "10k", "hot": "1k", "condition": "cold", "pins": ["OUT", "GND"]}
REG = {"ref": "U1", "kind": "REG", "model": "7805", "pins": {"IN": "VCC", "OUT": "OUT", "GND": "GND"}}
LOAD = {"ref": "R9", "kind": "R", "value": "1k", "pins": ["OUT", "GND"]}


def accepted():
    print("Written right, accepted; written wrong, refused")
    ok(not check(circuit(POT, LOAD)), "a pot across the supply, its wiper loaded")
    ok(not check(circuit(RS, ZENER, LOAD)), "a zener behind its resistor")
    ok(not check(circuit({"ref": "R1", "kind": "R", "value": "100", "pins": ["VCC", "G"]},
                         {"ref": "R2", "kind": "R", "value": "100k", "pins": ["G", "GND"]}, MOS,
                         {"ref": "R3", "kind": "R", "value": "120", "pins": ["VCC", "D"]})), "a MOSFET switch")
    ok(not check(circuit(SW, {"ref": "R1", "kind": "R", "value": "330", "pins": ["SWO", "GND"]})), "a switch")
    ok(not check(circuit(LDR, NTC)), "an LDR over an NTC")
    ok(not check(circuit(REG, LOAD)), "a regulator with its load")
    ok(not check(circuit({"ref": "D2", "kind": "D", "model": "1N5817", "pins": {"A": "VCC", "K": "OUT"}}, LOAD)),
       "a Schottky diode")
    for part, why in [(dict(POT, position=1.5), "a pot's position past 1"), (dict(POT, value="-5"), "a negative track"),
                      (dict(POT, pins={"A": "GND", "B": "VCC"}), "a pot without its wiper"),
                      (dict(ZENER, model="BZX55C99"), "a zener model that does not exist"),
                      (dict(MOS, model="2N2222"), "a MOSFET with a bipolar's model"),
                      (dict(MOS, pins={"C": "D", "B": "G", "E": "GND"}), "a MOSFET with C, B, E"),
                      (dict(SW, state="half"), "a switch half closed"), (dict(LDR, condition="dim"), "an LDR in a condition it has not"),
                      (dict(NTC, hot="-1k"), "an NTC with a negative value"), (dict(REG, model="7809"), "a regulator model that does not exist"),
                      (dict(REG, pins={"IN": "VCC", "OUT": "OUT"}), "a regulator without its GND")]:
        problems = check(circuit(part, LOAD))
        ok(bool(problems), f"{why}: refused ({(problems or [''])[0][:60]})")


def spice():
    print("SPICE lines, healthy and with each fault")
    lines = "\n".join(netlist(circuit(POT)))
    ok("RP1a n_gnd n_out 2500" in lines.replace("n_gnd", "0").replace(" 0 n_out", " n_gnd n_out") or "RP1a 0 n_out 2500" in lines,
       "a pot at 0.25 is 2.5 k from A to the wiper")
    ok("RP1b n_out n_vcc 7500" in lines, "and 7.5 k from the wiper to B")
    turned = "\n".join(netlist(with_fault(circuit(POT), "P1:turned_up")))
    ok("RP1a 0 n_out 10000" in turned and "RP1b n_out n_vcc 0.5" in turned, "turned all the way up: the wiper at B")
    worn = "\n".join(netlist(with_fault(circuit(POT), "P1:wiper_open")))
    ok("RP1a 0 n_vcc 10000" in worn and "RP1b n_out n_vcc 1e12" in worn, "a worn wiper: the track whole, the wiper on nothing")
    ok(" BZX55C5V1" in "\n".join(netlist(circuit(RS, ZENER, LOAD))), "the zener by its model")
    ok("DD1 n_out 0 BZX55C5V1" in "\n".join(netlist(with_fault(circuit(RS, ZENER, LOAD), "D1:reversed"))),
       "a zener the wrong way round")
    lines = "\n".join(netlist(circuit(MOS, LOAD)))
    ok("MQ1 n_d n_g 0 0 IRLZ44N" in lines, "a MOSFET with its bulk on the source")
    ok("RQ1_short n_d 0 0.05" in "\n".join(netlist(with_fault(circuit(MOS, LOAD), "Q1:ds_short"))), "a dead MOSFET: drain to source")
    ok("MQ1" not in "\n".join(netlist(with_fault(circuit(MOS, LOAD), "Q1:removed"))), "a MOSFET missing")
    ok("RSW1 n_vcc n_swo 0.05" in "\n".join(netlist(circuit(SW))), "a closed switch is a few hundredths of an ohm")
    ok("RSW1 n_vcc n_swo 1e+12" in "\n".join(netlist(circuit(dict(SW, state="open")))), "an open one is nothing")
    ok("RSW1 n_vcc n_swo 47" in "\n".join(netlist(with_fault(circuit(SW), "SW1:dirty"))), "dirty contacts: 47 ohm")
    ok("RSW1 n_vcc n_swo 1e+12" in "\n".join(netlist(with_fault(circuit(SW), "SW1:stuck_open"))), "a dead switch never closes")
    ok("RR2 n_vcc n_out 10000" in "\n".join(netlist(circuit(LDR, NTC))), "an LDR lit is its lit value")
    ok("RR2 n_vcc n_out 1e+06" in "\n".join(netlist(with_fault(circuit(LDR, NTC), "R2:flipped"))), "covered, its dark value")
    ok("RR3 n_out 0 1000" in "\n".join(netlist(with_fault(circuit(LDR, NTC), "R3:flipped"))), "an NTC warmed up")
    ok("XU1 n_vcc n_out 0 REG vreg=5 drop=2" in "\n".join(netlist(circuit(REG, LOAD))), "a 7805 as the REG subcircuit")
    ok("REG_DEAD" in "\n".join(netlist(with_fault(circuit(REG, LOAD), "U1:dead"))), "a dead regulator")
    ok("RU1_short n_vcc n_out 0.05" in "\n".join(netlist(with_fault(circuit(REG, LOAD), "U1:shorted"))),
       "a regulator with its input on its output")
    ids = [f for f, _ in faults(circuit(POT, ZENER, MOS, SW, LDR, NTC, REG))]
    ok({"P1:turned_down", "P1:wiper_open", "D1:reversed", "Q1:ds_short", "SW1:stuck_closed", "R2:flipped", "R3:flipped",
        "U1:dead", "U1:shorted"} <= set(ids), f"every typical fault is listed: {len(ids)} of them")
    said = dict(faults(circuit(LDR)))["R2:flipped"]
    ok("dark" in said, f"a flipped sensor says its other condition: {said}")


def reviewed():
    print("The review before and after")
    errors, _ = review.before(circuit(dict(ZENER, pins={"A": "GND", "K": "VCC"}), LOAD))
    ok(any("series resistor" in e for e in errors), f"a zener straight across the supply: {errors[:1]}")
    errors, warnings = review.before(circuit(RS, ZENER, LOAD))
    ok(not errors, "behind a resistor, fine")
    errors, _ = review.before(circuit(MOS, LOAD))
    ok(any("no DC path" in e and "G" in e for e in errors), f"a floating gate has no DC path: {errors[:1]}")
    _, warnings = review.before(circuit(REG, LOAD, supply_v=6.0))
    ok(any("will not regulate" in w for w in warnings), f"a 7805 from 6 V: {warnings[:1]}")
    _, warnings = review.before(circuit(REG, LOAD, supply_v=24.0))
    ok(any("heat" in w for w in warnings), f"a 7805 from 24 V: {warnings[:1]}")
    errors, _ = review.before(circuit({"ref": "SW1", "kind": "SW", "state": "closed", "pins": ["VCC", "GND"]}))
    ok(any("dead short" in e for e in errors), "a switch closed across the supply")
    sim = {"nets": {"G": {"mean_v": 2.5, "max_v": 5.0, "min_v": 0.0}, "GND": {"mean_v": 0.0}, "D": {"mean_v": 1.0},
                    "VCC": {"mean_v": 12.0}},
           "currents": {"Q1": {"peak_a": 0.1, "mean_a": 0.05}}}
    warnings = review.after(circuit(dict(MOS, model="IRF540"), LOAD), sim)
    ok(any("gate" in w and "10" not in w[:3] for w in warnings), f"an IRF540 from 5 V: {warnings[:1]}")
    ok(not review.after(circuit(MOS, LOAD), sim), "an IRLZ44N from 5 V: nothing to say")
    sim = {"nets": {"OUT": {"mean_v": 5.1}}, "currents": {"D1": {"peak_a": 0.2, "mean_a": 0.2}}}
    warnings = review.after(circuit(RS, ZENER, LOAD), sim)
    ok(any("past its 0.5 W" in w for w in warnings), f"a zener at 200 mA: {warnings[:1]}")
    sim = {"nets": {"OUT": {"mean_v": 3.9}}, "currents": {}}
    warnings = review.after(circuit(REG, LOAD), sim)
    ok(any("instead of 5" in w for w in warnings), f"a regulator giving 3.9 V: {warnings[:1]}")


def designs():
    print("Every new design, laid out on the breadboard and rebuilt from its holes")
    for name, params in [("astable_555_adjustable", {"f_min_hz": 10e3, "f_max_hz": 15e3}),
                         ("mosfet_switch", {}), ("zener_regulator", {}), ("linear_regulator", {}),
                         ("pot_divider", {}), ("light_sensor", {}), ("pushbutton_led", {})]:
        circuit_, spec, notes = TEMPLATES[name](**params)
        problems = check(circuit_)
        errors, _ = review.before(circuit_)
        try:
            plan = breadboard.plan(circuit_)
            laid = f"{len(plan['steps'])} steps, {len(plan['wires'])} wires"
        except ValueError as e:
            laid = f"NOT laid out: {e}"
        ok(not problems and not errors and "NOT" not in laid, f"{name}: {circuit_['name']}; {laid}")
    c, _, notes = TEMPLATES["astable_555_adjustable"](f_min_hz=10e3, f_max_hz=15e3)
    pot = next(p for p in c["parts"] if p["kind"] == "POT")
    ok(pot["pins"]["W"] == pot["pins"]["B"] and "all in" in notes[0], f"the 555's pot as a rheostat: {notes[0][:90]}")
    for params in ({"f_min_hz": 1000, "f_max_hz": 1010}, {"f_min_hz": 1, "f_max_hz": 1e6}):
        try:
            TEMPLATES["astable_555_adjustable"](**params)
            ok(False, f"{params}: should refuse")
        except ValueError as e:
            ok(True, f"{params}: refused ({str(e)[:50]})")
    c, _, _ = TEMPLATES["zener_regulator"](vz=5.0, vin=9.0, load_ma=100.0)
    ok(next(p for p in c["parts"] if p["kind"] == "ZENER")["model"] == "1N4733A", "100 mA from 9 V picks the 1 W zener")
    c, _, _ = TEMPLATES["zener_regulator"](vz=5.0, vin=9.0, load_ma=20.0)
    ok(next(p for p in c["parts"] if p["kind"] == "ZENER")["model"] == "BZX55C5V1", "20 mA from 9 V, the 500 mW one does")


if __name__ == "__main__":
    accepted()
    spice()
    reviewed()
    designs()
    print(f"\n{'all passed' if not FAILED else f'{len(FAILED)} FAILED'}")
    sys.exit(1 if FAILED else 0)
