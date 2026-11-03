"""Chips as data: python tests/test_catalog.py (needs ngspice)

Every chip in the catalog must pass its own test in the simulator and lay out on a breadboard, with no code written for
it. A chip Lunes learns by itself must enter the catalog only if it passes, and a subcircuit that tries anything but
electricity must never reach ngspice: what the model writes, and what comes from a web page, is not trusted.
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lunes import breadboard, catalog, review  # noqa: E402
from lunes.circuit import check, faults, netlist, with_fault  # noqa: E402
from lunes.simulator import expected  # noqa: E402

FAILED = []
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def ok(condition, what):
    print(("  ok   " if condition else "  FAIL ") + what)
    if not condition:
        FAILED.append(what)


def inverter(name="TESTINV", goals=None, spice=None):
    """A made-up chip, as the model would write it: an inverter in a DIP8."""
    sub = catalog.spice_name(name)
    return {"name": name, "package": "DIP8",
            "pins": {"1": "A", "2": "Y", "3": "NC1", "4": "GND", "5": "NC2", "6": "NC3", "7": "NC4", "8": "VCC"},
            "power": ["VCC", "GND"], "inputs": ["A"], "outputs": ["Y"], "dc_links": [["VCC", "GND"], ["Y", "GND"]],
            "supply_v": [2, 6], "notes": "made up", "source": "test",
            "spice": spice or (f".subckt {sub} a y nc1 gnd nc2 nc3 nc4 vcc\nRs vcc gnd 10k\n"
                               f"Bq d gnd V = v(vcc,gnd)*(1-lunes_step(v(a,gnd)-v(vcc,gnd)/2,30))\nRo d y 50\n.ends {sub}"),
            "test": {"circuit": {"name": "inverter", "supply_v": 5, "parts": [
                {"ref": "U1", "kind": "IC", "model": name, "pins": {"VCC": "VCC", "GND": "GND", "A": "VCC", "Y": "OUT"}},
                {"ref": "R1", "kind": "R", "value": "10k", "pins": ["OUT", "GND"]}]},
                "goals": goals or [{"net": "OUT", "min_v_below": 0.2}]}}


def seeded():
    print("Every chip of the catalog, with no code of its own")
    ok(len(catalog.names()) >= 7, f"chips: {catalog.names()}")
    for name in catalog.names():
        entry = catalog.get(name)
        ok(not catalog.problems(entry), f"{name}: a sound entry")
        result = catalog.prove(entry)
        ok(result["ok"], f"{name}: passes its test " + str([g["measured"] for g in result.get("goals", [])]))
        circuit = entry["test"]["circuit"]
        try:
            plan = breadboard.plan(circuit)
            laid = f"{len(plan['steps'])} steps"
        except ValueError as e:
            plan, laid = None, f"NOT laid out: {e}"
        ok(plan is not None, f"{name} ({entry['package']}): on the breadboard and rebuilt from its holes, {laid}")
    ok(catalog.get("lm393n") is catalog.get("LM393"), "a chip is found by another name it goes by")


def faulty():
    print("Its faults, from what the catalog says")
    circuit = catalog.get("LM393")["test"]["circuit"]
    ids = [f for f, _ in faults(circuit)]
    ok({"U1:removed", "U1:dead", "U1:pin1_open", "U1:pin8_open"} <= set(ids), f"typical faults of a chip with no code: {len(ids)}")
    ok("U1:pin5_open" in ids, "a leg not in its row, for each pin that is wired")
    healthy, broken = expected(circuit), expected(with_fault(circuit, "U1:pin8_open"))
    ok(healthy["nets"]["B"]["mean_v"] < 0.5 and broken["nets"]["B"]["mean_v"] > 4.0,
       f"pin 8 not in its row: no supply, output B floats up ({healthy['nets']['B']['mean_v']} to {broken['nets']['B']['mean_v']} V)")
    lines = "\n".join(netlist(with_fault(circuit, "U1:pin3_open")))
    ok("nc_u1_3" in lines, "an open leg is a node nothing else touches")
    ok("XU1" not in "\n".join(netlist(with_fault(circuit, "U1:removed"))), "a chip removed")


def refusals():
    print("What is refused before anything runs")
    bad = [("shell", ".subckt {s} a y nc1 gnd nc2 nc3 nc4 vcc\n.control\nshell del C:\\\\x\n.endc\n.ends {s}"),
           ("include", ".subckt {s} a y nc1 gnd nc2 nc3 nc4 vcc\n.include C:\\\\secret.txt\n.ends {s}"),
           ("a command as a part", ".subckt {s} a y nc1 gnd nc2 nc3 nc4 vcc\nshell del x\n.ends {s}"),
           ("hidden on a continuation", ".subckt {s} a y nc1 gnd nc2 nc3 nc4 vcc\nRa a y 1k\n+ .control\n.ends {s}"),
           ("the wrong node count", ".subckt {s} a y gnd vcc\nRa a y 1k\n.ends {s}"),
           ("another chip's name", ".subckt CHIP_LM393 a y nc1 gnd nc2 nc3 nc4 vcc\nRa a y 1k\n.ends CHIP_LM393")]
    for why, text in bad:
        entry = inverter(spice=text.replace("{s}", catalog.spice_name("TESTINV")))
        problems = catalog.problems(entry)
        ok(bool(problems), f"{why}: {problems[:1]}")
    entry = inverter()
    entry["pins"]["3"] = "A"
    ok(bool(catalog.problems(entry)), "two pins with one name")
    entry = inverter()
    entry["package"] = "DIP14"
    ok(bool(catalog.problems(entry)), "a DIP14 with 8 pins")
    entry = inverter()
    del entry["test"]
    ok(bool(catalog.problems(entry)), "no test, no entry")


def learnt():
    print("A chip learnt by Lunes enters only if it passes")
    name = "TESTINV"
    path = catalog.FOLDER / f"{name}.json"
    try:
        result = catalog.add(inverter(goals=[{"net": "OUT", "mean_v": 5.0, "tolerance": 0.05}]))
        ok(not result["ok"] and catalog.get(name) is None and not path.exists(), f"its test fails: left out ({result.get('error')})")
        result = catalog.add(inverter())
        ok(result["ok"] and catalog.get(name) is not None and path.exists(), f"its test passes: in the catalog, {result.get('goals')}")
        ok(catalog.get(name)["origin"] == "learnt", "marked as learnt, not as checked by a person")
        circuit = copy.deepcopy(inverter()["test"]["circuit"])
        ok(not check(circuit) and breadboard.plan(circuit), "and every part of Lunes works with it at once")
        before = catalog.get(name)
        worse = inverter(goals=[{"net": "OUT", "mean_v": 5.0, "tolerance": 0.05}])
        ok(not catalog.add(worse)["ok"] and catalog.get(name) is before, "a worse version does not replace it")
    finally:
        path.unlink(missing_ok=True)
        catalog._load().pop(name, None)
        catalog.write_lib()
    ok(catalog.get(name) is None, "cleaned up")


def reviewed():
    print("The review, from the catalog")
    circuit = copy.deepcopy(catalog.get("LM393")["test"]["circuit"])
    chip = circuit["parts"][0]
    del chip["pins"]["IN2+"], chip["pins"]["IN2-"], chip["pins"]["OUT2"]
    circuit["parts"] = [p for p in circuit["parts"] if p["ref"] != "R6"]
    _, warnings = review.before(circuit)
    ok(sum("floating" in w for w in warnings) == 2, f"unused inputs left floating: {warnings[:1]}")
    bad = copy.deepcopy(catalog.get("74HC00")["test"]["circuit"])
    bad["supply_v"] = 12
    warnings = review.after(bad, expected(bad))
    ok(any("works from" in w for w in warnings), f"a 74HC00 at 12 V: {warnings[:1]}")
    circuit = {"name": "x", "supply_v": 5, "parts": [{"ref": "U1", "kind": "IC", "model": "NOPE", "pins": {"A": "B"}}]}
    ok(any("chip_learn" in p for p in check(circuit)), f"a chip not in the catalog says how to learn it: {check(circuit)[0][:70]}")
    circuit = {"name": "x", "supply_v": 5, "parts": [{"ref": "U1", "kind": "IC", "model": "LM393", "pins": {"OUT1": "A", "XX": "B"}}]}
    ok(len(check(circuit)) >= 2, "a pin it does not have, and its power not connected")


if __name__ == "__main__":
    seeded()
    faulty()
    refusals()
    learnt()
    reviewed()
    print(f"\n{'all passed' if not FAILED else f'{len(FAILED)} FAILED'}")
    sys.exit(1 if FAILED else 0)
