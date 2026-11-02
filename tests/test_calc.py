"""Arithmetic with units and formulas solved by the code, no model: python tests/test_calc.py

What an engineer writes on paper must come back right, in the unit that fits; a mix of units that makes no sense
must be an error, not a number; a formula must be solved for its unknown with the known values' units carried
through; and nothing but arithmetic may run, whatever the text says.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lunes.calc import run  # noqa: E402

FAILED = []
if hasattr(sys.stdout, "reconfigure"):  # Ω and µ on a Windows console
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def ok(condition, what):
    print(("  ok   " if condition else "  FAIL ") + what)
    if not condition:
        FAILED.append(what)


def close(got, want, tol=0.002):
    return got.get("ok") and abs(got["value"] - want) <= tol * abs(want)


def arithmetic():
    print("Expressions with units")
    for expression, to, value, units in [("12 V / 20 mA", None, 600.0, "ohm"), ("4.7 kohm * 2 mA", None, 9.4, "volt"),
                                         ("12 V / 20 mA", "kohm", 0.6, "kiloohm"), ("(5 V - 2 V) * 20 mA", "mW", 60.0, "milliwatt"),
                                         ("1/(2*pi*10 kohm*100 nF)", None, 159.15, "hertz"),
                                         ("10 kohm * 100 nF", "ms", 1.0, "millisecond"), ("sqrt(2) * 230 V", None, 325.27, "volt"),
                                         ("sin(30 degree)", None, 0.5, "dimensionless"), ("2.2 kΩ * 1.5 mA", "V", 3.3, "volt"),
                                         ("10kohm * 2mA", None, 20.0, "volt"), ("1e3 ohm * 5 mA", None, 5.0, "volt"),
                                         ("5 V ** 2 / 100 ohm", None, 250.0, "milliwatt"), ("log10(1000)", None, 3.0, "dimensionless"),
                                         ("3 V / 1.5 V", None, 2.0, "dimensionless"), ("47 uF", "nF", 47000.0, "nanofarad")]:
        got = run({"expression": expression, "to": to})
        ok(close(got, value) and got["units"] == units, f"{expression}" + (f" in {to}" if to else "") + f" = {got.get('result', got.get('error'))}")
    got = run({"expression": "12 V / 20 mA"})
    ok(got["result"] == "600 Ω", f"said as a person would: {got['result']}")
    got = run({"expression": "1/(2*pi*10 kohm*100 nF) to kHz"})
    ok(close(got, 0.15915) and got["units"] == "kilohertz", f"'to kHz' written at the end: {got.get('result')}")
    got = run({"expression": "2 sqrt(2) V"})
    ok(close(got, 2.8284), f"a number before a function: {got.get('result')}")
    ok(run({"expression": "1/(2*pi*10 kohm*100 nF)"})["result"] == "159.2 Hz", "with the prefix that fits")


def refusals():
    print("What is refused")
    for expression, why in [("5 V + 2 A", "volts and amperes do not add"), ("12 V / 20 mA", "ohms are not watts"),
                            ("3 furlongs_of_nonsense", "an unknown unit"), ("__import__('os').system('dir')", "python"),
                            ("open('x')", "a function not allowed"), ("(5 V", "a broken expression"),
                            ("log(5 V)", "a log of volts"), ("1 V / 0", "division by zero"), ("[1, 2]", "a list")]:
        got = run({"expression": expression, "to": "W" if "watts" in why else None})
        ok(not got["ok"] and "error" in got, f"{expression}: refused ({why}): {got.get('error', '')[:70]}")
    ok(not run({})["ok"], "nothing given: refused")


def formulas():
    print("Formulas solved for their unknown")
    got = run({"formula": "V = I * R", "solve_for": "R", "values": {"V": "12 V", "I": "20 mA"}})
    ok(close(got, 600.0) and got["units"] == "ohm" and got["solved"] == "R = V/I", f"V = I R for R: {got.get('result')} ({got.get('solved')})")
    got = run({"formula": "f = 1 / (2 * pi * R * C)", "solve_for": "C", "values": {"f": "1 kHz", "R": "10 kohm"}, "to": "nF"})
    ok(close(got, 15.915) and got["units"] == "nanofarad", f"RC cutoff for C: {got.get('result')}")
    got = run({"formula": "P = V**2 / R", "solve_for": "R", "values": {"P": "0.25 W", "V": "5 V"}})
    ok(close(got, 100.0) and got["units"] == "ohm", f"P = V^2/R for R: {got.get('result')}")
    got = run({"formula": "R = (Vs - Vf) / I", "solve_for": "R", "values": {"Vs": "5 V", "Vf": "2 V", "I": "15 mA"}})
    ok(close(got, 200.0), f"an LED's resistor: {got.get('result')}")
    got = run({"formula": "x**2 = 4 * a", "solve_for": "x", "values": {"a": "9"}})
    ok(got["ok"] and len(got.get("results", [])) == 2 and {round(r["value"]) for r in got["results"]} == {-6, 6},
       f"a quadratic gives both solutions: {[r['result'] for r in got.get('results', [])]}")
    got = run({"formula": "V = I * R", "solve_for": "R", "values": {"V": "12 V"}})
    ok(not got["ok"] and "I" in got["error"], f"a missing value is asked for: {got['error'][:60]}")
    got = run({"formula": "V = I + R", "solve_for": "R", "values": {"V": "12 V", "I": "20 mA"}})
    ok(not got["ok"], f"a formula whose units do not add up is refused: {got['error'][:60]}")
    got = run({"formula": "V = I * R", "solve_for": "Z", "values": {"V": "12 V", "I": "1 A"}})
    ok(not got["ok"], "an unknown not in the formula is refused")
    got = run({"formula": "V = __import__('os')", "solve_for": "V", "values": {}})
    ok(not got["ok"], "python in a formula is refused")


if __name__ == "__main__":
    arithmetic()
    refusals()
    formulas()
    print(f"\n{'all passed' if not FAILED else f'{len(FAILED)} FAILED'}")
    sys.exit(1 if FAILED else 0)
