"""The board's own code tried on the simulated circuit, with no board: python tests/test_dryrun.py

- A latching threshold alarm: a pot into A0, an LED on D5. The code turns the LED on past half scale, keeps it on
  when the pot comes back (it latches) and clears it when "reset" arrives on the serial port. The dry run, case
  after case from one reset, must show all of that, and what the code printed.
- A pin the circuit drives that the code never sets is called out (it floats on the board).
- An endless loop that never reads anything stops with a clear reason; code that reaches files is refused.
- Through Hume's tools: mcu_code keeps a sketch that compiles, mcu_program refuses it before a passing dry run and
  takes it after, mcu_read points to the serial monitor, and the simulated board's monitor shows what it prints.

Needs arduino-cli with the arduino:avr core, and g++ (on Windows, in WSL Ubuntu); skipped otherwise.
"""
import shutil
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lunes import dryrun, mcu, report  # noqa: E402
from lunes.hume import Hume  # noqa: E402

report.REPORTS = Path(tempfile.mkdtemp(prefix="lunes-reports-"))
FAILED = []

CIRCUIT = {"name": "Latching threshold alarm", "supply_v": 5, "parts": [
    {"ref": "P1", "kind": "POT", "value": "10k", "position": 0.3, "pins": {"A": "GND", "W": "VIN", "B": "VCC"}},
    {"ref": "U1", "kind": "ARDUINO_NANO", "pins": {"5V": "VCC", "GND": "GND", "A0": "VIN", "D5": "ALM", "D6": "AUX"},
     "program": {"A0": "analog_in", "D5": "low", "D6": "high"}},
    {"ref": "R1", "kind": "R", "value": "330", "pins": ["ALM", "LED_A"]},
    {"ref": "D1", "kind": "LED", "color": "red", "pins": {"A": "LED_A", "K": "GND"}},
    {"ref": "R2", "kind": "R", "value": "10k", "pins": ["AUX", "GND"]}]}

LATCH = r"""
bool alarm = false;
void setup() {
  Serial.begin(115200);
  pinMode(5, OUTPUT);
}
void loop() {
  if (!alarm && analogRead(A0) > 512) { alarm = true; Serial.println(F("ALARM")); }
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd == "reset") { alarm = false; Serial.println("cleared"); }
  }
  digitalWrite(5, alarm ? HIGH : LOW);
}
"""


def ok(condition, what):
    print(("  ok   " if condition else "  FAIL ") + what)
    if not condition:
        FAILED.append(what)


def circuit_with(code):
    import copy
    c = copy.deepcopy(CIRCUIT)
    c["parts"][1]["code"] = code
    return c


def latch():
    print("A latching alarm, case after case")
    cases = [{"label": "low", "set": {"P1": {"position": 0.3}}, "expect": {"D1": "off", "D5": "LOW"}},
             {"label": "past half", "set": {"P1": {"position": 0.8}}, "expect": {"D1": "on", "serial_contains": "ALARM"}},
             {"label": "back low", "set": {"P1": {"position": 0.3}}, "expect": {"D1": "on"}},
             {"label": "reset", "set": {"P1": {"position": 0.3}}, "serial": "reset\n",
              "expect": {"D1": "off", "serial_contains": "cleared"}}]
    out = dryrun.dry_run(circuit_with(LATCH), "U1", cases)
    ok(out["ok"] and out["all_expectations_met"], "every expectation met: " +
       str([(r["case"], r.get("failed")) for r in out.get("cases", [])] or out))
    if out["ok"]:
        on = out["cases"][1]
        ok(on["leds"]["D1"]["ma"] > 5 and on["inputs"]["A0"]["analogRead"] > 700,
           f"past half: A0 reads {on['inputs']['A0']['analogRead']}, the LED carries {on['leds']['D1']['ma']} mA")
        ok(any("D6" in r.get("warning", "") for r in out["cases"]), "D6, driven in the circuit but never by the code, "
                                                                       "is called out")
    wrong = dryrun.dry_run(circuit_with(LATCH), "U1", [{"label": "low", "set": {"P1": {"position": 0.3}},
                                                        "expect": {"D1": "on"}}])
    ok(wrong["ok"] and not wrong["all_expectations_met"], "a wrong expectation is reported as missed")


def stuck_and_refused():
    print("Code that never returns, and code that is not a sketch")
    out = dryrun.dry_run(circuit_with("void setup(){}\nvoid loop(){ volatile long x = 0; for(;;) x++; }"), "U1",
                         [{"label": "on"}])
    ok(not out["ok"] and "endless loop" in out.get("error", ""), f"an endless loop: {out.get('error', out)}"[:140])
    out = dryrun.dry_run(circuit_with('void setup(){ system("ls"); }\nvoid loop(){}'), "U1", [{"label": "on"}])
    ok(not out["ok"] and "system" in " ".join(out.get("errors", [])), "a call to system() is refused before compiling")
    out = dryrun.dry_run(circuit_with("void setup(){ pinMode(5, OUTPUT) }\nvoid loop(){}"), "U1", [{"label": "on"}])
    ok(not out["ok"] and out.get("errors"), f"a syntax error comes back: {(out.get('errors') or [''])[0][:90]}")


def through_hume():
    print("Through Hume's tools on a simulated board")
    machine = types.SimpleNamespace(s=types.SimpleNamespace(port="COM8"), config={})
    hume = Hume(machine, None, mcu=mcu.SimulatedBoard(None))
    hume.run_tool("circuit_set", {"circuit": CIRCUIT})
    kept, _ = hume.run_tool("mcu_code", {"code": LATCH})
    ok(kept.get("kept") is True, f"mcu_code keeps a sketch that compiles: {kept.get('board_compile') or kept}")
    bad, _ = hume.run_tool("mcu_code", {"code": "void setup(){ pinMode(5, OUTPUT) }\nvoid loop(){}"})
    ok(not bad["ok"] and hume._mcu_part()[0]["code"] == LATCH.strip(), "a sketch that does not compile is not kept")
    refused, _ = hume.run_tool("mcu_program", {})
    ok(not refused["ok"] and "dry run" in refused["error"], "mcu_program refuses code without a passing dry run")
    run, _ = hume.run_tool("mcu_dry_run", {"cases": [
        {"label": "low", "set": {"P1": {"position": 0.3}}, "expect": {"D1": "off"}},
        {"label": "high", "set": {"P1": {"position": 0.8}}, "expect": {"D1": "on"}}]})
    ok(run["ok"] and run["all_expectations_met"], "the dry run passes")
    flashed, _ = hume.run_tool("mcu_program", {})
    ok(flashed["ok"], f"and then mcu_program takes it: {flashed.get('error', 'ok')}")
    read, _ = hume.run_tool("mcu_read", {})
    ok(not read["ok"] and "serial_monitor" in read["error"], "mcu_read points to the serial monitor")
    hume.mcu.circuit["parts"][0]["position"] = 0.8
    shown, _ = hume.run_tool("serial_monitor", {"seconds": 1})
    ok(shown["ok"] and "ALARM" in shown["lines"], f"the simulated board prints what the code prints: {shown.get('lines')}")


if __name__ == "__main__":
    if not mcu.cli_exe() or not (shutil.which("g++") or shutil.which("wsl")):
        print("skipped: needs arduino-cli and g++ (WSL on Windows)")
        sys.exit(0)
    latch()
    stuck_and_refused()
    through_hume()
    print("\nFAILED: " + "; ".join(FAILED) if FAILED else "\nall ok")
    sys.exit(1 if FAILED else 0)
