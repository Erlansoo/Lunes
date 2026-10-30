"""A microcontroller in the circuit: one pin program that is simulated, turned into a sketch, flashed and read back.

The circuit gives the board as a part, with what each pin does:

    {"ref": "U2", "kind": "ARDUINO_UNO", "pins": {"D9": "PWM", "A0": "FILT", "GND": "GND", "5V": "VCC"},
     "program": {"D9": {"pwm": 0.3}, "A0": "analog_in"}}

Pin modes: "high", "low", {"pwm": duty 0..1} (on the board's PWM pins), {"square": hz, "duty": 0.5} (toggled by
the sketch, up to 5 kHz), "input", "input_pullup", "analog_in" (on the analogue pins).

The same program does three things, so the bench and the simulation cannot disagree:
- In ngspice each pin is what it is electrically: an output is a 0 / VCC source behind the pin's own resistance
  (about 25 ohm on an ATmega328P), an input a high impedance, a pull-up 35 k to the board's supply.
- sketch() writes the Arduino sketch: pinMode, the outputs, and a JSON line every 250 ms with every input. The
  board reads its inputs as fast as it can between lines (about 9000 times a second for one analogue input on an
  Uno), so an analogue input comes as [mean, lowest, highest] counts and a digital one as the share of reads that
  were HIGH: {"A0": [307.4, 301, 312], "D4": 1.0}. The board is then a slow voltmeter on every net its analogue
  inputs touch. compile() and upload() go through arduino-cli; upload never goes to the CNC's port.
- read() listens to those lines over USB serial, expectations() says what each input should read according to the
  simulation, and compare() puts the two side by side. to_reading() turns an analogue input into a measurement of
  its net for the diagnosis, like the multimeter but with the swing too.

The sampling slows the loop (about 0.1 ms per analogue input on an Uno), so a square output toggled by the sketch
keeps its frequency but its edges move by up to that much.

SimulatedBoard is a board on a made-up bench: it prints what the sketch would on the simulated circuit, with a
fault planted if asked, to try all of this without a board.
"""
import copy
import json
import random
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLI = Path.home() / "AppData" / "Local" / "Programs" / "Arduino IDE" / "resources" / "app" / "lib" / "backend" / \
    "resources" / "arduino-cli.exe"
SKETCHES = ROOT / "runs" / "sketches"

_AVR = {"vcc": 5.0, "adc_bits": 10, "pin_r": 25.0, "pullup": 35e3, "rec_pin_a": 0.02, "max_pin_a": 0.04,
        "max_total_a": 0.2, "vih": 0.6, "vil": 0.3, "power": ("5V", "3V3", "GND", "VIN"),
        "digital": tuple(f"D{i}" for i in range(14)), "analog": tuple(f"A{i}" for i in range(6)),
        "pwm": {"D3": 490, "D5": 980, "D6": 980, "D9": 490, "D10": 490, "D11": 490}, "upload": True}
BOARDS = {
    "ARDUINO_UNO": dict(_AVR, fqbn="arduino:avr:uno", name="Arduino Uno"),
    "ARDUINO_NANO": dict(_AVR, fqbn="arduino:avr:nano", name="Arduino Nano",
                         analog=tuple(f"A{i}" for i in range(8)), analog_only=("A6", "A7")),
    "BLUEPILL": {"name": "STM32F103 Blue Pill", "fqbn": "stm32duino:STM32F1:genericSTM32F103C", "vcc": 3.3,
                 "adc_bits": 12, "pin_r": 40.0, "pullup": 40e3, "rec_pin_a": 0.008, "max_pin_a": 0.025,
                 "max_total_a": 0.15, "vih": 0.7, "vil": 0.3, "power": ("5V", "3V3", "GND"),
                 "digital": tuple(f"PA{i}" for i in range(16)) + tuple(f"PB{i}" for i in range(16)) +
                 ("PC13", "PC14", "PC15"),
                 "analog": tuple(f"PA{i}" for i in range(8)) + ("PB0", "PB1"),
                 "pwm": {p: 1000 for p in ("PA0", "PA1", "PA2", "PA3", "PA6", "PA7", "PA8", "PA9", "PA10", "PB0",
                                          "PB1", "PB6", "PB7", "PB8", "PB9")}, "upload": False},
}
OUTPUTS = ("high", "low", "pwm", "square")
ARDUINO_VIDS = ("2341", "2A03")  # USB vendor ids of genuine Arduino boards; clones use common serial chips
NO_READINGS = "no readings: is it this port, and is Lunes' sketch on the board?"


def mode_of(entry):
    """("pwm", {"pwm": 0.3}) for a program entry given as a word or a dict."""
    if isinstance(entry, str):
        return entry, {}
    if isinstance(entry, dict):
        for k in ("pwm", "square", "blink"):
            if k in entry:
                return ("square" if k == "blink" else k), entry
    return None, entry


def check(part):
    """Problems with a board part's pins and program, in plain words."""
    board = BOARDS[part["kind"]]
    problems, program = [], part.get("program") or {}
    pins = part.get("pins") or {}
    known = set(board["digital"]) | set(board["analog"]) | set(board["power"])
    for pin in pins:
        if pin not in known:
            problems.append(f"{part['ref']}: a {board['name']} has no pin {pin}")
    for pin, entry in program.items():
        mode, args = mode_of(entry)
        if pin not in pins:
            problems.append(f"{part['ref']}: {pin} is programmed but not connected to any net")
        if mode not in ("high", "low", "pwm", "square", "input", "input_pullup", "analog_in"):
            problems.append(f"{part['ref']} {pin}: unknown mode {entry!r}")
        if mode == "pwm" and pin not in board["pwm"]:
            problems.append(f"{part['ref']} {pin}: no PWM on this pin; PWM pins are {sorted(board['pwm'])}")
        if mode == "pwm" and not 0 <= float(args["pwm"]) <= 1:
            problems.append(f"{part['ref']} {pin}: pwm is a duty from 0 to 1")
        if mode == "square" and not 0 < float(args.get("square", args.get("blink", 0))) <= 5000:
            problems.append(f"{part['ref']} {pin}: square needs a frequency up to 5000 Hz")
        if mode == "analog_in" and pin not in board["analog"]:
            problems.append(f"{part['ref']} {pin}: analog_in only on {list(board['analog'])}")
        if mode in OUTPUTS and pin in board.get("analog_only", ()):
            problems.append(f"{part['ref']} {pin}: {pin} is an analogue input only")
    for pin in pins:
        if pin not in board["power"] and pin not in program:
            problems.append(f"{part['ref']} {pin}: connected but not programmed; say what the pin does")
    return problems


def spice(part, net):
    """SPICE lines for a board: each programmed pin as it is electrically. net(name) gives the SPICE node."""
    board, ref, lines = BOARDS[part["kind"]], part["ref"], []
    pins, program, fault = part["pins"], part.get("program") or {}, part.get("_fault") or ""
    vcc_node = net(pins["5V"]) if board["vcc"] == 5.0 and "5V" in pins else net(pins.get("3V3", "VCC"))
    for supply, volts in (("5V", 5.0), ("3V3", 3.3)):
        if supply in pins and net(pins[supply]) not in ("0", net("VCC")):
            lines.append(f"VMCU_{ref}_{supply} mcu_{ref}_{supply} 0 DC {volts:g}")
            lines.append(f"RMCU_{ref}_{supply}_out mcu_{ref}_{supply} {net(pins[supply])} 0.5")
    running = fault != "not_running"
    for pin, entry in program.items():
        mode, args = mode_of(entry)
        node, name = net(pins[pin]), f"MCU_{ref}_{pin}"
        if mode in OUTPUTS and running and fault != f"{pin}_floating":
            if fault == f"{pin}_stuck_low" or mode == "low":
                source = "DC 0"
            elif mode == "high":
                source = f"DC {board['vcc']:g}"
            else:
                hz = board["pwm"][pin] if mode == "pwm" else float(args.get("square", args.get("blink")))
                duty = float(args["pwm"]) if mode == "pwm" else float(args.get("duty", 0.5))
                period, edge = 1 / hz, 50e-9
                if duty <= 0:
                    source = "DC 0"
                elif duty >= 1:
                    source = f"DC {board['vcc']:g}"
                else:
                    source = f"PULSE(0 {board['vcc']:g} 0 {edge:g} {edge:g} {duty * period - edge:g} {period:g})"
            lines.append(f"V{name} mcu_{ref}_{pin} 0 {source}")
            lines.append(f"R{name} mcu_{ref}_{pin} {node} {board['pin_r']:g}")
        else:  # an input, or an output the sketch never drives: a high impedance with a little capacitance, and the
            # pin's protection diodes, so a net pushed past the supply (12 V through a resistor) sits at about
            # VCC + 0.5 V, as on the real chip, not at 12 V
            lines.append(f"R{name}_in {node} 0 100Meg")
            lines.append(f"C{name}_in {node} 0 10p")
            lines.append(f"D{name}_up {node} {vcc_node} DMCU_{ref}")
            lines.append(f"D{name}_down 0 {node} DMCU_{ref}")
            if mode == "input_pullup" and running:
                lines.append(f"R{name}_pullup {node} {vcc_node} {board['pullup']:g}")
    if any(line.startswith("D") for line in lines):
        lines.append(f".model DMCU_{ref} D(IS=1e-14 N=1.2 RS=100)")
    return lines


def faults(part):
    """[(fault id, words)] for a board: each driven pin dead low or never driven, and the sketch not running."""
    out = [(f"{part['ref']}:not_running", f"{part['ref']} not running its sketch (not uploaded, wrong board, or "
                                          f"held in reset): every pin floats")]
    for pin, entry in (part.get("program") or {}).items():
        mode, _ = mode_of(entry)
        if mode in OUTPUTS and mode != "low":
            out.append((f"{part['ref']}:{pin}_stuck_low", f"{part['ref']} pin {pin} stuck low (a dead pin, or "
                                                          f"shorted to GND on the board)"))
            out.append((f"{part['ref']}:{pin}_floating", f"{part['ref']} pin {pin} never driven (the sketch does "
                                                         f"not set it as an output, or it is wired to another pin)"))
    return out


def _counts(v, board):
    return round(min(max(v, 0.0), board["vcc"]) / board["vcc"] * (2 ** board["adc_bits"] - 1))


def _level(v, board):
    """1, 0, or None between the logic thresholds."""
    return 1 if v >= board["vih"] * board["vcc"] else 0 if v <= board["vil"] * board["vcc"] else None


def expectations(part, nets):
    """What each input should read on the board, from the simulated nets: analogRead counts (mean, lowest and
    highest), and for a digital input its level and the share of the time it reads HIGH."""
    board, out = BOARDS[part["kind"]], {}
    for pin, entry in (part.get("program") or {}).items():
        mode, _ = mode_of(entry)
        m = nets.get(part["pins"][pin])
        if m is None or mode not in ("analog_in", "input", "input_pullup"):
            continue
        v = m["mean_v"]
        low, high = m.get("min_v", v), m.get("max_v", v)
        if mode == "analog_in":
            out[pin] = {"analogRead": _counts(v, board), "lowest": _counts(low, board),
                        "highest": _counts(high, board), "volts": round(v, 3)}
            if m.get("frequency_hz") and m["vpp_v"] > 0.02 * board["vcc"]:
                out[pin]["note"] = (f"the net moves ({m['frequency_hz']:.4g} Hz, {m['vpp_v']:.2f} Vpp): the mean is "
                                    f"over many periods, lowest and highest are its swing")
        else:
            level = _level(v, board)
            out[pin] = {"digitalRead": level, "volts": round(v, 3)}
            if m.get("frequency_hz") and _level(low, board) == 0 and _level(high, board) == 1:
                out[pin]["high_share"] = round(float(m.get("duty") or 0.5), 2)  # it toggles: HIGH for its duty
            elif _level(low, board) == _level(high, board) is not None:
                out[pin]["high_share"] = float(_level(low, board))
            else:
                out[pin]["note"] = "between the logic thresholds: it may read either way"
    return out


def compare(part, got, expect):
    """{pin: what the board read beside what the simulation says, and whether they agree}. got: read()'s result.
    An analogue input agrees within 5 % plus 1 % of the scale: the parts' tolerance, and the board's 5 V from USB
    is often 4.8 V, which every reading of an outside voltage scales with."""
    board = BOARDS[part["kind"]]
    top = 2 ** board["adc_bits"] - 1
    out = {}
    for pin, e in expect.items():
        g = got["pins"].get(pin)
        if g is None:
            out[pin] = {"expected": e, "read": None, "ok": False,
                        "note": "the board sent nothing for this pin: its sketch is not this circuit's"}
            continue
        if "analogRead" in e:
            tolerance = 0.05 * e["analogRead"] + 0.01 * top
            row = {"net": part["pins"][pin], "expected_v": e["volts"], "read_v": g.get("mean_v"),
                   "expected_counts": e["analogRead"], "read_counts": g.get("analogRead"),
                   "read_swing_v": g.get("vpp_v"), "ok": g.get("analogRead") is not None
                   and abs(g["analogRead"] - e["analogRead"]) <= tolerance}
            swing = e["highest"] - e["lowest"]
            if g.get("highest") is not None and abs((g["highest"] - g["lowest"]) - swing) > 0.1 * swing + 0.02 * top:
                row["ok"] = False
                row["note"] = (f"it swings {g['highest'] - g['lowest']} counts where the simulation swings {swing}: "
                               f"{'the net moves and should not' if swing < 0.02 * top else 'the net is not moving as it should'}")
        else:
            row = {"net": part["pins"][pin], "expected_high_share": e.get("high_share"),
                   "read_high_share": g.get("high_share"),
                   "ok": e.get("high_share") is None or abs(g.get("high_share", -1) - e["high_share"]) <= 0.1}
        if e.get("note"):
            row.setdefault("note", e["note"])
        out[pin] = row
    return out


def to_reading(pin_read, part):
    """An analogue input's reading as a measurement of its net, for lunes.diagnosis: its mean and its swing. No
    frequency: the board does not measure one, and the engine then does not judge by it. The ADC stops at 0 and at
    its supply: a reading that touched the top says the net is at least that high (its swing at least that big),
    one that touched 0 that it is at most that high, as on a scope that clips."""
    top = 2 ** BOARDS[part["kind"]]["adc_bits"] - 1
    reading = {"mean_v": pin_read["mean_v"], "vpp_v": pin_read["vpp_v"]}
    if pin_read["highest"] >= top - 1:
        reading["clipped_high"] = True
    if pin_read["lowest"] <= 0:
        reading["clipped_low"] = True
    return reading


def review(part, sim):
    """Warnings about the board's pins from the simulation: currents past the pin's ratings, inputs beyond its
    supply."""
    board, warnings = BOARDS[part["kind"]], []
    total = 0.0
    for pin, entry in (part.get("program") or {}).items():
        mode, _ = mode_of(entry)
        c = sim.get("currents", {}).get(f"{part['ref']}.{pin}")
        if c:
            total += c["mean_a"]
            if c["peak_a"] > board["max_pin_a"]:
                warnings.append(f"{part['ref']} pin {pin} would carry {c['peak_a'] * 1e3:.0f} mA, past the "
                                f"{board['max_pin_a'] * 1e3:.0f} mA a {board['name']} pin survives: drive the load "
                                f"through a transistor")
            elif c["peak_a"] > board["rec_pin_a"]:
                warnings.append(f"{part['ref']} pin {pin} carries {c['peak_a'] * 1e3:.0f} mA, above the "
                                f"{board['rec_pin_a'] * 1e3:.0f} mA recommended: it works, but hot and with the "
                                f"output sagging")
        m = sim["nets"].get(part["pins"][pin], {})
        if mode in ("input", "input_pullup", "analog_in") and m:
            if m.get("max_v", m["mean_v"]) > board["vcc"] + 0.5 or m.get("min_v", m["mean_v"]) < -0.5:
                warnings.append(f"{part['ref']} pin {pin} sees {m.get('min_v', 0):.2f} .. {m.get('max_v', 0):.2f} V, "
                                f"outside its {board['vcc']:g} V supply: it can damage the pin (a divider or a "
                                f"series resistor and clamp)")
            if mode == "input" and len(m.get("touches", [])) <= 1:
                warnings.append(f"{part['ref']} pin {pin} is an input with nothing on it: it reads noise (use "
                                f"input_pullup or a resistor)")
    if total > board["max_total_a"]:
        warnings.append(f"{part['ref']} pins together carry {total * 1e3:.0f} mA, past the {board['name']}'s "
                        f"{board['max_total_a'] * 1e3:.0f} mA total")
    return warnings


# ---- the sketch ---------------------------------------------------------------------------------------------------
def _var(pin):
    return re.sub(r"\W", "_", pin)


def _code_pin(pin, board):
    return pin[1:] if pin.startswith("D") and board["vcc"] == 5.0 else pin  # D9 -> 9, A0 -> A0, PA0 -> PA0


def sketch(part, circuit_name="circuit"):
    """The Arduino sketch for the part's pin program, as text."""
    board = BOARDS[part["kind"]]
    program = part.get("program") or {}
    head, setup, loop, analog, digital = [], [], [], [], []
    for pin, entry in program.items():
        mode, args = mode_of(entry)
        p = _code_pin(pin, board)
        if mode in ("high", "low"):
            setup += [f"  pinMode({p}, OUTPUT);", f"  digitalWrite({p}, {'HIGH' if mode == 'high' else 'LOW'});"]
        elif mode == "pwm":
            level = round(float(args["pwm"]) * 255)
            setup += [f"  pinMode({p}, OUTPUT);", f"  analogWrite({p}, {level});  // duty {float(args['pwm']):.0%}"]
        elif mode == "square":
            hz, duty = float(args.get("square", args.get("blink"))), float(args.get("duty", 0.5))
            hi, lo = round(1e6 / hz * duty), round(1e6 / hz * (1 - duty))
            v = re.sub(r"\W", "_", pin)
            head += [f"const unsigned long HIGH_US_{v} = {hi}UL, LOW_US_{v} = {lo}UL;  // {hz:g} Hz, duty {duty:.0%}",
                     f"unsigned long edge_{v} = 0;", f"bool on_{v} = false;"]
            setup += [f"  pinMode({p}, OUTPUT);"]
            loop += [f"  if (now - edge_{v} >= (on_{v} ? HIGH_US_{v} : LOW_US_{v})) {{",
                     f"    edge_{v} += on_{v} ? HIGH_US_{v} : LOW_US_{v};", f"    on_{v} = !on_{v};",
                     f"    digitalWrite({p}, on_{v} ? HIGH : LOW);", "  }"]
        elif mode in ("input", "input_pullup"):
            setup.append(f"  pinMode({p}, {'INPUT_PULLUP' if mode == 'input_pullup' else 'INPUT'});")
            digital.append(pin)
        elif mode == "analog_in":
            analog.append(pin)
    for pin in analog:
        v = _var(pin)
        head += [f"unsigned long sum_{v} = 0;", f"int lowest_{v} = 32767, highest_{v} = -1;"]
    for pin in digital:
        head.append(f"unsigned long high_{_var(pin)} = 0;")
    if analog or digital:
        head.append("unsigned long reads = 0, last_report = 0;")
    lines = [f"// Written by Lunes from the pin program of \"{circuit_name}\" ({board['name']}).",
             "// The same program is what Lunes simulated, so the bench and the simulation should agree.",
             "// Between reports it reads every input as fast as it can. Every 250 ms it prints one JSON line for",
             "// Lunes: an analogue input as [mean, lowest, highest] counts, a digital one as the share of HIGH reads.",
             ""]
    lines += head + ["", "void setup() {", "  Serial.begin(115200);"] + setup + ["}", "", "void loop() {",
                                                                              "  unsigned long now = micros();"]
    lines += loop
    if analog or digital:
        for pin in analog:
            v, p = _var(pin), _code_pin(pin, board)
            lines += [f"  int a_{v} = analogRead({p});", f"  sum_{v} += a_{v};",
                      f"  if (a_{v} < lowest_{v}) lowest_{v} = a_{v};",
                      f"  if (a_{v} > highest_{v}) highest_{v} = a_{v};"]
        for pin in digital:
            lines.append(f"  high_{_var(pin)} += digitalRead({_code_pin(pin, board)});")
        lines += ["  reads++;", "  if (millis() - last_report >= 250) {", "    last_report = millis();"]
        for k, pin in enumerate(analog + digital):
            v = _var(pin)
            start = "{" if k == 0 else ","
            if pin in analog:
                lines += [f'    Serial.print("{start}\\"{pin}\\":[");', f"    Serial.print((float)sum_{v} / reads, 1);",
                          '    Serial.print(",");', f"    Serial.print(lowest_{v});", '    Serial.print(",");',
                          f"    Serial.print(highest_{v});", '    Serial.print("]");',
                          f"    sum_{v} = 0;", f"    lowest_{v} = 32767;", f"    highest_{v} = -1;"]
            else:
                lines += [f'    Serial.print("{start}\\"{pin}\\":");', f"    Serial.print((float)high_{v} / reads, 2);",
                          f"    high_{v} = 0;"]
        lines += ['    Serial.println("}");', "    reads = 0;", "  }"]
    lines.append("}")
    return "\n".join(lines) + "\n"


def _cli(*args, timeout=300):
    if not CLI.exists():
        raise RuntimeError(f"arduino-cli is not at {CLI} (it comes with the Arduino IDE)")
    r = subprocess.run([str(CLI), *args], capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr).strip()


def compile_sketch(part, circuit_name="circuit"):
    """Writes the sketch to runs/sketches/<name>/ and compiles it: {"ok", "path", "size" or "errors"}."""
    board = BOARDS[part["kind"]]
    name = re.sub(r"\W+", "_", f"lunes_{circuit_name}_{part['ref']}").strip("_")[:60]
    folder = SKETCHES / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.ino").write_text(sketch(part, circuit_name), encoding="utf-8")
    code, out = _cli("compile", "--fqbn", board["fqbn"], "--output-dir", str(folder / "build"), str(folder))
    result = {"ok": code == 0, "path": str(folder / f"{name}.ino"), "fqbn": board["fqbn"]}
    if code == 0:
        m = re.search(r"Sketch uses (\d+) bytes \((\d+)%\)", out)
        result["size"] = m.group(0) if m else "compiled"
    else:
        result["errors"] = [l for l in out.splitlines() if "error" in l.lower()][:8] or out.splitlines()[-8:]
    return result


def ports():
    """[{port, description, usb id}] of the serial ports now, as pyserial sees them."""
    from serial.tools import list_ports
    return [{"port": p.device, "description": p.description,
             "usb_id": f"{p.vid:04X}:{p.pid:04X}" if p.vid is not None else None} for p in list_ports.comports()]


def upload(part, port, forbidden=(), circuit_name="circuit"):
    """Compiles and flashes the sketch to port. Never to a forbidden port (the CNC's)."""
    board = BOARDS[part["kind"]]
    if not board["upload"]:
        return {"ok": False, "error": f"Lunes does not flash a {board['name']} yet (it needs an ST-Link); the "
                                      f"sketch is compiled, flash it from the Arduino IDE"}
    if str(port).upper() in {str(f).upper() for f in forbidden}:
        return {"ok": False, "error": f"{port} is the CNC's motion board: Lunes never flashes it"}
    built = compile_sketch(part, circuit_name)
    if not built["ok"]:
        return built
    folder = Path(built["path"]).parent
    code, out = _cli("upload", "-p", str(port), "--fqbn", board["fqbn"], "--input-dir", str(folder / "build"),
                     timeout=120)
    if code != 0 and board["fqbn"] == "arduino:avr:nano":  # older Nanos have the old bootloader
        code, out = _cli("upload", "-p", str(port), "--fqbn", "arduino:avr:nano:cpu=atmega328old", "--input-dir",
                         str(folder / "build"), timeout=120)
    return {**built, "ok": code == 0, "port": port,
            **({} if code == 0 else {"error": [l for l in out.splitlines() if l.strip()][-6:]})}


def read(port, part, seconds=3.0, forbidden=()):
    """What the board's inputs read over a few seconds, from the sketch's JSON lines (see summarize). Opening the
    port resets the board, so it waits for it to start first."""
    import serial
    if str(port).upper() in {str(f).upper() for f in forbidden}:
        return {"ok": False, "error": f"{port} is the CNC's motion board, not the Arduino"}
    rows = []
    try:
        with serial.Serial(port, 115200, timeout=0.3) as s:
            time.sleep(2.0)
            s.reset_input_buffer()
            t0 = time.time()
            while time.time() - t0 < seconds:
                line = s.readline().decode(errors="ignore").strip()
                if line.startswith("{"):
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        pass
    except serial.SerialException as e:
        return {"ok": False, "error": f"cannot open {port}: {e}"}
    return summarize(rows, part) if rows else {"ok": False, "error": NO_READINGS}


BAUDS = (115200, 9600, 57600, 38400, 19200, 4800, 2400, 1200, 230400, 250000, 500000, 1000000)


def text_share(data):
    """Share of the bytes that are plain text (printable, tab, line ends): at the wrong baud rate it is low."""
    return sum(1 for b in data if 32 <= b < 127 or b in (9, 10, 13)) / len(data) if data else 0.0


def _listen(port, baud, seconds, send=None, reset=False):
    import serial
    s = serial.Serial()
    s.port, s.baudrate, s.timeout = port, int(baud), 0.1
    if not reset:  # DTR held low from the start: most boards keep running instead of restarting
        s.dtr = s.rts = False
    s.open()
    try:
        if send is not None:
            s.write((str(send).rstrip("\r\n") + "\n").encode())
        data, t0 = bytearray(), time.time()
        while time.time() - t0 < seconds:
            data += s.read(4096)
        return bytes(data)
    finally:
        s.close()


def monitor(port, baud=115200, seconds=3.0, send=None, reset=False, forbidden=()):
    """What runs on a board prints on its serial port, like the Arduino IDE's serial monitor: any sketch, not only
    Lunes' own. baud "auto" tries the usual rates and keeps the one whose bytes read as text. send: a line written
    right after opening (a newline is added), for sketches that take commands. reset=False keeps the running
    sketch going (on most boards); True restarts it, to catch what it prints when it starts."""
    import serial
    if str(port).upper() in {str(f).upper() for f in forbidden}:
        return {"ok": False, "error": f"{port} is the CNC's motion board: Lunes does not open it as a monitor"}
    try:
        if str(baud).lower() == "auto":
            tried = {}
            for rate in BAUDS:
                sample = _listen(port, rate, 0.8)
                tried[rate] = round(text_share(sample), 2) if sample else None
                if sample and text_share(sample) >= 0.95:
                    baud = rate
                    break
            else:
                heard = {r: t for r, t in tried.items() if t is not None}
                if not heard:
                    return {"ok": False, "error": "the board printed nothing at any usual baud rate in 0.8 s each",
                            "note": "it may print only when something happens, or only at start (reset=true)"}
                baud = max(heard, key=heard.get)
        data = _listen(port, baud, seconds, send, reset)
    except serial.SerialException as e:
        return {"ok": False, "error": f"cannot open {port}: {e}"}
    text = data.decode("utf-8", errors="replace")
    lines = [line[:200] for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n") if line.strip()]
    out = {"ok": True, "port": port, "baud": int(baud), "seconds": seconds, "bytes": len(data),
           "lines": lines[-60:], "text_share": round(text_share(data), 2)}
    if len(lines) > 60:
        out["note"] = f"{len(lines)} lines; the last 60 are here"
    if data and text_share(data) < 0.9:
        out["warning"] = ("these bytes do not read as text: the baud rate is probably wrong (try baud 'auto'), or "
                          "the sketch sends binary data")
    if not data:
        out["note"] = "nothing printed in that time" + ("" if reset else ": try reset=true to catch the start")
    return out


def summarize(rows, part):
    """The sketch's lines over a few seconds as one reading per input: an analogue one as its mean, lowest and
    highest counts and volts (and the swing, highest minus lowest), a digital one as the share of HIGH reads."""
    board = BOARDS[part["kind"]]
    volts = board["vcc"] / (2 ** board["adc_bits"] - 1)
    pins = {}
    for pin in sorted({k for r in rows for k in r}):
        values = [r[pin] for r in rows if pin in r]
        if isinstance(values[0], list):  # [mean, lowest, highest] of one 250 ms report
            mean = sum(v[0] for v in values) / len(values)
            low, high = min(v[1] for v in values), max(v[2] for v in values)
            pins[pin] = {"analogRead": round(mean, 1), "lowest": low, "highest": high,
                         "mean_v": round(mean * volts, 3), "min_v": round(low * volts, 3),
                         "max_v": round(high * volts, 3), "vpp_v": round((high - low) * volts, 3)}
        else:
            pins[pin] = {"high_share": round(sum(values) / len(values), 2)}
    return {"ok": True, "reports": len(rows), "pins": pins}


def simulated_rows(part, nets, seconds=3.0, rng=None):
    """The lines Lunes' sketch would print over a few seconds on a board whose nets read like nets (a simulation),
    with a count or two of noise."""
    rng = rng or random.Random(1)
    board = BOARDS[part["kind"]]
    top = 2 ** board["adc_bits"] - 1
    rows = []
    for _ in range(max(1, int(seconds / 0.25))):
        row = {}
        for pin, entry in (part.get("program") or {}).items():
            mode, _ = mode_of(entry)
            m = nets.get(part["pins"][pin])
            if m is None:
                continue
            if mode == "analog_in":
                low, high = _counts(m.get("min_v", m["mean_v"]), board), _counts(m.get("max_v", m["mean_v"]), board)
                mean = min(max(_counts(m["mean_v"], board) + rng.gauss(0, 0.7), 0), top)
                row[pin] = [round(mean, 1), max(low - rng.randint(0, 2), 0), min(high + rng.randint(0, 2), top)]
            elif mode in ("input", "input_pullup"):
                low, high = _level(m.get("min_v", m["mean_v"]), board), _level(m.get("max_v", m["mean_v"]), board)
                if m.get("frequency_hz") and low == 0 and high == 1:
                    row[pin] = round(float(m.get("duty") or 0.5), 2)
                else:
                    level = _level(m["mean_v"], board)
                    row[pin] = float(level) if level is not None else float(rng.random() < 0.5)
        rows.append(row)
    return rows


class SimulatedBoard:
    """A board on a made-up bench, to try Lunes without one. Programming it keeps the circuit; reading it gives what
    Lunes' sketch would print on that circuit as ngspice simulates it, with one fault planted if asked (an id of
    lunes.circuit.faults, like "R1:open" or "U1:not_running")."""
    port = "simulated"

    def __init__(self, fault=None, seed=1):
        self.fault, self.rng = fault, random.Random(seed)
        self.circuit = self.ref = None

    def program(self, circuit, ref):
        self.circuit, self.ref = copy.deepcopy(circuit), ref

    def read(self, seconds=3.0):
        if self.circuit is None or self.fault == f"{self.ref}:not_running":
            return {"ok": False, "error": NO_READINGS}
        part = next(p for p in self.circuit["parts"] if p["ref"] == self.ref)
        return summarize(self._rows(seconds), part)

    def _rows(self, seconds):
        from lunes.circuit import with_fault
        from lunes.simulator import expected
        bench = self.circuit
        if self.fault:
            try:
                bench = with_fault(self.circuit, self.fault)
            except ValueError:  # a part this circuit does not have: the fault is not on this bench
                pass
        part = next(p for p in self.circuit["parts"] if p["ref"] == self.ref)
        return simulated_rows(part, expected(bench)["nets"], seconds, self.rng)

    def monitor(self, seconds=3.0):
        """What the serial monitor would show: the lines of Lunes' sketch."""
        if self.circuit is None or self.fault == f"{self.ref}:not_running":
            return {"ok": True, "port": "simulated", "baud": 115200, "seconds": seconds, "bytes": 0, "lines": [],
                    "note": "nothing printed in that time"}
        lines = [json.dumps(r, separators=(",", ":")) for r in self._rows(seconds)]
        return {"ok": True, "port": "simulated", "baud": 115200, "seconds": seconds,
                "bytes": sum(len(line) + 2 for line in lines), "lines": lines[-60:], "text_share": 1.0}
