"""A microcontroller in the circuit: one pin program that is simulated, turned into a sketch, flashed and read back.

The circuit gives the board as a part, with what each pin does:

    {"ref": "U2", "kind": "ARDUINO_UNO", "pins": {"D9": "PWM", "A0": "FILT", "GND": "GND", "5V": "VCC"},
     "program": {"D9": {"pwm": 0.3}, "A0": "analog_in"}}

Pin modes: "high", "low", {"pwm": duty 0..1} (on the board's PWM pins), {"square": hz, "duty": 0.5} (toggled by
the sketch, up to 5 kHz), "input", "input_pullup", "analog_in" (on the analogue pins).

The same program does three things, so the bench and the simulation cannot disagree:
- In ngspice each pin is what it is electrically: an output is a 0 / VCC source behind the pin's own resistance
  (about 25 ohm on an ATmega328P), an input a high impedance, a pull-up 35 k to the board's supply.
- sketch() writes the Arduino sketch: pinMode, the outputs, and a JSON line every 250 ms with every input
  ({"A0": 307, "D4": 1}). compile() and upload() go through arduino-cli; upload never goes to the CNC's port.
- read() listens to those lines over USB serial, and expectations() says what each input should read according
  to the simulation (analogRead counts, digitalRead levels).
"""
import json
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
        else:  # an input, or an output the sketch never drives: a high impedance with a little capacitance
            lines.append(f"R{name}_in {node} 0 100Meg")
            lines.append(f"C{name}_in {node} 0 10p")
            if mode == "input_pullup" and running:
                lines.append(f"R{name}_pullup {node} {vcc_node} {board['pullup']:g}")
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


def expectations(part, nets):
    """What each input should read on the board, from the simulated nets: analogRead counts, digitalRead levels."""
    board, out = BOARDS[part["kind"]], {}
    top = 2 ** board["adc_bits"] - 1
    for pin, entry in (part.get("program") or {}).items():
        mode, _ = mode_of(entry)
        m = nets.get(part["pins"][pin])
        if m is None or mode not in ("analog_in", "input", "input_pullup"):
            continue
        v = m["mean_v"]
        if mode == "analog_in":
            out[pin] = {"analogRead": round(min(max(v, 0), board["vcc"]) / board["vcc"] * top), "volts": round(v, 3)}
            if m.get("frequency_hz"):
                out[pin]["note"] = (f"the net moves ({m['frequency_hz']:.4g} Hz, {m['vpp_v']:.2f} Vpp): single "
                                    f"readings scatter around the mean")
        else:
            level = 1 if v >= board["vih"] * board["vcc"] else 0 if v <= board["vil"] * board["vcc"] else None
            out[pin] = {"digitalRead": level, "volts": round(v, 3)}
            if level is None:
                out[pin]["note"] = "between the logic thresholds: it may read either way"
    return out


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
def _code_pin(pin, board):
    return pin[1:] if pin.startswith("D") and board["vcc"] == 5.0 else pin  # D9 -> 9, A0 -> A0, PA0 -> PA0


def sketch(part, circuit_name="circuit"):
    """The Arduino sketch for the part's pin program, as text."""
    board = BOARDS[part["kind"]]
    program = part.get("program") or {}
    head, setup, loop, report = [], [], [], []
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
            report.append((pin, f"digitalRead({p})"))
        elif mode == "analog_in":
            report.append((pin, f"analogRead({p})"))
    lines = [f"// Written by Lunes from the pin program of \"{circuit_name}\" ({board['name']}).",
             "// The same program is what Lunes simulated, so the bench and the simulation should agree.",
             "// Every 250 ms it prints one JSON line with every input, for Lunes to read over USB.", ""]
    lines += head + ["", "void setup() {", "  Serial.begin(115200);"] + setup + ["}", "", "void loop() {",
                                                                              "  unsigned long now = micros();"]
    lines += loop
    if report:
        lines += ["  static unsigned long last_report = 0;", "  if (millis() - last_report >= 250) {",
                  "    last_report = millis();"]
        for k, (pin, expr) in enumerate(report):
            lines.append(f'    Serial.print("{"{" if k == 0 else ","}\\"{pin}\\":");')
            lines.append(f"    Serial.print({expr});")
        lines += ['    Serial.println("}");', "  }"]
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


def read(port, seconds=3.0, forbidden=()):
    """The sketch's JSON lines over serial for a few seconds: {"readings": n, "mean": {pin: value},
    "last": {...}}. Opening the port resets the board, so it waits for it to start first."""
    import serial
    if str(port).upper() in {str(f).upper() for f in forbidden}:
        return {"ok": False, "error": f"{port} is the CNC's motion board, not the Arduino"}
    rows = []
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
    if not rows:
        return {"ok": False, "error": "no readings: is it this port, and is Lunes' sketch on the board?"}
    keys = sorted({k for r in rows for k in r})
    mean = {k: round(sum(r[k] for r in rows if k in r) / sum(1 for r in rows if k in r), 1) for k in keys}
    return {"ok": True, "readings": len(rows), "mean": mean, "last": rows[-1]}
