"""A circuit as data: its parts, their values and the nets they join, and the SPICE netlist ngspice simulates.

The same small format is what Lunes writes when it designs a circuit or copies one Erlan describes, so the model
fills it in and the code checks it:

    {"name": "555 astable at 1 kHz", "supply_v": 5,
     "parts": [
       {"ref": "U1", "kind": "NE555", "pins": {"GND": "GND", "TRIG": "TH", "OUT": "OUT", "RESET": "VCC",
                                              "CTRL": "CV", "THRES": "TH", "DISCH": "DIS", "VCC": "VCC"}},
       {"ref": "R1", "kind": "R", "value": "1k", "pins": ["VCC", "DIS"]},
       {"ref": "C1", "kind": "C", "value": "68n", "pins": ["TH", "GND"]},
       {"ref": "D1", "kind": "LED", "color": "red", "pins": {"A": "LED", "K": "GND"}}]}

Nets are names; GND is ground and VCC the supply (supply_v volts, from a bench supply or a battery). A
microcontroller board (ARDUINO_UNO, ARDUINO_NANO, BLUEPILL) comes with the program of its pins; see lunes.mcu. Kinds: R, C,
L, D (model 1N4148, 1N4007, 1N5819), LED (color red, yellow, green, blue, white), NPN and PNP (model 2N2222,
2N3904, 2N3906, BC547, BC557; pins C, B, E), NE555, OPAMP (pins OUT, IN-, IN+, V+, V-), and V for an extra source
(pins +, -; value "3.3", or a wave: {"sine": [offset, amplitude, hz]} or {"square": [low, high, hz]}, with an
optional fourth number for the duty: a PWM).

with_fault() gives the same circuit with one thing broken, the way parts break on a bench: open, shorted, the
wrong value (a colour code read wrong: ten times too much or too little), a diode or transistor the wrong way round,
a dead chip.
"""
import copy
import re

from lunes import mcu

UNITS = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "m": 1e-3, "k": 1e3, "K": 1e3, "M": 1e6,
         "G": 1e9, "R": 1.0}
KINDS = {"R", "C", "L", "D", "LED", "NPN", "PNP", "NE555", "OPAMP", "V"} | set(mcu.BOARDS)
PINS = {"D": ("A", "K"), "LED": ("A", "K"), "NPN": ("C", "B", "E"), "PNP": ("C", "B", "E"),
        "NE555": ("GND", "TRIG", "OUT", "RESET", "CTRL", "THRES", "DISCH", "VCC"),
        "OPAMP": ("OUT", "IN-", "IN+", "V+", "V-"), "V": ("+", "-")}
PIN_NUMBER = {"NE555": {name: i + 1 for i, name in enumerate(PINS["NE555"])}}
MODELS = {"D": {"1N4148", "1N4007", "1N5819"}, "NPN": {"2N2222", "2N3904", "BC547"}, "PNP": {"2N3906", "BC557"},
          "LED": {"red", "yellow", "green", "blue", "white"}}


def value(text):
    """ "4k7" -> 4700.0, "68n" -> 6.8e-08, "1M" -> 1e6, "330" -> 330.0, "2.2u" -> 2.2e-06."""
    if isinstance(text, (int, float)):
        return float(text)
    t = str(text).strip().replace("Ω", "").replace("ohm", "").replace("F", "").replace("H", "").replace(" ", "")
    t = re.sub("meg", "M", t, flags=re.IGNORECASE)
    m = re.fullmatch(r"(\d+)([fpnuµmkKMGR])(\d*)", t)  # 4k7, 2R2, 10k
    if m:
        whole, unit, frac = m.groups()
        return float(f"{whole}.{frac or 0}") * UNITS[unit]
    m = re.fullmatch(r"([0-9]*\.?[0-9]+(?:e-?\d+)?)([fpnuµmkKMG]?)", t)
    if not m:
        raise ValueError(f"cannot read the value {text!r}")
    return float(m.group(1)) * UNITS.get(m.group(2), 1.0)


def _net(name):
    """SPICE node name for a net: ground is 0, the rest lower case letters, digits and underscores."""
    n = str(name).strip()
    if n.upper() in ("GND", "0", "TIERRA", "MASA"):
        return "0"
    return "n_" + re.sub(r"[^a-z0-9_]", "_", n.lower())


def check(circuit):
    """Problems in a circuit description, in plain words; an empty list means it can be simulated."""
    problems, refs = [], set()
    if not isinstance(circuit.get("parts"), list) or not circuit["parts"]:
        return ["the circuit has no parts"]
    try:
        if value(circuit.get("supply_v", 5)) <= 0:
            problems.append("supply_v must be positive")
    except ValueError as e:
        problems.append(str(e))
    for p in circuit["parts"]:
        ref, kind = p.get("ref"), p.get("kind")
        if not ref or ref in refs:
            problems.append(f"every part needs its own ref ({ref!r})")
        refs.add(ref)
        if kind not in KINDS:
            problems.append(f"{ref}: kind {kind!r} is not one of {sorted(KINDS)}")
            continue
        pins = p.get("pins")
        if kind in mcu.BOARDS:
            if not isinstance(pins, dict) or not pins:
                problems.append(f"{ref}: pins must map the board's pins to nets, like {{\"D9\": \"OUT\"}}")
            else:
                problems += mcu.check(p)
            continue
        if kind in ("R", "C", "L"):
            if not (isinstance(pins, list) and len(pins) == 2):
                problems.append(f"{ref}: give its two nets as pins: [net, net]")
            try:
                if value(p.get("value", "")) <= 0:
                    problems.append(f"{ref}: the value must be positive")
            except ValueError as e:
                problems.append(f"{ref}: {e}")
        else:
            need = PINS[kind]
            if not isinstance(pins, dict) or set(pins) != set(need):
                problems.append(f"{ref}: pins must name exactly {list(need)}")
        if kind in MODELS:
            key = "color" if kind == "LED" else "model"
            if p.get(key, "red" if kind == "LED" else None) not in MODELS[kind]:
                problems.append(f"{ref}: {key} must be one of {sorted(MODELS[kind])}")
    nets = nets_of(circuit) if not problems else {}
    for net, joined in nets.items():
        if len(joined) < 2 and net not in ("GND", "VCC"):
            problems.append(f"net {net} touches only {joined[0]}: a lead left in the air?")
    return problems


def nets_of(circuit):
    """{net: ["R1", "U1 pin 3 (OUT)", ...]}: what touches each net, as Erlan would find it on the board."""
    nets = {}
    for p in circuit["parts"]:
        pins = p["pins"]
        items = (zip(("1", "2"), pins) if isinstance(pins, list) else pins.items())
        for pin, net in items:
            if isinstance(pins, list):
                where = p["ref"]
            elif p["kind"] in mcu.BOARDS:
                where = f"{p['ref']} pin {pin}"
            elif p["kind"] in PIN_NUMBER:
                where = f"{p['ref']} pin {PIN_NUMBER[p['kind']][pin]} ({pin})"
            else:
                where = f"{p['ref']} {pin}"
            nets.setdefault(str(net), []).append(where)
    return nets


def _source(text_or_wave):
    if isinstance(text_or_wave, dict) and "sine" in text_or_wave:
        off, amp, hz = (value(x) for x in text_or_wave["sine"])
        return f"SIN({off:g} {amp:g} {hz:g})"
    if isinstance(text_or_wave, dict) and "square" in text_or_wave:
        args = [value(x) for x in text_or_wave["square"]]
        lo, hi, hz = args[:3]
        duty = args[3] if len(args) > 3 else 0.5  # a fourth number is the duty cycle: a PWM
        period = 1 / hz
        edge = period / 1000
        return f"PULSE({lo:g} {hi:g} 0 {edge:g} {edge:g} {duty * period - edge:g} {period:g})"
    return f"DC {value(text_or_wave):g}"


def netlist(circuit):
    """The SPICE lines for the circuit and its supply (without the analysis)."""
    lines = [f"* {circuit.get('name', 'circuit')}", f"Vsupply {_net('VCC')} 0 DC {value(circuit.get('supply_v', 5)):g}"]
    for p in circuit["parts"]:
        ref, kind, pins = p["ref"], p["kind"], p["pins"]
        fault = p.get("_fault")
        if fault == "removed":
            continue
        if kind in ("R", "C", "L"):
            a, b = (_net(n) for n in pins)
            if fault == "short":
                lines.append(f"R{ref}_short {a} {b} 0.01")
            elif fault == "open" and kind != "C":
                lines.append(f"R{ref}_open {a} {b} 1e12")
            elif fault == "open":
                continue
            else:
                lines.append(f"{kind}{ref} {a} {b} {value(p['value']):g}")
        elif kind in ("D", "LED"):
            a, k = _net(pins["A"]), _net(pins["K"])
            model = p.get("model", "1N4148") if kind == "D" else f"LED_{p.get('color', 'red').upper()}"
            if fault == "open":
                continue
            if fault == "short":
                lines.append(f"R{ref}_short {a} {k} 0.01")
            elif fault == "reversed":
                lines.append(f"D{ref} {k} {a} {model}")
            else:
                lines.append(f"D{ref} {a} {k} {model}")
        elif kind in ("NPN", "PNP"):
            c, b, e = (_net(pins[x]) for x in ("C", "B", "E"))
            if fault == "ce_short":
                lines.append(f"R{ref}_short {c} {e} 0.01")
                continue
            if fault == "reversed":
                c, e = e, c
            lines.append(f"Q{ref} {c} {b} {e} {p['model']}")
        elif kind == "NE555":
            order = " ".join(_net(pins[x]) for x in PINS["NE555"])
            lines.append(f"X{ref} {order} {'NE555_DEAD' if fault == 'dead' else 'NE555'}")
        elif kind == "OPAMP":
            order = " ".join(_net(pins[x]) for x in PINS["OPAMP"])
            lines.append(f"X{ref} {order} OPAMP")
        elif kind == "V":
            lines.append(f"V{ref} {_net(pins['+'])} {_net(pins['-'])} {_source(p.get('value', 0))}")
        elif kind in mcu.BOARDS:
            lines += mcu.spice(p, _net)
    # every breadboard row and its wires have a few picofarads to ground; it also keeps a broken oscillator (its
    # timing capacitor gone) from racing at the speed of the simulator's own delays
    for net in sorted({_net(n) for n in nets_of(circuit)} - {"0"}):
        lines.append(f"Cstray_{net} {net} 0 5p")
    return lines


FAULTS = {"R": ("open", "short", "x10", "x0.1"), "C": ("open", "short", "x10", "x0.1"), "L": ("open", "short"),
          "D": ("open", "short", "reversed"), "LED": ("open", "short", "reversed"),
          "NPN": ("removed", "ce_short", "reversed"), "PNP": ("removed", "ce_short", "reversed"),
          "NE555": ("dead",), "OPAMP": (), "V": ()}
SAYS = {"open": "{ref} open (broken or a lead not in its row)", "short": "{ref} shorted",
        "x10": "{ref} ten times too big ({wrong} instead of {right}: the colour code read wrong?)",
        "x0.1": "{ref} ten times too small ({wrong} instead of {right})",
        "reversed": "{ref} the wrong way round", "removed": "{ref} missing or not making contact",
        "ce_short": "{ref} shorted collector to emitter (a dead transistor)", "dead": "{ref} dead (the chip itself)"}


def _si(x, unit):
    for factor, prefix in ((1e6, "M"), (1e3, "k"), (1, ""), (1e-3, "m"), (1e-6, "u"), (1e-9, "n"), (1e-12, "p")):
        if abs(x) >= factor:
            return f"{x / factor:g} {prefix}{unit}"
    return f"{x:g} {unit}"


def faults(circuit):
    """[(fault id, what it means in words)] for every typical fault of every part."""
    out = []
    for p in circuit["parts"]:
        if p["kind"] in mcu.BOARDS:
            out += mcu.faults(p)
            continue
        unit = {"R": "ohm", "C": "F", "L": "H"}.get(p["kind"], "")
        for f in FAULTS[p["kind"]]:
            right = value(p["value"]) if "value" in p and p["kind"] in ("R", "C", "L") else None
            wrong = right * (10 if f == "x10" else 0.1) if right is not None else None
            out.append((f"{p['ref']}:{f}", SAYS[f].format(ref=p["ref"], right=right and _si(right, unit),
                                                          wrong=wrong and _si(wrong, unit))))
    return out


def with_fault(circuit, fault_id):
    """A copy of the circuit with one fault (an id from faults())."""
    ref, f = fault_id.split(":")
    broken = copy.deepcopy(circuit)
    for p in broken["parts"]:
        if p["ref"] == ref:
            if f in ("x10", "x0.1"):
                p["value"] = value(p["value"]) * (10 if f == "x10" else 0.1)
            else:
                p["_fault"] = f
            return broken
    raise ValueError(f"no part {ref}")
