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
microcontroller board (ARDUINO_UNO, ARDUINO_NANO, BLUEPILL) comes with the program of its pins; see lunes.mcu. Kinds: R,
C, L (value, pins [a, b]); D (model 1N4148, 1N4007, 1N5819, 1N5817, 1N5822, BAT85; pins A, K); ZENER (model
BZX55C3V3 .. BZX55C12, 1N4733A, 1N4742A: the zener voltage is in the name; pins A, K); LED (color red, yellow, green,
blue, white); NPN and PNP (model 2N2222, 2N3904, 2N3906, BC547, BC557; pins C, B, E); NMOS (model 2N7000, BS170,
IRLZ44N, IRF540) and PMOS (IRF9540), pins D, G, S; POT, a potentiometer (value the whole track, position 0 to 1 of
the wiper from A to B, 0.5 if left out; pins A, W, B); SW, a switch or push button (state "closed" or "open", pins
[a, b]); LDR, a light-dependent resistor (dark and lit values, condition "lit" or "dark") and NTC, a thermistor
(cold and hot values, condition "cold" or "hot"), pins [a, b]; REG, a linear regulator (model 7805, 7812,
LM1117-3.3, LM1117-5; pins IN, OUT, GND); NE555; OPAMP (pins OUT, IN-, IN+, V+, V-); and V for an extra source
(pins +, -; value "3.3", or a wave: {"sine": [offset, amplitude, hz]} or {"square": [low, high, hz]}, with an
optional fourth number for the duty: a PWM).

with_fault() gives the same circuit with one thing broken, the way parts break on a bench: open, shorted, the
wrong value (a colour code read wrong: ten times too much or too little), a diode or transistor the wrong way round,
a dead chip, a pot turned all the way or with a worn wiper, a switch that never closes or is welded, a sensor in
the other condition (covered, hot), a regulator with nothing out or its input straight on its output.
"""
import copy
import re

from lunes import catalog, mcu

UNITS = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "m": 1e-3, "k": 1e3, "K": 1e3, "M": 1e6,
         "G": 1e9, "R": 1.0}
KINDS = {"R", "C", "L", "D", "ZENER", "LED", "NPN", "PNP", "NMOS", "PMOS", "POT", "SW", "LDR", "NTC", "REG", "NE555",
         "OPAMP", "V", "IC"} | set(mcu.BOARDS)
TWO_LEADS = ("R", "C", "L", "SW", "LDR", "NTC")  # pins as [a, b]
PINS = {"D": ("A", "K"), "ZENER": ("A", "K"), "LED": ("A", "K"), "NPN": ("C", "B", "E"), "PNP": ("C", "B", "E"),
        "NMOS": ("D", "G", "S"), "PMOS": ("D", "G", "S"), "POT": ("A", "W", "B"), "REG": ("IN", "OUT", "GND"),
        "NE555": ("GND", "TRIG", "OUT", "RESET", "CTRL", "THRES", "DISCH", "VCC"),
        "OPAMP": ("OUT", "IN-", "IN+", "V+", "V-"), "V": ("+", "-")}
PIN_NUMBER = {"NE555": {name: i + 1 for i, name in enumerate(PINS["NE555"])}}
ZENER_V = {"BZX55C3V3": 3.3, "BZX55C4V7": 4.7, "BZX55C5V1": 5.1, "BZX55C6V2": 6.2, "BZX55C9V1": 9.1,
           "BZX55C12": 12.0, "1N4733A": 5.1, "1N4742A": 12.0}  # the zener voltage, from the name
ZENER_W = {name: (1.0 if name.startswith("1N") else 0.5) for name in ZENER_V}  # what each can dissipate
REG_V = {"7805": 5.0, "7812": 12.0, "LM1117-3.3": 3.3, "LM1117-5": 5.0}  # the regulator's output
REG_DROPOUT = {"7805": 2.0, "7812": 2.0, "LM1117-3.3": 1.2, "LM1117-5": 1.2}  # the input must exceed the output by this
REG_MAX_A = {"7805": 1.0, "7812": 1.0, "LM1117-3.3": 0.8, "LM1117-5": 0.8}
MOS_MAX_A = {"2N7000": 0.2, "BS170": 0.5, "IRLZ44N": 20.0, "IRF540": 20.0, "IRF9540": 15.0}  # on a breadboard, no heatsink
MOS_VGS_FULL = {"2N7000": 4.5, "BS170": 4.5, "IRLZ44N": 4.0, "IRF540": 8.0, "IRF9540": 8.0}  # gate drive for a full switch
MODELS = {"D": {"1N4148", "1N4007", "1N5819", "1N5817", "1N5822", "BAT85"}, "ZENER": set(ZENER_V),
          "NPN": {"2N2222", "2N3904", "BC547"}, "PNP": {"2N3906", "BC557"},
          "NMOS": {"2N7000", "BS170", "IRLZ44N", "IRF540"}, "PMOS": {"IRF9540"}, "REG": set(REG_V),
          "LED": {"red", "yellow", "green", "blue", "white"}}
CONDITIONS = {"LDR": ("lit", "dark"), "NTC": ("cold", "hot")}  # the first is the usual one


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
        if kind == "IC":  # any chip of the catalog: its pins by the names the catalog gives them
            entry = catalog.get(p.get("model"))
            if entry is None:
                problems.append(f"{ref}: chip {p.get('model')!r} is not in the catalog ({', '.join(catalog.names())}): "
                                f"learn it from its datasheet with chip_learn")
            elif not isinstance(pins, dict) or not pins:
                problems.append(f"{ref}: pins must map the chip's pins to nets, like {{\"VCC\": \"VCC\"}}")
            else:
                known = set(catalog.pin_numbers(entry))
                unknown = sorted(set(pins) - known)
                if unknown:
                    problems.append(f"{ref}: {entry['name']} has no pins {unknown}; its pins are {sorted(known)}")
                missing = [x for x in entry["power"] if x not in pins]
                if missing:
                    problems.append(f"{ref}: {entry['name']} needs {missing} connected")
            continue
        if kind in mcu.BOARDS:
            if not isinstance(pins, dict) or not pins:
                problems.append(f"{ref}: pins must map the board's pins to nets, like {{\"D9\": \"OUT\"}}")
            else:
                problems += mcu.check(p)
            continue
        if kind in TWO_LEADS:
            if not (isinstance(pins, list) and len(pins) == 2):
                problems.append(f"{ref}: give its two nets as pins: [net, net]")
            keys = {"SW": (), "LDR": ("dark", "lit"), "NTC": ("cold", "hot")}.get(kind, ("value",))
            for key in keys:
                try:
                    if value(p.get(key, "")) <= 0:
                        problems.append(f"{ref}: {key} must be positive")
                except ValueError as e:
                    problems.append(f"{ref}: {key}: {e}")
            if kind == "SW" and p.get("state", "closed") not in ("open", "closed"):
                problems.append(f"{ref}: state must be 'closed' or 'open'")
            if kind in CONDITIONS and p.get("condition", CONDITIONS[kind][0]) not in CONDITIONS[kind]:
                problems.append(f"{ref}: condition must be one of {list(CONDITIONS[kind])}")
        else:
            need = PINS[kind]
            if not isinstance(pins, dict) or set(pins) != set(need):
                problems.append(f"{ref}: pins must name exactly {list(need)}")
        if kind == "POT":
            try:
                if value(p.get("value", "")) <= 0:
                    problems.append(f"{ref}: the value (the whole track) must be positive")
                if not 0 <= float(p.get("position", 0.5)) <= 1:
                    problems.append(f"{ref}: position is where the wiper stands, 0 (at A) to 1 (at B)")
            except (ValueError, TypeError) as e:
                problems.append(f"{ref}: {e}")
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
            elif p["kind"] == "IC":
                where = f"{p['ref']} pin {catalog.pin_numbers(catalog.get(p['model']))[pin]} ({pin})"
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
        elif kind == "SW":  # closed: a few hundredths of an ohm; open: nothing; dirty: tens of ohms
            a, b = (_net(n) for n in pins)
            closed = p.get("state", "closed") == "closed"
            ohms = 47.0 if fault == "dirty" else 0.05 if (fault == "stuck_closed" or (closed and fault != "stuck_open")) else 1e12
            lines.append(f"R{ref} {a} {b} {ohms:g}")
        elif kind in CONDITIONS:  # a sensor: the resistance of its condition
            a, b = (_net(n) for n in pins)
            if fault == "short":
                lines.append(f"R{ref}_short {a} {b} 0.01")
            elif fault == "open":
                lines.append(f"R{ref}_open {a} {b} 1e12")
            else:
                lines.append(f"R{ref} {a} {b} {value(p[p.get('condition', CONDITIONS[kind][0])]):g}")
        elif kind == "POT":
            a, w, b = (_net(pins[x]) for x in ("A", "W", "B"))
            r, pos = value(p["value"]), float(p.get("position", 0.5))
            if fault == "open":  # the track broken: nothing conducts
                lines += [f"R{ref}a {a} {w} 1e12", f"R{ref}b {w} {b} 1e12"]
            elif fault == "wiper_open":  # the track whole, the wiper touching nothing
                lines += [f"R{ref}a {a} {b} {r:g}", f"R{ref}b {w} {b} 1e12"]
            else:  # position 0: the wiper at A (nothing between A and W); 1: at B
                lines += [f"R{ref}a {a} {w} {max(r * pos, 0.5):g}", f"R{ref}b {w} {b} {max(r * (1 - pos), 0.5):g}"]
        elif kind in ("D", "ZENER", "LED"):
            a, k = _net(pins["A"]), _net(pins["K"])
            model = p.get("model", "1N4148") if kind in ("D", "ZENER") else f"LED_{p.get('color', 'red').upper()}"
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
        elif kind in ("NMOS", "PMOS"):
            d, g, s = (_net(pins[x]) for x in ("D", "G", "S"))
            if fault == "ds_short":
                lines.append(f"R{ref}_short {d} {s} 0.05")
                continue
            if fault == "reversed":
                d, s = s, d
            lines.append(f"M{ref} {d} {g} {s} {s} {p['model']}")  # the bulk on the source, as in the package
        elif kind == "REG":
            i, o, g = (_net(pins[x]) for x in ("IN", "OUT", "GND"))
            if fault == "shorted":  # the pass element shorted: the input straight on the output
                lines.append(f"R{ref}_short {i} {o} 0.05")
            elif fault == "dead":
                lines.append(f"X{ref} {i} {o} {g} REG_DEAD")
            else:
                lines.append(f"X{ref} {i} {o} {g} REG vreg={REG_V[p['model']]:g} drop={REG_DROPOUT[p['model']]:g}")
        elif kind == "IC":
            entry = catalog.get(p["model"])
            if fault == "removed":
                continue
            ground = _net(pins[entry["power"][-1]])
            if fault == "dead":  # no life in it: its supply and its outputs pulled down, its inputs nothing
                lines.append(f"R{ref}_dead {_net(pins[entry['power'][0]])} {ground} 1k")
                lines += [f"R{ref}_dead_{i} {_net(pins[x])} {ground} 10"
                          for i, x in enumerate(entry.get("outputs", [])) if x in pins]
                continue
            # a pin left out, or a leg not in its row, is a node of its own that nothing else touches
            nodes = [_net(pins[x]) if x in pins and fault != f"pin{n}_open" else f"nc_{ref.lower()}_{n}"
                     for n, x in catalog.ordered_pins(entry)]
            lines.append(f"X{ref} {' '.join(nodes)} {catalog.spice_name(entry['name'])}")
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
          "D": ("open", "short", "reversed"), "ZENER": ("open", "short", "reversed"), "LED": ("open", "short", "reversed"),
          "NPN": ("removed", "ce_short", "reversed"), "PNP": ("removed", "ce_short", "reversed"),
          "NMOS": ("removed", "ds_short", "reversed"), "PMOS": ("removed", "ds_short", "reversed"),
          "POT": ("open", "wiper_open", "x10", "x0.1", "turned_down", "turned_up"),
          "SW": ("stuck_open", "stuck_closed", "dirty"), "LDR": ("open", "short", "flipped"),
          "NTC": ("open", "short", "flipped"), "REG": ("dead", "shorted"),
          "NE555": ("dead",), "OPAMP": (), "V": ()}
SAYS = {"open": "{ref} open (broken or a lead not in its row)", "short": "{ref} shorted",
        "x10": "{ref} ten times too big ({wrong} instead of {right}: the colour code read wrong?)",
        "x0.1": "{ref} ten times too small ({wrong} instead of {right})",
        "reversed": "{ref} the wrong way round", "removed": "{ref} missing or not making contact",
        "ce_short": "{ref} shorted collector to emitter (a dead transistor)", "dead": "{ref} dead (the chip itself)",
        "ds_short": "{ref} shorted drain to source (a dead MOSFET)",
        "wiper_open": "{ref} wiper not touching the track (worn, or its middle leg in the air)",
        "turned_down": "{ref} turned all the way to its A end", "turned_up": "{ref} turned all the way to its B end",
        "stuck_open": "{ref} never closes (a dead switch, or not pressed)",
        "stuck_closed": "{ref} stuck closed (contacts welded)", "dirty": "{ref} dirty contacts (tens of ohms across them)",
        "flipped": "{ref} in its other condition ({other})",
        "shorted": "{ref} input shorted to output (a dead regulator: the full input on its output)"}
OTHER_CONDITION = {"lit": "dark: covered or in the dark", "dark": "lit", "cold": "hot: warmed up", "hot": "cold"}


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
        if p["kind"] == "IC":
            entry, ref = catalog.get(p["model"]), p["ref"]
            out += [(f"{ref}:removed", SAYS["removed"].format(ref=ref)), (f"{ref}:dead", SAYS["dead"].format(ref=ref))]
            out += [(f"{ref}:pin{n}_open", f"{ref} pin {n} ({name}) not making contact (a bent leg, or not in its row)")
                    for n, name in catalog.ordered_pins(entry) if name in p["pins"]]
            continue
        unit = {"R": "ohm", "C": "F", "L": "H", "POT": "ohm"}.get(p["kind"], "")
        for f in FAULTS[p["kind"]]:
            right = value(p["value"]) if "value" in p and p["kind"] in ("R", "C", "L", "POT") else None
            wrong = right * (10 if f == "x10" else 0.1) if right is not None else None
            now = p.get("condition", CONDITIONS[p["kind"]][0]) if p["kind"] in CONDITIONS else None
            out.append((f"{p['ref']}:{f}", SAYS[f].format(ref=p["ref"], right=right and _si(right, unit),
                                                          wrong=wrong and _si(wrong, unit),
                                                          other=OTHER_CONDITION.get(now, ""))))
    return out


def with_fault(circuit, fault_id):
    """A copy of the circuit with one fault (an id from faults())."""
    ref, f = fault_id.split(":")
    broken = copy.deepcopy(circuit)
    for p in broken["parts"]:
        if p["ref"] == ref:
            if f in ("x10", "x0.1"):
                p["value"] = value(p["value"]) * (10 if f == "x10" else 0.1)
            elif f in ("turned_down", "turned_up"):
                p["position"] = 0.0 if f == "turned_down" else 1.0
            elif f == "flipped":
                usual, other = CONDITIONS[p["kind"]]
                p["condition"] = other if p.get("condition", usual) == usual else usual
            else:
                p["_fault"] = f
            return broken
    raise ValueError(f"no part {ref}")
