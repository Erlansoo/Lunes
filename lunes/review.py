"""A senior engineer's review of a circuit: before simulating it (how it is wired) and after (what its parts go
through). Plain rules, no model; each finding says what is wrong, why it matters and what to do.

Before: every net needs a DC path to ground (a net reached only through capacitors or chip inputs floats, and
the simulator then solves nonsense), the supply must not be shorted, a part with both leads in one net does nothing
(two leads in the same breadboard row), a 555's RESET must not float. After: an LED past 30 mA or under half a
milliamp, a resistor past a quarter watt, a transistor past its rating, a supply asked for more than a USB port
gives, a net beyond the supply rails.
"""
from lunes.circuit import PINS, _net, nets_of, value

# which pins of a part conduct DC between them (inputs of chips do not)
DC_LINKS = {"NE555": [("GND", "VCC"), ("CTRL", "VCC"), ("CTRL", "GND"), ("OUT", "GND"), ("DISCH", "GND")],
            "OPAMP": [("OUT", "V-"), ("IN+", "IN-"), ("V+", "V-")],
            "NPN": [("C", "E"), ("B", "E")], "PNP": [("C", "E"), ("B", "E")]}
CHIP_INPUTS = {"NE555": ("TRIG", "THRES", "RESET"), "OPAMP": ("IN+", "IN-")}
IC_MAX_A = {"2N2222": 0.6, "2N3904": 0.2, "2N3906": 0.2, "BC547": 0.1, "BC557": 0.1}
DIODE_MAX_A = {"1N4148": 0.3, "1N4007": 1.0, "1N5819": 1.0}


def _pins(part):
    pins = part["pins"]
    return dict(zip(("1", "2"), pins)) if isinstance(pins, list) else dict(pins)


def before(circuit):
    """(errors that make simulating pointless, warnings) for how the circuit is wired."""
    errors, warnings = [], []
    nets = nets_of(circuit)
    links = {n: set() for n in nets}
    links.setdefault("GND", set())
    links.setdefault("VCC", set())

    def link(a, b):
        links.setdefault(a, set()).add(b)
        links.setdefault(b, set()).add(a)

    link("VCC", "GND")  # the supply
    for p in circuit["parts"]:
        pins, kind = _pins(p), p["kind"]
        if kind in ("R", "L", "D", "LED", "V"):
            a, b = list(pins.values())
            link(a, b)
            if kind == "L" and {a, b} == {"VCC", "GND"}:
                errors.append(f"{p['ref']} sits straight across the supply: at DC a coil is a short")
        for x, y in DC_LINKS.get(kind, []):
            link(pins[x], pins[y])
        values = list(pins.values())
        if kind in ("R", "C", "L", "D", "LED") and values[0] == values[1]:
            warnings.append(f"{p['ref']} has both leads in net {values[0]}: it does nothing (both in the same "
                            f"breadboard row?)")
        if kind == "V" and {values[0], values[1]} == {"VCC", "GND"}:
            errors.append(f"{p['ref']} is a second source straight across the supply: two sources fight")
        if kind in ("R",) and value(p["value"]) < 1:
            warnings.append(f"{p['ref']} is {value(p['value']):g} ohm: practically a wire")
        if kind == "C" and value(p["value"]) > 0.01:
            warnings.append(f"{p['ref']} is {value(p['value']) * 1e3:g} mF: unusually big, is the unit right?")
        if kind == "NE555" and pins["RESET"] not in ("VCC",) and len(nets.get(pins["RESET"], [])) < 2:
            warnings.append(f"{p['ref']} RESET (pin 4) is left floating: tie it to VCC, or the chip may stop")
        if kind == "NE555" and len(nets.get(pins["CTRL"], [])) < 2:
            warnings.append(f"{p['ref']} CTRL (pin 5) has nothing on it: a 10 nF capacitor to GND keeps noise out "
                            f"of the timing")
    # nets that cannot reach ground through anything that conducts DC
    seen, todo = {"GND"}, ["GND"]
    while todo:
        for nxt in links.get(todo.pop(), ()):
            if nxt not in seen:
                seen.add(nxt)
                todo.append(nxt)
    for net in nets:
        if net not in seen:
            what = ", ".join(nets[net])
            errors.append(f"net {net} ({what}) has no DC path to ground: it reaches the rest only through "
                          f"capacitors or chip inputs, so its voltage is undefined; connect it, or add a resistor "
                          f"to ground")
    return errors, warnings


def after(circuit, sim):
    """Warnings from what the simulation says the parts go through (sim from lunes.simulator.expected)."""
    warnings = []
    supply = value(circuit.get("supply_v", 5))
    currents = sim.get("currents", {})
    for p in circuit["parts"]:
        c = currents.get(p["ref"])
        if not c:
            continue
        peak, kind = c["peak_a"], p["kind"]
        if kind == "LED":
            if peak > 0.03:
                warnings.append(f"{p['ref']} carries {peak * 1e3:.0f} mA at its peak: an LED wants 5 to 20 mA, "
                                f"this burns it or dims fast; a bigger series resistor fixes it")
            elif c.get("mean_a", peak) < 0.0005 and peak < 0.001:
                warnings.append(f"{p['ref']} carries only {peak * 1e3:.2f} mA: it will barely light")
        if kind == "D" and peak > DIODE_MAX_A.get(p.get("model"), 1.0):
            warnings.append(f"{p['ref']} carries {peak:.2f} A, past what a {p.get('model')} takes")
        if kind == "R" and c.get("power_w", 0) > 0.25:
            watts = c["power_w"]
            size = "1/2 W" if watts < 0.45 else "1 W" if watts < 0.9 else f"{max(2, round(watts * 2)):.0f} W"
            warnings.append(f"{p['ref']} dissipates {watts:.2f} W: a common 1/4 W resistor overheats, use {size}")
        if kind in ("NPN", "PNP") and peak > IC_MAX_A.get(p.get("model"), 0.2):
            warnings.append(f"{p['ref']} collector carries {peak:.2f} A, past the {p.get('model')}'s "
                            f"{IC_MAX_A.get(p.get('model'), 0.2):g} A")
    s = currents.get("supply")
    if s and s.get("mean_a", 0) > 0.5:
        warnings.append(f"the circuit takes {s['mean_a']:.2f} A from the supply: more than a USB port gives")
    for net, m in sim.get("nets", {}).items():
        top = m.get("max_v", m["mean_v"])
        low = m.get("min_v", m["mean_v"])
        if top > supply + 0.6 or low < -0.6:
            warnings.append(f"net {net} goes to {low:.2f} .. {top:.2f} V, outside 0 .. {supply:g} V: a chip input "
                            f"there would clamp or be damaged; bias it inside the supply (a high-pass output sits "
                            f"around 0 V, a coil kicks when switched off)")
    return warnings


def spice_names(circuit):
    """{spice node: net name}, to say ngspice's complaints in the circuit's own words."""
    return {_net(n): n for n in nets_of(circuit)}


def chip_pins(kind):
    return PINS.get(kind, ())
