"""Circuits designed by formula, with values you can buy, and the goals the simulation must meet.

The model decides what is wanted ("a 555 at 2 Hz with a red LED"); these functions do the arithmetic an engineer
does, round to the standard series (E12 resistors, E6 capacitors) and say which formula they used. Each returns
(circuit, spec, notes): the spec is what the simulation of the circuit must show (a frequency within 8 %, an LED
current near its target), checked by verify() on lunes.simulator.expected's result, so a design is never handed over
on faith.
"""
import math

from lunes.circuit import value

E12 = (1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2)
E6 = (1.0, 1.5, 2.2, 3.3, 4.7, 6.8)
LED_VF = {"red": 1.85, "yellow": 1.95, "green": 2.15, "blue": 3.2, "white": 3.2}  # at about 10 mA, as the models give


def nearest(x, series=E12, mode="nearest"):
    """The value of a standard series nearest to x ("up": the next one up, "down": the next one down)."""
    decade = 10 ** math.floor(math.log10(x))
    options = [m * decade * k for k in (0.1, 1, 10) for m in series]
    if mode == "up":
        return min((o for o in options if o >= x * 0.999), default=max(options))
    if mode == "down":
        return max((o for o in options if o <= x * 1.001), default=min(options))
    return min(options, key=lambda o: abs(math.log(o / x)))


def text(x, unit=""):
    """4700 -> "4.7k", 6.8e-8 -> "68n": how the value is written on the part list."""
    for factor, prefix in ((1e6, "M"), (1e3, "k"), (1, ""), (1e-3, "m"), (1e-6, "u"), (1e-9, "n"), (1e-12, "p")):
        if x >= factor * 0.999:
            return f"{x / factor:.3g}{prefix}{unit}"
    return f"{x:g}{unit}"


def led(supply_v=5.0, color="red", current_ma=10.0, ref="D1", resistor="R1", source="VCC", anode="LED_A"):
    """An LED with its series resistor from source: R = (V - Vf) / I, rounded up so the current stays at or below
    the target."""
    vf = LED_VF[color]
    r = nearest((supply_v - vf) / (current_ma / 1000), mode="up")
    parts = [{"ref": resistor, "kind": "R", "value": text(r), "pins": [source, anode]},
             {"ref": ref, "kind": "LED", "color": color, "pins": {"A": anode, "K": "GND"}}]
    notes = [f"{resistor} = ({supply_v:g} V - {vf:g} V of the {color} LED) / {current_ma:g} mA = "
             f"{(supply_v - vf) / current_ma * 1000:.0f} ohm, rounded up to {text(r)}"]
    spec = [{"current_of": ref, "target_a": (supply_v - vf) / r, "tolerance": 0.3}]
    return parts, spec, notes


def led_resistor(supply_v=5.0, color="red", current_ma=10.0):
    parts, spec, notes = led(supply_v, color, current_ma)
    return {"name": f"{color} LED at {current_ma:g} mA from {supply_v:g} V", "supply_v": supply_v,
            "parts": parts}, spec, notes


def astable_555(frequency_hz, duty=0.55, supply_v=5.0, led_color=None):
    """A 555 astable: f = 1.44 / ((R1 + 2 R2) C), duty = (R1 + R2) / (R1 + 2 R2), so the duty stays above 50 %.
    The capacitor is picked so both resistors land between 1 k and 470 k, then everything is rounded to the series
    and the frequency recomputed with the rounded values."""
    duty = min(max(float(duty), 0.51), 0.95)
    best = None
    for exp in range(-11, -2):
        for m in E6:
            c = m * 10 ** exp
            s = 1.44 / (frequency_hz * c)
            r2, r1 = (1 - duty) * s, (2 * duty - 1) * s
            r1 = max(r1, 1e3)
            r2 = (s - r1) / 2
            if not (1e3 <= r2 <= 470e3 and r1 <= 470e3):
                continue
            r1n, r2n = nearest(r1), nearest(r2)
            f = 1.44 / ((r1n + 2 * r2n) * c)
            d = (r1n + r2n) / (r1n + 2 * r2n)
            score = abs(math.log(f / frequency_hz)) + 0.5 * abs(d - duty) + 0.02 * abs(math.log(r2n / 47e3))
            if best is None or score < best[0]:
                best = (score, c, r1n, r2n, f, d)
    if best is None:
        raise ValueError(f"no 555 astable with standard parts for {frequency_hz:g} Hz")
    _, c, r1, r2, f, d = best
    parts = [
        {"ref": "U1", "kind": "NE555", "pins": {"GND": "GND", "TRIG": "TH", "OUT": "OUT", "RESET": "VCC",
                                               "CTRL": "CV", "THRES": "TH", "DISCH": "DIS", "VCC": "VCC"}},
        {"ref": "R1", "kind": "R", "value": text(r1), "pins": ["VCC", "DIS"]},
        {"ref": "R2", "kind": "R", "value": text(r2), "pins": ["DIS", "TH"]},
        {"ref": "C1", "kind": "C", "value": text(c), "pins": ["TH", "GND"]},
        {"ref": "C2", "kind": "C", "value": "10n", "pins": ["CV", "GND"]}]
    notes = [f"f = 1.44 / ((R1 + 2 R2) C1) = 1.44 / (({text(r1)} + 2 x {text(r2)}) x {text(c, 'F')}) = {f:.4g} Hz",
             f"duty = (R1 + R2) / (R1 + 2 R2) = {d:.0%}",
             "pin 4 (RESET) to VCC so it never stops; 10 nF on pin 5 (CTRL) keeps noise out of the timing"]
    if c >= 1e-6:
        notes.append(f"C1 of {text(c, 'F')} is an electrolytic: its minus to GND")
    if abs(f / frequency_hz - 1) > 0.02:
        notes.append(f"{abs(f / frequency_hz - 1):.0%} {'below' if f < frequency_hz else 'above'} the "
                     f"{frequency_hz:g} Hz asked, from rounding to standard values; a trimmer in series with R2 "
                     f"tunes it exactly")
    spec = [{"net": "OUT", "frequency_hz": frequency_hz, "tolerance": 0.08},
            {"net": "OUT", "duty": d, "tolerance": 0.06}]
    if led_color:
        lp, ls, ln = led(supply_v - 1.5, led_color, 10.0, ref="D1", resistor="R3", source="OUT", anode="LED_A")
        parts += lp
        notes += [n.replace(f"{supply_v - 1.5:g} V", f"{supply_v - 1.5:g} V (the 555's output when high)") for n in ln]
    return {"name": f"555 astable at {frequency_hz:g} Hz" + (f" with a {led_color} LED" if led_color else ""),
            "supply_v": supply_v, "parts": parts}, spec, notes


def npn_switch(load_current_ma=20.0, supply_v=5.0, logic_v=5.0, input_hz=1.0, color="red", model="2N2222"):
    """An NPN transistor switching an LED from a logic pin (a microcontroller's, here a square wave on IN). Rc sets
    the LED current: (V - Vf - 0.2 V across the saturated transistor) / I; Rb gives ten times less base current than
    collector current, so the transistor saturates whatever its gain."""
    vf = LED_VF[color]
    i = load_current_ma / 1000
    rc = nearest((supply_v - vf - 0.2) / i, mode="up")
    ic = (supply_v - vf - 0.2) / rc
    rb = nearest((logic_v - 0.7) / (ic / 10), mode="down")
    parts = [{"ref": "V1", "kind": "V", "value": {"square": [0, logic_v, input_hz]}, "pins": {"+": "IN", "-": "GND"}},
             {"ref": "R1", "kind": "R", "value": text(rb), "pins": ["IN", "B"]},
             {"ref": "Q1", "kind": "NPN", "model": model, "pins": {"C": "C", "B": "B", "E": "GND"}},
             {"ref": "R2", "kind": "R", "value": text(rc), "pins": ["VCC", "LED_A"]},
             {"ref": "D1", "kind": "LED", "color": color, "pins": {"A": "LED_A", "K": "C"}}]
    notes = [f"R2 = ({supply_v:g} - {vf:g} - 0.2) V / {load_current_ma:g} mA, rounded up to {text(rc)}: "
             f"{ic * 1000:.1f} mA in the LED",
             f"R1 = ({logic_v:g} - 0.7) V / ({ic * 1000:.1f} mA / 10) rounded down to {text(rb)}: the base gets a "
             f"tenth of the collector current, enough to saturate a {model}",
             "V1 stands for the pin that drives it (an Arduino pin, a 555): 0 or "
             f"{logic_v:g} V at {input_hz:g} Hz"]
    spec = [{"current_of": "D1", "target_a": ic, "tolerance": 0.3}, {"net": "C", "min_v_below": 0.4}]
    return {"name": f"{model} switching a {color} LED at {ic * 1000:.0f} mA", "supply_v": supply_v,
            "parts": parts}, spec, notes


def divider(vout, supply_v=5.0, current_ma=0.5):
    """A resistive divider: Vout = V x R2 / (R1 + R2), the pair picked from the series for the closest ratio near
    the asked current. It holds its voltage only into a load much bigger than R2."""
    total = supply_v / (current_ma / 1000)
    best = None
    for r2 in {nearest(total * vout / supply_v * k) for k in (0.7, 0.85, 1, 1.2, 1.4)}:
        r1 = nearest(r2 * (supply_v - vout) / vout)
        v = supply_v * r2 / (r1 + r2)
        if best is None or abs(v - vout) < abs(best[2] - vout):
            best = (r1, r2, v)
    r1, r2, v = best
    parts = [{"ref": "R1", "kind": "R", "value": text(r1), "pins": ["VCC", "OUT"]},
             {"ref": "R2", "kind": "R", "value": text(r2), "pins": ["OUT", "GND"]}]
    notes = [f"Vout = {supply_v:g} V x {text(r2)} / ({text(r1)} + {text(r2)}) = {v:.3f} V",
             f"it draws {supply_v / (r1 + r2) * 1000:.2f} mA; any load on OUT must be much bigger than {text(r2)}"]
    return {"name": f"divider for {vout:g} V from {supply_v:g} V", "supply_v": supply_v, "parts": parts}, \
        [{"net": "OUT", "mean_v": v, "tolerance": 0.03}], notes


def rc_filter(cutoff_hz, kind="lowpass", supply_v=5.0):
    """A first-order RC filter: fc = 1 / (2 pi R C), the capacitor picked so R lands between 1 k and 100 k. A test
    sine at the cutoff on IN (around half the supply) must come out at 70.7 % of its size."""
    best = None
    for exp in range(-11, -3):
        for m in E6:
            c = m * 10 ** exp
            r = nearest(1 / (2 * math.pi * cutoff_hz * c))
            if 1e3 <= r <= 100e3:
                fc = 1 / (2 * math.pi * r * c)
                score = abs(math.log(fc / cutoff_hz)) + 0.02 * abs(math.log(r / 10e3))
                if best is None or score < best[0]:
                    best = (score, r, c, fc)
    if best is None:
        raise ValueError(f"no RC filter with standard parts for {cutoff_hz:g} Hz")
    _, r, c, fc = best
    low = kind == "lowpass"
    parts = [{"ref": "V1", "kind": "V", "value": {"sine": [supply_v / 2, 1.0, fc]}, "pins": {"+": "IN", "-": "GND"}},
             {"ref": "R1", "kind": "R", "value": text(r), "pins": ["IN", "OUT"] if low else ["OUT", "GND"]},
             {"ref": "C1", "kind": "C", "value": text(c), "pins": ["OUT", "GND"] if low else ["IN", "OUT"]}]
    notes = [f"fc = 1 / (2 pi x {text(r)} x {text(c, 'F')}) = {fc:.4g} Hz",
             f"V1 is a test sine at fc (1 V amplitude on {supply_v / 2:g} V): at fc the output is 70.7 % of the "
             f"input; {'below fc it passes, above it falls' if low else 'above fc it passes, below it falls'} "
             f"20 dB per decade"]
    return {"name": f"RC {kind} at {fc:.3g} Hz", "supply_v": supply_v, "parts": parts}, \
        [{"net": "OUT", "of": "IN", "ratio_vpp": 0.707, "tolerance": 0.08}], notes


def opamp_noninverting(gain, supply_v=5.0, input_offset_v=1.0, input_amplitude_v=0.1, input_hz=1000.0):
    """A non-inverting amplifier on one supply: gain = 1 + Rf / Rg. The output of this op-amp swings from about 0 to
    1.5 V below the supply, so the input's offset and size are checked against that."""
    rg = 10e3
    rf = nearest((gain - 1) * rg)
    g = 1 + rf / rg
    top = g * (input_offset_v + input_amplitude_v)
    notes = [f"gain = 1 + Rf / Rg = 1 + {text(rf)} / {text(rg)} = {g:.3g}"]
    if top > supply_v - 1.5 or g * (input_offset_v - input_amplitude_v) < 0.05:
        notes.append(f"the output would need {g * (input_offset_v - input_amplitude_v):.2f} .. {top:.2f} V, outside "
                     f"0 .. {supply_v - 1.5:g} V: it will clip; lower the offset or the gain")
    parts = [{"ref": "V1", "kind": "V", "value": {"sine": [input_offset_v, input_amplitude_v, input_hz]},
              "pins": {"+": "IN", "-": "GND"}},
             {"ref": "U1", "kind": "OPAMP", "pins": {"OUT": "OUT", "IN-": "FB", "IN+": "IN", "V+": "VCC", "V-": "GND"}},
             {"ref": "R1", "kind": "R", "value": text(rf), "pins": ["OUT", "FB"]},
             {"ref": "R2", "kind": "R", "value": text(rg), "pins": ["FB", "GND"]}]
    spec = [{"net": "OUT", "of": "IN", "ratio_vpp": g, "tolerance": 0.08},
            {"net": "OUT", "mean_v": g * input_offset_v, "tolerance": 0.08}]
    return {"name": f"non-inverting amplifier, gain {g:.3g}", "supply_v": supply_v, "parts": parts}, spec, notes


TEMPLATES = {"led_resistor": led_resistor, "astable_555": astable_555, "npn_switch": npn_switch, "divider": divider,
             "rc_filter": rc_filter, "opamp_noninverting": opamp_noninverting}


def verify(sim, spec):
    """[{goal, measured, ok}] for each goal of the spec, read off the simulation."""
    out = []
    nets, currents = sim["nets"], sim.get("currents", {})
    for goal in spec:
        tol = goal.get("tolerance", 0.05)
        if "current_of" in goal:
            got = currents.get(goal["current_of"], {}).get("peak_a")
            ok = got is not None and abs(got / goal["target_a"] - 1) <= tol
            out.append({"goal": f"{goal['current_of']} carries {goal['target_a'] * 1e3:.1f} mA (±{tol:.0%})",
                        "measured": got and f"{got * 1e3:.2f} mA", "ok": ok})
            continue
        m = nets.get(goal["net"], {})
        if "min_v_below" in goal:
            got = m.get("min_v")
            ok = got is not None and got < goal["min_v_below"]
            out.append({"goal": f"{goal['net']} goes below {goal['min_v_below']} V (the switch closes fully)",
                        "measured": got is not None and f"{got:.3f} V", "ok": ok})
            continue
        if "ratio_vpp" in goal:
            ref = nets.get(goal["of"], {})
            got = m.get("vpp_v", 0) / ref["vpp_v"] if ref.get("vpp_v") else None
            ok = got is not None and abs(got / goal["ratio_vpp"] - 1) <= tol
            out.append({"goal": f"{goal['net']} is {goal['ratio_vpp']:.3g} times {goal['of']} peak to peak "
                                f"(±{tol:.0%})", "measured": got and f"{got:.3g} times", "ok": ok})
            continue
        for q in ("frequency_hz", "duty", "mean_v"):
            if q in goal:
                got = m.get(q)
                ok = got is not None and abs(got - goal[q]) <= tol * abs(goal[q])
                out.append({"goal": f"{goal['net']} {q} {goal[q]:.4g} (±{tol:.0%})", "measured": got, "ok": ok})
    return out


def as_spec(goals):
    """A spec as the model writes it ({"net", "frequency_hz", ...}) with numbers, for verify()."""
    out = []
    for g in goals or []:
        g = dict(g)
        for k in ("frequency_hz", "duty", "mean_v", "target_a", "ratio_vpp", "tolerance", "min_v_below"):
            if k in g:
                g[k] = value(g[k]) if isinstance(g[k], str) else float(g[k])
        out.append(g)
    return out
