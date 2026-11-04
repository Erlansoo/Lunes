"""A solderless breadboard: where each hole is in mm, and where each part of a circuit goes, so the laser can show Erlan
where every lead, wire and probe belongs.

The board: rows 1 to 63, columns a to e and f to j with the centre channel between them, every 2.54 mm. The five holes of
a row on one side are one strip. Two rails run along the left edge: + for VCC and - for GND.

plan() lays a circuit out by rule, with no model: chips straddle the channel, each pin's strip takes that pin's net,
every other net gets a strip of its own, and strips of one net are joined with wires. A part with two leads always lies
straight: along a column, across the channel, or from a strip down to a rail, never aslant; only wires run any way, VCC's
red and GND's black. verify() then rebuilds the connections from the holes alone and compares them with the circuit, so
a plan that would not be the circuit is never shown. Boards and sources stay off the breadboard and come in on wires.

Placement says where the board lies on the machine, from two holes whose position in mm is known: the laser's dot on them.
"""
import json
import math

from lunes import catalog, mcu
from lunes.circuit import PIN_NUMBER, value

ROWS, PITCH = 63, 2.54
LEFT, RIGHT = "abcde", "fghij"
COLUMN = {c: i for i, c in enumerate(LEFT)} | {c: 7 + i for i, c in enumerate(RIGHT)}  # the channel takes two pitches
RAIL_COLUMN = {"-": -3, "+": -4}  # beyond column a, as a mosaic of the real board shows: blue first, red at the edge
OPAMP_PINS = {"OUT": 1, "IN-": 2, "IN+": 3, "V-": 4, "V+": 8}  # an LM358's first amplifier
# the legs of a TO-92 as they stand with its flat face towards you, left to right; a TO-220 with its tab away
TRANSISTOR_LEGS = {"BC547": "CBE", "BC557": "CBE", "2N3904": "EBC", "2N3906": "EBC", "2N2222": "EBC",
                   "2N7000": "SGD", "BS170": "SGD", "IRLZ44N": "GDS", "IRF540": "GDS", "IRF9540": "GDS"}
REG_LEGS = {"7805": ("IN", "GND", "OUT"), "7812": ("IN", "GND", "OUT"), "LM1117-3.3": ("GND", "OUT", "IN"),
            "LM1117-5": ("GND", "OUT", "IN")}  # left to right, the marking towards you
THREE_LEGS = ("NPN", "PNP", "NMOS", "PMOS", "POT", "REG")
TWO_LEADS = ("R", "C", "L", "SW", "LDR", "NTC")
DIODES = ("D", "ZENER", "LED")
RAIL_OF = {"VCC": "+", "GND": "-"}
SPAN = {"R": 4, "D": 4, "ZENER": 4, "L": 4, "C": 2, "SW": 2, "LDR": 2, "LED": 1, "NTC": 1}  # pitches between a lying part's leads
WIRE_COLORS = ("azul", "verde", "naranja", "morado", "amarillo", "gris")  # after VCC's red and GND's black


def on_rail(row):
    """Whether a rail has a hole beside this row. As a mosaic of the real board shows, they go in groups of five from row 3
    to row 61, with no hole beside rows 8, 14, 20 and so on."""
    return 3 <= row <= 61 and row % 6 != 2


def board_xy(hole):
    """A hole's place on the board in mm, from hole a1: "c12", or a rail as "+12" (the rail beside row 12)."""
    column, row = hole[0], int(hole[1:])
    across = RAIL_COLUMN[column] if column in RAIL_COLUMN else COLUMN[column]
    return (row - 1) * PITCH, across * PITCH


def strip_of(hole):
    """The hole's electrical node on the board: its row and side, or its rail."""
    if hole[0] in RAIL_COLUMN:
        return "rail" + hole[0]
    return f"{hole[1:]}{'L' if hole[0] in LEFT else 'R'}"


class Placement:
    """Where the board lies: machine mm of any hole, from two known ones, turned and shifted (never scaled). mirrored is
    for a machine whose X and Y make a left-handed pair seen from above: then going up the rows and across them is
    the other way round from any turn of the board, and two holes alone cannot tell."""

    def __init__(self, hole_a, at_a, hole_b, at_b, mirrored=False):
        self.flip = -1.0 if mirrored else 1.0
        (ax, ay), (bx, by) = self.board(hole_a), self.board(hole_b)
        board, machine = (bx - ax, by - ay), (at_b[0] - at_a[0], at_b[1] - at_a[1])
        length = math.hypot(*board)
        if length < 20:
            raise ValueError("the two holes must be at least 20 mm apart for the board's angle to be good")
        self.scale_error = math.hypot(*machine) / length - 1
        if abs(self.scale_error) > 0.04:
            raise ValueError(f"those holes are {math.hypot(*machine):.1f} mm apart on the machine and {length:.1f} mm on "
                             f"a breadboard: one of them is wrong")
        self.angle = math.atan2(machine[1], machine[0]) - math.atan2(board[1], board[0])
        self.origin, self.anchor = (ax, ay), at_a

    def board(self, hole):
        x, y = board_xy(hole)
        return x, self.flip * y

    def mm(self, hole):
        x, y = self.board(hole)
        dx, dy = x - self.origin[0], y - self.origin[1]
        c, s = math.cos(self.angle), math.sin(self.angle)
        return round(self.anchor[0] + c * dx - s * dy, 2), round(self.anchor[1] + s * dx + c * dy, 2)


class BoardGrid:
    """Where the board lies, from three of its corners measured in machine mm: a1, a63 and j1. Unlike Placement it needs
    no word on how the machine's axes turn, and it holds each axis's own scale: on a home-made machine 62 pitches do not
    measure the same along X and along Y."""

    def __init__(self, a1, a63, j1):
        self.a1, self.along, self.across = a1, (a63[0] - a1[0], a63[1] - a1[1]), (j1[0] - a1[0], j1[1] - a1[1])

    def mm(self, hole):
        x, y = board_xy(hole)
        u, v = x / (62 * PITCH), y / (11 * PITCH)
        return round(self.a1[0] + u * self.along[0] + v * self.across[0], 2), round(self.a1[1] + u * self.along[1] + v * self.across[1], 2)

    def across_of(self, x, y):
        """How far across the board a point in machine mm lies, in pitches from column a (j is 11, the channel 5.5)."""
        (lx, ly), (cx, cy) = self.along, self.across
        return 11 * (lx * (y - self.a1[1]) - ly * (x - self.a1[0])) / (lx * cy - ly * cx)

    def holds_chip(self, x, y):
        """Whether a DIP centred here straddles this grid's channel, as every DIP on a breadboard does: a grid that puts
        it elsewhere took the rails or the board's edge for holes (Nov 3: one put the chip on column a, 13 mm off)."""
        return abs(self.across_of(x, y) - 5.5) <= 1.0


class _Layout:
    def __init__(self):
        self.net_of, self.used, self.covered, self.row = {}, set(), set(), 3
        self.color_of = {"VCC": "rojo", "GND": "negro"}
        self.wires, self.holes, self.notes, self.first_pin, self.chip_pins = [], {}, [], {}, {}

    def take(self, row, side, net):
        self.net_of[(row, side)] = net
        return row, side

    def free(self, strip):
        row, side = strip
        holes = [f"{column}{row}" for column in ("cbade" if side == "L" else "hijgf")]
        for hole in sorted(holes, key=lambda h: h in self.covered):  # one under a part's body, only if nothing else is left
            if hole not in self.used:
                self.used.add(hole)
                return hole
        return None

    def room(self, strip):
        """Whether a part can land on a strip: two holes stay free on each for the wires that join it to its net."""
        row, side = strip
        return sum(f"{column}{row}" not in self.used for column in (LEFT if side == "L" else RIGHT)) >= 3

    def strips(self, net):
        return [s for s, n in self.net_of.items() if n == net]

    def new_strip(self, net):
        while (self.row, "L") in self.net_of:
            self.row += 1
        if self.row > ROWS:
            raise ValueError("the circuit does not fit on one breadboard")
        return self.take(self.row, "L", net)

    def fresh_row(self, span=0):
        """The first row from here on with its strips free on both sides, and span more like it after it."""
        row = self.row
        while any((r, side) in self.net_of for r in range(row, row + span + 1) for side in "LR"):
            row += 1
        if row + span > ROWS:
            raise ValueError("the circuit does not fit on one breadboard")
        return row

    def rail_row(self, near, rails=("+", "-")):
        """The row nearest this one where these rails have a free hole."""
        rows = [r for r in range(1, ROWS + 1) if on_rail(r) and not {f"{rail}{r}" for rail in rails} & self.used]
        if not rows:
            raise ValueError("the circuit does not fit on one breadboard")
        return min(rows, key=lambda r: abs(r - near))

    def hole_on(self, net, near=None):
        """A free hole of the net: its rail, or one of its strips, adding a strip when they are full; join() wires it."""
        if net in RAIL_OF:
            row = self.rail_row(near or self.row, [RAIL_OF[net]])
            self.used.add(f"{RAIL_OF[net]}{row}")
            return f"{RAIL_OF[net]}{row}"
        for strip in self.strips(net):
            if self.room(strip):
                return self.free(strip)
        return self.free(self.new_strip(net))

    def wire(self, a, b, net):
        color = self.color_of.setdefault(net, WIRE_COLORS[(len(self.color_of) - 2) % len(WIRE_COLORS)])
        self.wires.append({"from": a, "to": b, "net": net, "color": color})

    def join(self):
        """Every strip of a supply goes to its rail, and the strips of any other net are chained."""
        for net in sorted({n for n in self.net_of.values() if n}):
            strips = self.strips(net)
            if net in RAIL_OF:
                for strip in strips:  # a chip's supply pins often face each other: their wires must not lie one over the other
                    self.wire(self.free(strip), self.hole_on(net, self.rail_row(strip[0])), net)
            for one, other in ([] if net in RAIL_OF else zip(strips, strips[1:])):
                self.wire(self.free(one), self.free(other), net)


def _dip(layout, part, pins, count=8):
    """A chip astride the channel: pin 1 at the top left, numbers running down that side and back up the other."""
    top, half = layout.row, count // 2
    layout.first_pin[part["ref"]] = f"e{top}"
    layout.chip_pins[part["ref"]] = count
    for number in range(1, count + 1):
        row, column, side = (top + number - 1, "e", "L") if number <= half else (top + count - number, "f", "R")
        layout.take(row, side, pins.get(number))
        hole = f"{column}{row}"
        layout.used.add(hole)
        if number in pins:
            layout.holes.setdefault(part["ref"], {})[number] = hole
    layout.row = top + half + 2


def _chips(layout, part):
    if part["kind"] == "IC":
        entry = catalog.get(part["model"])
        numbers = catalog.pin_numbers(entry)
        _dip(layout, part, {numbers[name]: net for name, net in part["pins"].items()}, len(numbers))
        return
    if part["kind"] == "NE555":
        pins = {PIN_NUMBER["NE555"][name]: net for name, net in part["pins"].items()}
    else:
        pins = {OPAMP_PINS[name]: net for name, net in part["pins"].items()}
    _dip(layout, part, pins)


def is_chip(part):
    """A part that straddles the channel."""
    return part["kind"] in ("NE555", "OPAMP") or (part["kind"] == "IC" and catalog.is_dip(catalog.get(part["model"])))


def is_three_legged(part):
    return part["kind"] in THREE_LEGS or (part["kind"] == "IC" and not catalog.is_dip(catalog.get(part["model"])))


def legs_of(part):
    """The pins of a three-legged part in the order its legs stand, left to right."""
    kind = part["kind"]
    if kind == "IC":
        return tuple(name for _, name in catalog.ordered_pins(catalog.get(part["model"])))
    if kind == "POT":
        return ("A", "W", "B")  # the wiper is the middle leg of a trimmer and of a panel pot
    if kind == "REG":
        return REG_LEGS.get(part.get("model"), ("IN", "GND", "OUT"))
    return tuple(TRANSISTOR_LEGS.get(part.get("model"), "EBC" if kind in ("NPN", "PNP") else "GDS"))


def _transistor(layout, part):
    """Any three-legged part down one side, a row per leg, in the order its legs stand."""
    top = layout.row
    for k, leg in enumerate(legs_of(part)):
        layout.take(top + k, "L", part["pins"][leg])
        layout.used.add(f"e{top + k}")
        layout.holes.setdefault(part["ref"], {})[leg] = f"e{top + k}"
    layout.row = top + 4


def _along(layout, strip_a, strip_b):
    """A part's holes along one column of two strips of a side, or None; the holes under its body stay out of the way."""
    (row_a, side), (row_b, _) = strip_a, strip_b
    rows = range(min(row_a, row_b), max(row_a, row_b) + 1)
    for column in ("cbad" if side == "L" else "hijg"):
        if not any(f"{column}{row}" in layout.used | layout.covered for row in rows):
            layout.covered.update(f"{column}{row}" for row in rows)
            layout.used.update((f"{column}{row_a}", f"{column}{row_b}"))
            return f"{column}{row_a}", f"{column}{row_b}"
    return None


def hole_words(hole, volts=5, prep=None):
    """A hole said so Erlan understands it: "+18" is not 18 volts but the hole of the + rail beside row 18, so it is
    said as "el riel + (5 V), fila 18"; a strip hole such as "c12" stays "c12". prep "de", "a" or "en" puts the
    preposition in front, contracted as Spanish wants ("del riel", "al riel", "de c12")."""
    hole = str(hole)
    if hole[:1] == "+":
        words, article = f"riel + ({volts:g} V), fila {hole[1:]}", True
    elif hole[:1] == "-":
        words, article = f"riel - (GND), fila {hole[1:]}", True
    else:
        words, article = hole, False
    if prep is None:
        return f"el {words}" if article else words
    if article:
        return {"de": "del ", "a": "al ", "en": "en el "}[prep] + words
    return f"{prep} {words}"


def _to_rail(layout, strip, net, span):
    """A part's holes straight from a strip on the rails' side down to its rail, or None: from the strip's hole that
    stands closest to the part's own length from the rail, since a long body does not fit between it and column a."""
    row, side = strip
    rail = f"{RAIL_OF[net]}{row}"
    if side != "L" or rail in layout.used or not on_rail(row):
        return None
    for column in sorted("abcd", key=lambda c: abs(COLUMN[c] - RAIL_COLUMN[RAIL_OF[net]] - span)):
        hole = f"{column}{row}"
        if hole not in layout.used:
            layout.used.update((hole, rail, f"-{row}"))  # the way to the far rail passes over the near one
            layout.covered.update(f"{c}{row}" for c in LEFT if COLUMN[c] < COLUMN[column])
            return hole, rail
    return None


def _lay_down(layout, part, a, b):
    """The holes of a two-lead part between nets a and b, in that order, always straight. A supply: down to its rail. Two
    other nets: between strips they have, if some stand a part's length apart; else from one of them to a new strip;
    else on new strips, across the channel if the part is long enough, or along a column. Wires do the rest."""
    span = SPAN[part["kind"]]
    if a in RAIL_OF and b in RAIL_OF and a != b:
        row = layout.rail_row(layout.row)
        layout.used.update((f"+{row}", f"-{row}"))
        return f"{RAIL_OF[a]}{row}", f"{RAIL_OF[b]}{row}"
    if (a in RAIL_OF) != (b in RAIL_OF):
        supply, other = (a, b) if a in RAIL_OF else (b, a)
        for strip in layout.strips(other):
            holes = layout.room(strip) and _to_rail(layout, strip, supply, span)
            if holes:
                return holes if b == supply else holes[::-1]
        row = layout.fresh_row()
        while f"{RAIL_OF[supply]}{row}" in layout.used or not on_rail(row):
            layout.row, row = row + 1, layout.fresh_row(1)
        holes = _to_rail(layout, layout.take(row, "L", other), supply, span)
        layout.row = row + 1
        return holes if b == supply else holes[::-1]
    for strip_a in layout.strips(a):
        for strip_b in layout.strips(b):
            if strip_a[1] == strip_b[1] and span <= abs(strip_a[0] - strip_b[0]) <= span + 2 and layout.room(strip_a)                     and layout.room(strip_b):
                holes = _along(layout, strip_a, strip_b)
                if holes:
                    return holes
    for near, far in ((a, b), (b, a)):
        for strip in layout.strips(near):
            for row in (strip[0] + span, strip[0] - span):
                if layout.room(strip) and 1 <= row <= ROWS and (row, strip[1]) not in layout.net_of:
                    holes = _along(layout, strip, (row, strip[1]))
                    if holes:
                        layout.take(row, strip[1], far)
                        return holes if near == a else holes[::-1]
    if span >= 3:
        row = layout.fresh_row()
        layout.take(row, "L", a)
        layout.take(row, "R", b)
        layout.used.update((f"e{row}", f"f{row}"))
        layout.row = row + 2  # a row between two parts across the channel: their bodies do not fit side by side
        return f"e{row}", f"f{row}"
    row = layout.fresh_row(span)
    strip_a, strip_b = layout.take(row, "L", a), layout.take(row + span, "L", b)
    layout.row = row + span + 1
    return _along(layout, strip_a, strip_b)


def plan(circuit, sim=None, first_row=None):
    """{"holes": {ref: {pin: hole}}, "wires", "probe": {net: hole}, "steps", "notes"} for a circuit on a breadboard.
    first_row is the row of a chip's pin 1 when it already stands on the board: the plan is laid out around it."""
    layout = _Layout()
    parts = circuit["parts"]
    layout.row = first_row or layout.row
    for part in parts:
        if is_chip(part):
            _chips(layout, part)
    if first_row:  # the rest beside the chip, on whichever side of it the board has rows left
        # before the chip only when a whole block fits there; else after it. Starting at row 1 with the chip at row 3
        # laid parts across the chip's own strips (3 Nov: "OUT_LOW split, VCC and OUT_LOW shorted")
        last = max((r for r, _ in layout.net_of), default=first_row)
        layout.row = first_row - 14 if first_row - 14 >= 1 else last + 2
    for part in parts:
        if is_three_legged(part):
            _transistor(layout, part)
    layout.row += 1
    for part in parts:
        if part["kind"] in TWO_LEADS:
            a, b = part["pins"]
            layout.holes[part["ref"]] = dict(zip(("1", "2"), _lay_down(layout, part, a, b)))
        elif part["kind"] in DIODES:
            pins = part["pins"]
            layout.holes[part["ref"]] = dict(zip(("A", "K"), _lay_down(layout, part, pins["A"], pins["K"])))
    external = []
    for part in parts:
        if part["kind"] in mcu.BOARDS or part["kind"] == "V":
            for pin, net in part["pins"].items():
                hole = layout.hole_on(net)
                layout.holes.setdefault(part["ref"], {})[pin] = hole
                external.append({"ref": part["ref"], "pin": pin, "hole": hole, "net": net})
    probe = {}
    for net in sorted({n for p in parts for n in (p["pins"].values() if isinstance(p["pins"], dict) else p["pins"])}):
        probe[net] = layout.hole_on(net, 2)
    layout.join()
    result = {"holes": layout.holes, "wires": layout.wires, "external": external, "probe": probe,
              "chips": {ref: {"first": hole, "pins": layout.chip_pins[ref]} for ref, hole in layout.first_pin.items()},
              "steps": _steps(circuit, layout, external, sim)}
    problems = verify(circuit, result)
    if problems:
        raise ValueError("; ".join(problems))
    return result


def _named(part):
    words = {"R": "resistencia", "C": "capacitor", "L": "bobina", "D": "diodo", "ZENER": "zener", "LED": "LED",
             "SW": "interruptor", "LDR": "fotorresistencia", "NTC": "termistor", "POT": "potenciómetro",
             "NMOS": "MOSFET", "PMOS": "MOSFET", "REG": "regulador", "NPN": "transistor", "PNP": "transistor",
             "IC": "integrado"}
    detail = part.get("value", part.get("model", part.get("color", "")))
    if part["kind"] in ("LDR", "NTC"):
        detail = "/".join(str(part.get(k, "?")) for k in (("dark", "lit") if part["kind"] == "LDR" else ("cold", "hot")))
    return f"{words[part['kind']]} {part['ref']} ({detail})"


def _capacitor_note(part, sim):
    """Which way an electrolytic goes: its minus to the net with the lower voltage."""
    if value(part["value"]) < 1e-6:
        return ""
    a, b = part["pins"]
    nets = (sim or {}).get("nets", {})
    low = a if nets.get(a, {}).get("mean_v", 0) <= nets.get(b, {}).get("mean_v", 0) else b
    return f"; es electrolítico: su pata negativa va a {low}"


def _steps(circuit, layout, external, sim):
    steps, volts = [], circuit.get("supply_v", 5)
    for part in circuit["parts"]:
        ref = part["ref"]
        if is_chip(part):
            hole = layout.first_pin[ref]
            steps.append({"kind": "chip", "ref": ref, "holes": [hole],
                          "says": f"Pon el chip {ref} sobre el canal central, con el pin 1 (la muesca a la izquierda) en "
                                  f"{hole}."})
        elif is_three_legged(part):
            holes = layout.holes[ref]
            order = " ".join(f"{leg} en {hole}" for leg, hole in holes.items())
            hint = {"POT": " El cursor es la pata del medio.", "REG": " Con la marca hacia ti, de izquierda a derecha.",
                    "NMOS": " Revisa su hoja de datos.", "PMOS": " Revisa su hoja de datos."}.get(part["kind"],
                                                                                                   " Revisa su hoja de datos.")
            steps.append({"kind": "transistor", "ref": ref, "holes": list(holes.values()),
                          "says": f"Pon {_named(part)}: {order}.{hint}"})
    for part in circuit["parts"]:
        ref = part["ref"]
        if part["kind"] in TWO_LEADS:
            a, b = layout.holes[ref]["1"], layout.holes[ref]["2"]
            extra = _capacitor_note(part, sim) if part["kind"] == "C" else ""
            steps.append({"kind": "part", "ref": ref, "holes": [a, b],
                          "says": f"Pon {_named(part)} {hole_words(a, volts, 'de')} {hole_words(b, volts, 'a')}{extra}."})
        elif part["kind"] in DIODES:
            a, k = layout.holes[ref]["A"], layout.holes[ref]["K"]
            leg = "la pata larga" if part["kind"] == "LED" else "el lado sin banda"
            steps.append({"kind": "part", "ref": ref, "holes": [a, k],
                          "says": f"Pon {_named(part)}: {leg} (ánodo) {hole_words(a, volts, 'en')} y el otro lado"
                                  f"{' (el de la banda)' if part['kind'] != 'LED' else ''} {hole_words(k, volts, 'en')}."})
    for wire in layout.wires:
        steps.append({"kind": "wire", "holes": [wire["from"], wire["to"]],
                      "says": f"Pon un cable {wire['color']} {hole_words(wire['from'], volts, 'de')} "
                              f"{hole_words(wire['to'], volts, 'a')} ({wire['net']})."})
    for item in external:
        steps.append({"kind": "external", "ref": item["ref"], "holes": [item["hole"]],
                      "says": f"Conecta {item['pin']} de {item['ref']} {hole_words(item['hole'], volts, 'en')} ({item['net']})."})
    if any("V" != p["kind"] and p["kind"] not in mcu.BOARDS for p in circuit["parts"]):
        steps.append({"kind": "power", "holes": ["+3", "-3"], "says": f"Con la fuente apagada y en {volts:g} V: positivo "
                                                                      f"al riel +, negativo al riel -."})
    return steps


def pin_key(part, pin):
    """The key of a part's pin in plan()'s holes."""
    if part["kind"] == "NE555":
        return PIN_NUMBER["NE555"][pin]
    if part["kind"] == "OPAMP":
        return OPAMP_PINS[pin]
    return catalog.pin_numbers(catalog.get(part["model"]))[pin] if is_chip(part) and part["kind"] == "IC" else pin


def verify(circuit, result):
    """What differs between the plan and the circuit, rebuilt from the holes alone: a wrong plan is never shown."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for wire in result["wires"]:
        parent[find(strip_of(wire["from"]))] = find(strip_of(wire["to"]))
    nodes, problems = {}, []
    for part in circuit["parts"]:
        ref, pins = part["ref"], part["pins"]
        items = list(zip(("1", "2"), pins)) if isinstance(pins, list) else list(pins.items())
        holes = result["holes"].get(ref, {})
        for pin, net in items:
            key = pin_key(part, pin)
            hole = holes.get(key, holes.get(str(key)))  # a plan read back from JSON has its chip pins as text
            if hole is None:
                problems.append(f"{ref} {pin} has no hole")
                continue
            nodes.setdefault(net, set()).add(find(strip_of(hole)))
    for net, found in nodes.items():
        if len(found) > 1:
            problems.append(f"net {net} is split on the board")
    seen = {}
    for net, found in nodes.items():
        for node in found:
            if node in seen and seen[node] != net:
                problems.append(f"nets {net} and {seen[node]} are shorted on the board")
            seen[node] = net
    return problems


def rebuild(circuit, edited, sim=None):
    """The circuit as the breadboard now has it, after Erlan moved leads or added and removed wires by hand in the web
    lab: (new circuit, its plan, what changed). edited is a plan ({"holes": {ref: {pin: hole}}, "wires": [{"from",
    "to", "color"?}]}); every lead's net comes from the strips and wires it touches, so the board is the truth. A node
    keeps the name of the net most of its leads had (VCC on the + rail, GND on the - rail); a net split in two keeps
    its name on the larger part and the other becomes NAME_2; nets joined by a new wire take one name."""
    holes = {ref: {str(k): v for k, v in pins.items()} for ref, pins in (edited.get("holes") or {}).items()}
    wires = [dict(w) for w in (edited.get("wires") or []) if w.get("from") and w.get("to")]
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for w in wires:
        parent[find(strip_of(w["from"]))] = find(strip_of(w["to"]))
    leads, problems = [], []
    for part in circuit["parts"]:
        pins = part["pins"]
        items = list(zip(("1", "2"), pins)) if isinstance(pins, list) else list(pins.items())
        for pin, net in items:
            hole = holes.get(part["ref"], {}).get(str(pin_key(part, pin)))
            if hole is None:
                problems.append(f"{part['ref']} {pin} has no hole")
                continue
            leads.append((part, pin, net, find(strip_of(hole))))
    if problems:
        raise ValueError("; ".join(problems))
    # name each node: the rails are VCC and GND; otherwise the net most of its leads had
    names, taken = {}, set()
    rail_net = {find("rail" + r): net for net, r in RAIL_OF.items() if ("rail" + r) in parent}
    for node, net in rail_net.items():
        names[node] = net
        taken.add(net)
    by_node = {}
    for part, pin, net, node in leads:
        by_node.setdefault(node, []).append(net)
    for node, nets in sorted(by_node.items(), key=lambda kv: -len(kv[1])):
        if node in names:
            continue
        best = max(set(nets), key=nets.count)
        name, k = best, 2
        while name in taken:
            name, k = f"{best}_{k}", k + 1
        names[node] = name
        taken.add(name)
    new = json.loads(json.dumps(circuit))
    changes = []
    by_ref = {p["ref"]: p for p in new["parts"]}
    for part, pin, net, node in leads:
        target = by_ref[part["ref"]]
        if isinstance(target["pins"], list):
            target["pins"][int(pin) - 1] = names[node]
        else:
            target["pins"][pin] = names[node]
        if names[node] != net:
            changes.append(f"{part['ref']} {pin}: {net} -> {names[node]}")
    for w in wires:
        w["net"] = names.get(find(strip_of(w["from"])), "?")
        w.setdefault("color", "amarillo")
    first_pin = {ref: h.get("1") for ref, h in holes.items() if is_chip(by_ref.get(ref, {"kind": ""}))}
    external = [dict(e, net=names.get(find(strip_of(e["hole"])), e.get("net"))) for e in (edited.get("external") or [])]
    probe = {}
    for part, pin, net, node in leads:
        probe.setdefault(names[node], holes[part["ref"]][str(pin_key(part, pin))])
    view = type("Edited", (), {})()
    view.holes, view.wires, view.first_pin = holes, wires, first_pin
    result = {"holes": holes, "wires": wires, "external": external, "probe": probe,
              "chips": {ref: {"first": hole, "pins": len(holes[ref])} for ref, hole in first_pin.items()},
              "steps": _steps(new, view, external, sim), "edited_by_hand": True}
    problems = verify(new, result)
    if problems:  # cannot happen when the names come from the board itself; said anyway rather than hidden
        raise ValueError("; ".join(problems))
    return new, result, changes


def nearest_hole(placement, x_mm, y_mm, max_mm=1.5):
    """The hole of the placed board nearest a machine point, or None if none is within max_mm: the point turned back
    into the board's frame, then rounded to the grid."""
    if isinstance(placement, BoardGrid):
        holes = [f"{c}{r}" for r in range(1, ROWS + 1) for c in COLUMN] + [f"{c}{r}" for r in range(1, ROWS + 1) for c in RAIL_COLUMN if on_rail(r)]
        hole = min(holes, key=lambda h: math.dist(placement.mm(h), (x_mm, y_mm)))
        return hole if math.dist(placement.mm(hole), (x_mm, y_mm)) <= max_mm else None
    c, s = math.cos(placement.angle), math.sin(placement.angle)
    dx, dy = x_mm - placement.anchor[0], y_mm - placement.anchor[1]
    bx = placement.origin[0] + c * dx + s * dy
    by = placement.origin[1] - s * dx + c * dy
    row = int(round(bx / PITCH)) + 1
    across = int(round(placement.flip * by / PITCH))
    columns = {v: k for k, v in COLUMN.items()} | {v: k for k, v in RAIL_COLUMN.items()}
    if row < 1 or row > ROWS or across not in columns:
        return None
    hole = f"{columns[across]}{row}"
    hx, hy = placement.mm(hole)
    return hole if math.hypot(hx - x_mm, hy - y_mm) <= max_mm else None


def chip_pin1_hole(layout, ref):
    """The hole the plan gave a chip's pin 1, or None."""
    holes = layout["holes"].get(ref, {})
    return holes.get(1)
