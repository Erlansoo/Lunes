"""A solderless breadboard: where each hole is in mm, and where each part of a circuit goes, so the laser can show Erlan
where every lead, wire and probe belongs.

The board: rows 1 to 63, columns a to e and f to j with the centre channel between them, every 2.54 mm. The five holes of
a row on one side are one strip. Two rails run along the left edge: + for VCC and - for GND.

plan() lays a circuit out by rule, with no model: chips straddle the channel, each pin's strip takes that pin's net,
every other net gets a strip of its own, and strips of one net are joined with wires. verify() then rebuilds the
connections from the holes alone and compares them with the circuit, so a plan that would not be the circuit is never
shown. Boards and sources stay off the breadboard and come in on wires.

Placement says where the board lies on the machine, from two holes whose position in mm is known: the laser's dot on them.
"""
import math

from lunes import mcu
from lunes.circuit import PIN_NUMBER, value

ROWS, PITCH = 63, 2.54
LEFT, RIGHT = "abcde", "fghij"
COLUMN = {c: i for i, c in enumerate(LEFT)} | {c: 7 + i for i, c in enumerate(RIGHT)}  # the channel takes two pitches
RAIL_COLUMN = {"-": -3, "+": -2}  # beyond column a; a board with other rails only moves these
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
    """Where the board lies: machine mm of any hole, from two known ones, turned and shifted (never scaled)."""

    def __init__(self, hole_a, at_a, hole_b, at_b):
        (ax, ay), (bx, by) = board_xy(hole_a), board_xy(hole_b)
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

    def mm(self, hole):
        x, y = board_xy(hole)
        dx, dy = x - self.origin[0], y - self.origin[1]
        c, s = math.cos(self.angle), math.sin(self.angle)
        return round(self.anchor[0] + c * dx - s * dy, 2), round(self.anchor[1] + s * dx + c * dy, 2)


class _Layout:
    def __init__(self):
        self.net_of, self.used, self.row = {}, set(), 3
        self.wires, self.holes, self.notes = [], {}, []

    def take(self, row, side, net):
        self.net_of[(row, side)] = net
        return row, side

    def free(self, strip):
        row, side = strip
        for column in ("cbade" if side == "L" else "hijgf"):
            hole = f"{column}{row}"
            if hole not in self.used:
                self.used.add(hole)
                return hole
        return None

    def strips(self, net):
        return [s for s, n in self.net_of.items() if n == net]

    def new_strip(self, net):
        while (self.row, "L") in self.net_of:
            self.row += 1
        if self.row > ROWS:
            raise ValueError("the circuit does not fit on one breadboard")
        return self.take(self.row, "L", net)

    def hole_on(self, net, near=None):
        """A free hole of the net: its rail, or one of its strips, adding a strip joined by a wire when they are full."""
        if net in RAIL_OF:
            row = max(1, min(ROWS, near or self.row))
            while f"{RAIL_OF[net]}{row}" in self.used:
                row += 1
            self.used.add(f"{RAIL_OF[net]}{row}")
            return f"{RAIL_OF[net]}{row}"
        strips = self.strips(net) or [self.new_strip(net)]
        for strip in strips:
            hole = self.free(strip)
            if hole:
                return hole
        strip = self.new_strip(net)
        self.wire(self.free(strips[0]) or self.free(strip), self.free(strip), net)
        return self.free(strip)

    def wire(self, a, b, net):
        self.wires.append({"from": a, "to": b, "net": net})


def _dip(layout, part, pins):
    """A chip astride the channel: pin 1 at the top left, numbers running down that side and back up the other."""
    top = layout.row
    for number in range(1, 9):
        row, column, side = (top + number - 1, "e", "L") if number <= 4 else (top + 8 - number, "f", "R")
        layout.take(row, side, pins.get(number))
        hole = f"{column}{row}"
        layout.used.add(hole)
        if number in pins:
            layout.holes.setdefault(part["ref"], {})[number] = hole
    layout.row = top + 6


def _chips(layout, part):
    if part["kind"] == "NE555":
        pins = {PIN_NUMBER["NE555"][name]: net for name, net in part["pins"].items()}
    else:
        pins = {OPAMP_PINS[name]: net for name, net in part["pins"].items()}
    _dip(layout, part, pins)


def legs_of(part):
    """The pins of a three-legged part in the order its legs stand, left to right."""
    kind = part["kind"]
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


def plan(circuit, sim=None):
    """{"holes": {ref: {pin: hole}}, "wires", "probe": {net: hole}, "steps", "notes"} for a circuit on a breadboard."""
    layout = _Layout()
    parts = circuit["parts"]
    for part in parts:
        if part["kind"] in ("NE555", "OPAMP"):
            _chips(layout, part)
    for part in parts:
        if part["kind"] in THREE_LEGS:
            _transistor(layout, part)
    layout.row += 1
    # strips of one net are joined, and a strip of a supply net goes to its rail
    for net in sorted({n for n in layout.net_of.values() if n}):
        strips = layout.strips(net)
        for (row, side) in strips:
            if net in RAIL_OF:
                layout.wire(layout.free((row, side)), layout.hole_on(net, row), net)
        if net not in RAIL_OF:
            first = strips[0]
            for other in strips[1:]:
                layout.wire(layout.free(first), layout.free(other), net)
    for part in parts:
        if part["kind"] in TWO_LEADS:
            a, b = part["pins"]
            hole_a = layout.hole_on(a)
            layout.holes[part["ref"]] = {"1": hole_a, "2": layout.hole_on(b, int(hole_a[1:]))}
        elif part["kind"] in DIODES:
            hole_a = layout.hole_on(part["pins"]["A"])
            layout.holes[part["ref"]] = {"A": hole_a, "K": layout.hole_on(part["pins"]["K"], int(hole_a[1:]))}
    external = []
    for part in parts:
        if part["kind"] in mcu.BOARDS or part["kind"] == "V":
            for pin, net in part["pins"].items():
                hole = layout.hole_on(net)
                layout.holes.setdefault(part["ref"], {})[pin] = hole
                external.append({"ref": part["ref"], "pin": pin, "hole": hole, "net": net})
    probe = {}
    for net in {n for p in parts for n in (p["pins"].values() if isinstance(p["pins"], dict) else p["pins"])}:
        probe[net] = layout.hole_on(net, 2)
    result = {"holes": layout.holes, "wires": layout.wires, "external": external, "probe": probe,
              "steps": _steps(circuit, layout, external, sim)}
    problems = verify(circuit, result)
    if problems:
        raise ValueError("; ".join(problems))
    return result


def _named(part):
    words = {"R": "resistencia", "C": "capacitor", "L": "bobina", "D": "diodo", "ZENER": "zener", "LED": "LED",
             "SW": "interruptor", "LDR": "fotorresistencia", "NTC": "termistor", "POT": "potenciómetro",
             "NMOS": "MOSFET", "PMOS": "MOSFET", "REG": "regulador", "NPN": "transistor", "PNP": "transistor"}
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
    steps = []
    for part in circuit["parts"]:
        ref = part["ref"]
        if part["kind"] in ("NE555", "OPAMP"):
            hole = layout.holes[ref][1] if 1 in layout.holes[ref] else layout.holes[ref][min(layout.holes[ref])]
            steps.append({"kind": "chip", "ref": ref, "holes": [hole],
                          "says": f"Pon el chip {ref} sobre el canal central, con el pin 1 (la muesca a la izquierda) en "
                                  f"{hole}."})
        elif part["kind"] in THREE_LEGS:
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
            steps.append({"kind": "part", "ref": ref, "holes": [a, b], "says": f"Pon {_named(part)} de {a} a {b}{extra}."})
        elif part["kind"] in DIODES:
            a, k = layout.holes[ref]["A"], layout.holes[ref]["K"]
            leg = "la pata larga" if part["kind"] == "LED" else "el lado sin banda"
            steps.append({"kind": "part", "ref": ref, "holes": [a, k],
                          "says": f"Pon {_named(part)}: {leg} (ánodo) en {a} y el otro lado{' (el de la banda)' if part['kind'] != 'LED' else ''} en {k}."})
    for wire in layout.wires:
        steps.append({"kind": "wire", "holes": [wire["from"], wire["to"]],
                      "says": f"Pon un cable de {wire['from']} a {wire['to']} ({wire['net']})."})
    for item in external:
        steps.append({"kind": "external", "ref": item["ref"], "holes": [item["hole"]],
                      "says": f"Conecta {item['pin']} de {item['ref']} en {item['hole']} ({item['net']})."})
    volts = circuit.get("supply_v", 5)
    if any("V" != p["kind"] and p["kind"] not in mcu.BOARDS for p in circuit["parts"]):
        steps.append({"kind": "power", "holes": ["+1", "-1"], "says": f"Con la fuente apagada y en {volts:g} V: positivo "
                                                                      f"al riel +, negativo al riel -."})
    return steps


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
            key = PIN_NUMBER["NE555"][pin] if part["kind"] == "NE555" else OPAMP_PINS[pin] if part["kind"] == "OPAMP" \
                else pin
            hole = holes.get(key)
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


def nearest_hole(placement, x_mm, y_mm, max_mm=1.5):
    """The hole of the placed board nearest a machine point, or None if none is within max_mm: the point turned back
    into the board's frame, then rounded to the grid."""
    c, s = math.cos(placement.angle), math.sin(placement.angle)
    dx, dy = x_mm - placement.anchor[0], y_mm - placement.anchor[1]
    bx = placement.origin[0] + c * dx + s * dy
    by = placement.origin[1] - s * dx + c * dy
    row = int(round(bx / PITCH)) + 1
    across = int(round(by / PITCH))
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
