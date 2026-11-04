"""What is on the shelves of the workshop (config/inventory.json): the values a design should land on, and the parts
Lunes can name when Erlan asks what he has. Data written by Erlan (Lunes adds a line when he dictates one). Without the
file there is no inventory: designs round to the standard series as before, and the tool says so.

- stocked(x, kind, mode) is what designs.nearest asks: the stocked value within STOCK_TOLERANCE of the ideal, else
  None and the standard series decides.
- stock_of(parts) says, part by part, what is on the shelf and what is not (and the nearest value there is).
- summary() is the paragraph the conductor knows, so "what resistors do I have" needs no tool.
"""
import json
import math
from pathlib import Path

FILE = Path(__file__).resolve().parent.parent / "config" / "inventory.json"
STOCK_TOLERANCE = 0.2  # a stocked value this close to the ideal (as a ratio) is used instead of the series value
ON_SHELF = 0.03  # a designed value this close to a stocked one counts as that part
_cache = {"mtime": None, "data": None}


def load():
    """The inventory as saved, or None without a file; read again whenever the file changes."""
    try:
        mtime = FILE.stat().st_mtime
    except OSError:
        return None
    if _cache["mtime"] != mtime:
        _cache.update(mtime=mtime, data=json.loads(FILE.read_text(encoding="utf-8")))
    return _cache["data"]


def values(kind):
    """Sorted stocked values of one kind: "R" (ohm), "C" (farad) or "POT" (ohm). [] without an inventory."""
    inv = load() or {}
    if kind == "R":
        raw = inv.get("resistors_ohm", [])
    elif kind == "POT":
        raw = inv.get("potentiometers_ohm", [])
    elif kind == "C":
        raw = [c.get("farad") for c in inv.get("capacitors", [])]
    else:
        raw = []
    return sorted(float(v) for v in raw if v)


def _closest(have, x):
    return min(have, key=lambda v: abs(math.log(v / x)))


def stocked(x, kind, mode="nearest", tolerance=STOCK_TOLERANCE):
    """The stocked value to use for an ideal x, or None when none is within tolerance: the nearest one, or with
    mode "up" / "down" the nearest one that way (a resistor rounded up keeps an LED's current at or under target)."""
    have = values(kind)
    if not have or x <= 0:
        return None
    if mode == "up":
        options = [v for v in have if v >= x * 0.999]
    elif mode == "down":
        options = [v for v in have if v <= x * 1.001]
    else:
        options = have
    if not options:
        return None
    best = _closest(options, x)
    return best if abs(math.log(best / x)) <= math.log(1 + tolerance) else None


def text(x, unit=""):
    """4700 -> "4.7k", 1e-7 -> "100n" (designs.text, kept here so this module imports nothing of Lunes)."""
    for factor, prefix in ((1e6, "M"), (1e3, "k"), (1, ""), (1e-3, "m"), (1e-6, "u"), (1e-9, "n"), (1e-12, "p")):
        if x >= factor * 0.999:
            return f"{x / factor:.3g}{prefix}{unit}"
    return f"{x:g}{unit}"


def stock_of(parts):
    """{ "R1 2.2k": "on the shelf", "C2 10n": "not on the shelf; the nearest there is 100n" } for the resistors,
    capacitors and pots of a circuit; None without an inventory."""
    if load() is None:
        return None
    from lunes.circuit import value as parse
    out = {}
    for part in parts:
        kind = part.get("kind")
        if kind not in ("R", "C", "POT") or part.get("value") in (None, ""):
            continue
        try:
            x = float(parse(part["value"]))
        except (TypeError, ValueError):
            continue
        have = values(kind)
        unit = "F" if kind == "C" else ""
        label = f"{part.get('ref')} {text(x, unit)}"
        if not have:
            out[label] = f"no {('capacitors' if kind == 'C' else 'pots' if kind == 'POT' else 'resistors')} listed in the inventory"
            continue
        near = _closest(have, x)
        if abs(math.log(near / x)) <= math.log(1 + ON_SHELF):
            out[label] = "on the shelf"
        else:
            out[label] = f"not on the shelf; the nearest there is {text(near, unit)}"
    return out


def listing(search=None):
    """The inventory for the model: values written as on a part list, and every part; search (a word, a value)
    keeps only what mentions it."""
    inv = load()
    if inv is None:
        return {"ok": False, "error": f"there is no inventory ({FILE.name}): Erlan can dictate what he has and it is "
                                      f"written there"}
    out = {"resistors": [text(v) for v in values("R")],
           "capacitors": [dict(c, farad=text(float(c["farad"]), "F")) for c in inv.get("capacitors", [])],
           "potentiometers": [text(v) for v in values("POT")],
           "parts": inv.get("parts", []),
           "notes": [v for k, v in inv.items() if k.endswith("_note")]}
    if search:
        q = str(search).lower()
        hit = lambda item: q in json.dumps(item, ensure_ascii=False).lower()  # noqa: E731
        out = {k: [i for i in v if hit(i)] if isinstance(v, list) else v for k, v in out.items()}
        out["search"] = search
    out["note"] = (f"circuit_design already rounds to these values when one is within {STOCK_TOLERANCE:.0%} of the "
                   f"ideal, and says which parts are not on the shelf. simulated: true means Lunes has a model of it")
    return out


def summary():
    """One Spanish paragraph of what the workshop has, for the conductor's briefing; "" without an inventory."""
    inv = load()
    if inv is None:
        return ""
    parts = [p["part"] for p in inv.get("parts", [])]
    bits = []
    if values("R"):
        bits.append("resistencias de " + ", ".join(text(v) for v in values("R")) + " ohm")
    if inv.get("capacitors"):
        bits.append("capacitores de " + ", ".join(f"{text(float(c['farad']), 'F')} ({c.get('part', '')})"
                                                   for c in inv["capacitors"]))
    if values("POT"):
        bits.append("potenciómetros de " + ", ".join(text(v) for v in values("POT")))
    if parts:
        bits.append("y además " + ", ".join(parts))
    return ("En el taller Erlan tiene (inventario en config/inventory.json, herramienta inventory): " + "; ".join(bits)
            + ". Si falta algo ahí, él lo dicta y se agrega.")
