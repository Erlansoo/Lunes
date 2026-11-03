"""Chips as data, not code: a new chip is a file here, so the simulation, the review, the faults, the breadboard and
the diagnosis all work with it without a line written for it.

An entry in lunes/catalog/<NAME>.json holds what a datasheet gives: the package, what each pin is called, which pins
take power, which are inputs and outputs, the supply's range, notes for whoever uses it, and a SPICE subcircuit with
its nodes in pin-number order. It also holds a test: a small circuit with the chip and the goals the simulation must
meet. Nothing enters the catalog without passing that test, whether Erlan, Claude or Lunes itself wrote the entry.

Lunes learns a chip by itself (add): the model reads the datasheet and writes the entry, and the code decides. What
the model writes runs in ngspice, and a netlist can carry commands that reach the computer (.control, .include,
shell), so only plain electrical cards are accepted. No model ever writes code here: a chip is data.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FOLDER = Path(__file__).resolve().parent / "catalog"
LIB = ROOT / "runs" / "catalog.lib"
PACKAGES = {"TO-92": 3, "TO-220": 3}  # DIPn says its own count
CARD = re.compile(r"^[RCLDQMJBEGFHVISKXA]\w*\s")  # a line of a subcircuit: an electrical part, nothing else
DOTS = (".subckt", ".ends", ".model", ".param", ".func")
BANNED = re.compile(r"\b(shell|system|exec|control|endc|include|lib|wrdata|write|load|source|echo|quit|run)\b", re.I)
_entries = None


def _name(text):
    return re.sub(r"\s+", "", str(text or "")).upper()


def spice_name(name):
    """The subcircuit's name: a SPICE name cannot safely start with a digit, and 74HC00 does."""
    return "CHIP_" + re.sub(r"[^A-Z0-9]", "_", _name(name))


def _load():
    global _entries
    if _entries is None:
        _entries = {}
        for path in sorted(FOLDER.glob("*.json")):
            entry = json.loads(path.read_text(encoding="utf-8"))
            _entries[_name(entry["name"])] = entry
    return _entries


def get(name):
    """The chip's entry, by its name or another it goes by (LM393N, 74LS00), or None."""
    wanted = _name(name)
    entries = _load()
    if wanted in entries:
        return entries[wanted]
    return next((e for e in entries.values() if wanted in {_name(a) for a in e.get("aka", [])}), None)


def names():
    return sorted(_load())


def pin_numbers(entry):
    """{pin name: pin number}."""
    return {name: int(number) for number, name in entry["pins"].items()}


def ordered_pins(entry):
    """[(number, name)] in pin-number order: the order of the subcircuit's nodes."""
    return sorted(((int(n), name) for n, name in entry["pins"].items()))


def is_dip(entry):
    return entry["package"].startswith("DIP")


def lib_path():
    """The file ngspice includes, with every chip's subcircuit; written when it is missing."""
    if not LIB.exists():
        write_lib()
    return LIB


def write_lib():
    LIB.parent.mkdir(exist_ok=True)
    LIB.write_text("* Lunes' chip catalog, written from lunes/catalog/*.json\n" +
                   "\n".join(e["spice"] for _, e in sorted(_load().items())) + "\n", encoding="ascii", errors="replace")
    return LIB


def fingerprint():
    """Changes when any chip changes, so cached fault dictionaries are simulated again."""
    return json.dumps({k: e["spice"] for k, e in sorted(_load().items())}, sort_keys=True)


def safe_spice(text, name, pin_count):
    """What is wrong with a subcircuit's text, or None. It runs in ngspice, and may come from a web page or a model."""
    lines = [line.strip() for line in str(text).replace("\r", "").split("\n") if line.strip()]
    if not lines:
        return "the subcircuit is empty"
    header = lines[0].split()
    if len(header) < 2 or header[0].lower() != ".subckt" or header[1].upper() != spice_name(name):
        return f"it must start with .subckt {spice_name(name)}, then its nodes in pin-number order"
    nodes = [w for w in header[2:] if "=" not in w]
    if len(nodes) != pin_count:
        return f"the subcircuit has {len(nodes)} nodes and the chip {pin_count} pins"
    if lines[-1].split()[0].lower() != ".ends":
        return "it must end with .ends"
    for line in lines:
        if line.startswith(("*", "+")):
            if line.startswith("+") and BANNED.search(line):
                return f"not allowed in a subcircuit: {line[:50]}"
            continue
        word = line.split()[0].lower()
        if word.startswith("."):
            if word not in DOTS:
                return f"not allowed in a subcircuit: {word}; only {', '.join(DOTS)}"
        elif not CARD.match(line):
            return f"not an electrical part: {line[:50]}"
        if BANNED.search(line):
            return f"not allowed in a subcircuit: {line[:50]}"
    return None


def problems(entry):
    """What is wrong with an entry, in words, before anything of it runs."""
    if not isinstance(entry, dict):
        return ["an entry is an object"]
    missing = [k for k in ("name", "package", "pins", "power", "spice", "test") if k not in entry]
    if missing:
        return [f"missing: {', '.join(missing)}"]
    out = []
    name, package, pins = entry["name"], str(entry["package"]), entry["pins"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\-]{1,23}", str(name)):
        out.append(f"name {name!r}: letters, digits and dashes")
    dip = re.fullmatch(r"DIP(\d+)", package)
    count = int(dip.group(1)) if dip else PACKAGES.get(package)
    if count is None or (dip and (count % 2 or not 4 <= count <= 40)):
        out.append(f"package {package!r}: DIP4 to DIP40, TO-92 or TO-220")
    if not isinstance(pins, dict):
        return out + ['pins maps pin numbers to names, like {"1": "OUT1"}']
    if count is not None and sorted(pins, key=lambda n: int(n) if str(n).isdigit() else 0) != [str(i + 1) for i in range(count)]:
        out.append(f"a {package} has pins 1 to {count}, each named once")
    labels = [str(v) for v in pins.values()]
    if len(set(labels)) != len(labels):
        out.append("two pins share a name: call them apart (OUT1, OUT2)")
    for key in ("power", "inputs", "outputs"):
        unknown = [p for p in entry.get(key, []) if p not in labels]
        if unknown:
            out.append(f"{key} names pins the chip does not have: {unknown}")
    for a, b in entry.get("dc_links", []):
        if a not in labels or b not in labels:
            out.append(f"dc_links names pins the chip does not have: {a}, {b}")
    if not entry["power"]:
        out.append("power names the pins that must be connected: the positive one first, then ground")
    if count is not None:
        bad = safe_spice(entry["spice"], name, count)
        if bad:
            out.append(bad)
    test = entry["test"]
    if not isinstance(test, dict) or not test.get("circuit") or not test.get("goals"):
        out.append("test needs a circuit with this chip and the goals the simulation must meet: nothing enters untested")
    return out


def prove(entry):
    """{"ok", "goals" or "error"}: the entry's own test, run with the entry in the catalog for a moment."""
    from lunes import review
    from lunes.circuit import check
    from lunes.designs import as_spec, verify
    from lunes.simulator import SimulationError, expected
    circuit, goals = entry["test"]["circuit"], entry["test"]["goals"]
    if not any(p.get("kind") == "IC" and get(p.get("model")) is entry for p in circuit.get("parts", [])):
        return {"ok": False, "error": "the test circuit must use this chip: a part of kind IC with its model"}
    bad = check(circuit)
    if bad:
        return {"ok": False, "error": "the test circuit is not valid", "problems": bad}
    errors, _ = review.before(circuit)
    if errors:
        return {"ok": False, "error": "the test circuit's wiring is wrong", "problems": errors}
    try:
        sim = expected(circuit)
    except SimulationError as e:
        return {"ok": False, "error": "ngspice could not simulate it", "problems": e.problems}
    results = verify(sim, as_spec(goals))
    good = bool(results) and all(g["ok"] for g in results)
    return {"ok": good, "goals": results, **({} if good else {"error": "the simulation does not meet the test's goals"})}


def add(entry, origin="learnt"):
    """Puts a chip in the catalog if it is sound and passes its test: {"ok", "name"} or why not. A chip already there
    is replaced only by one that passes, and a failed attempt leaves everything as it was."""
    bad = problems(entry)
    if bad:
        return {"ok": False, "error": "the entry is not valid", "problems": bad}
    entries = _load()
    key = _name(entry["name"])
    before = entries.get(key)
    entry = dict(entry, origin=origin)  # set here, never by the entry: learnt chips stay marked as learnt
    entries[key] = entry
    write_lib()
    try:
        result = prove(entry)
    except Exception as e:  # noqa: BLE001 - a bad entry is an answer for the model, never a crash
        result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    if not result["ok"]:
        if before is None:
            del entries[key]
        else:
            entries[key] = before
        write_lib()
        return result
    FOLDER.mkdir(exist_ok=True)
    (FOLDER / f"{re.sub(r'[^A-Z0-9-]', '_', key)}.json").write_text(json.dumps(entry, ensure_ascii=False, indent=1) + "\n",
                                                                    encoding="utf-8")
    return dict(result, name=entry["name"], note="in the catalog from now on: use it as a part of kind IC")
