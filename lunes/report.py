"""The report of a circuit checked on the bench, as Markdown in runs/reports/: what Erlan asked, the parts and what
is on the shelf, what the simulation says net by net, what was measured (the scope, his multimeter, the board's own
inputs, the supply's display), the two side by side with the criterion behind every "agrees" or "differs", the
diagnosis step by step with its verdict, and the pictures. Written by itself when a diagnosis reaches a verdict, and
by the report tool when Erlan asks. The web lab shows the newest one on its report page (runs/reports/latest.md).

Everything here is read off Hume's memory of the session; nothing is computed anew, so the report says only what
the tools said.
"""
import math
import re
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "runs" / "reports"
# the criterion for "agrees": the simulation's own tolerances (designs.py specs), and the parts' real tolerances
TOLERANCE = {"frequency_hz": 0.08, "duty": 0.06, "volts_ratio": 0.05, "volts_floor": 0.15}


def _si(x, unit):
    if x is None:
        return "-"
    for factor, prefix in ((1e6, "M"), (1e3, "k"), (1, "")):
        if abs(x) >= factor * 0.999:
            return f"{x / factor:.3g} {prefix}{unit}"
    return f"{x:.3g} {unit}"


def _v(x):
    return "-" if x is None else f"{x:.2f} V"


def _pct(x):
    return "-" if x is None else f"{x * 100:.0f} %"


def compare(simulated, measured):
    """[(what, simulated as text, measured as text, difference, agrees)] for one net, by the criterion:
    frequency within 8 %, duty within 6 points, volts within 5 % plus 0.15 V."""
    rows = []
    s, g = simulated.get("frequency_hz"), measured.get("frequency_hz")
    if "frequency_hz" in measured:
        if s and g:
            ok = abs(math.log(g / s)) <= math.log(1 + TOLERANCE["frequency_hz"])
            rows.append(("frequency", _si(s, "Hz"), _si(g, "Hz"), f"{(g / s - 1) * 100:+.0f} %", ok))
        else:
            rows.append(("frequency", _si(s, "Hz") if s else "steady", _si(g, "Hz") if g else "steady", "-", bool(s) == bool(g)))
    s, g = simulated.get("duty"), measured.get("duty")
    if s is not None and g is not None:
        rows.append(("duty", _pct(s), _pct(g), f"{(g - s) * 100:+.0f} points", abs(g - s) <= TOLERANCE["duty"]))
    for key, what in (("mean_v", "mean"), ("vpp_v", "peak to peak")):
        s, g = simulated.get(key), measured.get(key)
        if s is not None and g is not None:
            allowed = TOLERANCE["volts_ratio"] * abs(s) + TOLERANCE["volts_floor"]
            rows.append((what, _v(s), _v(g), f"{g - s:+.2f} V (allowed ±{allowed:.2f})", abs(g - s) <= allowed))
    return rows


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def build(hume, title=None):
    """The report's Markdown text, from what Hume holds now."""
    circuit = hume.circuit or {}
    name = circuit.get("name") or "circuit"
    now = datetime.now()
    lines = [f"# {title or name}", "", f"Lunes, {now:%d %b %Y %H:%M}. Supply {circuit.get('supply_v', '?')} V."]
    if getattr(hume, "request", ""):
        lines += ["", f"**Asked:** {hume.request}"]

    # the parts, and what is on the shelf
    parts = circuit.get("parts") or []
    if parts:
        from lunes import inventory
        stock = inventory.stock_of(parts) or {}
        rows = []
        for p in parts:
            value = p.get("value") if isinstance(p.get("value"), str) else (p.get("model") or p.get("color") or "")
            label = next((k for k in stock if k.startswith(f"{p.get('ref')} ")), None)
            rows.append((p.get("ref", ""), p.get("kind", ""), value, stock.get(label, "") if label else ""))
        lines += ["", "## Parts", "", _table(("ref", "kind", "value", "on the shelf"), rows)]

    # what the simulation says
    sim_nets = ((hume.expected or {}).get("nets") or {})
    if sim_nets:
        rows = [(net, m.get("shape", ""), _v(m.get("mean_v")), _v(m.get("vpp_v")),
                 _si(m.get("frequency_hz"), "Hz") if m.get("frequency_hz") else "steady", _pct(m.get("duty")))
                for net, m in sim_nets.items()]
        lines += ["", "## Simulation (ngspice)", "", "What every net should read, from the circuit as written.", "",
                  _table(("net", "shape", "mean", "peak to peak", "frequency", "duty"), rows)]

    # what was measured
    readings = []
    diagnosis = hume.diagnosis
    if diagnosis is not None:
        for net, r in diagnosis.readings:
            readings.append((net, r, "multimeter" if "vpp_v" not in r else "scope"))
    last = getattr(hume, "last_scope", None)
    if last and last.get("result") and not readings:
        m = (last["result"].get("CH1") or {})
        if m:
            readings.append((last.get("net") or "CH1", {"mean_v": m.get("mean_v"), "vpp_v": m.get("vpp_v"),
                                                        "frequency_hz": m.get("frequency_hz"), "duty": m.get("duty")}, "scope"))
    measured_lines = []
    if readings:
        rows = [(net, src, _v(r.get("mean_v")), _v(r.get("vpp_v")),
                 _si(r.get("frequency_hz"), "Hz") if r.get("frequency_hz") else ("steady" if "frequency_hz" in r else "-"),
                 _pct(r.get("duty"))) for net, r, src in readings]
        measured_lines += [_table(("net", "instrument", "mean", "peak to peak", "frequency", "duty"), rows)]
    supply = getattr(hume, "supply", None)
    if supply:
        measured_lines += ["", f"Supply, read on its display by the bench camera: {supply.get('volts')} V, "
                               f"{supply.get('amps')} A, output {'on' if supply.get('out') else 'OFF'}."]
    board = getattr(hume, "last_mcu", None)
    if isinstance(board, dict) and board.get("pins"):
        rows = [(pin, d.get("net", ""), _v(d.get("read_v")), _v(d.get("expected_v")), "yes" if d.get("ok") else "no")
                for pin, d in board["pins"].items() if isinstance(d, dict)]
        measured_lines += ["", "The board's own inputs:", "", _table(("pin", "net", "read", "simulated", "agrees"), rows)]
    if measured_lines:
        lines += ["", "## Measured on the bench", ""] + measured_lines
    else:
        lines += ["", "## Measured on the bench", "", "Nothing measured yet."]

    # side by side, with the criterion
    if readings and sim_nets:
        lines += ["", "## Simulation against the bench", "",
                  f"Criterion: frequency within {TOLERANCE['frequency_hz']:.0%}, duty within {TOLERANCE['duty'] * 100:.0f} "
                  f"points, volts within {TOLERANCE['volts_ratio']:.0%} plus {TOLERANCE['volts_floor']} V. Differences "
                  f"inside that come from the parts' tolerances (5 % resistors, 10 to 20 % capacitors), the probe's "
                  f"factor and the supply's exact voltage; outside it, the diagnosis says what explains them."]
        disagree = 0
        for net, r, src in readings:
            if net not in sim_nets:
                continue
            rows = compare(sim_nets[net], r)
            if not rows:
                continue
            disagree += sum(not ok for *_, ok in rows)
            lines += ["", f"**{net}** ({src})", "",
                      _table(("", "simulated", "measured", "difference", "agrees"),
                             [(w, s, g, d, "yes" if ok else "**no**") for w, s, g, d, ok in rows])]
        lines += ["", "Every measured net agrees with the simulation: the circuit does what it was designed to do."
                  if not disagree else f"{disagree} reading(s) outside the criterion: see the diagnosis."]

    # the diagnosis
    if diagnosis is not None:
        state = diagnosis.state()
        lines += ["", "## Diagnosis", "",
                  "Each reading updates the probability of every fault the simulator worked out for this circuit; the "
                  "next net is the one that tells the hypotheses apart best."]
        if state.get("likely"):
            lines += ["", _table(("explanation", "probability"),
                                 [(h["says"], f"{h['p'] * 100:.0f} %") for h in state["likely"]])]
        if state.get("verdict"):
            lines += ["", f"**Verdict:** {state['verdict']}"]
            if state.get("note"):
                lines += ["", state["note"]]
        elif readings:
            lines += ["", "No explanation reached 90 % yet."]

    # pictures and the rest
    pictures = []
    surveys = getattr(hume, "surveys", None) or []
    if surveys and surveys[-1].get("file"):
        pictures.append(("board mosaic (microscope)", surveys[-1]["file"]))
    chip = (getattr(hume, "world", None) or {}).get("chip") or {}
    if chip.get("file"):
        pictures.append(("the chip", chip["file"]))
    if pictures:
        lines += ["", "## Pictures", ""]
        for what, file in pictures:
            rel = str(file).replace("\\", "/")
            if rel.startswith(str(ROOT).replace("\\", "/")):
                rel = rel[len(str(ROOT).replace("\\", "/")) + 1:]
            lines += [f"![{what}]({rel})", "", f"*{what}: `{rel}`*", ""]
    verification = getattr(hume, "last_verification", None)
    if isinstance(verification, dict) and verification.get("says"):
        lines += ["", "## Verifier", "", str(verification["says"])]
    lines += ["", "---", "", "Written by Lunes from its tools' results: simulated values by ngspice, measurements by the "
              "instruments, probabilities by the diagnosis engine. Nothing here was guessed."]
    return "\n".join(lines) + "\n"


def write(hume, title=None):
    """Writes the report and copies it to latest.md: {"ok", "path", "latest", "sections"}."""
    text = build(hume, title)
    folder = Path(REPORTS)  # tests point REPORTS at a temporary folder, so the bench's list keeps only real reports
    folder.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", ((hume.circuit or {}).get("name") or "circuit").lower()).strip("-")[:40] or "circuit"
    path = folder / f"{slug}-{datetime.now():%Y%m%d-%H%M}.md"
    path.write_text(text, encoding="utf-8")
    latest = folder / "latest.md"
    shutil.copyfile(path, latest)
    sections = [l[3:] for l in text.splitlines() if l.startswith("## ")]

    def shown(p):
        try:
            return str(p.relative_to(ROOT)).replace("\\", "/")
        except ValueError:
            return str(p)

    return {"ok": True, "path": shown(path), "latest": shown(latest), "sections": sections,
            "note": "tell Erlan the report is written and the web lab shows it on its report page"}
