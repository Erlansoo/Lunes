"""Lunes' diagnosis with Muse Spark itself at the wheel, on the simulated bench, many times: the measured record.

The no-hardware tests drive Hume's tools from code. This lets the model do the driving, as it does with Erlan: a
job in Spanish, the model designs the circuit, programs the made-up board, starts the diagnosis, reads the board,
asks for the probe net by net (a script answers for Erlan: the probe is there) and names the fault, or says the
circuit is sound. Every answer is kept, with the engine's own last state, so the verdicts can be read again.

    python tools/reliability.py                       3 rounds of the 7 cases with Muse Spark (about 2.7 M tokens)
    python tools/reliability.py --rounds 1 --cases arduino_pwm_rc:R1:open
    python tools/reliability.py --glimmer             the local model instead, for comparison
    python tools/reliability.py --rescore runs/reliability_2026-11-02_0912.jsonl    the verdicts again, no tokens

Writes runs/reliability_<when>.jsonl (every run, in full) and docs/reliability.md (the table).
"""
import argparse
import json
import re
import sys
import time
import types
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lunes import mcu  # noqa: E402
from lunes.circuit import with_fault  # noqa: E402
from lunes.designs import TEMPLATES  # noqa: E402
from lunes.hantek import SimulatedScope  # noqa: E402
from lunes.hume import Hume  # noqa: E402
from lunes.meta_api import LOG, tokens_today  # noqa: E402
from lunes.simulator import periodic_signal, waveform  # noqa: E402

# the same cases the no-hardware test drives from code (tests/test_simulated_bench.py)
CASES = [("arduino_pwm_rc", {"duty": 0.3}, None), ("arduino_pwm_rc", {"duty": 0.3}, "R1:open"),
         ("arduino_pwm_rc", {"duty": 0.3}, "C1:short"), ("arduino_pwm_rc", {"duty": 0.3}, "U1:D9_floating"),
         ("arduino_pwm_rc", {"duty": 0.3}, "U1:not_running"), ("arduino_read_divider", {"vin": 12}, "R1:x10"),
         ("arduino_read_divider", {"vin": 12}, "R2:open")]
PROBE_TURNS = 6  # how many times the script answers "the probe is there" for Erlan
WORDS = {"uno": "1", "dos": "2", "tres": "3", "cuatro": "4", "cinco": "5", "seis": "6", "siete": "7", "ocho": "8",
         "nueve": "9", "diez": "10"}
REF = re.compile(r"\b([RCDLQU])[ -]?(\d+|" + "|".join(WORDS) + r")\b", re.I)  # R1, or "R uno" as it is spoken
KINDS = {"R": ("resistencia", "resistor"), "C": ("capacitor", "condensador"), "U": ("arduino", "placa", "micro"),
         "D": ("led", "diodo"), "Q": ("transistor",), "L": ("bobina", "inductor")}  # a part named by its kind
SOUND = re.compile(r"\b(san[oa]|funciona bien|está bien|sin falla|no (?:hay|encontré|veo) (?:ninguna |una )?falla|"
                   r"no hay componente fallado|correctamente|como debe)\b", re.I)


def build_job(template, params):
    return (f"Arma el circuito de la plantilla {template} con los parámetros {json.dumps(params)} y prográmalo en el "
            f"Arduino que está conectado. Solo eso por ahora: cuando esté programado dime y paramos.")


def diagnose_job():
    return ("Ese circuito que acabamos de armar no me convence: diagnostícalo. Empieza el diagnóstico, lee la placa, y "
            "mide con el osciloscopio las redes que pida el motor. La punta CH1 ya está en la red que pidas, cada vez: "
            "mide directo, no me preguntes si ya la puse. Al final dime qué componente falló y cómo, o si el circuito "
            "está sano.")


def parts_of(template, params):
    circuit, _, _ = TEMPLATES[template](**params)
    return [p["ref"] for p in circuit["parts"]]


def refs_in(text):
    return {m.group(1).upper() + WORDS.get(m.group(2).lower(), m.group(2)) for m in REF.finditer(text)}


def named_parts(text, parts):
    """The circuit's parts the answer names: by ref (R1, or "R uno" as it is spoken), or by kind when the circuit
    has one part of that kind ("la resistencia", "el Arduino"). A pin (D9) is not a part."""
    named = {r for r in refs_in(text) if r in parts}
    low = text.lower()
    for letter, words in KINDS.items():
        same = [r for r in parts if r[0] == letter]
        if len(same) == 1 and any(w in low for w in words):
            named.add(same[0])
    return named


def verdict(truth, answers, engine, parts):
    """found / unsure / wrong, by a plain rule anyone can check against the answers. The engine groups faults that
    read the same at the measured nets (R1 open and C1 short both leave the filter at zero) into one class. A fault
    is found when the engine settled on a class holding the planted one and the answer names its part; wrong when
    the engine settled on a class without it, or the answer names other parts and not it; unsure otherwise. A sound
    circuit is found when the answer says so and the engine did not settle on a fault; wrong when it did."""
    text = " ".join(answers[1:]) if len(answers) > 1 else " ".join(answers)  # the diagnosis, not the building
    named = named_parts(text, parts)
    top = (engine or {}).get("likely", [{}])[0] if engine else {}
    faults, clear = top.get("faults", []), bool((engine or {}).get("clear"))
    if truth is None:
        if clear and "healthy" not in faults:
            return "wrong"
        return "found" if SOUND.search(text) else "unsure"
    ref = truth.split(":")[0]
    if clear and truth in faults and ref in named:
        return "found"
    if (clear and truth not in faults) or (named and ref not in named):
        return "wrong"
    return "unsure"


def run_case(template, params, fault, spark, think):
    machine = types.SimpleNamespace(s=types.SimpleNamespace(port="COM8"), config={})
    hume = Hume(machine, None, mcu=mcu.SimulatedBoard(fault), spark=spark, think=think)
    on_bench = {"circuit": None}
    calls = []

    def on_tool(name, args):
        calls.append(name)
        if name == "diagnose_measure" and hume.circuit is not None:  # the scope shows the planted fault on that net
            if on_bench["circuit"] is None:
                on_bench["circuit"] = with_fault(hume.circuit, fault) if fault else hume.circuit
            try:
                rate, v = waveform(on_bench["circuit"], str(args.get("net", "")), hume.expected["simulated_s"])
                hume.scope = SimulatedScope({"CH1": periodic_signal(rate, v), "CH2": lambda t: np.zeros_like(t)})
            except Exception:  # a net the circuit does not have: the tool says so
                hume.scope = None

    lines_before, t0 = LOG.read_text(encoding="utf-8").count("\n") if LOG.exists() else 0, time.time()
    answers = [hume.handle(build_job(template, params), on_tool)]
    answers.append(hume.handle(diagnose_job(), on_tool))
    for _ in range(PROBE_TURNS):  # Erlan, played by the script: the probe is wherever the model asked
        state = hume.diagnosis.state() if hume.diagnosis is not None else None
        if state is None or state.get("clear") or "next" not in state:
            break
        if not re.search(r"\?|punta|sonda|probe", answers[-1], re.I):
            break
        answers.append(hume.handle("Listo, la punta ya está ahí. Mide y sigue.", on_tool))
    engine = hume.diagnosis.state() if hume.diagnosis is not None else None
    rows = LOG.read_text(encoding="utf-8").splitlines()[lines_before:] if LOG.exists() else []
    tokens = sum((json.loads(r).get("usage") or {}).get("total_tokens") or 0 for r in rows if r.strip())
    parts = parts_of(template, params)
    return {"template": template, "params": params, "truth": fault or "healthy",
            "verdict": verdict(fault, answers, engine, parts), "parts": parts,
            "answers": answers, "tools": calls, "engine": engine and {"clear": engine.get("clear"),
                                                                       "likely": engine.get("likely", [])[:2]},
            "tokens": tokens, "seconds": round(time.time() - t0, 1), "brain": hume.brain}


def rescored(row):
    """The row with its verdict worked out again, by the rule as it is now."""
    parts = row.get("parts") or parts_of(row["template"], row["params"])
    truth = None if row["truth"] == "healthy" else row["truth"]
    return dict(row, parts=parts, verdict=verdict(truth, row["answers"], row.get("engine"), parts))


def table(results, when, brain):
    counts = {}
    for r in results:
        c = counts.setdefault(r["truth"], {"found": 0, "unsure": 0, "wrong": 0, "tokens": 0, "runs": 0})
        c[r["verdict"]] += 1
        c["tokens"] += r["tokens"]
        c["runs"] += 1
    total = {k: sum(c[k] for c in counts.values()) for k in ("found", "unsure", "wrong", "tokens", "runs")}
    lines = [f"# Diagnosis with {brain} at the wheel: the measured record",
             "",
             f"Made by `tools/reliability.py` on {when}: the model drives Hume's tools on the simulated bench "
             f"(lunes.mcu.SimulatedBoard, ngspice for the board's readings and the scope), one job in Spanish per "
             f"case, a script answering for Erlan that the probe is placed. The engine groups faults that read the "
             f"same at the measured nets into one class (R1 open and C1 short both leave the filter at zero; a board "
             f"not running and its pin stuck low look alike from outside). A run is **found** when the engine settled "
             f"on a class holding the planted fault and the answer names its part (by ref, or by kind when the "
             f"circuit has one such part); **wrong** when the engine settled on a class without it, or the answer "
             f"blames other parts and not it (or any fault on a sound circuit); **unsure** otherwise, the engine "
             f"still undecided and the answer saying so. The full answers are in `runs/reliability_*.jsonl`.",
             "",
             "| Circuit | Planted fault | Runs | Found | Unsure | Wrong | Tokens per run |",
             "|---|---|---|---|---|---|---|"]
    for (template, _, fault) in CASES:
        truth = fault or "healthy"
        if truth not in counts:
            continue
        c = counts[truth]
        lines.append(f"| {template} | {truth} | {c['runs']} | {c['found']} | {c['unsure']} | {c['wrong']} | "
                     f"{c['tokens'] // max(1, c['runs']):,} |")
    lines.append(f"| **all** | | {total['runs']} | **{total['found']}** | {total['unsure']} | **{total['wrong']}** | "
                 f"{total['tokens'] // max(1, total['runs']):,} |")
    lines += ["", f"Tokens in all: {total['tokens']:,} ({len(results)} runs)."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--cases", default=None, help="comma-separated template:fault (fault 'healthy' for none)")
    ap.add_argument("--glimmer", action="store_true", help="the local model instead of Muse Spark")
    ap.add_argument("--think", default="low")
    ap.add_argument("--rescore", default=None, help="a runs/reliability_*.jsonl to judge again, no model calls")
    args = ap.parse_args()
    if args.rescore:
        src = Path(args.rescore)
        results = [rescored(json.loads(line)) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
        stamp = src.stem.replace("reliability_", "")  # 2026-11-02_0907
        when = f"{stamp[:10]} {stamp[11:13]}:{stamp[13:15]}" if len(stamp) == 15 else stamp
    else:
        cases = CASES
        if args.cases:
            wanted = set(args.cases.split(","))
            cases = [c for c in CASES if f"{c[0]}:{c[2] or 'healthy'}" in wanted]
        when = datetime.now().strftime("%Y-%m-%d %H:%M")
        out = ROOT / "runs" / f"reliability_{datetime.now().strftime('%Y-%m-%d_%H%M')}.jsonl"
        out.parent.mkdir(exist_ok=True)
        results = []
        print(f"{len(cases)} cases x {args.rounds} rounds; tokens today so far {tokens_today():,}", flush=True)
        for k in range(args.rounds):
            for template, params, fault in cases:
                r = run_case(template, params, fault, not args.glimmer, args.think)
                results.append(r)
                with out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")
                print(f"round {k + 1} {template} {r['truth']:>16}: {r['verdict']:6} {r['tokens']:>7,} tokens "
                      f"{r['seconds']:5.0f} s tools {' '.join(r['tools'])}", flush=True)
                print(f"      said: {r['answers'][-1][:220]}", flush=True)
        print("rows:", out)
    brain = results[0]["brain"] if results else "?"
    report = table(results, when, "Muse Spark" if brain == "spark" else "Muse Glimmer")
    (ROOT / "docs" / "reliability.md").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
