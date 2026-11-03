"""How far Lunes goes with circuits nobody prepared: python tools/unseen.py

Each case is a sentence in Spanish, as Erlan would say it, about a circuit with no template, some with a chip that is
not in the catalog, which Lunes must learn from its datasheet by itself. The model does all of it. The verdict is not
its own: the code simulates the circuit it left and checks what the sentence asked for, with criteria the model never
sees, so goals written to pass prove nothing here.

Writes runs/unseen_<when>.jsonl (every run in full) and docs/unseen.md (the table). Chips learnt stay in the catalog,
marked as learnt.
"""
import argparse
import json
import sys
import tempfile
import time
import types
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lunes import catalog, mcu  # noqa: E402
from lunes import hume as hume_module  # noqa: E402
from lunes.hume import Hume  # noqa: E402
from lunes.meta_api import DAILY_TOKENS, LOG, tokens_today  # noqa: E402

hume_module.MEMORY = Path(tempfile.gettempdir()) / "lunes_unseen_memory.jsonl"  # a made-up bench is not to be remembered
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def uses(circuit, *names):
    wanted = {n.upper() for n in names}
    found = {str(p.get("model", p.get("kind"))).upper() for p in circuit["parts"]}
    found |= {catalog.get(p["model"])["name"].upper() for p in circuit["parts"] if p["kind"] == "IC" and catalog.get(p["model"])}
    return wanted <= found


def moves_at(sim, hz, tolerance=0.3):
    return any(m.get("frequency_hz") and abs(m["frequency_hz"] / hz - 1) <= tolerance for m in sim["nets"].values())


def holds(sim, volts, tolerance=0.1):
    return any(net not in ("VCC", "GND") and abs(m["mean_v"] / volts - 1) <= tolerance for net, m in sim["nets"].items())


def swings(sim, vpp, tolerance=0.3):
    return any(abs(m["vpp_v"] / vpp - 1) <= tolerance for net, m in sim["nets"].items() if net not in ("VCC", "GND"))


CASES = [
    ("oscillator with a Schmitt inverter", "Hazme un oscilador de 1 kHz con un 74HC14 a 5 voltios.",
     lambda c, s: uses(c, "74HC14") and moves_at(s, 1000)),
    ("comparator with a reference", "Con un LM393 a 5 voltios, compara 3 voltios de un divisor contra 2.5 voltios de otro "
                                    "divisor, y que la salida quede en alto.",
     lambda c, s: uses(c, "LM393") and holds(s, 3.0, 0.1) and holds(s, 2.5, 0.1) and any(m["mean_v"] > 4.5 for n, m in s["nets"].items() if n != "VCC")),
    ("adjustable regulator", "Dame 9 voltios desde 12 con un LM317 y una carga de unos 50 miliamperios.",
     lambda c, s: uses(c, "LM317") and holds(s, 9.0, 0.08)),
    ("amplifier", "Un amplificador no inversor de ganancia 5 con un LM358 a 9 voltios, para una señal de 100 milivoltios "
                  "de amplitud a 1 kHz montada sobre 1 voltio.",
     lambda c, s: uses(c, "LM358") and moves_at(s, 1000) and swings(s, 1.0)),
    ("Arduino through a driver", "Un Arduino Uno que haga parpadear a 2 Hz un LED rojo a través de un ULN2003, con el "
                                 "LED alimentado de 5 voltios.",
     lambda c, s: uses(c, "ULN2003", "ARDUINO_UNO") and moves_at(s, 2)),
    ("two NAND gates", "Un oscilador de unos 500 Hz con compuertas NAND de un 74HC00 a 5 voltios.",
     lambda c, s: uses(c, "74HC00") and moves_at(s, 500, 0.5)),
    ("isolated signal", "Aísla con un PC817 una señal cuadrada de 5 voltios a 100 Hz, y que del otro lado salga en una "
                        "fuente de 12 voltios.",
     lambda c, s: uses(c, "PC817") and moves_at(s, 100) and swings(s, 12.0, 0.4)),
    ("a chip it must learn: quad comparator", "Quiero usar un LM339 a 5 voltios como comparador: compara 3 voltios contra "
                                              "2.5 voltios y que la salida quede en alto.",
     lambda c, s: uses(c, "LM339") and any(m["mean_v"] > 4.5 for n, m in s["nets"].items() if n != "VCC")),
    ("a chip it must learn: shunt reference", "Quiero una referencia de 2.5 voltios con un TL431 alimentado de 5 voltios.",
     lambda c, s: uses(c, "TL431") and holds(s, 2.5, 0.05)),
    ("a chip it must learn: inverter oscillator", "Hazme un oscilador de 1 kHz con un CD4093 a 5 voltios.",
     lambda c, s: uses(c, "CD4093") and moves_at(s, 1000, 0.5)),
]
JOB = "\n(Solo diseña y simula, no muevas la máquina. Si te falta algo, resuélvelo tú; no hay nadie a quien preguntarle.)"


def run_case(name, sentence, judge, brain, effort):
    machine = types.SimpleNamespace(s=types.SimpleNamespace(port="COM8"), config={})
    hume = Hume(machine, None, mcu=mcu.SimulatedBoard(None), spark=True, brain_model=brain, effort=effort)
    calls = []
    lines_before, t0 = LOG.read_text(encoding="utf-8").count("\n") if LOG.exists() else 0, time.time()
    known = set(catalog.names())
    try:
        answer = hume.handle(sentence + JOB, lambda tool, args: calls.append(tool))
    except Exception as e:  # noqa: BLE001 - a crash is a result here
        answer = f"{type(e).__name__}: {e}"
    rows = LOG.read_text(encoding="utf-8").splitlines()[lines_before:] if LOG.exists() else []
    tokens = sum((json.loads(r).get("usage") or {}).get("total_tokens") or 0 for r in rows if r.strip())
    circuit, sim = hume.circuit, hume.expected
    passed = False
    why = "it left no simulated circuit"
    if circuit and sim:
        try:
            passed = bool(judge(circuit, sim))
            why = "" if passed else "the simulation does not do what was asked"
        except Exception as e:  # noqa: BLE001
            why = f"{type(e).__name__}: {e}"
    return {"case": name, "sentence": sentence, "passed": passed, "why": why, "answer": answer, "tools": calls,
            "learnt": sorted(set(catalog.names()) - known), "circuit": circuit, "tokens": tokens,
            "seconds": round(time.time() - t0, 1), "fallback": "glimmer" if not tokens else ""}


def table(results, when, brain, effort):
    lines = [f"# Circuits nobody prepared, with {brain} at effort {effort}", "",
             f"Made by `tools/unseen.py` on {when}. Each case is one sentence in Spanish with no template behind it, and "
             f"the last ones name a chip that was not in the catalog, which the model must learn from its datasheet. The "
             f"verdict is the code's: it simulates what the model left and checks what the sentence asked for, with "
             f"criteria the model never sees.", "",
             "| Case | Works | Learnt | Tools | Tokens | Time |", "|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['case']} | {'yes' if r['passed'] else 'no: ' + r['why']} | {', '.join(r['learnt']) or '-'} | "
                     f"{len(r['tools'])} | {r['tokens']:,} | {r['seconds']:.0f} s |")
    good = sum(r["passed"] for r in results)
    lines += ["", f"**{good} of {len(results)} work.** Tokens in all: {sum(r['tokens'] for r in results):,}."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--brain", default="muse-wonder-1.0")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--only", default=None, help="only the cases whose name holds this")
    ap.add_argument("--reserve", type=int, default=400000, help="stop when fewer tokens than this are left today")
    args = ap.parse_args()
    when = datetime.now().strftime("%Y-%m-%d %H:%M")
    out = ROOT / "runs" / f"unseen_{datetime.now().strftime('%Y-%m-%d_%H%M')}.jsonl"
    out.parent.mkdir(exist_ok=True)
    results = []
    for name, sentence, judge in CASES:
        if args.only and args.only not in name:
            continue
        if DAILY_TOKENS - tokens_today() < args.reserve:
            print(f"stopped before '{name}': only {DAILY_TOKENS - tokens_today():,} tokens left today", flush=True)
            break
        r = run_case(name, sentence, judge, args.brain, args.effort)
        results.append(r)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")
        print(f"{'PASS' if r['passed'] else 'FAIL'} {name}: {r['tokens']:,} tokens {r['seconds']:.0f} s learnt {r['learnt']} "
              f"tools {' '.join(r['tools'])}" + (f" | {r['why']}" if r["why"] else ""), flush=True)
        print(f"      said: {r['answer'][:240]}", flush=True)
    if results:
        (ROOT / "docs" / "unseen.md").write_text(table(results, when, "Muse Wonder" if "wonder" in args.brain else args.brain,
                                                       args.effort), encoding="utf-8")
    print(f"{sum(r['passed'] for r in results)} of {len(results)} work; rows: {out}", flush=True)


if __name__ == "__main__":
    main()
