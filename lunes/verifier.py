"""Checking what Lunes is about to say against what its tools showed, before it is said aloud.

Two steps, so most answers cost nothing extra:
- The code first: every number in the answer is looked for in the evidence (what Erlan said and what the tools
  returned in this conversation), allowing for the ways a number gets said: rounded ("977" for 976.6), as a
  percentage ("53 %" for 0.529), in another unit ("14.7 mA" for 0.0147 A, "6.8 µF" for 6.8e-6).
- Only if a number is not found, or the answer claims to have seen or measured something when no tool ran in this
  turn, a model is asked to compare the answer with the evidence and fix only what is not supported. Muse Spark at
  minimal effort does it, or Glimmer on the PC when there is no internet.

If no model answers, the answer goes out as it was: the verifier never silences Lunes. Each check is logged in
runs/verifier.jsonl.
"""
import json
import math
import re
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from lunes.speakable import HUNDREDS, TENS, UNITS

LOG = Path(__file__).resolve().parent.parent / "runs" / "verifier.jsonl"
OLLAMA = "http://127.0.0.1:11434/api/chat"
# a number standing alone or with a unit letter after it ("47k", "3,3 V"); not the digits inside a part name ("2N2222")
SAID = re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)?)(?=[a-zA-Zµ]{0,3}\b|[^\w]|$)")
GIVEN = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)([pnuµmkKMG])?(?![a-zA-Z0-9])")
SCALE = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "m": 1e-3, "k": 1e3, "K": 1e3, "M": 1e6, "G": 1e9}
CLAIM = re.compile(r"\b(veo|vi|se ve|medí|mide|midió|mido|marca|leí|da unos|está en)\b", re.IGNORECASE)
SYSTEM = """You check what Lunes, an assistant on an electronics bench, is about to say aloud, against the evidence:
what Erlan said and what Lunes' tools returned in this conversation. Every number presented as measured, simulated,
designed or seen, and every claim of having seen, measured or read something, must come from the evidence or be
correct arithmetic on it. General knowledge said as such is fine ("un LED suele querer unos 20 mA", "en general el
NE555 aguanta 16 V"). Fix only what is not supported: remove it, or say it as unknown or as an estimate ("no lo medí", "más o menos"). Keep everything else,
in the same short spoken Spanish, with no lists or markdown. Answer only JSON: {"ok": true} if nothing needs
changing, or {"ok": false, "answer": "the fixed answer", "fixed": ["what you changed, briefly"]}."""
SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}, "answer": {"type": "string"},
                                           "fixed": {"type": "array", "items": {"type": "string"}}},
          "required": ["ok"]}


WORD_VALUE = {w: i for i, w in enumerate(UNITS)} | {"un": 1, "una": 1, "veintiún": 21} | \
    {w: 10 * k for k, w in TENS.items()} | {w: 100 * k for k, w in HUNDREDS.items()} | {"cien": 100}


def word_numbers(text):
    """Numbers written in Spanish words: "ciento cincuenta ohmios" -> 150, "treinta y dos" -> 32."""
    out, value, started = [], 0, False
    words = re.findall(r"[a-záéíóúñ]+", text.lower())
    for i, w in enumerate(words):
        if w in WORD_VALUE:
            value += WORD_VALUE[w]
            started = True
        elif w == "mil" and started:
            value = max(value, 1) * 1000
        elif w == "mil":
            value, started = 1000, True
        elif w == "y" and started and i + 1 < len(words) and words[i + 1] in WORD_VALUE \
                and WORD_VALUE[words[i + 1]] < 10 and value % 10 == 0 and value >= 30:
            continue  # "treinta y dos"
        else:
            if started:
                out.append(float(value))
            value, started = 0, False
    if started:
        out.append(float(value))
    return [n for n in out if n > 2]


def said_numbers(text):
    """The numbers in a spoken answer, in figures ("3,3 micros" -> 3.3) or in words ("ciento cincuenta" -> 150)."""
    return [float(m.group(1).replace(",", ".")) for m in SAID.finditer(text)] + word_numbers(text)


def evidence_values(texts):
    """Every number in the evidence, in every form it may be said in."""
    out = set()
    for text in texts:
        out.update(float(d) for d in re.findall(r"\d+", text))  # part names too: "NE555", "BC547", "pin 8"
        for m in GIVEN.finditer(text):
            try:
                v = abs(float(m.group(1)))
            except ValueError:
                continue
            if m.group(2):
                out.add(v)  # "47k" may be said "47 kiloohmios"
                v *= SCALE[m.group(2)]
            out.update(_forms(v))
    return out


def _forms(v):
    """The ways a value may be said: as it is, and in the units a person would use for its size (a fraction as a
    percentage or in milli, a tiny value in micro, nano or pico, a big one in kilo or mega)."""
    forms = [v]
    if 0 < v < 1:
        forms += [v * 100, v * 1e3]
    if 0 < v < 1e-3:
        forms.append(v * 1e6)
    if 0 < v < 1e-6:
        forms.append(v * 1e9)
    if 0 < v < 1e-9:
        forms.append(v * 1e12)
    if v >= 1e3:
        forms.append(v / 1e3)
    if v >= 1e6:
        forms.append(v / 1e6)
    return forms


def _round_sig(x, sig):
    return 0.0 if x == 0 else round(x, sig - 1 - int(math.floor(math.log10(abs(x)))))


def supported(n, values):
    """Is the said number n one of the evidence's values, as a person would round or convert it?"""
    if n <= 2 and float(n).is_integer():  # "dos", "un", "el 1": counting words, not measurements
        return True
    figures = (1, 2, 3) if n < 10 else (2, 3)  # "unos 2 voltios" for 2.08; but 150 is not "about" 147
    for v in values:
        if v and abs(n - v) <= 0.02 * abs(v):
            return True
        if v and any(abs(_round_sig(v, f) - n) < 1e-9 * max(1, abs(n)) for f in figures):
            return True
    return False


def findings(answer, evidence, used_tools):
    """(numbers not found in the evidence, whether it claims an observation with no tool run this turn)."""
    values = evidence_values(evidence)
    missing = [n for n in said_numbers(answer) if not supported(n, values)]
    return missing, (not used_tools and bool(CLAIM.search(answer)))


def _ask_spark(spark, prompt):
    msg = spark.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                     purpose="Lunes verifier", effort="minimal", max_tokens=900, timeout=20)
    return _parse(msg.content)


def _ask_glimmer(model, prompt):
    body = {"model": model, "stream": False, "think": False, "format": SCHEMA, "keep_alive": -1,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            "options": {"temperature": 0.1, "num_predict": 400, "num_ctx": 8192}}
    req = urllib.request.Request(OLLAMA, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return _parse(json.loads(r.read())["message"].get("content"))


def _parse(text):
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    return json.loads(m.group(0)) if m else None


def verify(answer, request, evidence, used_tools, spark=None, glimmer=None, recent=None):
    """(answer to say, report). evidence: the texts the answer may draw on; recent: the part of it to show the
    model (this turn's tool results), to keep the check short."""
    t0 = time.time()
    missing, unseen = findings(answer, evidence + [request], used_tools)
    report = {"at": datetime.now().isoformat(timespec="seconds"), "answer": answer, "missing": missing,
              "unseen_claim": unseen}
    if not missing and not unseen:
        report["outcome"] = "supported by the code check"
        return answer, _log(report, t0)
    shown = "\n".join((recent or evidence)[-8:])[-7000:]
    prompt = (f"Erlan said: {request}\n\nEvidence (tool results):\n{shown or '(no tool ran in this turn)'}\n\n"
              f"Numbers in the answer not found in the evidence: {missing or 'none'}\n"
              + ("The answer claims to have seen or measured something, but no tool ran in this turn.\n"
                 if unseen else "") + f"\nAnswer to check:\n{answer}")
    result, who = None, None
    for who, ask in (("spark", lambda: _ask_spark(spark, prompt) if spark else None),
                     ("glimmer", lambda: _ask_glimmer(glimmer, prompt) if glimmer else None)):
        try:
            result = ask()
        except Exception as e:  # no internet, no Ollama, a slow or bad reply: the next one, or say it as it was
            report[f"{who}_error"] = str(e)[:200]
            result = None
        if result is not None:
            break
    report["checked_by"] = who if result is not None else None
    if result and result.get("ok") is False and (result.get("answer") or "").strip():
        report.update(outcome="corrected", fixed=result.get("fixed", []), said=result["answer"].strip())
        return result["answer"].strip(), _log(report, t0)
    report["outcome"] = "confirmed by the model" if result else "not checked (no model answered)"
    return answer, _log(report, t0)


def _log(report, t0):
    report["seconds"] = round(time.time() - t0, 2)
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False) + "\n")
    return report
