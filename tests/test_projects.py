"""Projects keep their own conversation, memory and history: python tests/test_projects.py

On a made-up bench whose files live in a temporary folder:
- a circuit set with no project open starts one; what its tools did goes into its history, one JSON line each;
- a fact remembered belongs to the project open, a bench fact (bench=true) to every project;
- a new project (project_new) has no circuit, no history of the other, and the other's fact is not in its memory;
- the worker's conversation starts again for it, and talk.py hears of the change (old id, new id);
- a redesign stays in the same project, with the circuit as it was kept in versions/;
- after a restart the open project comes back from runs/circuit.json, even with no circuit yet.
"""
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lunes import hume as H  # noqa: E402

FAILED = []
DIVIDER = {"name": "Divider", "supply_v": 5, "parts": [
    {"ref": "R1", "kind": "R", "value": "10k", "pins": ["VCC", "MID"]},
    {"ref": "R2", "kind": "R", "value": "10k", "pins": ["MID", "GND"]}]}


def ok(condition, what):
    print(("  ok   " if condition else "  FAIL ") + what)
    if not condition:
        FAILED.append(what)


def bench(tmp):
    machine = types.SimpleNamespace(s=types.SimpleNamespace(port="COM8"), config={})
    hume = H.Hume(machine, None)
    hume.keep_world(tmp / "bench_world.json")
    return hume


def main():
    tmp = Path(tempfile.mkdtemp(prefix="lunes-projects-"))
    H.CIRCUIT, H.PROJECTS, H.MEMORY = tmp / "circuit.json", tmp / "projects", tmp / "memory.jsonl"
    try:
        print("A project of its own")
        hume = bench(tmp)
        changes = []
        hume.on_project_change = lambda old, new: changes.append((old, new))
        hume.run_tool("circuit_set", {"circuit": DIVIDER})
        first = hume.project_id()
        ok(first is not None and (tmp / "circuits" / f"{first}.json").exists(), f"the circuit started a project: {first}")
        hume.run_tool("remember", {"fact": "MID read 2.49 V", "evidence": "multimeter"})
        hume.run_tool("remember", {"fact": "the Kinect sits 40 cm away", "evidence": "tape", "bench": True})
        history = hume.project_history()
        ok(any(r["kind"] == "tool" and r.get("tool") == "circuit_set" for r in history),
           f"circuit_set is in its history: {history[-1]['text'][:70] if history else None}")
        ok("MID read 2.49 V" in hume._memory_text(), "its fact is in its memory")

        print("A new project, kept apart")
        out = hume.project_new("Blinker 555")
        second = hume.project_id()
        ok(out["ok"] and second != first and hume.circuit is None, f"a new, empty project: {out.get('file')}")
        ok(changes == [(first, second)], f"talk.py hears of the change: {changes}")
        memory = hume._memory_text()
        ok("MID read 2.49 V" not in memory and "Kinect sits 40 cm" in memory,
           "the other project's fact stays there; the bench's fact is here too")
        ok(hume.reset_pending, "the worker's conversation starts again before its next request")
        hume._reset_conversation()
        prompt = hume.messages[0]["content"]
        ok("Blinker 555" in prompt and "circuit_set" not in prompt.split("Proyecto abierto")[-1],
           "its prompt names it and has none of the other's history")
        listed = hume._circuit_list()[0]
        mine = next(c for c in listed["circuits"] if c["file"] == f"{second}.json")
        ok(mine["name"] == "Blinker 555" and mine["empty"], "the list shows it by its name, empty")

        print("Back to the first, and a redesign")
        hume._circuit_load(f"{first}.json")
        ok(hume.project_id() == first and hume.circuit["name"] == "Divider", "opening it brings its circuit back")
        hume._reset_conversation()
        ok("circuit_set" in hume.messages[0]["content"], "and its history to the worker")
        redesign = json.loads(json.dumps(DIVIDER))
        redesign["parts"][1]["value"] = "4.7k"
        hume.run_tool("circuit_set", {"circuit": redesign})
        versions = list((tmp / "projects" / first / "versions").glob("*.json"))
        files = [f.stem for f in (tmp / "circuits").glob("*.json")]
        ok(hume.project_id() == first and len(files) == 2, f"a redesign stays in the project: {sorted(files)}")
        ok(len(versions) == 1 and json.loads(versions[0].read_text("utf-8"))["circuit"]["parts"][1]["value"] == "10k",
           "the circuit as it was is kept in versions/")

        print("After a restart")
        hume.project_new("Empty one")
        empty = hume.project_id()
        again = bench(tmp)
        ok(again.project_id() == empty and again.circuit is None, "the open project comes back, empty as it was")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nFAILED: " + "; ".join(FAILED) if FAILED else "\nall passed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
