"""What the web lab shows of the bench right now, ready for JSON: the machine, the instruments, the last scope reading,
the circuit, its diagnosis and the board in it. It only reads what Hume already keeps, from the page's own thread, so it
never talks to the machine or the scope: those belong to the thread that is working.
"""
from lunes import catalog, mcu
import time

import numpy as np

SCOPE_KEYS = ("shape", "vpp_v", "mean_v", "frequency_hz", "duty", "rise_s", "baud_if_uart", "range_v", "notes")
NET_KEYS = ("mean_v", "vpp_v", "shape", "frequency_hz", "touches")


def _ago(moment):
    return None if moment is None else round(time.time() - moment, 1)


def _mind(hume):
    """What Lunes has in mind of the bench, for the map: all of it in machine mm, with how long ago it was seen."""
    world = hume.world
    return {"chip": world["chip"] and dict(world["chip"], ago_s=_ago(world["chip"]["at"])),
            "laser": world["laser"] and dict(world["laser"], ago_s=_ago(world["laser"]["at"])),
            "mosaic": world["mosaic"], "notes": world["notes"]}


def _machine(hume):
    from lunes.hume import WORKSPACE
    microscope = hume.m.config.get("microscope", {})
    scale = microscope.get("px_per_mm") or {}
    check = hume.last_kinect_check or {}
    board = hume.board or hume.world["board"] or {}
    return {"at_mm": {k: round(float(v), 2) for k, v in (getattr(hume.m, "at_mm", None) or {}).items()},
            "area_mm": WORKSPACE, "focus_z_mm": hume.focus_z,
            "board": {k: board[k] for k in ("center_mm", "size_mm") if k in board} or None,
            "hold": hume.hold["why"] if hume.hold else None, "doubtful": bool(hume.position_doubtful),
            "kinect": {"agrees": check.get("agrees"), "off_mm": round(check.get("off_mm", 0), 1),
                       "ago_s": _ago(check.get("at"))} if check else None,
            "px_per_mm": float(np.mean(list(scale.values()))) if scale else None,
            "image": microscope.get("resolution") or [640, 480], "mosaics": len(hume.surveys), "mind": _mind(hume),
            "bed": hume.world["bed"]}


def _scope(hume):
    scope, last = hume.scope, hume.last_scope
    if scope is None:
        return {"connected": False}
    out = {"connected": True, "simulated": getattr(scope, "signals", None) is not None, "probe": dict(scope.probe)}
    if last:
        result = last["result"]
        out.update(ago_s=_ago(last["at"]), rate=result.get("sample_rate"),
                   channels={ch: {k: result[ch].get(k) for k in SCOPE_KEYS}
                             for ch in ("CH1", "CH2") if ch in result})
    if hume.last_trace:
        out["trace"] = hume.last_trace
    return out


def _circuit(hume):
    circuit, expected = hume.circuit, hume.expected
    if not circuit:
        return None
    nets = {name: {k: net.get(k) for k in NET_KEYS} for name, net in (expected or {}).get("nets", {}).items()
            if name != "GND"}
    parts = [{k: part[k] for k in ("ref", "kind", "value", "model", "color") if k in part}
             for part in circuit.get("parts", [])]
    return {"name": circuit.get("name"), "supply_v": circuit.get("supply_v", 5), "parts": parts, "nets": nets,
            "goals": hume.goals}


def _diagnosis(hume):
    """The engine's state, worked out once per reading: choosing the next net takes it a moment."""
    diagnosis = hume.diagnosis
    if diagnosis is None:
        return None
    key = (id(diagnosis), len(diagnosis.readings))
    if hume.lab_diagnosis is None or hume.lab_diagnosis[0] != key:
        hume.lab_diagnosis = key, diagnosis.state()
    return hume.lab_diagnosis[1]


def _arduino(hume):
    from lunes import mcu
    simulated = isinstance(hume.mcu, mcu.SimulatedBoard)
    if not (simulated or hume.mcu_port or hume.last_mcu):
        return None
    last = hume.last_mcu or {}
    return {"simulated": simulated, "port": "simulated" if simulated else hume.mcu_port, "ago_s": _ago(last.get("at")),
            "pins": last.get("pins"), "all_as_simulated": last.get("all_as_simulated"), "serial": hume.last_serial}


def _parts(hume, layout):
    """What the page needs to draw each part of the plan: all of it comes from the part's own data, so a chip learnt a
    minute ago is drawn like any other, with no drawing made for it."""
    out = []
    for part in (hume.circuit or {}).get("parts", []):
        ref, holes = part["ref"], layout["holes"].get(part["ref"])
        if not holes or part["kind"] == "V" or part["kind"] in mcu.BOARDS:
            continue
        item = {"ref": ref, "kind": part["kind"], "value": part.get("value"), "color": part.get("color"),
                "label": part.get("model") or part["kind"], "holes": {str(k): v for k, v in holes.items()}}
        if ref in layout.get("chips", {}):
            item["chip"] = layout["chips"][ref]
        if part["kind"] == "IC":
            entry = catalog.get(part["model"])
            item.update(package=entry["package"], label=entry["name"], learnt=entry.get("origin") == "learnt")
        out.append(item)
    return out


def _breadboard(hume):
    layout = hume.plan_now()
    if layout is None:
        return None
    steps = layout["steps"]
    return {"holes": [h for pins in layout["holes"].values() for h in pins.values()], "parts": _parts(hume, layout),
            "wires": [[w["from"], w["to"], w["color"]] for w in layout["wires"]], "total": len(steps),
            "placed": hume.placement is not None, "guide": hume.guide}


def _chip(hume):
    seen = hume.chip_seen
    if not seen:
        return None
    out = {k: seen.get(k) for k in ("seen", "pin1_in_picture", "pin1_hole", "planned_hole", "as_planned", "how", "sure")}
    out["ago_s"] = _ago(seen.get("at"))
    return out


def state(hume):
    machine, scope = _machine(hume), _scope(hume)
    arduino = _arduino(hume)
    return {"machine": machine, "scope": scope, "circuit": _circuit(hume), "diagnosis": _diagnosis(hume),
            "arduino": arduino, "brain": hume.brain, "supply": hume.supply_state(),
            "breadboard": _breadboard(hume), "chip": _chip(hume),
            "instruments": {"cnc": bool(machine["at_mm"]), "microscope": hume.cam is not None,
                            "bench": hume.bench is not None, "kinect": hume.kinect is not None,
                            "scope": scope["connected"], "arduino": arduino is not None}}
