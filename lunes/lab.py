"""What the web lab shows of the bench right now, ready for JSON: the machine, the instruments, the last scope reading,
the circuit, its diagnosis and the board in it. It only reads what Hume already keeps, from the page's own thread, so it
never talks to the machine or the scope: those belong to the thread that is working.
"""
import time

import numpy as np

SCOPE_KEYS = ("shape", "vpp_v", "mean_v", "frequency_hz", "duty", "rise_s", "baud_if_uart", "range_v", "notes")
NET_KEYS = ("mean_v", "vpp_v", "shape", "frequency_hz", "touches")


def _ago(moment):
    return None if moment is None else round(time.time() - moment, 1)


def _machine(hume):
    from lunes.hume import WORKSPACE
    microscope = hume.m.config.get("microscope", {})
    scale = microscope.get("px_per_mm") or {}
    check = hume.last_kinect_check or {}
    board = hume.board or {}
    return {"at_mm": {k: round(float(v), 2) for k, v in (getattr(hume.m, "at_mm", None) or {}).items()},
            "area_mm": WORKSPACE, "focus_z_mm": hume.focus_z,
            "board": {k: board[k] for k in ("center_mm", "size_mm") if k in board} or None,
            "hold": hume.hold["why"] if hume.hold else None, "doubtful": bool(hume.position_doubtful),
            "kinect": {"agrees": check.get("agrees"), "off_mm": round(check.get("off_mm", 0), 1),
                       "ago_s": _ago(check.get("at"))} if check else None,
            "px_per_mm": float(np.mean(list(scale.values()))) if scale else None,
            "image": microscope.get("resolution") or [640, 480], "mosaics": len(hume.surveys)}


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


def _breadboard(hume):
    layout = hume.layout
    if layout is None:
        return None
    steps = layout["steps"]
    return {"holes": [h for pins in layout["holes"].values() for h in pins.values()],
            "wires": [[w["from"], w["to"]] for w in layout["wires"]], "total": len(steps), "placed": hume.placement is not None,
            "guide": hume.guide}


def state(hume):
    machine, scope = _machine(hume), _scope(hume)
    arduino = _arduino(hume)
    return {"machine": machine, "scope": scope, "circuit": _circuit(hume), "diagnosis": _diagnosis(hume),
            "arduino": arduino, "brain": hume.brain, "supply": hume.supply_state(),
            "breadboard": _breadboard(hume),
            "instruments": {"cnc": bool(machine["at_mm"]), "microscope": hume.cam is not None,
                            "bench": hume.bench is not None, "kinect": hume.kinect is not None,
                            "scope": scope["connected"], "arduino": arduino is not None}}
