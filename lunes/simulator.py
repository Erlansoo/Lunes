"""What Lunes expects to measure on a circuit: ngspice simulates it, and each net is read with the same measurements
the oscilloscope uses (lunes.signals.measure), so a reading on the bench and a simulated one compare number to number.

A circuit is solved for its DC operating point and also run in time from the moment the supply comes on. A net that keeps moving after the start
(an oscillator, a clocked output) is described by its waveform after the first third of the run: mean, peak to peak,
frequency, duty cycle, shape. A net that settles is described by its DC voltage. When something is still moving but
not repeating (a slow oscillator, a big capacitor charging), the run is made ten times longer, up to ten seconds.

ngspice (BSD licence) lives in ngspice/ next to the code; see the README for where to get it.
"""
import math
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from lunes.circuit import _net, faults, netlist, nets_of, value, with_fault
from lunes.signals import measure

ROOT = Path(__file__).resolve().parent.parent
NGSPICE = ROOT / "ngspice" / "bin" / "ngspice_con.exe"
MODELS = Path(__file__).resolve().parent / "spice" / "models.lib"
POINTS = 40000  # samples per run: plenty for the measurements, quick for ngspice
OPTIONS = ".options reltol=1e-3 abstol=1e-10 vntol=1e-5 method=gear itl4=50"


def _run(lines, control, timeout=120):
    """Runs ngspice in batch mode; returns (written data file text or None, ngspice's own output)."""
    if not NGSPICE.exists():
        raise RuntimeError(f"ngspice is not installed at {NGSPICE} (see the README)")
    with tempfile.TemporaryDirectory() as folder:
        cir, data = Path(folder) / "circuit.cir", Path(folder) / "data.txt"
        cir.write_text("\n".join(["* Lunes", f'.include "{MODELS.as_posix()}"', OPTIONS, *lines, ".control",
                                  *control(data.as_posix()), "quit", ".endc", ".end"]) + "\n", encoding="ascii")
        r = subprocess.run([str(NGSPICE), "-b", str(cir)], capture_output=True, text=True, timeout=timeout,
                           cwd=NGSPICE.parent)
        return (data.read_text() if data.exists() else None), r.stdout + r.stderr


def _table(text):
    """wrdata with a single time column and names: (names, numpy array with time first)."""
    rows = [line.split() for line in text.strip().splitlines()]
    names = rows[0]
    values = np.array([[float(x) for x in row] for row in rows[1:] if len(row) == len(names)])
    return names, values


def operating_point(circuit):
    """{spice node: DC volts} at the operating point, or None if ngspice cannot solve it."""
    nodes = sorted({_net(n) for n in nets_of(circuit)} - {"0"})
    text, log = _run(netlist(circuit), lambda out: ["set wr_singlescale", "set wr_vecnames", "op",
                                                    f"wrdata {out} " + " ".join(f"v({n})" for n in nodes)])
    if text is None:
        return None
    names, values = _table(text)
    return {n: float(values[0][names.index(f"v({n})")]) for n in nodes if f"v({n})" in names}


def transient(circuit, stop_s, nodes, timeout=15):
    """(rate, {spice node: volts array}) for a run of stop_s seconds, on an even time grid; None if it fails. A run
    that takes too long (something racing at megahertz) is tried once more ten times shorter."""
    try:
        return _transient(circuit, stop_s, nodes, timeout)
    except subprocess.TimeoutExpired:
        try:
            return _transient(circuit, stop_s / 10, nodes, timeout)
        except subprocess.TimeoutExpired:
            return None


def _transient(circuit, stop_s, nodes, timeout):
    step = stop_s / POINTS
    # from zero, as when the supply is switched on: every capacitor empty. Started from its DC operating point instead,
    # an oscillator can sit at the balance point between its two states and never start.
    for extra in (["uic"], []):
        text, log = _run(netlist(circuit), lambda out: [
            "set wr_singlescale", "set wr_vecnames", f"tran {step:g} {stop_s:g} 0 {step:g} {' '.join(extra)}",
            "linearize", f"wrdata {out} " + " ".join(f"v({n})" for n in nodes)], timeout=timeout)
        if text:
            names, values = _table(text)
            return 1 / step, {n: values[:, names.index(f"v({n})")] for n in nodes if f"v({n})" in names}
    return None


def expected(circuit, stop_s=0.02):
    """{net: what the scope should read there}: the healthy circuit's simulation, read like a measurement."""
    nets = nets_of(circuit)
    nodes = {net: _net(net) for net in nets if _net(net) != "0"}
    dc = operating_point(circuit) or {}
    out = {}
    while True:
        run = transient(circuit, stop_s, sorted(set(nodes.values())))
        if run is None:
            raise RuntimeError("ngspice could not simulate this circuit")
        rate, waves = run
        moving_unsettled = False
        for net, node in nodes.items():
            v = waves[node][len(waves[node]) // 3:]
            m = measure(v, rate)
            swing = m["vpp_v"] > max(0.05, 0.02 * value(circuit.get("supply_v", 5)))
            if swing and m.get("frequency_hz") is None:
                moving_unsettled = True
            out[net] = {"touches": nets[net], "mean_v": m["mean_v"], "vpp_v": m["vpp_v"], "shape": m["shape"]}
            if m.get("frequency_hz"):
                out[net].update(frequency_hz=m["frequency_hz"], duty=m.get("duty"), min_v=m["min_v"],
                                max_v=m["max_v"])
            elif not swing and node in dc:
                out[net].update(mean_v=round(dc[node], 4), vpp_v=0.0, shape="dc")
        if not moving_unsettled or stop_s >= 10:
            break
        stop_s *= 10
    out["GND"] = {"touches": nets.get("GND", []), "mean_v": 0.0, "vpp_v": 0.0, "shape": "dc"}
    return {"nets": out, "simulated_s": stop_s}


CACHE = ROOT / "runs" / "fault_dictionaries"


def fault_dictionary(circuit, stop_s=None, workers=6):
    """{fault id: {"says": words, "nets": expected nets or None if the simulation failed}} plus "healthy". Kept on
    disk by the circuit's content, so the same circuit is simulated once (about a minute for twenty faults)."""
    import hashlib
    import json
    key = hashlib.sha1((json.dumps(circuit, sort_keys=True) + MODELS.read_text()).encode()).hexdigest()[:16]
    cached = CACHE / f"{key}.json"
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    healthy = expected(circuit, stop_s or 0.02)
    stop = healthy["simulated_s"]
    found = faults(circuit)

    def one(item):
        fid, says = item
        try:
            return fid, {"says": says, "nets": expected(with_fault(circuit, fid), stop)["nets"]}
        except RuntimeError:
            return fid, {"says": says, "nets": None}

    with ThreadPoolExecutor(workers) as pool:
        table = dict(pool.map(one, found))
    table["healthy"] = {"says": "works as designed", "nets": healthy["nets"]}
    CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(table), encoding="utf-8")
    return table


def waveform(circuit, net, stop_s=0.02):
    """(rate, volts) of one net after the start, for a simulated scope or a picture."""
    node = _net(net)
    rate, waves = transient(circuit, stop_s, [node])
    v = waves[node]
    return rate, v[len(v) // 3:]


def periodic_signal(rate, v, frequency_hz=None):
    """A function of time that repeats the simulated waveform, so a simulated scope can capture any window of it."""
    if frequency_hz:
        period = int(round(rate / frequency_hz))
        whole = max(1, (len(v) // period)) * period
        v = v[-whole:]
    n = len(v)
    return lambda t: np.interp((np.asarray(t) * rate) % n, np.arange(n), v)
