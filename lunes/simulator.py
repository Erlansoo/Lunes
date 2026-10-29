"""What Lunes expects to measure on a circuit: ngspice simulates it, and each net is read with the same measurements
the oscilloscope uses (lunes.signals.measure), so a reading on the bench and a simulated one compare number to number.

A circuit is solved for its DC operating point and also run in time from the moment the supply comes on. A net that
keeps moving after the start (an oscillator, a clocked output) is described by its waveform after the first third of
the run: mean, peak to peak, frequency, duty cycle, shape. A net that settles is described by its DC voltage. When
something is still moving but not repeating (a slow oscillator, a big capacitor charging), the run is made ten times
longer, up to ten seconds. The current through every resistor, diode, LED and transistor and from the supply is
recorded too, for lunes.review.

Made not to fail silently:
- ngspice runs as its own process with a time limit, killed if it hangs; a run that is too slow is tried once more,
  shorter.
- A run that does not converge is tried again with more tolerant settings, and says so.
- ngspice's complaints become sentences in the circuit's own words ("net TH has no DC path to ground"), and a result
  that came out of a singular matrix is refused instead of trusted.
- selftest() checks that ngspice and its digital models answer right on two known circuits.

ngspice (BSD licence) lives in ngspice/ next to the code; see the README for where to get it.
"""
import hashlib
import json
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from lunes.circuit import _net, faults, netlist, nets_of, value, with_fault
from lunes.signals import measure

ROOT = Path(__file__).resolve().parent.parent
NGSPICE = ROOT / "ngspice" / "bin" / "ngspice_con.exe"
MODELS = Path(__file__).resolve().parent / "spice" / "models.lib"
CACHE = ROOT / "runs" / "fault_dictionaries"
POINTS = 40000  # samples per run: plenty for the measurements, quick for ngspice
VERSION = "2"  # of how a simulation is read; a change makes the cached fault dictionaries be simulated again
SETTINGS = [  # tried in order until a run converges
    ("normal", ".options reltol=1e-3 abstol=1e-10 vntol=1e-5 method=gear itl4=50"),
    ("relaxed", ".options reltol=1e-2 abstol=1e-9 vntol=1e-4 method=gear itl4=200 gmin=1e-10"),
    ("trapezoidal", ".options reltol=5e-3 abstol=1e-9 vntol=1e-4 method=trap itl4=500 gmin=1e-9"),
]


class SimulationError(RuntimeError):
    """ngspice could not simulate the circuit; problems says why, in the circuit's words."""

    def __init__(self, problems):
        self.problems = problems
        super().__init__("; ".join(problems) or "ngspice could not simulate this circuit")


def _run(lines, control, timeout=120, options=SETTINGS[0][1]):
    """Runs ngspice in batch mode; returns (written data file text or None, ngspice's own output)."""
    if not NGSPICE.exists():
        raise SimulationError([f"ngspice is not installed at {NGSPICE} (see the README)"])
    with tempfile.TemporaryDirectory() as folder:
        cir, data = Path(folder) / "circuit.cir", Path(folder) / "data.txt"
        cir.write_text("\n".join(["* Lunes", f'.include "{MODELS.as_posix()}"', options, *lines, ".control",
                                  *control(data.as_posix()), "quit", ".endc", ".end"]) + "\n", encoding="ascii")
        r = subprocess.run([str(NGSPICE), "-b", str(cir)], capture_output=True, text=True, timeout=timeout,
                           cwd=NGSPICE.parent)
        return (data.read_text() if data.exists() else None), r.stdout + r.stderr


def problems(log, circuit=None):
    """ngspice's complaints as sentences: (fatal problems, warnings)."""
    names = {_net(n): n for n in nets_of(circuit)} if circuit else {}
    fatal, warn = [], []

    def add(where, text):
        if text not in where:
            where.append(text)

    for raw in log.splitlines():
        line = raw.strip().lower()
        m = re.search(r"singular matrix:\s+check node (\S+)", line)
        if m:
            node = m.group(1)
            if node.endswith("#branch"):
                dev = node[:-len("#branch")]
                who = "the supply" if dev == "vsupply" else dev[1:].upper() if len(dev) > 1 else dev
                add(fatal, f"{who} closes a loop of sources or coils with nothing to limit the current (a coil or a "
                           f"second source straight across a source)")
            else:
                add(fatal, f"net {names.get(node, node)} has no DC path to ground: its voltage is undefined")
            continue
        m = re.search(r"can't find model '([^']+)'", line)
        if m:
            add(fatal, f"unknown model or value '{m.group(1)}'")
            continue
        m = re.search(r"unknown subckt: .* (\S+)$", line)
        if m:
            add(fatal, f"unknown chip '{m.group(1)}'")
            continue
        m = re.search(r"unknown parameter \(([^)]*)\)", line)
        if m:
            add(fatal, f"a value ngspice cannot read: '{m.group(1)}'")
            continue
        if "timestep too small" in line:
            add(fatal, "the simulation could not follow the circuit in time: an edge too fast to resolve, or "
                       "values that make it race or never settle")
        elif "gmin stepping failed" in line or "source stepping failed" in line or "dc solution failed" in line:
            add(warn, "ngspice found no DC operating point (it starts the run from zero instead)")
        elif "unknown model type" in line or "could not find a valid modelname" in line:
            add(fatal, f"ngspice lacks a model it needs ({raw.strip()[:80]}); are its code models installed?")
    return fatal, warn


def _table(text):
    """wrdata with a single time column and names: (names, numpy array with time first)."""
    rows = [line.split() for line in text.strip().splitlines()]
    names = rows[0]
    values = np.array([[float(x) for x in row] for row in rows[1:] if len(row) == len(names)])
    return names, values


def _devices(lines, circuit):
    """{part ref: ngspice current vector} for the resistors, diodes and transistors that are in the netlist as
    themselves (a fault replaces some), plus the supply."""
    refs = {p["ref"]: p["kind"] for p in circuit["parts"]}
    out = {"supply": "i(vsupply)"}
    for line in lines:
        name = line.split()[0] if line.strip() and not line.startswith("*") else ""
        letter, ref = name[:1], name[1:]
        if ref in refs:
            if letter == "R" and refs[ref] == "R":
                out[ref] = f"@r{ref.lower()}[i]"
            elif letter == "D" and refs[ref] in ("D", "LED"):
                out[ref] = f"@d{ref.lower()}[id]"
            elif letter == "Q" and refs[ref] in ("NPN", "PNP"):
                out[ref] = f"@q{ref.lower()}[ic]"
    return out


def operating_point(circuit):
    """{spice node: DC volts} at the operating point, or {} if ngspice cannot solve it."""
    nodes = sorted({_net(n) for n in nets_of(circuit)} - {"0"})
    text, log = _run(netlist(circuit), lambda out: ["set wr_singlescale", "set wr_vecnames", "op",
                                                    f"wrdata {out} " + " ".join(f"v({n})" for n in nodes)],
                     timeout=30)
    if text is None or problems(log, circuit)[0]:
        return {}
    names, values = _table(text)
    return {n: float(values[0][names.index(f"v({n})")]) for n in nodes if f"v({n})" in names}


def transient(circuit, stop_s, nodes, timeout=15):
    """(rate, {spice node or "@" + part ref: array}, settings used, warnings) for a run of stop_s seconds on an even
    time grid. Raises SimulationError when no setting gives a result ngspice itself trusts."""
    lines = netlist(circuit)
    devices = _devices(lines, circuit)
    last = []
    for label, options in SETTINGS:
        timed_out = 0
        for stop in (stop_s, stop_s / 10):  # a run that is too slow (racing at megahertz) is tried shorter
            step = stop / POINTS
            vectors = [f"v({n})" for n in nodes] + list(devices.values())
            try:
                # from zero, as when the supply is switched on: every capacitor empty. Started from its DC operating
                # point instead, an oscillator can sit at the balance point between its two states and never start.
                text, log = _run(lines, lambda out: [
                    "set wr_singlescale", "set wr_vecnames", "save all " + " ".join(devices.values()),
                    f"tran {step:g} {stop:g} 0 {step:g} uic", "linearize", f"wrdata {out} " + " ".join(vectors)],
                    timeout=timeout, options=options)
            except subprocess.TimeoutExpired:
                last = ["the simulation took too long (something racing at megahertz?)"]
                timed_out += 1
                continue
            fatal, warn = problems(log, circuit)
            if fatal:
                if any("no DC path" in f or "loop of sources" in f or "unknown" in f or "cannot read" in f
                       or "lacks a model" in f for f in fatal):
                    raise SimulationError(fatal)  # no setting fixes a wiring or naming mistake
                last = fatal
                break  # try the next, more tolerant setting
            if not text:
                last = ["ngspice wrote no results"]
                break
            names, values = _table(text)
            data = {n: values[:, names.index(f"v({n})")] for n in nodes if f"v({n})" in names}
            for ref, vec in devices.items():
                if vec in names:
                    data["@" + ref] = values[:, names.index(vec)]
            if label != "normal":
                warn.append(f"simulated with {label} settings, after the normal ones did not converge: trust the "
                            f"numbers a little less")
            if stop != stop_s:
                warn.append(f"simulated only {stop:g} s instead of {stop_s:g} s: the full run took too long")
            return 1 / step, data, label, warn
        if timed_out == 2:
            break  # more tolerant settings make a racing circuit no faster
    raise SimulationError(last or ["ngspice could not simulate this circuit"])


def expected(circuit, stop_s=0.02):
    """{"nets": {net: what the scope should read there}, "currents": {part: peak, mean, power}, "warnings": [...],
    "simulated_s"}: the healthy circuit's simulation, read like a measurement."""
    nets = nets_of(circuit)
    nodes = {net: _net(net) for net in nets if _net(net) != "0"}
    # the DC operating point is the settled state only when every source is steady; with a square or sine source it
    # is just the instant before the first edge
    steady_sources = not any(p["kind"] == "V" and isinstance(p.get("value"), dict) for p in circuit["parts"])
    dc = operating_point(circuit) if steady_sources else {}
    while True:
        rate, waves, label, warn = transient(circuit, stop_s, sorted(set(nodes.values())))
        out, moving_unsettled = {}, False
        for net, node in nodes.items():
            v = waves[node][len(waves[node]) // 3:]
            m = measure(v, rate)
            swing = m["vpp_v"] > max(0.05, 0.02 * value(circuit.get("supply_v", 5)))
            if swing and m.get("frequency_hz") is None:
                moving_unsettled = True
            out[net] = {"touches": nets[net], "mean_v": m["mean_v"], "vpp_v": m["vpp_v"], "shape": m["shape"],
                        "min_v": m["min_v"], "max_v": m["max_v"]}
            if m.get("frequency_hz"):
                out[net].update(frequency_hz=m["frequency_hz"], duty=m.get("duty"))
            elif not swing and node in dc:
                out[net].update(mean_v=round(dc[node], 4), vpp_v=0.0, shape="dc", min_v=round(dc[node], 4),
                                max_v=round(dc[node], 4))
        if not moving_unsettled or stop_s >= 10:
            break
        stop_s *= 10
    currents = {}
    resistors = {p["ref"]: value(p["value"]) for p in circuit["parts"] if p["kind"] == "R" and "value" in p}
    for key, i in waves.items():
        if key.startswith("@"):
            ref, i = key[1:], np.abs(i[len(i) // 3:])
            currents[ref] = {"peak_a": float(f"{i.max():.4g}"), "mean_a": float(f"{i.mean():.4g}")}
            if ref in resistors:
                currents[ref]["power_w"] = float(f"{float(np.mean(i * i)) * resistors[ref]:.4g}")
    out["GND"] = {"touches": nets.get("GND", []), "mean_v": 0.0, "vpp_v": 0.0, "shape": "dc", "min_v": 0.0,
                  "max_v": 0.0}
    return {"nets": out, "currents": currents, "warnings": warn, "settings": label, "simulated_s": stop_s}


def fault_dictionary(circuit, stop_s=None, workers=6):
    """{fault id: {"says": words, "nets": expected nets, or None and "problem" if it could not be simulated}} plus
    "healthy". Kept on disk by the circuit's content, so the same circuit is simulated once (about half a minute
    for twenty faults)."""
    key = hashlib.sha1((json.dumps(circuit, sort_keys=True) + MODELS.read_text() + VERSION).encode()).hexdigest()[:16]
    cached = CACHE / f"{key}.json"
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    healthy = expected(circuit, stop_s or 0.02)
    stop = healthy["simulated_s"]

    def one(item):
        fid, says = item
        try:
            return fid, {"says": says, "nets": expected(with_fault(circuit, fid), stop)["nets"]}
        except SimulationError as e:
            return fid, {"says": says, "nets": None, "problem": str(e)}

    with ThreadPoolExecutor(workers) as pool:
        table = dict(pool.map(one, faults(circuit)))
    table["healthy"] = {"says": "works as designed", "nets": healthy["nets"]}
    CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(table), encoding="utf-8")
    return table


def waveform(circuit, net, stop_s=0.02):
    """(rate, volts) of one net after the start, for a simulated scope or a picture."""
    node = _net(net)
    rate, waves, _, _ = transient(circuit, stop_s, [node])
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


SELFTEST_555 = {"name": "self-test 555", "supply_v": 5, "parts": [
    {"ref": "U1", "kind": "NE555", "pins": {"GND": "GND", "TRIG": "TH", "OUT": "OUT", "RESET": "VCC", "CTRL": "CV",
                                           "THRES": "TH", "DISCH": "DIS", "VCC": "VCC"}},
    {"ref": "R1", "kind": "R", "value": "1k", "pins": ["VCC", "DIS"]},
    {"ref": "R2", "kind": "R", "value": "10k", "pins": ["DIS", "TH"]},
    {"ref": "C1", "kind": "C", "value": "68n", "pins": ["TH", "GND"]},
    {"ref": "C2", "kind": "C", "value": "10n", "pins": ["CV", "GND"]}]}
SELFTEST_DIVIDER = {"name": "self-test divider", "supply_v": 5, "parts": [
    {"ref": "R1", "kind": "R", "value": "10k", "pins": ["VCC", "MID"]},
    {"ref": "R2", "kind": "R", "value": "4.7k", "pins": ["MID", "GND"]}]}


def selftest():
    """Does ngspice answer right? A divider (1.599 V) and a 555 astable (about 1 kHz, needs the digital models)."""
    t0, found = time.time(), {}
    try:
        found["divider_v"] = expected(SELFTEST_DIVIDER)["nets"]["MID"]["mean_v"]
        found["astable_hz"] = expected(SELFTEST_555)["nets"]["OUT"].get("frequency_hz")
    except (SimulationError, subprocess.TimeoutExpired, OSError) as e:
        return {"ok": False, "problem": str(e), "seconds": round(time.time() - t0, 1)}
    ok = (abs(found["divider_v"] - 1.5986) < 0.01 and found["astable_hz"] is not None
          and abs(found["astable_hz"] / 1000 - 1) < 0.05)
    return {"ok": bool(ok), **found, "seconds": round(time.time() - t0, 1),
            **({} if ok else {"problem": "ngspice answers, but wrong: expected 1.599 V and about 1000 Hz"})}


def kill_strays():
    """Ends ngspice processes of this installation left running (Lunes stopped in the middle of a simulation);
    returns how many."""
    script = (f"$p = Get-CimInstance Win32_Process -Filter \"Name='ngspice_con.exe'\" | Where-Object "
              f"{{ $_.ExecutablePath -eq '{NGSPICE}' }}; $p | ForEach-Object {{ Stop-Process -Id $_.ProcessId "
              f"-Force }}; @($p).Count")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True, timeout=30)
    try:
        return int(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return 0
