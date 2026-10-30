"""Finding the fault from measurements: which net to measure next, and what each reading says about the circuit.

Every hypothesis (the circuit healthy, or one part broken one typical way) comes with what the scope would read at
each net (lunes.simulator.fault_dictionary). Each reading makes the hypotheses that predicted it more likely and the
rest less (Bayes' rule), with room for the parts' tolerances (5 % resistors, 10 to 20 % capacitors) and the scope's
error. The next net to measure is the one whose reading is expected to tell the remaining hypotheses apart best: the
largest expected drop in entropy.

Faults that read the same at every net are reported together: no measurement separates them, but a look can (a
colour code, a part missing). One more hypothesis, "none of these", stands for what the dictionary does not hold (a
wire in the wrong row, two faults at once, no supply): it wins when no hypothesis explains the readings, so Lunes
says it does not know instead of naming the closest wrong fault.
"""
import math

import numpy as np

OTHER = "other"
UNPREDICTABLE = 0.05  # likelihood of any reading under a hypothesis whose simulation failed
VISIBLE_SWING_V = 0.1  # a swing smaller than this is lost in the scope's steps over a few volts of range


def _gauss(x, sigma):
    return math.exp(-0.5 * (x / sigma) ** 2) / sigma


def likelihood(pred, meas):
    """How likely the reading meas is if the net really behaves as pred (both like signals.measure's output)."""
    if pred is None:
        return UNPREDICTABLE
    like = 1.0
    high, low = meas.get("clipped_high"), meas.get("clipped_low")
    if meas.get("mean_v") is not None and not (high and low):
        off = meas["mean_v"] - pred["mean_v"]
        if high:  # the instrument stopped at its top: the net is at least this high
            off = max(off, 0.0)
        elif low:  # stopped at its bottom: at most this high
            off = min(off, 0.0)
        like *= _gauss(off, 0.06 + 0.05 * abs(pred["mean_v"]))
    if "frequency_hz" in meas:
        fp, fm = pred.get("frequency_hz"), meas.get("frequency_hz")
        if fp and fm is None and pred["vpp_v"] < VISIBLE_SWING_V:
            pass  # a ripple too small for the scope to see: not seeing it says nothing
        elif (fp is None) != (fm is None):
            like *= 0.02  # one oscillates and the other does not
        elif fp and fm:
            like *= _gauss(math.log(fm / fp), 0.15)
            if pred.get("duty") is not None and meas.get("duty") is not None:
                like *= _gauss(meas["duty"] - pred["duty"], 0.06)
    if meas.get("vpp_v") is not None:
        off = meas["vpp_v"] - pred["vpp_v"]
        if high or low:  # clipped: the swing is at least what was seen
            off = min(off, 0.0)
        like *= _gauss(off, 0.1 + 0.08 * pred["vpp_v"])
    return max(like, 1e-300)


def _seen_frequency(pred):
    """The frequency a scope would find on a net that behaves like pred: none when its swing is too small to see."""
    return pred.get("frequency_hz") if pred["vpp_v"] >= VISIBLE_SWING_V else None


def _other_likelihood(meas, supply_v):
    """A reading that no hypothesis predicts: about as likely as any value in the supply's range."""
    like = 1.0
    if meas.get("mean_v") is not None:
        like *= 1 / (supply_v + 1)
    if "frequency_hz" in meas:
        like *= 0.5 if meas.get("frequency_hz") is None else 0.5 / math.log(1e6)
    if meas.get("vpp_v") is not None:
        like *= 1 / (supply_v + 1)
    return like


def _entropy(p):
    q = np.array([x for x in p if x > 0])
    return float(-(q * np.log2(q)).sum())


class Diagnosis:
    def __init__(self, circuit, table, prior_healthy=0.4, prior_other=0.03):
        """table: lunes.simulator.fault_dictionary(circuit)."""
        self.circuit, self.table = circuit, table
        self.supply_v = float(circuit.get("supply_v", 5))
        self.hypotheses = [h for h in table if h != "healthy"] + ["healthy", OTHER]
        n_faults = len(self.hypotheses) - 2
        self.p = {h: (1 - prior_healthy - prior_other) / max(n_faults, 1) for h in self.hypotheses}
        self.p["healthy"], self.p[OTHER] = prior_healthy, prior_other
        self.nets = [n for n in table["healthy"]["nets"] if n != "GND"]
        self.readings = []  # (net, reading)
        self.groups = self._groups()

    def predicted(self, h, net):
        if h == OTHER:
            return None
        nets = self.table[h]["nets"]
        return None if nets is None else nets.get(net)

    def _groups(self):
        """Hypotheses that read the same at every net, within what a measurement can tell apart."""
        groups = []
        for h in self.hypotheses:
            if h == OTHER:
                groups.append([h])
                continue
            for g in groups:
                ref = g[0]
                if ref != OTHER and all(self._alike(self.predicted(ref, n), self.predicted(h, n)) for n in self.nets):
                    g.append(h)
                    break
            else:
                groups.append([h])
        return groups

    @staticmethod
    def _alike(a, b):
        if a is None or b is None:
            return a is b
        if abs(a["mean_v"] - b["mean_v"]) > 0.08 + 0.03 * abs(a["mean_v"]):
            return False
        fa, fb = _seen_frequency(a), _seen_frequency(b)
        if (fa is None) != (fb is None) or (fa and abs(math.log(fa / fb)) > 0.1):
            return False
        return abs(a["vpp_v"] - b["vpp_v"]) <= 0.1 + 0.05 * a["vpp_v"]

    def _group_p(self, p):
        return [sum(p[h] for h in g) for g in self.groups]

    def _posterior(self, net, reading, p=None):
        p = dict(p or self.p)
        for h in self.hypotheses:
            p[h] *= (_other_likelihood(reading, self.supply_v) if h == OTHER
                     else likelihood(self.predicted(h, net), reading))
        total = sum(p.values())
        return {h: v / total for h, v in p.items()}

    def update(self, net, reading):
        """A reading at net: {"mean_v", "vpp_v", "frequency_hz" (None if it does not oscillate), "duty"}. Only what
        the instrument measured: a multimeter gives mean_v alone, and leaving frequency_hz out (not None) says
        nothing about whether the net oscillates."""
        if net not in self.nets:
            raise ValueError(f"no net {net} in this circuit; nets: {self.nets}")
        reading = {k: reading[k] for k in ("mean_v", "vpp_v", "frequency_hz", "duty", "clipped_high", "clipped_low")
                   if k in reading}
        self.readings.append((net, reading))
        self.p = self._posterior(net, reading)

    def next_net(self):
        """(net, expected information in bits, the hypotheses it separates) to measure next, or None."""
        # a net counts as measured once the scope saw it; a multimeter or a board's input gives its mean only, so
        # the scope may still be worth putting there
        measured = {n for n, r in self.readings if "frequency_hz" in r}
        if "VCC" in self.nets and "VCC" not in {n for n, _ in self.readings}:
            return "VCC", None, ["the supply first: without it, nothing else means anything"]
        h0 = _entropy(self._group_p(self.p))
        best = None
        likely = [h for h in self.hypotheses if self.p[h] > 1e-4 and h != OTHER]
        for net in self.nets:
            if net in measured:
                continue
            expected_h = 0.0
            for h in likely:
                pred = self.predicted(h, net)
                if pred is None:
                    continue
                outcome = {"mean_v": pred["mean_v"], "vpp_v": pred["vpp_v"],
                           "frequency_hz": _seen_frequency(pred), "duty": pred.get("duty")}
                expected_h += self.p[h] * _entropy(self._group_p(self._posterior(net, outcome)))
            gain = h0 - expected_h / max(sum(self.p[h] for h in likely), 1e-12)
            if best is None or gain > best[1]:
                best = (net, gain)
        if best is None or best[1] < 0.02:
            return None
        net = best[0]
        values = {}
        for g in self._ranked()[:6]:
            pred = self.predicted(g[0], net)
            if pred is not None:
                values.setdefault(self._describe(pred), []).append(self._name(g))
        return net, round(best[1], 2), [f"{k}: {', '.join(v)}" for k, v in values.items()]

    @staticmethod
    def _describe(pred):
        if pred.get("frequency_hz"):
            return f"{pred['frequency_hz']:.4g} Hz, {pred['vpp_v']:.2f} Vpp, mean {pred['mean_v']:.2f} V"
        return f"steady {pred['mean_v']:.2f} V"

    def _name(self, group):
        if "healthy" in group:
            others = [self.table[h]["says"] for h in group if h != "healthy"]
            return "works as designed" + (f" (these faults would read the same and need a look instead: "
                                          f"{'; '.join(others)})" if others else "")
        return " or ".join("none of the typical faults (a wire in the wrong row, two faults, no supply?)"
                           if h == OTHER else self.table[h]["says"] for h in group)

    def _ranked(self):
        return sorted(self.groups, key=lambda g: -sum(self.p[h] for h in g))

    def state(self):
        """What Lunes reports: the likeliest explanations, whether one is clear, and what to measure next."""
        ranked = self._ranked()
        top = [{"faults": g, "says": self._name(g), "p": round(sum(self.p[h] for h in g), 3)} for g in ranked[:4]]
        clear = top[0]["p"] >= 0.9
        out = {"likely": top, "clear": clear, "readings": [{"net": n, **r} for n, r in self.readings]}
        if clear:
            out["verdict"] = top[0]["says"]
            if len(top[0]["faults"]) > 1:
                out["note"] = ("these read the same at every net, so measuring cannot separate them: look at the "
                               "parts (colour codes, a missing or loose lead)")
        else:
            nxt = self.next_net()
            if nxt is None:
                out["note"] = "no measurement left that separates these: look at the parts, or check the wiring"
            else:
                net, bits, why = nxt
                out["next"] = {"net": net, "touches": self.table["healthy"]["nets"][net]["touches"],
                               "expected_bits": bits, "would_read": why,
                               "if_healthy": self._describe(self.table["healthy"]["nets"][net])}
        return out
