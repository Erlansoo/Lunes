"""A signal measured the way a bench oscilloscope measures it, and a picture of it for Lunes and Erlan.

Plain signal processing on the scope's samples, no model:
- levels: min, max, peak to peak, mean (DC), RMS, and the low and high levels of a digital or PWM signal;
- timing: frequency, period, duty cycle, rise and fall times (10 to 90 %);
- a shape: dc, sine, triangle, square, pwm, digital (two levels, not periodic), periodic, irregular or noise.
notes() says in short sentences what the numbers imply for the measurement itself (it clips the range, too few
samples per period, flat at 0 V, mains hum), so Lunes knows how far to trust it before judging the circuit.

autoset() does what the Auto button of a bench scope does: it looks at three sample rates on the widest range, then
picks the narrowest range that holds the signal whole and a sample rate that shows about ten periods.
"""
import cv2
import numpy as np

from lunes.hantek import MAX_RATE, rates, ranges

COLOURS = {"CH1": (40, 220, 245), "CH2": (240, 200, 60)}  # BGR: yellow and cyan, as on most scopes


def si(value, unit, digits=3):
    """1234.5, "Hz" -> "1.23 kHz"."""
    if value is None:
        return "?"
    if value == 0:
        return f"0 {unit}"
    a = abs(value)
    for factor, prefix in ((1e6, "M"), (1e3, "k"), (1, ""), (1e-3, "m"), (1e-6, "u"), (1e-9, "n")):
        if a >= factor or factor == 1e-9:
            return f"{value / factor:.{digits}g} {prefix}{unit}"
    return f"{value:g} {unit}"


def _round(x, digits=4):
    return None if x is None else float(f"{x:.{digits}g}")


def _edges(v, level, hysteresis):
    """Sub-sample positions where v rises through level and where it falls; with hysteresis, so noise riding on a
    slow edge counts once."""
    s = np.where(v > level + hysteresis, 1, np.where(v < level - hysteresis, -1, 0))
    known = np.flatnonzero(s)
    if known.size == 0:
        return np.array([]), np.array([])
    last = np.zeros(len(s), np.int64)
    last[known] = known
    state = s[np.maximum.accumulate(last)]
    state[:known[0]] = s[known[0]]
    rising, falling = [], []
    for i in np.flatnonzero(np.diff(state)) + 1:
        up, k = state[i] > 0, i
        while k > 0 and ((v[k - 1] > level) if up else (v[k - 1] < level)):  # back to where it crossed level
            k -= 1
        a, b = v[k - 1], v[k]
        x = k - 1 + (level - a) / (b - a) if k > 0 and b != a else float(k)
        (rising if up else falling).append(x)
    return np.array(rising), np.array(falling)


def _transition(v, x, start, stop, rising):
    """Samples taken to go from 10 % to 90 % of the way (start to stop) around the edge at x."""
    lo, hi = start + 0.1 * (stop - start), start + 0.9 * (stop - start)
    j = int(x)
    a = j
    while a > 0 and ((v[a] > lo) if rising else (v[a] < lo)):
        a -= 1
    b = j
    while b < len(v) - 1 and ((v[b] < hi) if rising else (v[b] > hi)):
        b += 1
    return b - a


BAUDS = (300, 1200, 2400, 4800, 9600, 19200, 38400, 57600, 115200, 230400, 250000, 460800, 500000, 921600, 1e6)


def measure(v, rate, step_v=0.0):
    """What a bench scope's measurement menu shows for one channel, in volts and seconds. step_v is one step of
    the ADC at the range used: a swing of a few steps is the converter, not a signal."""
    v = np.asarray(v, np.float64)
    n = len(v)
    vmin, vmax, mean = float(v.min()), float(v.max()), float(v.mean())
    vpp = vmax - vmin
    ac_rms = float(np.std(v))
    noise = float(1.4826 * np.median(np.abs(np.diff(v))) / np.sqrt(2))  # robust: edges barely move the median
    noise = max(noise, step_v / np.sqrt(12))  # an 8-bit converter cannot tell less than its own step
    m = {"samples": n, "sample_rate": rate, "window_s": _round(n / rate), "min_v": _round(vmin), "max_v": _round(vmax),
         "vpp_v": _round(vpp), "mean_v": _round(mean), "rms_v": _round(float(np.sqrt(np.mean(v * v)))),
         "ac_rms_v": _round(ac_rms), "noise_v": _round(noise)}
    if vpp <= max(6 * noise, 3.5 * step_v, 0.02):
        m["shape"] = "dc"
        return m

    mid = (vmin + vmax) / 2
    low, high = float(np.median(v[v < mid])), float(np.median(v[v >= mid]))
    near = (np.abs(v - low) < 0.15 * vpp) | (np.abs(v - high) < 0.15 * vpp)
    two_level = float(near.mean()) > 0.8
    level = (low + high) / 2 if two_level else mean
    rising, falling = _edges(v, level, max(0.1 * vpp, 3 * noise))
    periodic = False
    if len(rising) >= 3:
        gaps = np.diff(rising)
        periodic = float(np.std(gaps) / np.mean(gaps)) < 0.08
    if periodic:
        period = float(np.mean(np.diff(rising))) / rate
        m.update(frequency_hz=_round(1 / period), period_s=_round(period), periods_seen=len(rising) - 1,
                 samples_per_period=_round(period * rate, 3))
    if two_level:
        m.update(low_v=_round(low), high_v=_round(high), amplitude_v=_round(high - low))
        rises = [_transition(v, x, low, high, True) for x in rising[:50]]
        falls = [_transition(v, x, high, low, False) for x in falling[:50]]
        if rises:
            m["rise_s"] = _round(float(np.median(rises)) / rate)
        if falls:
            m["fall_s"] = _round(float(np.median(falls)) / rate)
        m["overshoot"] = _round(max(vmax - high, low - vmin) / (high - low), 3)
        if periodic:
            highs = []
            for r in rising[:-1]:
                after = falling[falling > r]
                if after.size:
                    highs.append(after[0] - r)
            duty = float(np.mean(highs)) / (period * rate) if highs else None
            m["duty"] = _round(duty, 3)
            m["shape"] = "square" if duty is not None and abs(duty - 0.5) < 0.05 else "pwm"
        else:
            m["shape"] = "digital"
            edges = np.sort(np.concatenate([rising, falling]))
            if edges.size >= 6:  # the shortest pulse of serial data is one bit
                shortest = float(np.percentile(np.diff(edges), 5)) / rate
                m["shortest_pulse_s"] = _round(shortest)
                guess = min(BAUDS, key=lambda b: abs(np.log(b * shortest)))
                if abs(guess * shortest - 1) < 0.08:
                    m["baud_if_uart"] = guess
        return m

    if periodic:
        crest = (vmax - mean) / ac_rms if ac_rms else 0
        spectrum = np.abs(np.fft.rfft(v - mean)) ** 2
        k = int(round(m["frequency_hz"] * n / rate))
        purity = float(spectrum[max(1, k - 2):k + 3].sum() / spectrum.sum()) if spectrum.sum() else 0
        m["fundamental_share"] = _round(purity, 3)
        m["shape"] = ("sine" if 1.3 < crest < 1.55 and purity > 0.9 else
                      "triangle" if 1.6 < crest < 1.85 else "periodic")
    else:
        m["shape"] = "noise" if ac_rms < 4 * noise else "irregular"
    return m


def notes(m, range_v, clipped, widest_v, narrowest_v):
    """Short facts about the measurement itself, for Lunes to weigh before it judges the circuit."""
    out = []
    if clipped > 0.001:
        out.append(f"the signal goes past the ±{range_v:g} V range ({clipped:.0%} of the samples clip), so its top "
                   f"and bottom are cut off: " + ("use a wider range" if range_v < widest_v else
                                                  "it is past the widest range; set the probe to x10, or the "
                                                  "signal is too big for this scope"))
    if m["shape"] == "dc" and abs(m["mean_v"]) < max(0.05, 3 * m["noise_v"]):
        out.append("flat at about 0 V: the tip may not be touching, the ground clip may be off, or the node has no "
                   "voltage")
    if m["shape"] in ("irregular", "noise"):
        out.append("no repeating period in this window: the signal may be slower than the window (lower the sample "
                   "rate), irregular data, or noise")
    spp = m.get("samples_per_period")
    if spp is not None and spp < 10:
        out.append(f"only {spp:g} samples per period: raise the sample rate, the shape and timing are rough")
    f = m.get("frequency_hz")
    if f and 45 <= f <= 55 and m["vpp_v"] < 0.5:
        out.append("about 50 Hz and small: probably mains hum picked up by the probe (check the ground clip)")
    if f and f > 5e6:
        out.append("above 5 MHz the scope's 20 MHz front end rounds the edges: amplitude and rise time read low")
    rise = m.get("rise_s")
    if rise is not None and rise * m["sample_rate"] <= 2:
        out.append("the edges are faster than this sample rate resolves (normal for digital signals): the rise "
                   "time is only an upper bound")
    if m.get("overshoot") and m["overshoot"] > 0.15:
        out.append(f"overshoot of {m['overshoot']:.0%} on the edges: ringing (long ground lead, or the circuit)")
    if m["vpp_v"] < 0.1 * range_v and m["shape"] != "dc" and not clipped:
        out.append("the signal uses little of the range: a narrower range shows it finer" if range_v > narrowest_v
                   else "small for this probe: even the narrowest range shows it coarsely; set the probe to x1 for "
                        "signals under a volt or so")
    if m["shape"] == "dc" and range_v == narrowest_v and narrowest_v >= 5:
        out.append(f"with the x10 probe one step of the converter is {range_v / 125 * 1000:.0f} mV: ripple smaller "
                   f"than that does not show (x1 probe for small signals)")
    return out


def _pick_rate(target, channels, floor=0.0):
    """The available sample rate nearest to target, not below floor (both in samples per second)."""
    allowed = [r for r in rates(channels) if r >= floor] or [MAX_RATE[channels]]
    return min(allowed, key=lambda r: abs(np.log(r / target)))


def step_v(range_v):
    """One ADC step at a range (volts at the tip): the 8-bit converter spans 1.024 times the range each way."""
    return 2 * 1.024 * range_v / 256


def autoset(scope, channel="CH1", samples=20000, alone=False):
    """Range and sample rate for what is on channel. alone=True lets CH1 run by itself when a fast signal needs
    twice the sample rate (CH2 then reads nothing). Returns (capture, measurement, what it tried)."""
    widest = ranges(scope.probe[channel])[0]
    tried, best, peak = [], None, 0.0
    for rate in (20e3, 1e6, MAX_RATE[scope.channels]):
        scope.configure(rate=rate, range_v={channel: widest})
        cap = scope.capture(samples)
        m = measure(cap.volts[channel], rate, step_v(widest))
        tried.append({"rate": rate, "frequency_hz": m.get("frequency_hz"), "vpp_v": m["vpp_v"], "shape": m["shape"]})
        peak = max(peak, abs(m["min_v"]), abs(m["max_v"]))
        if cap.clipped[channel] > 0.001:
            return cap, m, tried  # past the widest range: nothing narrower will hold it
        if m.get("frequency_hz") and m.get("periods_seen", 0) >= 3:
            best = m  # the fastest rate that still sees three periods times them best
    fitting = [r for r in ranges(scope.probe[channel]) if r >= 1.15 * peak]
    range_v = fitting[-1] if fitting else widest
    channels = scope.channels
    if best:
        f = best["frequency_hz"]
        if alone and channel == "CH1" and 25 * f > MAX_RATE[channels]:
            channels = 1
        rate = _pick_rate(f * samples / 10, channels, floor=min(25 * f, MAX_RATE[channels]))
    elif tried[-1]["shape"] in ("digital", "irregular"):
        rate = MAX_RATE[scope.channels]
    else:
        rate = 1e6
    scope.configure(rate=rate, range_v={channel: range_v}, channels=channels)
    cap = scope.capture(samples)
    return cap, measure(cap.volts[channel], rate, step_v(range_v)), tried


def fit_range(scope, channel, samples=20000):
    """The narrowest range that holds what is on channel, at the sample rate already set; returns it."""
    allowed = ranges(scope.probe[channel])
    scope.configure(range_v={channel: allowed[0]})
    v = scope.capture(samples).volts[channel]
    peak = max(abs(float(v.min())), abs(float(v.max())))
    fitting = [r for r in allowed if r >= 1.15 * peak]
    scope.configure(range_v={channel: fitting[-1] if fitting else allowed[0]})
    return scope.range_v[channel]


def _start(v, m):
    """Where the picture starts: the first rising edge after a tenth of the window (a software trigger), so
    repeated pictures of a periodic signal line up."""
    if m.get("frequency_hz") is None:
        return 0
    level = (m["low_v"] + m["high_v"]) / 2 if "low_v" in m else m["mean_v"]
    rising, _ = _edges(v[len(v) // 10:], level, max(0.1 * m["vpp_v"], 3 * m["noise_v"]))
    return len(v) // 10 + int(rising[0]) if rising.size else 0


def picture(capture, measurements, title="", size=(960, 540)):
    """The capture drawn like a scope screen: 10 x 8 divisions, each channel at its own volts per division, the
    time per division and the main measurements written below."""
    w, h = size
    top, plot_h = 34, h - 34 - 78
    img = np.full((h, w, 3), 18, np.uint8)
    for i in range(11):
        x = int(i * (w - 1) / 10)
        cv2.line(img, (x, top), (x, top + plot_h), (70, 70, 70) if i != 5 else (110, 110, 110), 1)
    for j in range(9):
        y = top + int(j * plot_h / 8)
        cv2.line(img, (0, y), (w - 1, y), (70, 70, 70) if j != 4 else (110, 110, 110), 1)
    channels = list(capture.volts)
    first = capture.volts[channels[0]]
    start = _start(first, measurements[channels[0]])
    f = measurements[channels[0]].get("frequency_hz")
    shown = len(first) - start
    if f:  # about four periods across the screen
        shown = min(shown, max(50, int(4 * capture.rate / f)))
    seconds = shown / capture.rate
    header = [f"{title}  " if title else "", f"{si(seconds / 10, 's')}/div  {si(capture.rate, 'S/s')}"]
    cv2.putText(img, "".join(header), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    lines_below = []
    for ch in channels:
        v = capture.volts[ch][start:start + shown]
        full = capture.range_v[ch]
        cols = np.linspace(0, len(v), w + 1).astype(int)
        lo = np.array([v[a:max(b, a + 1)].min() for a, b in zip(cols[:-1], cols[1:])])
        hi = np.array([v[a:max(b, a + 1)].max() for a, b in zip(cols[:-1], cols[1:])])
        to_y = lambda x: np.clip(top + plot_h / 2 - x / full * plot_h / 2, top, top + plot_h).astype(np.int32)
        xs = np.arange(w)
        for x, y0, y1 in zip(xs, to_y(lo), to_y(hi)):
            cv2.line(img, (int(x), int(y0)), (int(x), int(y1)), COLOURS[ch], 1)
        pts = np.stack([xs, to_y((lo + hi) / 2)], axis=1).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, COLOURS[ch], 1, cv2.LINE_AA)
        m = measurements[ch]
        parts = [f"{ch} {si(full / 4, 'V')}/div", m["shape"], f"{si(m['vpp_v'], 'V')}pp", f"mean {si(m['mean_v'], 'V')}"]
        if m.get("frequency_hz"):
            parts.append(si(m["frequency_hz"], "Hz"))
        if m.get("duty") is not None:
            parts.append(f"duty {m['duty']:.0%}")
        if m.get("baud_if_uart"):
            parts.append(f"{m['baud_if_uart']:g} baud if UART")
        elif m.get("rise_s"):
            parts.append(f"rise {si(m['rise_s'], 's')}")
        lines_below.append((ch, "   ".join(parts)))
    for k, (ch, text) in enumerate(lines_below):
        cv2.putText(img, text, (8, top + plot_h + 28 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOURS[ch], 1,
                    cv2.LINE_AA)
    return img
