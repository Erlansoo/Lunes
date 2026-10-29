"""Live view of the Hantek 6022BE with Lunes' driver and measurements, for trying the scope without Lunes.

Keys: a = auto (range and sample rate for CH1), b = auto for both channels, c = 1 kHz on the calibration output,
s = save the picture, q or Esc = quit. --simulado shows made-up signals instead of the scope.

    python tools/scope_view.py
    python tools/scope_view.py --probe 1      (both probes at x1)
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lunes.hantek import Scope, SimulatedScope, ranges, sine, square  # noqa: E402
from lunes.signals import autoset, fit_range, measure, notes, picture, step_v  # noqa: E402

CAPTURES = ROOT / "captures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=float, default=10, help="the factor on both probes' switches (1 or 10)")
    ap.add_argument("--simulado", action="store_true")
    args = ap.parse_args()
    probe = {"CH1": args.probe, "CH2": args.probe}
    try:
        scope = (SimulatedScope({"CH1": square(1000, 0, 2.0), "CH2": sine(10e3, 1.0, 2.5)}, probe=probe)
                 if args.simulado else Scope(probe=probe))
    except RuntimeError as e:
        sys.exit(f"osciloscopio: {e}")
    print(scope.info())
    autoset(scope, "CH1")
    try:
        while True:
            cap = scope.capture(20000)
            shown = {ch: measure(v, cap.rate, step_v(cap.range_v[ch])) for ch, v in cap.volts.items()}
            cv2.imshow("Lunes - osciloscopio", picture(cap, shown, title="a: auto  b: ambos  c: 1 kHz cal  s: foto"))
            key = cv2.waitKey(100) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("a"):
                autoset(scope, "CH1")
            elif key == ord("b"):
                autoset(scope, "CH1")
                fit_range(scope, "CH2")
            elif key == ord("c"):
                print(f"salida de calibración: {scope.calibration_output(1000):g} Hz")
            elif key == ord("s"):
                CAPTURES.mkdir(exist_ok=True)
                path = CAPTURES / f"osciloscopio_{datetime.now():%Y%m%d_%H%M%S}.png"
                cv2.imwrite(str(path), picture(cap, shown))
                for ch, m in shown.items():
                    allowed = ranges(scope.probe[ch])
                    print(ch, m, notes(m, cap.range_v[ch], cap.clipped[ch], allowed[0], allowed[-1]))
                print(f"guardado {path}")
    finally:
        scope.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
