# Diagnosis with Muse Spark at the wheel: the measured record

Made by `tools/reliability.py` on 2026-11-02 09:07: the model drives Hume's tools on the simulated bench (lunes.mcu.SimulatedBoard, ngspice for the board's readings and the scope), one job in Spanish per case, a script answering for Erlan that the probe is placed. The engine groups faults that read the same at the measured nets into one class (R1 open and C1 short both leave the filter at zero; a board not running and its pin stuck low look alike from outside). A run is **found** when the engine settled on a class holding the planted fault and the answer names its part (by ref, or by kind when the circuit has one such part); **wrong** when the engine settled on a class without it, or the answer blames other parts and not it (or any fault on a sound circuit); **unsure** otherwise, the engine still undecided and the answer saying so. The full answers are in `runs/reliability_*.jsonl`.

| Circuit | Planted fault | Runs | Found | Unsure | Wrong | Tokens per run |
|---|---|---|---|---|---|---|
| arduino_pwm_rc | healthy | 3 | 3 | 0 | 0 | 111,973 |
| arduino_pwm_rc | R1:open | 3 | 3 | 0 | 0 | 133,218 |
| arduino_pwm_rc | C1:short | 3 | 3 | 0 | 0 | 132,467 |
| arduino_pwm_rc | U1:D9_floating | 3 | 3 | 0 | 0 | 162,362 |
| arduino_pwm_rc | U1:not_running | 3 | 3 | 0 | 0 | 157,178 |
| arduino_read_divider | R1:x10 | 3 | 3 | 0 | 0 | 100,233 |
| arduino_read_divider | R2:open | 3 | 0 | 3 | 0 | 130,807 |
| **all** | | 21 | **18** | 3 | **0** | 132,605 |

Tokens in all: 2,784,724 (21 runs).
