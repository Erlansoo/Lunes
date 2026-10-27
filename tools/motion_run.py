"""Sends a sequence of commands to the Lunes Motion firmware and waits for each move to finish.

Motors are enabled first and disabled at the end (unless --keep), so the drivers do not heat up holding still.

    python tools/motion_run.py COM8 "CFG X R400 A400" "REL X200 F100"
"""
import argparse
import json
import time

import serial

MOVES = ("MOVE", "REL", "TOUCH")


def read_msgs(s):
    out = []
    while True:
        raw = s.readline()
        if not raw:
            return out
        out.append(json.loads(raw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port")
    ap.add_argument("commands", nargs="+")
    ap.add_argument("--keep", action="store_true", help="leave the motors enabled at the end")
    args = ap.parse_args()

    with serial.Serial(args.port, 115200, timeout=0.05) as s:
        time.sleep(2.5)  # the board resets when the port opens
        read_msgs(s)
        try:
            for cmd in ["EN 1", *args.commands]:
                s.write(cmd.encode() + b"\n")
                time.sleep(0.05)
                print(cmd, "->", read_msgs(s))
                if cmd.split()[0] in MOVES:
                    t0 = time.time()
                    while True:
                        s.write(b"?")  # status polls keep the host watchdog quiet
                        done = [m for m in read_msgs(s) if m.get("ev") == "done"]
                        if done:
                            print(f"   done in {time.time() - t0:.2f} s: {done[0]}")
                            break
                        time.sleep(0.1)
        finally:
            if not args.keep:
                s.write(b"EN 0\n")
                time.sleep(0.05)
                print("EN 0 ->", read_msgs(s))


if __name__ == "__main__":
    main()
