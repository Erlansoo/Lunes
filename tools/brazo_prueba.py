"""First test of the arm board (Creality E3 Free-runs with firmware/lunes_brazo): are the four drivers answering with
the right current, and does each motor turn a little each way?

Each motor turns 1/16 of a turn (22.5 degrees at 16 microsteps) forward and back, one at a time, slowly. Nothing
moves until you press Enter, and Ctrl+C stops everything.

    python tools/brazo_prueba.py COM9            (all four motors)
    python tools/brazo_prueba.py COM9 --solo     (only what the drivers say, nothing moves)
    python tools/brazo_prueba.py COM9 --eje E    (only one motor)
"""
import argparse
import json
import sys
import time

import serial
import serial.tools.list_ports

TURN_STEPS = 200 * 16  # a 1.8-degree motor at 16 microsteps
NAMES = {"X": "X (motor de 1.8 A)", "Y": "Y (motor de 1.8 A)", "Z": "Z (motor de 1.2 A)", "E": "E (motor de 0.6 A)"}


class Board:
    def __init__(self, port):
        self.s = serial.Serial(port, 115200, timeout=0.1)
        self.boot = None
        t0 = time.time()
        while time.time() - t0 < 3:  # the board restarts when the port opens on some USB chips
            msg = self.read()
            if msg and msg.get("ev") == "boot":
                self.boot = msg
                break

    def read(self):
        raw = self.s.readline().decode(errors="replace").strip()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}

    def ask(self, line, until=("ok", "err", "fw", "tmc"), timeout=3.0):
        self.s.write((line + "\n").encode())
        t0 = time.time()
        while time.time() - t0 < timeout:
            msg = self.read()
            if msg and any(k in msg for k in until):
                return msg
        return {"err": f"sin respuesta a {line}"}

    def status(self):
        self.s.write(b"?")
        t0 = time.time()
        while time.time() - t0 < 1:
            msg = self.read()
            if msg and "st" in msg:
                return msg
        return None

    def move(self, axis, steps, rate):
        reply = self.ask(f"REL {axis}{steps} F{rate}")
        if "err" in reply:
            return reply
        while True:  # '?' every 0.2 s keeps the firmware's host-lost watchdog quiet
            self.s.write(b"?")
            t0 = time.time()
            while time.time() - t0 < 0.2:
                msg = self.read()
                if msg and msg.get("ev") == "done":
                    return msg

    def stop(self):
        self.s.write(b"\x18")
        self.ask("EN 0")


def print_drivers(board):
    tmc = board.ask("TMC", until=("tmc",), timeout=5).get("tmc", [])
    if not tmc:
        print("Los drivers no respondieron al pedido de estado.")
        return False
    good = True
    for d in tmc:
        line = (f"  {NAMES[d['axis']]}: responde {'sí' if d['answers'] else 'NO'}, configurado "
                f"{'sí' if d['set'] else 'NO'}, {d['ma']} mA (tope {d['cap']} mA), {d['ustep']} micropasos, {d['mode']}")
        if d.get("answers"):
            flags = [name for key, name in (("ot", "SOBRECALENTADO"), ("otpw", "caliente"), ("short", "CORTO"),
                                            ("undervoltage", "sin 24 V")) if d.get(key)]
            if flags:
                line += "  <- " + ", ".join(flags)
        print(line)
        good = good and d["answers"] and d["set"] and not d.get("undervoltage")
    return good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?")
    ap.add_argument("--solo", action="store_true", help="only read the drivers, move nothing")
    ap.add_argument("--eje", choices=list("XYZE"), help="test only this motor")
    ap.add_argument("--rate", type=int, default=400, help="steps/s (400 = 1/8 of a turn per second)")
    args = ap.parse_args()
    if not args.port:
        print("Puertos serie:", ", ".join(f"{p.device} ({p.description})" for p in serial.tools.list_ports.comports()))
        sys.exit("Dime cuál es la placa del brazo: python tools/brazo_prueba.py COMx")

    board = Board(args.port)
    print("arranque:", board.boot or "(no vi el mensaje de arranque; sigo)")
    print("firmware:", board.ask("INFO"))
    print("drivers:")
    if not print_drivers(board):
        sys.exit("Algún driver no responde o no tiene los 24 V. Revisa la fuente de 24 V antes de mover nada.")
    if args.solo:
        return

    axes = [args.eje] if args.eje else list("XYZE")
    input(f"Voy a girar {', '.join(axes)} 1/16 de vuelta hacia un lado y de vuelta, uno por uno. Enter para seguir, "
          f"Ctrl+C para cancelar... ")
    try:
        reply = board.ask("EN 1")
        if "err" in reply:
            sys.exit(f"No pude habilitar los motores: {reply}")
        for axis in axes:
            print(f"{NAMES[axis]}: adelante...", end=" ", flush=True)
            print(board.move(axis, TURN_STEPS // 16, args.rate).get("pos"), end="  ")
            time.sleep(0.5)
            print("atrás...", end=" ", flush=True)
            print(board.move(axis, -TURN_STEPS // 16, args.rate).get("pos"))
            time.sleep(0.5)
        print("estado:", board.status())
        print("drivers después de moverse:")
        print_drivers(board)
    except KeyboardInterrupt:
        print("\nparado")
    finally:
        board.stop()


if __name__ == "__main__":
    main()
