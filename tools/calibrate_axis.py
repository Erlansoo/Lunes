"""Calibrates steps per millimetre for one axis with a vernier caliper.

You choose how far to move, the machine moves, you type what the caliper reads, and the program computes the real
steps per millimetre and saves it in config/machine.json. Repeat with longer moves to refine it.

    python tools/calibrate_axis.py COM8 X
"""
import json
import sys
import time
from pathlib import Path

import serial

CONFIG = Path(__file__).resolve().parent.parent / "config" / "machine.json"
MAX_MOVE_MM = 100
RATE_CAP = 18000           # steps/s, the firmware's ceiling
DEFAULT_SPEED_MM_S = 2.0
DEFAULT_ACCEL_MM_S2 = 10.0
DEFAULT_STEPS_PER_MM = 100.0  # deliberately low: an unknown axis moves less than asked, not more
COUNTDOWN_S = 15              # time to walk from the keyboard to the machine before it moves
LEADS_MM = (1, 1.25, 2, 2.5, 4, 5, 8, 10, 12)


class Board:
    def __init__(self, port):
        self.s = serial.Serial(port, 115200, timeout=0.05)
        time.sleep(2.5)  # the board resets when the port opens
        self.read_msgs()

    def read_msgs(self):
        out = []
        while True:
            raw = self.s.readline()
            if not raw:
                return out
            out.append(json.loads(raw))

    def send(self, cmd):
        self.s.write(cmd.encode() + b"\n")
        time.sleep(0.05)
        return self.read_msgs()

    def move(self, axis, steps, rate):
        reply = self.send(f"REL {axis}{steps} F{rate}")
        if not any("ok" in m for m in reply):
            raise RuntimeError(f"la placa rechazó el movimiento: {reply}")
        while True:
            self.s.write(b"?")  # status polls keep the host watchdog quiet
            for m in self.read_msgs():
                if m.get("ev") == "done":
                    return m
            time.sleep(0.1)


def load_config():
    if CONFIG.exists():
        return json.loads(CONFIG.read_text(encoding="utf-8"))
    return {"axes": {}}


def save_config(cfg):
    CONFIG.parent.mkdir(exist_ok=True)
    CONFIG.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def microstep_hints(steps_per_mm):
    """Screw leads that turn this value into a standard microstep (200-step motor), within 3 %."""
    hints = []
    for lead in LEADS_MM:
        micro = steps_per_mm * lead / 200
        for m in (1, 2, 4, 8, 16, 32):
            if abs(micro - m) / m < 0.03:
                hints.append(f"tornillo de {lead} mm por vuelta con micropaso 1/{m}")
    return hints


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "COM8"
    axis = sys.argv[2].upper() if len(sys.argv) > 2 else "X"
    cfg = load_config()
    ax = cfg["axes"].setdefault(axis, {})
    fw = ax.get("socket", axis)  # firmware axis (driver socket) that drives this machine axis
    spm = ax.get("steps_per_mm", DEFAULT_STEPS_PER_MM)
    invert = ax.get("invert", False)
    speed = ax.get("speed_mm_s", DEFAULT_SPEED_MM_S)
    accel = ax.get("accel_mm_s2", DEFAULT_ACCEL_MM_S2)

    print(f"Calibración del eje {axis}. Pasos por mm actuales: {spm:.4f}"
          f"{'' if 'steps_per_mm' in ax else ' (valor inicial, todavía sin calibrar)'}")
    print("Comandos: un número = mm a mover (negativo = sentido contrario); con 'p' al final = pasos (ej. 4000p);")
    print("          'v 5' = velocidad en mm/s; 'a 20' = aceleración en mm/s²; 'i' = invertir el sentido del eje;")
    print("          's' = invertir solo el segundo motor (eje con dos motores); Enter vacío = salir.")
    if fw != axis:
        print(f"El eje {axis} se mueve con el driver del zócalo {fw} del firmware.")
    b = Board(port)
    slave = ax.get("slave")
    if slave:
        # Dual-motor axis: the second motor's driver copies every step; its own direction can be flipped.
        b.send(f"SLAVE {slave['socket']} {fw}")
        b.send(f"CFG {slave['socket']} I{1 if slave.get('invert') else 0}")
        print(f"Eje con dos motores: el zócalo {slave['socket']} copia a {axis}"
              f"{' con sentido invertido' if slave.get('invert') else ''}.")

    def apply_motion():
        """Sends speed and acceleration in steps; returns the step rate for moves."""
        rate = min(speed * spm, RATE_CAP)
        b.send(f"CFG {fw} R{rate:.0f} A{accel * spm:.0f} I{1 if invert else 0}")
        print(f"Velocidad {rate / spm:.2f} mm/s (máximo posible {RATE_CAP / spm:.2f}), aceleración {accel:.1f} mm/s²")
        return round(rate)

    rate = apply_motion()
    try:
        while True:
            txt = input(f"\n[{axis}] mm a mover: ").strip().lower().replace(",", ".")
            if not txt:
                break
            if txt[0] in "va" and len(txt) > 1:
                try:
                    value = float(txt[1:])
                except ValueError:
                    print("No entendí el número.")
                    continue
                if value <= 0:
                    print("Tiene que ser mayor que cero.")
                    continue
                if txt[0] == "v":
                    speed = value
                    ax["speed_mm_s"] = value
                else:
                    accel = value
                    ax["accel_mm_s2"] = value
                save_config(cfg)
                rate = apply_motion()
                continue
            if txt == "s":
                if not slave:
                    print("Este eje tiene un solo motor.")
                    continue
                slave["invert"] = not slave.get("invert", False)
                save_config(cfg)
                b.send(f"CFG {slave['socket']} I{1 if slave['invert'] else 0}")
                print(f"Segundo motor {'invertido' if slave['invert'] else 'normal'} (guardado).")
                continue
            if txt == "i":
                invert = not invert
                ax["invert"] = invert
                save_config(cfg)
                b.send(f"CFG {fw} I{1 if invert else 0}")
                print(f"Sentido {'invertido' if invert else 'normal'} (guardado).")
                continue
            try:
                steps = int(float(txt[:-1])) if txt.endswith("p") else round(float(txt) * spm)
            except ValueError:
                print("No entendí el número.")
                continue
            if steps == 0:
                continue
            if abs(steps) / spm > MAX_MOVE_MM:
                print(f"Son más de {MAX_MOVE_MM} mm según la calibración actual; pide algo más corto.")
                continue

            input(f"Presiona Enter y tienes {COUNTDOWN_S} s para ir a la CNC y poner el vernier en cero...")
            for left in range(COUNTDOWN_S, 0, -1):
                print(f"\r  Se mueve en {left:2d} s (Ctrl-C para cancelar) ", end="\a" if left <= 3 else "", flush=True)
                time.sleep(1)
            print("\r  Moviendo...                              ")
            # The motor is energized only while it moves: a small driver holding a 2 A motor overheats in minutes,
            # and the lead screw keeps the axis in place without holding torque.
            b.send("EN 1")
            time.sleep(0.2)
            try:
                done = b.move(fw, steps, rate)
            finally:
                b.send("EN 0")
            if done.get("stopped"):
                print(f"El movimiento se detuvo antes de terminar: {done}")
                continue
            print(f"Listo: {steps} pasos (unos {steps / spm:.3f} mm según la calibración actual).")

            meas = input("¿Cuánto marca el vernier, en mm? (Enter = no medí): ").strip().replace(",", ".")
            if not meas:
                continue
            try:
                d = abs(float(meas))
            except ValueError:
                print("No entendí la medida.")
                continue
            if d == 0:
                print("Con 0 mm no puedo calcular nada.")
                continue
            new = abs(steps) / d
            print(f"Pasos por mm medidos: {new:.4f} (antes {spm:.4f}, diferencia {100 * (new - spm) / spm:+.2f} %)")
            for h in microstep_hints(new):
                print(f"  Coincide con un {h}.")
            if input("¿Guardar este valor? (s/n): ").strip().lower().startswith("s"):
                spm = new
                ax["steps_per_mm"] = round(new, 4)
                save_config(cfg)
                print(f"Guardado en {CONFIG}")
                rate = apply_motion()
    finally:
        b.send("EN 0")
        print("Motor deshabilitado.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSaliste con Ctrl-C.")
