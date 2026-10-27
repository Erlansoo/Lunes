"""Keyboard jog window for the Lunes Motion firmware: hold keys to move, combine keys to move axes together.

Arrows move X and Y, Page Up / Page Down move Z, Home / End move A and Insert / Delete move B (only the axes that
config/machine.json defines). Hold Shift for fine speed. Space stops, Esc is the emergency stop. "Ir a" makes one
coordinated move to absolute millimetres with every axis arriving together.

    python tools/jog.py COM8
"""
import json
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import serial

CONFIG = Path(__file__).resolve().parent.parent / "config" / "machine.json"
RATE_CAP = 18000         # steps/s, the firmware's ceiling
JOG_PERIOD_MS = 80       # JV is repeated this often while a key is held (the firmware's deadman is 250 ms)
STATUS_PERIOD_MS = 120
IDLE_DISABLE_S = 20      # motors are released after this long without moving, so the drivers stay cool
FINE = 0.2               # speed factor while Shift is held
KEYS = {  # key -> (machine axis, direction)
    "Left": ("X", -1), "Right": ("X", 1), "Up": ("Y", 1), "Down": ("Y", -1),
    "Prior": ("Z", 1), "Next": ("Z", -1), "Home": ("A", 1), "End": ("A", -1),
    "Insert": ("B", 1), "Delete": ("B", -1),
}


class Board:
    """Serial link: a reader thread turns JSON lines into a queue; writes are serialized with a lock."""

    def __init__(self, port):
        self.s = serial.Serial(port, 115200, timeout=0.05)
        self.lock = threading.Lock()
        self.msgs = queue.Queue()
        self.alive = True
        threading.Thread(target=self._reader, daemon=True).start()
        time.sleep(2.5)  # the board resets when the port opens

    def _reader(self):
        while self.alive:
            try:
                raw = self.s.readline()
            except serial.SerialException as e:
                self.msgs.put({"lost": str(e)})
                return
            if raw:
                try:
                    self.msgs.put(json.loads(raw))
                except ValueError:
                    pass

    def write(self, data):
        with self.lock:
            self.s.write(data if isinstance(data, bytes) else data.encode() + b"\n")

    def close(self):
        self.alive = False
        try:
            self.write("JV")
            self.write("EN 0")
        finally:
            self.s.close()


class Axis:
    def __init__(self, name, cfg):
        self.name = name
        self.socket = cfg.get("socket", name)  # firmware axis letter that drives this machine axis
        self.spm = cfg["steps_per_mm"]
        self.speed = cfg.get("speed_mm_s", 2.0)
        self.accel = cfg.get("accel_mm_s2", 10.0)
        self.invert = cfg.get("invert", False)
        self.slave = cfg.get("slave")


class JogWindow:
    def __init__(self, root, port):
        self.root = root
        cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.axes = {}
        used = set()
        for name, ax_cfg in cfg["axes"].items():
            if "steps_per_mm" not in ax_cfg:
                continue
            ax = Axis(name, ax_cfg)
            sockets = {ax.socket} | ({ax.slave["socket"]} if ax.slave else set())
            if sockets & used:
                print(f"El eje {name} usa un zócalo ocupado ({sockets & used}); lo dejo fuera.")
                continue
            used |= sockets
            self.axes[name] = ax
        self.fw_index = {c: i for i, c in enumerate("XYZAB")}

        self.board = Board(port)
        for ax in self.axes.values():
            rate = min(ax.speed * ax.spm, RATE_CAP)
            self.board.write(f"CFG {ax.socket} R{rate:.0f} A{ax.accel * ax.spm:.0f} I{1 if ax.invert else 0}")
            if ax.slave:
                self.board.write(f"SLAVE {ax.slave['socket']} {ax.socket}")
                self.board.write(f"CFG {ax.slave['socket']} I{1 if ax.slave.get('invert') else 0}")

        self.pressed = set()
        self.shift = False
        self.enabled = False
        self.jogging = False
        self.last_motion = time.time()
        self.pos_steps = [0] * 5
        self.build_ui()
        root.bind("<KeyPress>", self.on_key)
        root.bind("<KeyRelease>", self.on_key_release)
        root.bind("<FocusOut>", lambda e: self.release_all())
        root.protocol("WM_DELETE_WINDOW", self.quit)
        root.after(JOG_PERIOD_MS, self.jog_loop)
        root.after(STATUS_PERIOD_MS, self.status_loop)

    def build_ui(self):
        r = self.root
        r.title("Lunes · mover con el teclado")
        frm = ttk.Frame(r, padding=12)
        frm.grid(sticky="nsew")
        self.pos_vars = {}
        for row, name in enumerate(self.axes):
            ttk.Label(frm, text=name, font=("Consolas", 18, "bold")).grid(row=row, column=0, sticky="w")
            v = tk.StringVar(value="0.000 mm")
            ttk.Label(frm, textvariable=v, font=("Consolas", 18), width=14, anchor="e").grid(row=row, column=1)
            self.pos_vars[name] = v
        n = len(self.axes)

        ttk.Label(frm, text="Velocidad").grid(row=n, column=0, sticky="w", pady=(10, 0))
        self.speed_pct = tk.DoubleVar(value=50)
        ttk.Scale(frm, from_=5, to=100, variable=self.speed_pct, orient="horizontal", length=220).grid(
            row=n, column=1, pady=(10, 0))
        self.speed_lbl = ttk.Label(frm, text="50 %")
        self.speed_lbl.grid(row=n, column=2, padx=6, pady=(10, 0))

        btns = ttk.Frame(frm)
        btns.grid(row=n + 1, column=0, columnspan=3, pady=8, sticky="w")
        self.en_btn = ttk.Button(btns, text="Habilitar motores", command=self.toggle_enable)
        self.en_btn.grid(row=0, column=0, padx=2)
        ttk.Button(btns, text="Cero aquí", command=self.zero).grid(row=0, column=1, padx=2)
        ttk.Button(btns, text="Quitar alarma", command=lambda: self.board.write("RESET")).grid(row=0, column=2, padx=2)

        goto = ttk.Frame(frm)
        goto.grid(row=n + 2, column=0, columnspan=3, sticky="w")
        self.goto_vars = {}
        for col, name in enumerate(self.axes):
            ttk.Label(goto, text=name).grid(row=0, column=2 * col)
            v = tk.StringVar()
            ttk.Entry(goto, textvariable=v, width=8).grid(row=0, column=2 * col + 1, padx=(2, 8))
            self.goto_vars[name] = v
        ttk.Button(goto, text="Ir a (mm)", command=self.goto).grid(row=0, column=2 * len(self.axes))

        self.state_var = tk.StringVar(value="conectando…")
        ttk.Label(frm, textvariable=self.state_var, font=("Consolas", 11)).grid(row=n + 3, column=0, columnspan=3,
                                                                                 sticky="w", pady=(8, 0))
        keys = ["←/→ X", "↑/↓ Y", "RePág/AvPág Z", "Inicio/Fin A", "Insert/Supr B"]
        keys = [k for k in keys if k.split()[-1] in self.axes]
        help_text = ("   ".join(keys) + "\nMantén varias teclas para mover ejes a la vez.  Shift = lento.\n"
                     "Espacio = parar.  Esc = parada de emergencia.  Ir a = todos los ejes llegan juntos.")
        ttk.Label(frm, text=help_text, foreground="#555").grid(row=n + 4, column=0, columnspan=3, sticky="w",
                                                               pady=(8, 0))

    # Keyboard ------------------------------------------------------------------------------------------------
    def on_key(self, e):
        if e.keysym in ("Shift_L", "Shift_R"):
            self.shift = True
        elif e.keysym == "space":
            self.release_all()
            self.board.write(b"!")
        elif e.keysym == "Escape":
            self.release_all()
            self.board.write(b"\x18")
        elif e.keysym in KEYS and KEYS[e.keysym][0] in self.axes:
            if isinstance(e.widget, ttk.Entry):
                return
            self.pressed.add(e.keysym)
            return "break"

    def on_key_release(self, e):
        if e.keysym in ("Shift_L", "Shift_R"):
            self.shift = False
        self.pressed.discard(e.keysym)

    def release_all(self):
        self.pressed.clear()
        if self.jogging:
            self.board.write("JV")
            self.jogging = False

    # Loops -----------------------------------------------------------------------------------------------------
    def jog_loop(self):
        pct = round(self.speed_pct.get())
        self.speed_lbl.config(text=f"{pct} %")
        parts = []
        for key in self.pressed:
            name, sign = KEYS[key]
            ax = self.axes[name]
            mm_s = ax.speed * pct / 100 * (FINE if self.shift else 1)
            parts.append(f"{ax.socket}{sign * mm_s * ax.spm:.0f}")
        if parts:
            if not self.enabled:
                self.set_enabled(True)
            self.board.write("JV " + " ".join(parts))
            self.jogging = True
            self.last_motion = time.time()
        elif self.jogging:
            self.board.write("JV")
            self.jogging = False
        if self.enabled and not self.jogging and time.time() - self.last_motion > IDLE_DISABLE_S:
            self.set_enabled(False)
        self.root.after(JOG_PERIOD_MS, self.jog_loop)

    def status_loop(self):
        self.board.write(b"?")
        while True:
            try:
                m = self.board.msgs.get_nowait()
            except queue.Empty:
                break
            if "lost" in m:
                self.state_var.set("SE PERDIÓ LA CONEXIÓN USB con la placa. Revisa el cable y vuelve a abrir.")
                return
            if "pos" in m:
                self.pos_steps = m["pos"]
                if m.get("st") not in (None, "idle"):
                    self.last_motion = time.time()
            if "st" in m:
                alarm = f"   ALARMA: {m['alarm']}" if m["alarm"] else ""
                self.state_var.set(f"estado: {m['st']}   motores: {'ON' if m['en'] else 'off'}{alarm}")
            if "err" in m:
                self.state_var.set(f"la placa respondió error: {m['err']}")
        for name, ax in self.axes.items():
            steps = self.pos_steps[self.fw_index[ax.socket]]
            self.pos_vars[name].set(f"{steps / ax.spm:9.3f} mm")
        self.root.after(STATUS_PERIOD_MS, self.status_loop)

    # Buttons ---------------------------------------------------------------------------------------------------
    def set_enabled(self, on):
        self.enabled = on
        self.board.write(f"EN {1 if on else 0}")
        self.en_btn.config(text="Deshabilitar motores" if on else "Habilitar motores")
        if on:
            self.last_motion = time.time()

    def toggle_enable(self):
        self.set_enabled(not self.enabled)

    def zero(self):
        self.release_all()
        self.board.write("ZERO")

    def goto(self):
        words = []
        for name, var in self.goto_vars.items():
            txt = var.get().strip().replace(",", ".")
            if txt:
                ax = self.axes[name]
                words.append(f"{ax.socket}{round(float(txt) * ax.spm)}")
        if not words:
            return
        self.release_all()
        if not self.enabled:
            self.set_enabled(True)
        self.last_motion = time.time()
        self.board.write(f"MOVE {' '.join(words)} F{RATE_CAP}")  # each axis is held to its own speed limit
        self.root.focus_set()

    def quit(self):
        self.board.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    JogWindow(root, sys.argv[1] if len(sys.argv) > 1 else "COM8")
    root.mainloop()


if __name__ == "__main__":
    main()
