"""Talk to Lunes like on a call.

The conductor is always on the line: it understands what Erlan wants, answers in under two seconds, asks when he
could mean two things, and takes a quick look or hands a job to the worker (Hume, with the bench tools), which runs
in the background while the conversation goes on. Lunes tells how the work goes at its milestones, not step by step
(Narrator). "Lunes" opens the conversation; it stays open while they talk and while a job runs. "Lunes, para" (or
"basta", "detente"...), Space or Esc in its window stop everything; "Lunes, adiós" ends the session. Every model is
a Meta model: the ears are Omnilingual ASR and the voice is MMS-TTS, on this PC; the conductor and the worker think
with Muse Spark through the Meta Model API (Muse Glimmer on this PC without --spark or without internet). Its memory,
Hume, keeps what it observed between sessions.
Erlan can also type to Lunes in the console while the microphone listens (a typed line is always for Lunes, no
wake word); "/mic off" and "/mic on" typed there switch the microphone. --text is keyboard only, to test without talking.

    python tools/talk.py COM8 --spark --mic fifine
    python tools/talk.py COM8 --text
"""
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
import difflib
import json
import os
import queue
import random
import re
import sys
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import sounddevice as sd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from lunes.conductor import Conductor  # noqa: E402
from lunes.cameras import cameras, find_camera  # noqa: E402
from lunes.dashboard import Dashboard  # noqa: E402
from lunes.hume import MODEL, PROJECTS, THINK, Hume  # noqa: E402
from lunes.hantek import Scope, SimulatedScope, sine, square  # noqa: E402
from lunes.lab import state as lab_state  # noqa: E402
from lunes.mcu import SimulatedBoard  # noqa: E402
from lunes.kinect import Kinect  # noqa: E402
from lunes.machine import Machine  # noqa: E402
from lunes.voice import FAST_EAR, Ear, Voice, find_microphone, latin, local_transcriber, sounds_like_speech  # noqa: E402
from microscope import Camera, sharpness  # noqa: E402

# The wake word. "el lunes", "este lunes"... are the weekday, not a call, so they do not count. The ear also writes
# "lúnes", "lúnese", "lunis", "lunas", "luna"; it splits "Hola" into "o la", so only "la luna" is left out, not
# "la lunes".
WAKE = re.compile(r"(?<!\bel )(?<!\beste )(?<!\bcada )(?<!\bpróximo )(?<!\bpasado )(?<!\bde )"
                  r"(\b(l[uú]n[aeéi]s\w*|lune|yunes)\b|(?<!\bla )\bluna\b)"
                  r"[,.!]?\s*", re.IGNORECASE)
BYE = re.compile(r"\b(adi[oó]s|cha[ou]|apágate|termina|descansa)\b", re.IGNORECASE)
# An interruption is a short phrase (the ear garbles the rest: "lune su para", "basta detintean"). Words that only
# mean stop count anywhere in it; "para" and "espera" only at its end, so "Lunes, ¿para qué sirve esto?" is not an
# order. ("alto" is left out: "mide el alto" is a job.)
CLEAR_STOP = re.compile(r"\b(basta|det[eé]nte|de\s?[td][eé]nte|det[ií]nte\w*|c[aá]llate|silencio|stop)\b", re.IGNORECASE)
SOFT_STOP = re.compile(r"\b(p[aá]r[aá](te)?|espera)(\W+(ya|lunes|por favor|ahí|un momento))*\W*$", re.IGNORECASE)


# The steps that take minutes: when one of them fails Erlan hears it, not only the retry's silence
LONG_STEPS = {"survey_board": "el recorrido con el microscopio", "measure_board": "la medida de la placa",
              "inspect_component": "la foto de cerca del componente", "calibrate_kinect": "la calibración del Kinect",
              "calibrate_bench": "la calibración de la cámara del banco", "laser_calibrate": "la calibración del láser"}


def is_stop(text):
    return len(re.findall(r"\w+", text)) <= 5 and bool(CLEAR_STOP.search(text) or SOFT_STOP.search(text))


# Said while the conductor thinks, if it takes long: neutral, so they never promise what it may decide not to do.
REACTIONS = ["A ver.", "Mmm."]
# "no hace falta que me cuentes": from then on Lunes only says the result; "cuéntame cómo vas" brings the updates back
QUIET = re.compile(r"\b(no (hace falta que )?me (cuentes|avises|vayas contando)|av[ií]same (solo )?(al final|cuando "
                   r"termines)|solo (al final|el resultado))\b", re.IGNORECASE)
TALKATIVE = re.compile(r"\b(cu[eé]ntame c[oó]mo vas|ve(te)? cont[aá]ndome|av[ií]same c[oó]mo va)\b", re.IGNORECASE)
REACTION_AFTER_S = 0.9  # said only if the conductor has not answered by then
THINKING_AFTER_S = 7.0  # "Déjame pensarlo", if it is still thinking by then
# The cameras by name, since their numbers change when one is plugged in: the MS5 microscope's USB id (its name,
# "USB CAMERA", is too common) and the 4K webcam that sees the whole work area.
MICROSCOPE, BENCH_CAMERA = "vid_1224&pid_2825", "EMEET"
# Creating this file closes Lunes from outside, to restart it: "Lunes, adiós" needs a working ear, Ctrl+C cannot
# reach a main thread waiting for the ear on Windows, and killing the process while it holds the CNC's port left the
# port jammed for minutes.
CLOSE_FILE = ROOT / "runs" / "cerrar_lunes"
# When Erlan leaves a sentence half-way, what he says within HALF_WAIT_S joins it; after NUDGE_AFTER_S of silence
# Lunes says "¿Sí? Te escucho" once.
HALF_WAIT_S, NUDGE_AFTER_S = 8.0, 3.0
WIN = "Lunes"
# "Lunes" opens the conversation; it stays open while they talk, while a job runs, and this long after the last
# exchange. Inside it everything Erlan says goes to the conductor, which judges what is for Lunes.
CONVERSATION_S = 90
# What each tool means to Erlan, said when it starts (if the narrator finds the moment) and kept as the current step
STEPS = {"measure_board": "mido la placa", "survey_board": "recorro la placa con el microscopio",
         "inspect_component": "me acerco al componente", "look_kinect": "miro con el Kinect",
         "look_bench": "miro el banco con la cámara", "find_in_bench": "busco en la foto del banco",
         "center_on_point": "centro el microscopio", "center_on_corner": "busco la esquina exacta de la placa",
         "measure_hole": "mido el agujero", "measure_distance": "mido la distancia",
         "kinect_distance": "mido con el Kinect", "autofocus": "enfoco el microscopio",
         "look_microscope": "miro por el microscopio", "look_survey": "reviso el mosaico",
         "go_to_mosaic_point": "voy a un punto del mosaico", "move_to": "muevo la máquina", "move_by": "muevo la máquina",
         "scope_measure": "mido con el osciloscopio", "scope_selftest": "pruebo el osciloscopio",
         "scope_test_signal": "pongo la señal de prueba", "scope_probe": "anoto la punta",
         "circuit_set": "simulo el circuito", "diagnose_start": "preparo el diagnóstico",
         "diagnose_measure": "mido y comparo con la simulación", "circuit_design": "diseño el circuito",
         "laser_calibrate": "calibro el láser", "laser_point": "apunto el láser", "laser_height": "mido la altura con el láser", "laser": "prendo el láser",
         "calibrate_bench": "calibro la cámara del banco", "teach_bench_height": "le enseño la altura a la cámara del banco",
         "reach_in_bench": "voy a lo que ve la cámara del banco", "see_chip": "miro el chip y lo enfoco",
         "center_on_chip": "centro el chip", "chip_catalog": "reviso mis integrados",
         "chip_learn": "aprendo el integrado nuevo y lo pruebo en el simulador",
         "circuit_guide": "repaso mis notas", "calculate": "hago la cuenta",
         "system_check": "reviso mis partes", "system_restart": "reinicio una parte",
         "ask_spark": "le pregunto a Muse Spark", "remember": "lo anoto en la memoria", "where": "veo dónde estoy"}
TOLD_AT_START = {"measure_board", "survey_board", "inspect_component", "look_kinect", "see_chip", "center_on_chip",
                 "laser_point", "reach_in_bench"}


def milestone(name, result):
    """The sentence a tool's result is worth saying at once, or None when the final answer will tell it."""
    if not isinstance(result, dict):
        return None
    check = result.get("kinect_check")
    if isinstance(check, dict) and check.get("agrees") is False:
        if not check.get("on_hold"):
            return None  # Erlan already allowed moving, and the mosaic agrees: nothing new to say
        off = check["difference_mm"]
        axis = max(off, key=lambda k: abs(off[k]))
        mosaic = check.get("mosaic_offset_mm")
        second = ("" if mosaic is None else
                  " Pero el microscopio se ubica bien en el mosaico, así que parece ser el Kinect." if mosaic <= 1.0
                  else f" Y en el mosaico también me veo corrido, unos {mosaic:.0f} milímetros.")
        return (f"Ojo: el Kinect no me ve donde creo estar, hay unos {abs(off[axis]):.0f} milímetros de diferencia en "
                f"{axis}.{second} No me muevo hasta que me des permiso.")
    if result.get("refused"):
        return None
    if name == "measure_board":
        if result.get("found"):
            w, h = result["size_mm"]
            return f"Ya la medí: {w:.0f} por {h:.0f} milímetros."
        return "No encuentro qué medir bajo el microscopio."
    if name == "survey_board" and result.get("ok", True):
        return f"Ya terminé de recorrer la {result.get('bed') or 'mesa'}."
    return None


stop_requested = threading.Event()
mic_toggle = threading.Event()  # the m key in the window: switch the microphone (a fridge or dogs next to the bench)
board = None  # the dashboard page, once it is up
OPEN = []  # how to close what is open on the bench, in the order it was opened; closed on any exit
OLLAMA_API = "http://127.0.0.1:11434/api/"
gpu_released = False


def release_gpu(model=None):
    """Asks Ollama to unload Muse Glimmer. Lunes loads it with keep_alive -1 so the bench never waits for it twice,
    and Ollama then keeps it in the GPU for ever: after Lunes exits, ~16 GB of VRAM stayed taken all night. A
    request with keep_alive 0 and no prompt unloads a model. Without Ollama running there is nothing to free."""
    global gpu_released
    if gpu_released:
        return
    gpu_released = True
    try:
        with urllib.request.urlopen(OLLAMA_API + "ps", timeout=3) as r:
            loaded = [m.get("name") or m.get("model") for m in json.load(r).get("models", [])]
    except Exception:  # noqa: BLE001 - Ollama is not running, or not on this PC
        return
    for name in loaded:
        if name and (name == model or name.startswith("muse-glimmer")):
            try:
                req = urllib.request.Request(OLLAMA_API + "generate", json.dumps({"model": name, "keep_alive": 0}).encode(),
                                             {"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=10).read()
                print(f"(GPU libre: {name} descargado de Ollama)", flush=True)
            except Exception as e:  # noqa: BLE001 - say so; it can still be done by hand with `ollama stop`
                print(f"(no pude descargar {name} de Ollama: {e})", flush=True)


def shut_down():
    """Closes the CNC's port, the cameras, the Kinect and the scope, each on its own, and ends the process whatever
    happens: a Ctrl-C while Lunes starts used to leave the process hanging, since libusb's clean-up at exit waits
    forever for the Kinect's transfers nobody cancelled, with the CNC's port still taken."""
    guard = threading.Timer(8.0, lambda: os._exit(0))  # whichever close hangs, the process still ends
    guard.daemon = True
    guard.start()
    for close in reversed(OPEN):
        try:
            close()
        except Exception:  # noqa: BLE001 - already closed, or unplugged: the next one still gets closed
            pass
    release_gpu()
    try:
        cv2.destroyAllWindows()
        sys.stdout.flush()
    except Exception:  # noqa: BLE001
        pass
    os._exit(0)


HISTORY_KINDS = {"tu": ("Tú: ", "Tú (parar): "), "lunes": ("Lunes: ",), "result": ("   (resultado completo) ",)}
project_log = None  # the open project's history (Hume.project_log), once Lunes is up


def log(kind, text):
    """One line of the session: printed for the log file, shown on the dashboard page, and what Erlan and Lunes said
    kept in the open project's history."""
    print(text)
    if board is not None:
        board.add(kind, text.strip())
    if project_log is not None and kind in HISTORY_KINDS:
        said = text
        for prefix in HISTORY_KINDS[kind]:
            if said.startswith(prefix):
                said = said[len(prefix):]
        try:
            project_log(kind, re.sub(r"\s+\(oído [\d.]+ s\)$", "", said.strip()))
        except Exception:  # noqa: BLE001  (the history never stops the conversation)
            pass


def show(frame, text):
    if board is not None:
        board.status = text
    f = frame.copy()
    for color, width in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.putText(f, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, width)
    cv2.imshow(WIN, f)
    key = cv2.waitKey(1)
    if key in (27, 32):  # Esc or Space in the window: stop
        stop_requested.set()
    elif key in (ord("m"), ord("M")):  # m in the window: microphone on/off (as "/mic off" typed in the console)
        mic_toggle.set()


class Narrator(threading.Thread):
    """Decides when Lunes speaks about the work, like an attentive colleague and not a stream of notices:
    - at milestones only, with at least GAP_S between updates nobody asked for; a newer update replaces an older one;
    - never while Erlan talks nor in the AFTER_ERLAN_S that follow, never cutting its own sentence;
    - never a dead silence: after SILENCE_S without news during a job it says in one sentence where it is;
    - the final result is always said, as soon as nobody is talking;
    - quiet (Erlan asked not to be told): only the final result."""
    GAP_S, AFTER_ERLAN_S, SILENCE_S, STALE_S = 15, 1.5, 40, 30

    def __init__(self, voice, ear, conductor, worker_running):
        super().__init__(daemon=True)
        self.voice, self.ear, self.conductor, self.worker_running = voice, ear, conductor, worker_running
        self.lock = threading.Lock()
        self.pending, self.final = None, []  # (when, text, still_true) / texts that must be said
        self.last_spoken = time.time()
        self.quiet = False

    def spoke(self):
        self.last_spoken = time.time()

    def update(self, text, still_true=lambda: True, result=False):
        """A newer update replaces the pending one, except that the start of a step never hides a result: "Ya la medí:
        43 por 48" was lost under "Ahora recorro la placa" otherwise."""
        with self.lock:
            if self.pending and self.pending[3] and not result:
                return
            self.pending = (time.time(), text, still_true, result)

    def must(self, text):
        with self.lock:
            self.final.append(text)

    def clear(self):
        with self.lock:
            self.pending, self.final = None, []

    def answered(self):
        """The conductor just answered Erlan from the state, which already has the pending update."""
        with self.lock:
            self.pending = None

    def say(self, text, kind="lunes", remember=True):
        """Says it now; remember=False when the conductor already has it in its turns."""
        log(kind, f"Lunes: {text}")
        if remember:
            self.conductor.heard_from_lunes(text)
        self.voice.say_soon(text)
        self.spoke()

    def run(self):
        while True:
            time.sleep(0.2)
            now = time.time()
            if self.voice.speaking:
                self.spoke()
                continue
            if self.ear is not None and (self.ear.hearing or now - self.ear.last_heard < self.AFTER_ERLAN_S):
                continue
            with self.lock:
                final = self.final.pop(0) if self.final else None
                pending = self.pending
            if final:
                self.say(final)
                continue
            if not self.worker_running() or self.quiet:
                continue
            if pending and now - self.last_spoken >= self.GAP_S:
                with self.lock:
                    self.pending = None
                when, text, still_true, _ = pending
                if now - when < self.STALE_S and still_true():
                    self.say(text, "progress")
            elif now - self.last_spoken >= self.SILENCE_S:
                self.spoke()  # even if the conductor fails, wait another SILENCE_S
                try:
                    text = self.conductor.tell("Hace rato que no le cuentas nada a Erlan. Dile en una frase corta en "
                                               "qué va el trabajo, sin inventar.", sentences=1)
                except Exception as e:
                    log("info", f"   (el conductor no respondió: {e})")
                    continue
                if text:
                    self.say(text, "progress")


class Worker:
    """Runs one job at a time on Hume in the background and feeds the narrator; a job asked for meanwhile waits in
    line and starts when the current one ends."""

    def __init__(self, hume, conductor, narrator_ref):
        self.hume, self.conductor, self.narrator_ref = hume, conductor, narrator_ref
        self.thread, self.job, self.started, self.done, self.last = None, "", 0.0, [], ""
        self.queued = []
        self.words = {}  # job -> Erlan's exact words, still being transcribed by the precise ear
        self.previous = ""  # the result of the job before the current one
        # what the bench did last, across jobs: what Erlan says next may be about it ("se movió hacia adelante")
        self.actions, self._args = collections.deque(maxlen=5), {}
        hume.on_result = self._result

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def submit(self, job, words=None):
        """words: a future with Erlan's words from the precise ear, handed to the worker with the job (part numbers
        like TP4056 come out right there)."""
        if words is not None:
            self.words[job] = words
        same = lambda a, b: difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() > 0.8  # noqa: E731
        if self.running and any(same(job, j) for j in [self.job] + self.queued):
            return  # the conductor restating the job it already handed over
        if self.running:
            self.queued.append(job)
            log("job", f"   [trabajo en cola] {job}")
        else:
            self.start(job)

    def start(self, job):
        log("job", f"   [trabajo] {job}")
        self.previous = self.last if self.job else ""
        self.job, self.started, self.done = job, time.time(), []
        if board is not None:
            board.busy = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _tool(self, name, args):
        log("tool", f"   [herramienta] {name} {args if args else ''}")
        self._args = args
        # The conductor has just said it starts; a step is worth telling when it comes later in the job.
        if name in TOLD_AT_START and time.time() - self.started > 10:
            bed = args.get("bed") or self.hume.world["bed"]
            step = STEPS[name].replace("la placa", f"la {bed}" if bed else "lo que hay en la mesa")
            self.narrator_ref().update(f"Ahora {step}.", lambda: self.hume.status == name)

    def _thought(self, text):
        """What the agent thinks as it works ("no se ve el punto en dos agujeros; la miro con el microscopio"): in the
        log, and said at the narrator's pace, a newer thought replacing one not yet said."""
        text = " ".join(text.split())
        if not text or text == getattr(self, "_last_thought", ""):
            return
        self._last_thought = text
        log("think", f"   (piensa) {text[:400]}")
        self.narrator_ref().update(text)

    def _result(self, name, result):
        if isinstance(result, dict) and result.get("ok") is False and result.get("error"):
            # a tool that failed shows here, with why: the worker may try again without a word. A long step that
            # failed is also said (Erlan, Nov 3: "la próxima vez por favor avísame")
            log("info", f"   ({name} falló: {str(result['error'])[:240]})")
            if name in LONG_STEPS and not result.get("refused"):
                self.narrator_ref().must(f"Ojo: se cortó {LONG_STEPS[name]}. Veo qué pasó y te digo.")
        if name not in ("where", "look_microscope", "look_bench", "look_kinect", "kinect_distance", "look_survey"):
            got = json.dumps(result, ensure_ascii=False, default=float)[:160] if isinstance(result, dict) else ""
            self.actions.append(f"{name} {json.dumps(self._args, ensure_ascii=False)} -> {got}")
        text = milestone(name, result)
        if text and text.startswith("Ojo"):  # the Kinect disagrees: said as soon as nobody talks, not queued
            self.done.append(text)
            self.narrator_ref().must(text)
        elif text:
            self.done.append(text)
            self.narrator_ref().update(text, result=True)

    def _run(self):
        narrator = self.narrator_ref()
        t0 = time.time()
        request, words = self.job, self.words.pop(self.job, None)
        if words is not None:
            try:
                exact = words.result(timeout=15)
            except Exception:
                exact = ""
            if exact:
                log("ear", f"   (oído preciso: {exact})")
            if exact and latin(exact):
                request += f"\n(Lo que dijo Erlan, palabra por palabra: «{exact}»)"
        # the conversation this job comes from, not only the conductor's one-line errand: without it the worker acted as
        # if it walked in from another room (Erlan, 3 Nov: "que trabaje con contexto")
        talk = [m["content"] for m in self.conductor.turns[-8:] if m.get("content")]
        if talk:
            request += "\n(La conversación reciente con Erlan, para el contexto:\n" + "\n".join(
                t if t.startswith("Erlan:") else f"Lunes: {t}" for t in talk) + ")"
        try:
            answer = self.hume.handle(request, self._tool, on_thought=self._thought)
        except Exception as e:
            log("info", f"   (el trabajo falló: {type(e).__name__}: {e})")
            answer = "Se me cayó el trabajo por un error; está en el registro."
        log("timing", f"   (trabajo terminado en {time.time() - t0:.1f} s: cerebro {self.hume.brain_seconds:.1f} s en "
            f"{self.hume.brain_steps} pasos)")
        self.last = answer
        if getattr(self, "on_done", None):  # the result on disk too, for a restart's continuity
            self.on_done()
        if board is not None:
            board.busy = False
        narrator.clear()  # what was pending is old news now
        if self.hume.stop.is_set():
            self.queued.clear()  # "para" stops everything, the line too
            self.words.clear()
            narrator.must("Listo, me detengo.")
            return
        log("result", f"   (resultado completo) {answer}")
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s]
        # Erlan, 3 Nov: "no importa que responda harto, que sea coherente e inteligente": only a very long answer is
        # told again, keeping the reasoning (what was seen, what it means, why), not cut to three sentences
        if len(sentences) > 8 or len(answer) > 900:
            try:
                short = self.conductor.tell(
                    f"El equipo terminó el trabajo «{self.job}» y respondió esto: «{answer}». Cuéntaselo a Erlan en "
                    f"pocas frases habladas, con los datos importantes y el razonamiento (qué se vio, qué significa y "
                    f"por qué), sin agregar nada que no esté ahí, sin preguntas ni ofrecimientos.",
                    history=False)
                answer = short or answer
            except Exception:
                pass
        narrator.must(answer)
        if self.queued:
            job = self.queued.pop(0)
            narrator.must("Ahora sigo con lo otro que me pediste.")
            self.start(job)


# When the brain cannot say hello in time: still not the same line every start
GREETINGS = ("Buenas, Erlan. Acá estoy.", "Listo, ya desperté. ¿Qué hacemos?", "Aquí estoy, cuando quieras.",
             "Buenas. ¿En qué andamos?", "Ya llegué al banco. Cuando quieras empezamos.")


def greeting(conductor, kinect):
    """What Lunes says on waking up: the conductor makes it up (the hour, what is on the bench), different each time;
    after ten seconds without an answer, one of a few plain ones."""
    missing = "" if kinect else " Dile también, de pasada, que el Kinect no está conectado."
    pool = ThreadPoolExecutor(1)
    try:
        hello = pool.submit(conductor.greet, time.localtime().tm_hour, missing).result(timeout=10)
    except Exception:
        hello = ""
    finally:
        pool.shutdown(wait=False)
    if not re.search(r"[A-Za-zÁÉÍÓÚáéíóúñÑ]", hello or ""):  # Glimmer once greeted with "..."
        hello = random.choice(GREETINGS) + ("" if kinect else " El Kinect no está conectado.")
    return hello


def wait_for_microphone(voice, error, ear):
    """When the microphone's cable moves, Windows drops it. Say so once and wait until it is back, instead of
    stopping Lunes."""
    print(f"(se perdió el micrófono: {error}; espero a que vuelva)")
    ear.hearing = False
    voice.say("Perdí el micrófono. Revisa el cable.")
    while True:
        time.sleep(3)
        sd._terminate()  # PortAudio keeps the old device list; start it again to see what is plugged in now
        sd._initialize()
        try:
            if ear.microphone:  # its number changes when it comes back
                ear.device = find_microphone(ear.microphone)
            sd.check_input_settings(device=ear.device, samplerate=16000, channels=1)
            print("(el micrófono volvió)")
            voice.say("Ya te escucho otra vez.")
            return
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", default="COM8")
    ap.add_argument("--camera", default=MICROSCOPE,
                    help="the microscope: its number, or part of its name or USB id (the MS5 is vid_1224&pid_2825)")
    ap.add_argument("--banco", default=BENCH_CAMERA,
                    help="the camera that sees the whole work area, by name or number ('no' to go without it)")
    ap.add_argument("--osciloscopio", default="hantek",
                    help="'hantek' (the 6022BE on the USB), 'simulado' (made-up signals, to try Lunes without it) "
                         "or 'no'")
    ap.add_argument("--arduino", default="usb",
                    help="'usb' (a real board, found by its port when Lunes programs it) or 'simulado' (a made-up "
                         "board that reads the simulated circuit, to try Lunes without one)")
    ap.add_argument("--falla", default=None,
                    help="with --arduino simulado: a fault planted on the made-up bench, e.g. R1:open or U1:not_running")
    ap.add_argument("--text", action="store_true",
                    help="keyboard only, no microphone (without it both work at once: talk, or type in the console; "
                         "/mic off and /mic on switch the microphone)")
    ap.add_argument("--no-wake", action="store_true", help="answer everything, not only sentences with 'Lunes'")
    ap.add_argument("--model", default=MODEL, help="Ollama model for the conductor and the local worker")
    ap.add_argument("--think", default=THINK, help="how much the local worker reasons: false, low, medium, high")
    ap.add_argument("--spark", action="store_true",
                    help="the worker uses the Meta Model API: the Meta model sees the pictures (with free credits Meta "
                         "may use them)")
    ap.add_argument("--brain", default="muse-wonder-1.0",
                    help="with --spark, the Meta model at the wheel of the work: muse-wonder-1.0 (the big one) or "
                         "muse-spark-1.3 (faster); the conductor of the conversation stays on Muse Spark")
    ap.add_argument("--effort", default="high",  # Erlan, 3 Nov: more thinking, even if each step takes longer
                    help="with --spark, how much the worker reasons per step: low, medium or high (medium takes a "
                         "few seconds more a step and thinks things through)")
    ap.add_argument("--mic", default=None, help="part of the microphone's name, e.g. fifine (Windows' default if left out)")
    ap.add_argument("--audifonos", action="store_true",
                    help="Lunes speaks into headphones, so the microphone never hears it: everything heard is Erlan, "
                         "even over Lunes' voice (set the headphones as Windows' sound output)")
    args = ap.parse_args()

    global board
    print("conectando la CNC, el microscopio, la cámara del banco y el Kinect...")
    index = find_camera(args.camera)
    if index is None:
        sys.exit(f"no encuentro el microscopio ({args.camera}); las cámaras conectadas son: {cameras()}")
    machine, cam = Machine(args.port), Camera(index)
    OPEN.extend([machine.close, cam.cap.release])
    print(f"posición en el banco: { {k: round(v, 2) for k, v in machine.position().items()} }")
    bench = None
    index = None if args.banco.lower() == "no" else find_camera(args.banco)
    if index is not None:
        try:
            bench = Camera(index, size=(3840, 2160))  # the EMEET gives this at 19 fps over USB; a Sony what it is set to
            OPEN.append(bench.release)
            h, w = bench.frame().shape[:2]
            print(f"cámara del banco: {args.banco}, {w}x{h}")
        except RuntimeError as e:
            print(f"(la cámara del banco {args.banco} no da imagen: {e}; sigo sin ella)")
    elif args.banco.lower() != "no":
        print(f"(la cámara del banco {args.banco} no está; revisa su cable USB)")
    try:
        kinect = Kinect()
        OPEN.append(kinect.close)
    except RuntimeError as e:
        kinect = None
        print(f"(sin Kinect: {e})")
    scope, probe = None, machine.config.get("scope", {}).get("probe", {})
    if args.osciloscopio.lower() == "simulado":  # a 1 kHz square on CH1 and a 10 kHz sine on CH2
        scope = SimulatedScope({"CH1": square(1000, 0.0, 2.0), "CH2": sine(10e3, 1.0, 2.5)}, probe=probe)
        print("osciloscopio: SIMULADO (señales inventadas, no del banco)")
    elif args.osciloscopio.lower() != "no":
        try:
            scope = Scope(probe=probe)
            OPEN.append(scope.close)
            print(f"osciloscopio: Hantek 6022BE, firmware {scope.firmware_version:#06x}, puntas {scope.probe}")
        except RuntimeError as e:
            print(f"(sin osciloscopio: {e})")
    mcu_board = None
    if args.arduino.lower() == "simulado":
        mcu_board = SimulatedBoard(args.falla)
        print("Arduino: SIMULADO (lee el circuito simulado, no el banco)" +
              (f", con la falla {args.falla} plantada" if args.falla else ""))
    think = False if args.think.lower() in ("false", "no", "0") else args.think
    worker_name = "Muse Wonder" if "wonder" in args.brain else "Muse Spark"
    brain = ("Muse Spark por la API de Meta (Muse Glimmer en esta PC si no hay internet)" if args.spark
             else "Muse Glimmer en esta PC")
    from lunes import inventory
    about = (f"Con qué funcionas: tu oído es Omnilingual ASR de Meta y tu voz es MMS-TTS de Meta, los dos en esta PC; "
             f"tú piensas con {brain}, y el equipo que trabaja con "
             f"{worker_name + ' por la API de Meta' if args.spark else brain}. Tu memoria, Hume, guarda en esta PC lo que "
             f"observaste. {inventory.summary()}".rstrip())
    hume = Hume(machine, cam, kinect, show=show, model=args.model, think=think,
                on_think=lambda t: log("think", "   (pensó) " + " ".join(t.split())[:300]), spark=args.spark,
                about=about, bench=bench, scope=scope, mcu=mcu_board, brain_model=args.brain, effort=args.effort)
    from lunes import simulator
    simulator_ok = simulator.selftest()
    simulator_ok["hanging_runs_ended"] = simulator.kill_strays()
    print("simulador: ngspice " + (f"listo ({simulator_ok['seconds']} s)" if simulator_ok["ok"] else
                                   f"CON PROBLEMAS: {simulator_ok.get('problem')}"))
    print("trabajador: " + (f"{hume.brain_name} ({hume.spark.model}, esfuerzo {hume.effort}) por la API de Meta; "
                            f"conductor: Muse Spark (si falla, Muse Glimmer local)"
                            if hume.brain == "spark"
                            else f"Muse Glimmer local ({args.model}); a la API de Meta no se le manda nada"))
    print(f"ubicación por el último mosaico: {hume.relocate() or 'no me encontré en el mosaico'}")
    if kinect and machine.config.get("kinect_head"):
        print(f"posición según el Kinect: {hume.check_position()}")
    conductor_model = hume.spark  # the conductor answers in a few seconds: Muse Spark, even when Wonder does the work
    if hume.spark is not None and "wonder" in hume.spark.model:
        from lunes.meta_api import MetaModels
        conductor_model = MetaModels()
    conductor = Conductor(conductor_model, args.model, about=about)
    warming = threading.Thread(target=conductor.warm, daemon=True)  # the first load takes most of a minute
    warming.start()
    voice = Voice()
    voice.prepare(*REACTIONS, "Listo, me detengo.", "Déjame pensarlo.")
    hume.keep_world(ROOT / "runs" / "bench_world.json")
    board = Dashboard(camera=cam, kinect=kinect, speaking=lambda: voice.speaking,
                      bench=bench, lab=lambda: lab_state(hume))
    hume.on_image = lambda image, what: board.set_seen(image, what)
    board.mosaic = hume.bench_mosaic
    board.on_tap = lambda x, y: hume.taps.append((x, y))
    # the web lab's list of saved circuits (each with its plan) and the one Erlan picks as the circuit on the bench
    board.circuits = lambda: hume._circuit_list()[0]["circuits"]
    board.on_circuit_load = lambda file: hume._circuit_load(file)[0]
    board.on_layout_edit = hume.layout_edit  # Erlan's breadboard edited by hand in the web lab
    board.on_scope_live = hume.scope_live_on

    def scope_live_loop():  # Erlan's continuous view: a few pictures a second, paused while Lunes measures
        while True:
            try:
                shown = hume.scope_live_frame()
            except Exception:  # noqa: BLE001  (never let the view's thread die)
                shown = False
            time.sleep(0.25 if shown else 0.5)

    threading.Thread(target=scope_live_loop, daemon=True).start()
    print(f"página en vivo: {board.url}")

    def watch_supply():  # always knows what the supply shows, and says when it changes
        last = None
        while True:
            try:
                seen = hume.supply_look()
            except Exception as e:
                seen = {"ok": False, "why": str(e)}
            now = (seen.get("volts"), seen.get("amps"), seen.get("out")) if seen.get("ok") else None
            if now != last:
                log("info", f"fuente: {now[0]:.2f} V, {now[1]:.3f} A, salida {'encendida' if now[2] else 'apagada'}"
                    if now else f"fuente: {seen.get('why')}")
                last = now
            time.sleep(5)
    if bench:
        threading.Thread(target=watch_supply, daemon=True).start()
    ear = None
    if not args.text:
        print("cargando los oídos (Omnilingual ASR de Meta, en esta PC)...")
        fast = local_transcriber(FAST_EAR, provider="cuda", threads=2)  # every utterance
        precise = local_transcriber()  # the 1B model, for the words of a job
        precise_pool = ThreadPoolExecutor(1)
        while True:  # a microphone unplugged at start is waited for, not a crash
            try:
                ear = Ear(fast, microphone=args.mic)
                floor = ear.calibrate()
                break
            except (RuntimeError, sd.PortAudioError) as e:
                print(f"(no encuentro el micrófono {args.mic or ''}: {e}; espero a que lo conectes)")
                time.sleep(3)
                sd._terminate()  # PortAudio keeps the old device list; start it again to see what is plugged in now
                sd._initialize()
        print(f"micrófono: {sd.query_devices(ear.device, 'input')['name'] if ear.device is not None else 'el de Windows'}")
        print(f"ruido de la habitación: {floor:.4f} (no hables durante este segundo)")

    narrator = None
    worker = Worker(hume, conductor, lambda: narrator)

    def state():
        """What the conductor knows on each turn: the job, its step, what came out, and the bench."""
        parts = []
        if worker.running:
            step = STEPS.get(hume.status, hume.status or "pensando")
            # "todavía no terminó": alone with the job's name, Glimmer told Erlan "ya moví la máquina" before it moved
            parts.append(f"trabajo en curso desde hace {time.time() - worker.started:.0f} s, todavía no terminó: "
                         f"«{worker.job}»; paso actual: {step}" + (f" ({board.status})" if board.status else ""))
            if worker.done:
                parts.append("lo que ya encontró el equipo: " + " ".join(worker.done[-3:]))
            if worker.queued:
                parts.append("en cola, para después: " + "; ".join(f"«{j}»" for j in worker.queued))
            if worker.previous:
                parts.append(f"resultado del trabajo anterior: {worker.previous[:400]}")
        elif worker.last:
            parts.append(f"nada en curso; el último trabajo fue «{worker.job}» y su resultado: {worker.last[:400]}")
        else:
            parts.append("nada en curso")
        if worker.actions:
            parts.append("lo último que hizo el equipo con la máquina: " + "; ".join(worker.actions))
        sides = "; ".join(f"{a}: {t}" for a, t in machine.config.get("como_lo_ve_erlan", {}).items())
        if sides:
            parts.append(f"hacia dónde va cada eje, visto desde Erlan: {sides}")
        parts.append(hume.brief())
        return "; ".join(parts) + "."

    frames_seen = {"count": 0, "at": time.time()}

    def senses():
        """What Lunes' senses and hands are doing right now, read from them on each turn, not remembered."""
        parts = []
        if ear is None:
            parts.append("oído: apagado, Erlan te escribe")
        else:
            ago = f", lo último que oíste fue hace {time.time() - ear.last_heard:.0f} s" if ear.last_heard else ""
            parts.append(f"oído: micrófono {args.mic or 'de Windows'} funcionando{ago}; escribe sin puntuación ni "
                         f"mayúsculas, a veces 'hola' como 'o la' u 'ola' y tu nombre como 'luna', 'lunas' o 'lúnes'")
        try:
            gray = cv2.cvtColor(cam.frame(), cv2.COLOR_BGR2GRAY)
            per_s = (cam._count - frames_seen["count"]) / max(time.time() - frames_seen["at"], 1e-3)
            frames_seen.update(count=cam._count, at=time.time())
            best = hume.surveys[-1]["summary"].get("sharpness") if hume.surveys else None
            parts.append(("microscopio: conectado y dando imagen" if per_s > 1 else
                          "microscopio: casi no manda imágenes, puede estar desconectado") +
                         f"; nitidez ahora {sharpness(gray):.0f}" +
                         (f" (enfocado sobre una placa da unos {best:.0f})" if best else ""))
        except Exception:
            parts.append("microscopio: no da imagen")
        # what the cameras point at, not what they see now: with "ve la CNC" in here it said "sí, la veo" without
        # looking
        if bench:
            _, problem = hume.bench_picture()
            parts.append("cámara del banco: conectada pero da la imagen en negro (¿tapa del lente puesta, o "
                         "se apagó?)" if problem else
                         "cámara del banco: conectada y apuntando a toda el área de trabajo, con mejor "
                         "imagen que el Kinect (lo que hay ahora, solo mirando)")
        else:
            parts.append("cámara del banco: no conectada")
        if hume.scope is None:  # hume's own: system_restart may open it again
            parts.append("osciloscopio: no conectado")
        else:
            kind = ("SIMULADO, con señales inventadas (no del banco)" if isinstance(hume.scope, SimulatedScope) else
                    "Hantek 6022BE conectado")
            parts.append(f"osciloscopio: {kind}, dos canales, puntas x{hume.scope.probe['CH1']:g} (CH1) y "
                         f"x{hume.scope.probe['CH2']:g} (CH2); mide voltajes y señales donde Erlan ponga la punta")
        if isinstance(hume.mcu, SimulatedBoard):
            parts.append("Arduino: SIMULADO, lee el circuito simulado (no el banco)")
        elif hume.mcu_port:
            parts.append(f"Arduino: programado en {hume.mcu_port}")
        power = hume.supply_state()
        if power["seen_now"]:
            parts.append(f"fuente KORAD: la ves en {power['volts']:.2f} V y {power['amps']:.3f} A, salida "
                         f"{'encendida' if power['out'] else 'apagada'}")
        else:
            parts.append(f"fuente KORAD: {power['why']}; {power['note']}")
        parts.append("simulador de circuitos (ngspice): " + ("listo" if simulator_ok["ok"] else
                                                             f"con problemas ({simulator_ok.get('problem')})"))
        parts.append("Kinect: conectado y apuntando a la CNC de frente y desde arriba; sirve para la profundidad y "
                     "para confirmar la posición" if kinect else "Kinect: no conectado")
        check = hume.last_kinect_check
        if kinect and not machine.config.get("kinect_head"):
            parts.append("posición: el Kinect todavía no está calibrado para confirmarla (calibrate_kinect)")
        elif check:
            parts.append(f"posición: hace {time.time() - check['at']:.0f} s el Kinect " +
                         ("la confirmó" if check["agrees"] else
                          f"vio el cabezal a {check['off_mm']:.0f} mm de donde crees que está"))
        at = {k: round(v) for k, v in machine.at_mm.items()}
        parts.append(f"CNC: conectada, {'trabajando' if worker.running else 'quieta'}, en X {at.get('X')}, "
                     f"Y {at.get('Y')}, Z {at.get('Z')} mm")
        parts.append("Se acabaron los tokens de hoy: estás con el modelo local y no puedes mover la máquina hasta "
                     "medianoche" if conductor.out_of_tokens else
                     "Muse Spark: disponible por internet" if hume.spark else "Muse Spark: no se usa (modo privado)")
        return ". ".join(parts) + "."

    conductor.context = lambda: (senses(), state())
    narrator = Narrator(voice, ear, conductor, lambda: worker.running)
    narrator.start()

    def interrupt():
        stop_requested.set()
        hume.stop.set()
        voice.stop()
        narrator.clear()
        machine.soft_stop()

    # The window's keys set stop_requested from inside show(); pass that on while a job runs.
    def watch_keys():
        while True:
            stop_requested.wait()
            interrupt()
            time.sleep(0.2)
            stop_requested.clear()

    threading.Thread(target=watch_keys, daemon=True).start()

    def close_all():
        """Leaves the CNC's port, the cameras and the Kinect closed, so the next Lunes finds them free."""
        if worker.running:
            interrupt()
            worker.thread.join(timeout=10)
        machine.close()
        cam.cap.release()
        if bench:
            bench.release()
        if kinect:
            kinect.close()
        if hume.scope:
            hume.scope.close()
        release_gpu(args.model)

    def watch_close_file():
        while not CLOSE_FILE.exists():
            time.sleep(0.5)
        CLOSE_FILE.unlink(missing_ok=True)
        log("info", "(me piden cerrar desde afuera: cierro la CNC y las cámaras)")
        close_all()
        os._exit(0)  # the main thread may be stuck waiting for the ear

    CLOSE_FILE.unlink(missing_ok=True)  # one left from before is not for this Lunes
    threading.Thread(target=watch_close_file, daemon=True).start()

    heard_q = queue.Queue()
    mic_on = threading.Event()  # cleared by "/mic off" typed in the console: Lunes then listens to the keyboard only
    mic_on.set()

    def set_mic(on, how="desde la web"):
        """Microphone on or off, said in the log: off while Erlan steps out, talks to someone else, or the fridge and
        the dogs are loud; he then types. Returns the state."""
        if on:
            mic_on.set()
            log("info", f"(micrófono encendido {how}: te oigo y te leo)")
        else:
            mic_on.clear()
            log("info", f"(micrófono apagado {how}: te leo solo por escrito)")
        return mic_on.is_set()

    # the web lab's text box and microphone button reach here (lunes.dashboard calls these)
    board.on_text = lambda line: heard_q.put(("text", line.strip())) if line and line.strip() else None
    board.on_mic = set_mic
    board.mic_is_on = mic_on.is_set

    def listen():
        while True:
            if mic_toggle.is_set():  # the m key in the window
                mic_toggle.clear()
                set_mic(not mic_on.is_set(), "con la tecla m")
            if not mic_on.is_set():
                time.sleep(0.2)
                continue
            try:
                got = ear.record_utterance(wait_s=5, max_s=20, silence_s=0.6)
            except sd.PortAudioError as e:
                wait_for_microphone(voice, e, ear)
                continue
            if got and mic_on.is_set():
                heard_q.put(("audio", got))

    def read_typed():
        """Typed lines go in beside the microphone's: a word Lunes mishears can be typed, and a noisy bench can
        be worked from the keyboard. Without a console (no stdin) this thread just ends."""
        while True:
            try:
                line = input("\nTú: ").strip()
            except (EOFError, OSError):
                return
            if line:
                heard_q.put(("text", line))

    # both channels at once: the microphone (unless --text) and the keyboard, always
    if not args.text:
        threading.Thread(target=listen, daemon=True).start()
    threading.Thread(target=read_typed, daemon=True).start()
    print("(también puedes escribirle aquí; /mic off apaga el micrófono, /mic on lo enciende)", flush=True)
    # Continuity: a restart within a few hours picks the conversation up where it was, with no greeting (Erlan, 3 Nov:
    # "ya estoy harto de que me diga buenas tardes... directamente continuidad del trabajo"). Each project keeps its
    # own conversation (runs/projects/<id>/conversation.json); the one in runs/ from before projects goes to the
    # project open at the first start.
    legacy_talk = ROOT / "runs" / "conversation.json"

    def talk_file(pid):
        return PROJECTS / pid / "conversation.json" if pid else legacy_talk

    def read_talk(pid, legacy=False):
        """A project's saved conversation; legacy=True also takes the one from before projects (only at start)."""
        for f in (talk_file(pid), legacy_talk) if legacy else (talk_file(pid),):
            try:
                return json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
        return {}

    def save_talk(pid=None):
        pid = pid or hume.project_id()
        f = talk_file(pid)
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps({"at": time.time(), "turns": conductor.turns, "last_job": worker.job,
                                     "last_result": worker.last}, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass

    resumed = False
    saved_talk = read_talk(hume.project_id(), legacy=True)
    if saved_talk.get("turns"):
        conductor.turns = saved_talk["turns"][-2 * conductor.keep_turns:]
        worker.last = saved_talk.get("last_result") or ""
        resumed = time.time() - float(saved_talk.get("at", 0)) < 3 * 3600
    if hume.project_id() and legacy_talk.exists():  # the conversation from before projects now lives in its project
        save_talk()
        legacy_talk.replace(legacy_talk.with_name("conversation.before-projects.json"))

    def show_project():
        """The page shows the open project's name and its conversation so far."""
        board.project = {"id": hume.project_id(), "name": hume.project_name()}
        with board._lock:  # opened before in this session: its lines are on the page already
            shown = any(e.get("project") == board.project["id"] for e in board.events)
        for turn in [] if shown else conductor.turns[-12:]:
            text = str(turn.get("content") or "").strip()
            if not text or turn.get("role") not in ("user", "assistant"):
                continue
            if turn["role"] == "user":
                board.add("tu", re.sub(r"^Erlan:\s*", "", text), at="antes")
            else:
                board.add("lunes", text, at="antes")

    def switch_project(old, new):
        """Another project opened (the web's list, the + button, or Lunes' own tools): the conductor's conversation
        is kept with the old one and the new one's comes back."""
        save_talk(old)
        saved = read_talk(new) if new else {}
        conductor.turns = list(saved.get("turns") or [])[-2 * conductor.keep_turns:]
        worker.last = saved.get("last_result") or ""
        name = hume.project_name()
        board.clear_seen()  # the AI analysis view starts blank: the last picture was the other project's
        show_project()
        log("info", f"proyecto abierto: {name}" + ("" if conductor.turns else " (nuevo: conversación en blanco)"))
        save_talk(new)
        if getattr(hume, "just_created", None):  # the web's "Iniciar proyecto": the one greeting, for this project
            hume.just_created = None
            narrator.say(f"Empezamos el proyecto {name}. ¿Qué armamos?")

    global project_log
    project_log = hume.project_log
    hume.on_project_change = switch_project
    board.on_project_new = hume.project_new
    show_project()
    worker.on_done = save_talk
    if resumed:
        name = hume.project_name()
        line = f"Sigo con {name}." if name else "Sigo donde quedamos."
        log("lunes", f"Lunes: {line}")
        voice.say_soon(line)
    else:  # no greeting at a restart (Erlan, 3 Nov): Lunes greets when a project is started with its button
        log("info", f"listo, en el proyecto {hume.project_name() or 'sin nombre'}")

    last_exchange, turn = 0.0, 0
    # Two things in a row that were not for Lunes (Erlan talking with his father) close the conversation until he
    # calls "Lunes" again; with it open, Lunes kept asking to repeat bits of their talk.
    not_for_me, closed, closed_at = 0, False, 0.0
    # a sentence Erlan left half-way, waiting for the rest: {"text", "at", "called", "audio"}
    half = {"now": None}

    def nudge(waiting):
        """Nothing came after the half sentence: like someone listening, Lunes invites him to go on, once."""
        if half["now"] is waiting:
            log("lunes", "Lunes: ¿Sí? Te escucho.")
            voice.say_soon("¿Sí? Te escucho.")

    try:
        while True:
            kind, got = heard_q.get()
            t_ear, over_lunes = 0.0, False
            if kind == "audio":
                audio, start, end = got
                t0 = time.time()
                heard = ear.transcribe(audio)
                t_ear = time.time() - t0
                own, rest = voice.own_words(heard, start, end)
                if not args.audifonos:  # from the speakers Lunes' own voice comes back into the microphone
                    over_lunes = bool(own) or voice.talking_during(start, end)
                    heard = rest if own else heard
                elif own and len(own.split()) >= 0.6 * len((own + " " + rest).split()):
                    # with headphones this should never happen; if Lunes hears itself anyway, answering its own
                    # words would start a loop
                    log("info", f"(oí mis propias palabras: «{heard}»; no respondo)")
                    continue
            else:
                heard = got
                if heard.startswith("/"):  # an order for the program, typed: not for Lunes
                    order = heard[1:].strip().lower()
                    if order in ("mic off", "mic apagar", "mic apagado"):
                        set_mic(False, "desde la consola")
                    elif order in ("mic on", "mic encender", "mic encendido"):
                        set_mic(True, "desde la consola")
                    elif order == "mic":
                        log("info", f"(micrófono {'encendido' if mic_on.is_set() else 'apagado'})")
                    else:
                        log("info", "(órdenes escritas: /mic off, /mic on, /mic; lo demás se lo digo a Lunes)")
                    continue
            if not heard:
                continue
            if is_stop(heard):
                log("tu", f"Tú (parar): {heard}   (oído {t_ear:.2f} s)")
                if worker.running or voice.speaking:
                    interrupt()
                continue
            called = bool(WAKE.search(heard))
            waiting = half["now"]
            if waiting and time.time() - waiting["at"] < HALF_WAIT_S:  # the rest of what he left half-way
                heard, called, half["now"] = f"{waiting['text']} {heard}", called or waiting["called"], None
                if kind == "audio" and waiting["audio"] is not None:
                    audio = np.concatenate([waiting["audio"], audio])
            # Over Lunes' own voice only an explicit call counts, or a long phrase of plain Spanish (Erlan starting
            # his request as Lunes ends its answer): the rest is mostly its own words coming back, garbled.
            if over_lunes and not called and not sounds_like_speech(heard):
                log("info", f"(oí «{heard}» encima de mi voz: no respondo)")
                continue
            in_conversation = (args.no_wake or args.text or kind == "text" or worker.running or  # typed: always for Lunes
                               # closed by noise, but Lunes spoke since (it asked him something): his answer counts
                               ((not closed or narrator.last_spoken > closed_at)
                                and time.time() - max(last_exchange, narrator.last_spoken) < CONVERSATION_S))
            if not (called or in_conversation) and kind == "audio" and len(audio) < 5 * 16000:
                # The fast ear sometimes garbles the name itself ("olaonaa", "onaola" for "hola Lunes"): before
                # dropping a short phrase, the precise ear hears it again (half a second, and only for these).
                second = precise(audio)
                if WAKE.search(second):
                    log("ear", f"   (el oído rápido oyó «{heard}», el preciso «{second}»)")
                    heard, called = second, True
            if not (called or in_conversation):
                log("info", f"(oí «{heard}», sin «Lunes» y fuera de una conversación: no respondo)")
                continue
            log("tu", f"Tú: {heard}" + (f"   (oído {t_ear:.2f} s)" if kind == "audio" else ""))
            # the precise ear starts at once; only a job waits for it (the conductor answers from the fast one)
            words = precise_pool.submit(precise, audio) if kind == "audio" else None
            if voice.speaking:  # Erlan talks over Lunes: Lunes lets him speak
                voice.stop()
            # The conductor gets the whole phrase, name included: with "Lunes" taken out, "o la lunes" (the ear's
            # "hola Lunes") was left as "o la" and it could not tell it was a greeting.
            request = heard
            if QUIET.search(request) or TALKATIVE.search(request):
                narrator.quiet = bool(QUIET.search(request))
                log("info", "   (desde ahora " + ("solo aviso al terminar)" if narrator.quiet else "cuento cómo voy)"))
            if BYE.search(request) and len(request.split()) <= 4:
                if worker.running:
                    interrupt()
                voice.say("Hasta luego, Erlan.")
                break
            if warming.is_alive():
                voice.say_soon("Dame un momento, estoy despertando.")
                warming.join()
            reaction, turn = REACTIONS[turn % len(REACTIONS)], turn + 1
            answered = threading.Event()
            threading.Timer(REACTION_AFTER_S, lambda: answered.is_set() or voice.say_soon(reaction)).start()
            # the conductor now reasons before it answers (several seconds): a second word so the wait is not dead air
            threading.Timer(THINKING_AFTER_S, lambda: answered.is_set() or voice.say_soon("Déjame pensarlo.")).start()
            t0 = time.time()
            try:
                say, action, todo = conductor.reply(request, called=called)
            except Exception as e:
                answered.set()
                log("info", f"   (el conductor no respondió: {e})")
                voice.say_soon("Perdón, se me trabó la cabeza. ¿Me lo repites?")
                continue
            answered.set()
            save_talk()  # every exchange on disk: a restart picks up from here
            narrator.answered()
            last_exchange = time.time()
            notice = conductor.tokens_notice()
            if notice:
                voice.say_soon(notice)
            if called:
                closed = False
            not_for_me = not_for_me + 1 if (not say and action == "ninguna") else 0
            if not_for_me >= 2 and not closed:
                closed, closed_at = True, time.time()
                log("info", "   (dos veces seguidas no era para mí: cierro la conversación hasta que me llamen)")
            if action == "esperar":
                half["now"] = {"text": request, "at": time.time(), "called": called,
                               "audio": audio if kind == "audio" else None}
                log("think", f"   (espero el resto de la frase: {conductor.understood})")
                threading.Timer(NUDGE_AFTER_S, nudge, args=(half["now"],)).start()
                continue
            if action == "parar":  # before saying so: stopping also silences what was queued to be said
                worker.queued.clear()
                if worker.running:
                    interrupt()
            if say:
                narrator.say(say, remember=False)
            if action == "mirar":  # a quick look, not a job: one picture and what it shows
                worker.submit(f"Mirada rápida, sin recorrer la placa: {todo}. Si es un chip y no sale entero, "
                              f"center_on_chip primero. Una sola foto y di en dos o tres frases qué se ve.", words)
            elif action == "trabajo":
                worker.submit(todo, words)
            log("think", f"   (entendió: {conductor.understood})")
            log("timing", f"   (conductor {conductor.brain} {time.time() - t0:.2f} s, {action}" +
                (f", oído {t_ear:.2f} s)" if kind == "audio" else ")"))
    except KeyboardInterrupt:
        pass
    finally:
        close_all()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # while Lunes starts (waiting for the microphone, loading the ears): before main's own
        pass
    finally:
        shut_down()
