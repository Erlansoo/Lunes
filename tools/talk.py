"""Talk to Lunes: listen, think with Muse Glimmer, act on the bench, answer out loud.

Start the sentence with "Lunes" so it knows you are talking to it (--no-wake to drop that), and "Lunes, adiós" to
stop. "Lunes, para" (or "basta", "detente", "cállate"), Space or Esc in its window interrupt it while it talks or
works. It only obeys Erlan's voice: the first time it learns it from five phrases (--enroll to learn it again). Its
memory, Hume, keeps what it observed between sessions.
--text reads typed lines instead of the microphone, to test without talking.

    python tools/talk.py COM8
    python tools/talk.py COM8 --text
"""
import argparse
import re
import sys
import threading
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from lunes.hume import MODEL, Hume  # noqa: E402
from lunes.kinect import Kinect, start_bridge  # noqa: E402
from lunes.machine import Machine  # noqa: E402
from lunes.speaker import Speakers  # noqa: E402
from lunes.voice import Ear, Voice  # noqa: E402
from microscope import Camera  # noqa: E402

# The wake word. "el lunes", "este lunes"... are the weekday, not a call, so they do not count.
WAKE = re.compile(r"(?<!el )(?<!este )(?<!cada )(?<!próximo )(?<!pasado )\b(lunes|lune|lunés)\b[,.!]?\s*", re.IGNORECASE)
BYE = re.compile(r"\b(adi[oó]s|chau|apágate|termina)\b", re.IGNORECASE)
# An interruption is a short order that ends what was heard ("¡basta!", "Lunes, para", or the same after Lunes' own
# words leaking in from the speakers); "Lunes, para qué sirve..." is a question, not an order.
STOP_WORD = r"(p[aá]ra(te)?|basta|det[eé]nte|alto|stop|c[aá]llate|silencio|espera)"
STOP = re.compile(rf"(^|[.!?,¡¿]\s*)\W*((lunes|lune)\W+)?{STOP_WORD}(\W+(ya|lunes|por favor|ahí|un momento))*\W*$",
                  re.IGNORECASE)
WIN = "Lunes"
FOLLOW_UP_S = 40  # seconds after an answer during which a follow-up needs no wake word
ENROLL_PHRASES = ["Lunes, mide la placa y dime cuánto mide.", "Lunes, recorre toda la placa y toma fotos.",
                  "Lunes, inspecciona el integrado y lee su número.",
                  "Hoy estoy trabajando en el banco de electrónica.", "Lunes, para un momento."]

stop_requested = threading.Event()


def show(frame, text):
    f = frame.copy()
    for color, width in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.putText(f, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, width)
    cv2.imshow(WIN, f)
    if cv2.waitKey(1) in (27, 32):  # Esc or Space in the window: stop
        stop_requested.set()


def enroll(voice, ear, speakers):
    """Learns Erlan's voice from five phrases he repeats after Lunes, plus Lunes' own voice to tell them apart."""
    voice.say("Erlan, voy a aprender tu voz, para obedecerte solo a ti. Repite cada frase después de que yo la diga.")
    samples = []
    for phrase in ENROLL_PHRASES:
        for _ in range(2):
            voice.say(phrase)
            audio = ear.record_utterance(wait_s=15, max_s=10)
            if audio is not None and len(audio) > 0.8 * 16000:
                samples.append(audio)
                print(f"   frase grabada ({len(audio) / 16000:.1f} s)")
                break
            voice.say("No te escuché bien. Otra vez.")
    if len(samples) < 3:
        voice.say("No alcancé a aprender tu voz. Te voy a escuchar a todos por ahora.")
        return
    own_voice = [(a.astype("float32") / 32768, r) for a, r in (voice.synth(p) for p in ENROLL_PHRASES)]
    stats = speakers.enroll(samples, own_voice)
    print(f"voz de Erlan aprendida: {stats}")
    voice.say("Listo, ya conozco tu voz.")


class StopListener(threading.Thread):
    """Listens while Lunes talks or works, only for an interruption. Lunes' own voice from the speakers is told
    apart by its voiceprint and by the shape of the phrase."""

    def __init__(self, ear, speakers, on_stop):
        super().__init__(daemon=True)
        self.ear, self.speakers, self.on_stop = ear, speakers, on_stop
        self.active = True

    def run(self):
        while self.active:
            audio = self.ear.record_utterance(wait_s=0.6, max_s=4, silence_s=0.4)
            if audio is None or not self.active:
                continue
            who, _ = self.speakers.who(audio)
            if who == "lunes":
                continue
            text = self.ear.transcribe(audio)
            if text and STOP.search(text):
                print(f"Tú (interrumpiendo): {text}")
                self.on_stop()

    def finish(self):
        self.active = False
        self.join(timeout=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", default="COM8")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--text", action="store_true", help="type instead of talking")
    ap.add_argument("--no-wake", action="store_true", help="answer everything, not only sentences with 'Lunes'")
    ap.add_argument("--enroll", action="store_true", help="learn Erlan's voice again")
    ap.add_argument("--model", default=MODEL, help="Ollama model for the brain (Muse Glimmer unless testing)")
    args = ap.parse_args()

    print("conectando la CNC, el microscopio y el Kinect...")
    machine, cam = Machine(args.port), Camera(args.camera)
    print(f"posición en el banco: { {k: round(v, 2) for k, v in machine.position().items()} }")
    kinect = Kinect() if start_bridge() else None
    hume = Hume(machine, cam, kinect, show=show, model=args.model)
    if args.model != MODEL:
        print(f"OJO: cerebro de prueba {args.model}; el de Lunes es {MODEL}")
    voice = Voice()
    ear = None if args.text else Ear(compute_type="int8_float16")
    speakers = None if args.text else Speakers()
    if ear:
        print(f"ruido de la habitación: {ear.calibrate():.4f} (no hables durante este segundo)")
        if args.enroll or not speakers.enrolled:
            enroll(voice, ear, speakers)

    said_now = set()

    def say_soon(text):
        """Each sentence is spoken as soon as the model finishes writing it, and never twice in one answer."""
        if text in said_now or stop_requested.is_set():
            return
        said_now.add(text)
        print(f"Lunes: {text}")
        voice.say_soon(text)

    def on_tool(name, a):
        print(f"   [herramienta] {name} {a if a else ''}")

    def interrupt():
        if not stop_requested.is_set():
            stop_requested.set()
        hume.stop.set()
        voice.stop()
        machine.soft_stop()

    # The window's keys set stop_requested from inside show(); pass that on to Hume while it works.
    def watch_keys():
        while True:
            stop_requested.wait()
            interrupt()
            time.sleep(0.2)
            while stop_requested.is_set():
                time.sleep(0.1)

    threading.Thread(target=watch_keys, daemon=True).start()

    last_answer = 0.0
    try:
        say_soon("Hola Erlan, soy Lunes. Dime Lunes y te escucho." + ("" if kinect else " El Kinect no está conectado."))
        while True:
            voice.wait()  # never listen while talking, or Lunes hears itself
            stop_requested.clear()
            said_now.clear()
            if args.text:
                heard = input("\nTú: ").strip()
            else:
                print("\n(escuchando...)")
                audio = ear.record_utterance(wait_s=60, max_s=20)
                if audio is None:
                    continue
                t_heard = time.time()
                who, sim = speakers.who(audio)
                if who != "erlan":
                    print(f"(habló {who}, no Erlan: parecido {sim:.2f}; no respondo)")
                    continue
                heard = ear.transcribe(audio)
                if not heard:
                    print(f"(oí {len(audio) / 16000:.1f} s de sonido, pero sin palabras claras)")
                    continue
                print(f"Tú: {heard}   (transcrito en {time.time() - t_heard:.1f} s, voz {sim:.2f})")
            if not heard:
                continue
            # Right after an answer the conversation is open: a follow-up needs no "Lunes".
            in_conversation = time.time() - last_answer < FOLLOW_UP_S
            if not args.no_wake and not args.text and not in_conversation and not WAKE.search(heard):
                print("(no dijiste «Lunes», así que no respondo)")
                continue
            if STOP.search(heard):
                last_answer = time.time()
                continue  # nothing running to stop
            request = WAKE.sub("", heard, count=1).strip() or heard
            if BYE.search(request) and len(request.split()) <= 4:
                say_soon("Hasta luego, Erlan.")
                voice.wait()
                break
            t0 = time.time()
            listener = StopListener(ear, speakers, interrupt) if ear else None
            if listener:
                listener.start()
            try:
                answer = hume.handle(request, on_tool, on_text=say_soon)
                if hume.stop.is_set():
                    voice.stop()
                    voice.say_soon("Listo, me detengo.")
                    print("Lunes: Listo, me detengo.")
                elif not answer:
                    say_soon("No tengo una respuesta para eso.")
                print(f"   (respuesta completa en {time.time() - t0:.1f} s)")
                while voice.speaking and not hume.stop.is_set():
                    time.sleep(0.1)
            finally:
                if listener:
                    listener.finish()
            last_answer = time.time()
    except KeyboardInterrupt:
        pass
    finally:
        machine.close()
        cam.cap.release()
        if kinect:
            kinect.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
