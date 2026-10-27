"""Hume's ear and voice, both local: speech to text with faster-whisper on the GPU, text to speech with Piper.

The ear listens on the Kinect's microphone array (Windows' default input here) and finds speech with the Silero
voice detector. The voice speaks from a queue on its own thread, so sentences can be handed over one by one while
the language model is still writing the rest, and stop() cuts it off mid-sentence when Erlan interrupts.
"""
import os
import re
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

ROOT = Path(__file__).resolve().parent.parent
PIPER_VOICE = ROOT / "models" / "piper" / "es_MX-ald-medium.onnx"  # a man's voice, the one Erlan prefers
RATE = 16000
BLOCK = 480  # 30 ms at 16 kHz
# Phrases Whisper makes up from noise (it learned them from the end of online videos); alone, they are not speech.
HALLUCINATION = re.compile(r"\W*(¡?suscr[ií]bete( al canal)?!?|gracias( por ver(lo)?)?\.?|subt[ií]tulos.*|"
                           r"amara\.org.*|m[uú]sica|\.\.\.)\W*", re.IGNORECASE)


def looping(text, times=3):
    """True when the text is one short phrase said over and over, which is Whisper looping on noise or distant
    voices ("no se está en el comienzo, no se está en el comienzo, ...")."""
    words = re.findall(r"\w+", text.lower())
    if len(words) >= 9 and len(set(words)) / len(words) < 0.45:  # a few words make up nearly everything said
        return True
    for n in range(2, 9):
        for start in range(0, max(0, len(words) - n * times) + 1):
            phrase = words[start:start + n]
            if len(phrase) == n and all(words[start + k * n:start + (k + 1) * n] == phrase for k in range(times)):
                return True
    return False


def _cuda_dlls():
    """On Windows the cuBLAS and cuDNN DLLs come in the nvidia-* wheels; make them findable for CTranslate2."""
    site = Path(sys.executable).resolve().parent.parent / "Lib" / "site-packages" / "nvidia"
    for bin_dir in site.glob("*/bin"):
        os.add_dll_directory(str(bin_dir))
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


class Voice:
    def __init__(self, model=PIPER_VOICE):
        from piper import PiperVoice
        self.voice = PiperVoice.load(str(model))
        self.lines = queue.Queue()
        self.generation = 0  # stop() moves it on; sentences queued before that are dropped
        threading.Thread(target=self._speaker, daemon=True).start()

    def synth(self, text):
        """(int16 audio, sample rate) of a sentence."""
        chunks = list(self.voice.synthesize(text))
        if not chunks:
            return np.zeros(0, np.int16), 22050
        return np.concatenate([c.audio_int16_array for c in chunks]), chunks[0].sample_rate

    def _speaker(self):
        while True:
            generation, text = self.lines.get()
            try:
                if generation == self.generation:
                    audio, rate = self.synth(text)
                    if generation == self.generation and len(audio):
                        sd.play(audio, rate)
                        sd.wait()
            finally:
                self.lines.task_done()

    @property
    def speaking(self):
        return self.lines.unfinished_tasks > 0

    def say_soon(self, text):
        """Queues a sentence and returns at once; sentences are spoken in order."""
        if text.strip():
            self.lines.put((self.generation, text))

    def stop(self):
        """Stops talking now and forgets what was still queued."""
        self.generation += 1
        sd.stop()

    def wait(self):
        """Blocks until everything queued has been spoken (so the ear does not hear Hume's own voice)."""
        self.lines.join()

    def say(self, text, wait=True):
        self.say_soon(text)
        if wait:
            self.wait()


class Ear:
    def __init__(self, model="large-v3-turbo", compute_type="float16"):
        """compute_type="int8_float16" halves the GPU memory, to share the card with the local language model."""
        _cuda_dlls()
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model, device="cuda", compute_type=compute_type)
        self.floor = None

    def calibrate(self, seconds=1.0):
        """Measures the room's noise level while nobody talks; call it before Hume speaks, not after."""
        audio = sd.rec(int(seconds * RATE), samplerate=RATE, channels=1, dtype="float32")
        sd.wait()
        levels = [float(np.sqrt(np.mean(audio[i:i + BLOCK, 0] ** 2))) for i in range(0, len(audio) - BLOCK, BLOCK)]
        self.floor = max(float(np.percentile(levels, 30)), 2e-4)
        return self.floor

    def record_utterance(self, max_s=15.0, silence_s=0.7, wait_s=10.0, kinect_sdk=False):
        """Records one utterance; returns float32 mono audio at 16 kHz, or None if nobody spoke within wait_s.

        Speech is found with the Silero voice detector that ships with faster-whisper, not by loudness: on this
        bench the voice reaching the Kinect is only about twice the fans' hum. Reads Windows' default input (the
        Kinect's microphone array here); kinect_sdk=True reads it through the bridge instead, which this Xbox 360
        Kinect refuses with E_FAIL."""
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        opts = VadOptions(threshold=0.35, min_speech_duration_ms=250, min_silence_duration_ms=300, speech_pad_ms=200)
        # The voice reaches the Kinect at a few thousandths of full scale; lift it so the detector sees normal speech.
        gain = float(np.clip(0.004 / self.floor, 1.0, 20.0)) if self.floor else 8.0
        if kinect_sdk:
            from lunes.kinect import KinectAudio
            source = KinectAudio()
            read = source.read
        else:
            source = sd.InputStream(samplerate=RATE, channels=1, dtype="float32", blocksize=BLOCK)
            source.start()
            read = lambda n: source.read(n)[0][:, 0]  # noqa: E731
        step = int(0.3 * RATE)  # run the detector every 0.3 s
        buf, t0 = np.zeros(0, np.float32), time.time()
        try:
            while True:
                buf = np.concatenate([buf, np.clip(read(step) * gain, -1.0, 1.0)])
                speech = get_speech_timestamps(buf, opts)
                if not speech:
                    if time.time() - t0 > wait_s:
                        return None
                    buf = buf[-RATE:]  # keep one second of lead-in while waiting
                    continue
                after_last = len(buf) - speech[-1]["end"]
                if after_last >= silence_s * RATE or len(buf) >= max_s * RATE:
                    return buf[max(0, speech[0]["start"] - int(0.2 * RATE)):]
        finally:
            source.close()

    def transcribe(self, audio, language="es"):
        peak = float(np.abs(audio).max()) if len(audio) else 0.0
        if peak > 1e-4:
            audio = audio * (0.5 / peak)  # Whisper is trained on normally recorded speech
        segments, _ = self.model.transcribe(audio, language=language, vad_filter=True, beam_size=5,
                                            condition_on_previous_text=False)
        # Whisper invents short phrases ("Gracias.") from noise; drop segments it is not confident are speech.
        kept = [s.text.strip() for s in segments if s.no_speech_prob < 0.6 and s.avg_logprob > -1.0]
        text = " ".join(kept).strip()
        if HALLUCINATION.fullmatch(text) or looping(text):
            return ""
        return text

    def listen(self, **kw):
        audio = self.record_utterance(**kw)
        return None if audio is None else self.transcribe(audio)
