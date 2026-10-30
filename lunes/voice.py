"""Lunes' ear and voice, with Meta models only.

The ear finds speech by its loudness against the room's measured noise (code, not a model) and hands each utterance
to a Meta speech-to-text model: Omnilingual ASR on the PC (local_transcriber), or Muse Voice Transcribe through the
Meta Model API. The voice is Meta's open MMS-TTS
Spanish model, run by sherpa-onnx on the PC. It speaks from a queue on its own thread; each sentence is synthesised
whole and played by Windows itself, so a busy PC does not chop the sound, and stop() cuts it off mid-sentence when
Erlan interrupts. The voice remembers what it said and when, so the ear can tell Lunes' own words coming back from
the speakers from Erlan's.
"""
import collections
import difflib
import os
import re
import queue
import sys
import tempfile
import threading
import time
import unicodedata
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

ROOT = Path(__file__).resolve().parent.parent
MMS_VOICE = ROOT / "models" / "mms" / "vits-mms-spa"  # Meta's open MMS-TTS Spanish voice
# Meta's open Omnilingual ASR, exported for sherpa-onnx (lowercase text, no punctuation). The 300M model in full
# precision runs on the GPU in 0.03 to 0.1 s for any phrase; the 1B int8 one hears names and part numbers better but
# needs 0.5 to 2 s on the CPU.
FAST_EAR = ROOT / "models" / "omnilingual" / "sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-2026-02-05"
LOCAL_EAR =ROOT / "models" / "omnilingual" / "sherpa-onnx-omnilingual-asr-1600-languages-1B-ctc-v2-int8-2026-02-05"
PACE = 1.1  # a little slower than the model's own pace, to be understood easily
PAUSE_S = 0.25  # between sentences
RATE = 16000
BLOCK = 480  # 30 ms at 16 kHz
START = 3.0  # speech starts when the level stays this many times above the room's noise...
START_BLOCKS = 5  # ...for this many of the last 8 blocks (150 ms)
STOP = 2.0  # and ends after silence_s below this many times the noise


def find_microphone(name):
    """Index of the input device whose name contains name (through Windows' MME, like the default device), or
    None for Windows' default input."""
    if not name:
        return None
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and name.lower() in d["name"].lower() and \
                sd.query_hostapis(d["hostapi"])["name"] == "MME":
            return i
    raise RuntimeError(f"no microphone named like {name!r}")


def words(text):
    return re.findall(r"\w+", text.lower())


def plain(word):
    """The word without accents ("cómo" -> "como")."""
    return unicodedata.normalize("NFD", word).encode("ascii", "ignore").decode()


def _cuda_dlls():
    """No CUDA toolkit on this PC: the CUDA and cuDNN DLLs come in the nvidia-* wheels; make them findable for ONNX
    Runtime's CUDA provider inside sherpa-onnx."""
    site = Path(sys.executable).resolve().parent.parent / "Lib" / "site-packages" / "nvidia"
    for bin_dir in site.glob("*/bin"):
        os.add_dll_directory(str(bin_dir))
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def local_transcriber(model_dir=LOCAL_EAR, threads=8, provider="cpu"):
    """Speech to text on the PC with Meta's Omnilingual ASR: a function from float32 16 kHz audio to text. Nothing
    leaves the PC. provider="cuda" runs it on the GPU, which only pays with a full-precision model: the int8 model's
    quantised layers stay on the CPU and it came out slower (4.4 s instead of 3 s for a 7 s phrase)."""
    import sherpa_onnx
    if provider == "cuda":
        _cuda_dlls()
    model_dir = Path(model_dir)
    recognizer = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
        model=str(next(model_dir.glob("*.onnx"))), tokens=str(model_dir / "tokens.txt"), num_threads=threads,
        provider=provider)

    def transcribe(audio):
        stream = recognizer.create_stream()
        stream.accept_waveform(RATE, np.asarray(audio, np.float32))
        recognizer.decode_stream(stream)
        return stream.result.text
    return transcribe


class Voice:
    def __init__(self, model=MMS_VOICE):
        import sherpa_onnx
        vits = sherpa_onnx.OfflineTtsVitsModelConfig(model=str(Path(model) / "model.onnx"),
                                                     tokens=str(Path(model) / "tokens.txt"))
        self.mms = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(vits=vits, num_threads=4)))
        self.lines = queue.Queue()
        self.generation = 0  # stop() moves it on; text queued before that is dropped
        self.said = collections.deque(maxlen=30)  # (start, end, sentence) of what was played
        self.ready = {}  # short phrases synthesised in advance, to be said at once ("a ver", "dale")
        self.wav = Path(tempfile.gettempdir()) / f"lunes_voice_{os.getpid()}.wav"
        threading.Thread(target=self._speaker, daemon=True).start()

    def prepare(self, *texts):
        """Synthesises phrases now so that saying them later starts without delay."""
        for text in texts:
            self.ready[text] = self.synth(text)

    def synth(self, text):
        """(int16 audio, sample rate) of a text; sentences are separated by a short pause."""
        if text in self.ready:
            return self.ready[text]
        from lunes.speakable import speakable
        parts, rate = [], 16000
        for sentence in re.split(r"(?<=[.!?])\s+", speakable(text)):
            sentence = re.sub(r"[^\w\sáéíóúüñÁÉÍÓÚÜÑ]", " ", sentence).strip()  # the voice has no punctuation
            if sentence:
                g = self.mms.generate(sentence, sid=0, speed=1.0 / PACE)
                rate = g.sample_rate
                parts += [np.clip(np.array(g.samples, np.float32), -1, 1), np.zeros(int(PAUSE_S * rate), np.float32)]
        audio = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        return (audio * 32767).astype(np.int16), rate

    def _play(self, audio, rate, generation):
        """Plays through Windows' own player (no Python callback that a busy interpreter could starve); returns
        early when stop() is called."""
        if sys.platform != "win32":
            sd.play(audio, rate)
            sd.wait()
            return
        import winsound
        with wave.open(str(self.wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(audio.tobytes())
        winsound.PlaySound(str(self.wav), winsound.SND_FILENAME | winsound.SND_ASYNC)
        end = time.time() + len(audio) / rate + 0.1
        while time.time() < end and generation == self.generation:
            time.sleep(0.05)
        if generation != self.generation:
            winsound.PlaySound(None, 0)

    def _speaker(self):
        """Says each queued text sentence by sentence: the next sentence is synthesised while the current one
        plays, so speech starts after the first sentence and still comes out whole (each one is a complete file)."""
        from concurrent.futures import ThreadPoolExecutor
        maker = ThreadPoolExecutor(1)
        while True:
            generation, text = self.lines.get()
            try:
                sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
                if generation != self.generation or not sentences:
                    continue
                coming = maker.submit(self.synth, sentences[0])
                for k in range(len(sentences)):
                    audio, rate = coming.result()
                    if k + 1 < len(sentences):
                        coming = maker.submit(self.synth, sentences[k + 1])
                    if generation != self.generation:
                        break
                    if len(audio):
                        start = time.time()
                        self.said.append([start, start + len(audio) / rate, sentences[k]])
                        self._play(audio, rate, generation)
                        self.said[-1][1] = time.time()
            finally:
                self.lines.task_done()

    def recently_said(self, start, end):
        """What Lunes said between two moments (with a margin for the room's echo), as one text."""
        return " ".join(s for a, b, s in list(self.said) if a < end + 0.5 and b > start - 1.5)

    def talking_during(self, start, end):
        """True when Lunes was speaking while someone else was (start is when the voice began, not the lead-in)."""
        return any(a < end and b > start - 0.2 for a, b, _ in list(self.said))

    def own_words(self, heard, start, end):
        """Splits what the ear heard between start and end into (Lunes' own words coming back from the speakers,
        the rest). The rest is what someone else said over Lunes' voice."""
        own = {plain(w) for w in words(self.recently_said(start, end))}
        heard_words = words(heard)
        if not own or not heard_words:
            return "", " ".join(heard_words)
        # Its own voice comes back garbled ("Hola Erlan, ¿cómo estás?" was heard as "ola jeslan kopusta"), so a
        # word counts as Lunes' when it is close to one it said, not only when it is the same.
        mine = {i for i, w in enumerate(heard_words)
                if any(difflib.SequenceMatcher(None, plain(w), o).ratio() >= 0.72 for o in own)}
        return (" ".join(w for i, w in enumerate(heard_words) if i in mine),
                " ".join(w for i, w in enumerate(heard_words) if i not in mine))

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
        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(None, 0)
        else:
            sd.stop()

    def wait(self):
        """Blocks until everything queued has been spoken."""
        self.lines.join()

    def say(self, text, wait=True):
        self.say_soon(text)
        if wait:
            self.wait()


class Ear:
    def __init__(self, transcriber, microphone=None):
        """transcriber: a Meta speech-to-text model, as a function from float32 16 kHz audio to text.
        microphone: part of the input device's name (e.g. "fifine"); Windows' default input when left out."""
        self.transcriber = transcriber
        self.microphone = microphone
        self.device = find_microphone(microphone)
        self.floor = None
        self.hearing = False  # someone is talking right now
        self.last_heard = 0.0  # when the last utterance ended

    def calibrate(self, seconds=1.0):
        """Measures the room's noise level while nobody talks."""
        audio = sd.rec(int(seconds * RATE), samplerate=RATE, channels=1, dtype="float32", device=self.device)
        sd.wait()
        levels = [float(np.sqrt(np.mean(audio[i:i + BLOCK, 0] ** 2))) for i in range(0, len(audio) - BLOCK, BLOCK)]
        self.floor = max(float(np.percentile(levels, 30)), 2e-4)
        return self.floor

    def record_utterance(self, max_s=15.0, silence_s=0.7, wait_s=10.0):
        """Records one utterance; returns (float32 mono audio at 16 kHz, time it started, time it ended), or None if
        nobody spoke within wait_s. Speech is whatever stays clearly above the room's noise; the noise level follows
        the room slowly while nobody talks."""
        if self.floor is None:
            self.calibrate()
        lead = int(0.3 * RATE / BLOCK)  # blocks kept from before speech started
        recent, blocks, loud = collections.deque(maxlen=lead), [], collections.deque(maxlen=8)
        quiet_since, voiced, t0, started = 0, 0, time.time(), None
        with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", blocksize=BLOCK, device=self.device) as s:
            while True:
                block = s.read(BLOCK)[0][:, 0].copy()
                level = float(np.sqrt(np.mean(block ** 2)))
                if started is None:
                    loud.append(level > START * self.floor)
                    recent.append(block)
                    if sum(loud) >= START_BLOCKS:
                        # the audio keeps 0.3 s of lead-in, but the utterance starts when the voice did: with the
                        # lead-in, Erlan answering right after Lunes looked like talking over it
                        started, blocks, voiced = time.time() - sum(loud) * BLOCK / RATE, list(recent), sum(loud)
                        self.hearing = True
                    else:
                        if not loud[-1]:
                            self.floor = 0.98 * self.floor + 0.02 * max(level, 2e-4)
                        if time.time() - t0 > wait_s:
                            return None
                    continue
                blocks.append(block)
                voiced += level > START * self.floor
                quiet_since = quiet_since + 1 if level < STOP * self.floor else 0
                if quiet_since * BLOCK >= silence_s * RATE or len(blocks) * BLOCK >= max_s * RATE:
                    self.hearing, self.last_heard = False, time.time()
                    if voiced * BLOCK < 0.3 * RATE:  # a knock or a cough, not a phrase: keep waiting
                        started, loud, quiet_since = None, collections.deque(maxlen=8), 0
                        continue
                    audio = np.concatenate(blocks)
                    return audio, started, time.time() - quiet_since * BLOCK / RATE

    def transcribe(self, audio):
        """The text, or "" when it is noise: the ear knows 1600 languages and cannot be told which one, so it writes
        some noises in Korean or Japanese ("아 예요"); Spanish only has Latin letters."""
        peak = float(np.abs(audio).max()) if len(audio) else 0.0
        if 1e-4 < peak < 0.3:  # quiet microphones (the Realtek input peaks at 0.05): bring the voice to a steady level
            audio = audio * min(20.0, 0.5 / peak)
        text = self.transcriber(audio).strip()
        letters = re.findall(r"[^\W\d_]", text)
        latin = re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]", text)
        return text if letters and len(latin) >= 0.8 * len(letters) else ""
