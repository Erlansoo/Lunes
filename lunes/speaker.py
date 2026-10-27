"""Who is talking: Erlan, someone else in the room, or Lunes' own voice coming back from the speakers.

A speaker-embedding network (WeSpeaker ResNet34, run by sherpa-onnx on the CPU) turns an utterance into a voiceprint
of 256 numbers; utterances whose voiceprint is close to the one enrolled for Erlan are his. Enrolling takes five
short phrases. The voiceprint never leaves this PC: it is kept in runs/, which is not published.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "speaker" / "wespeaker_en_voxceleb_resnet34_LM.onnx"
VOICEPRINT = ROOT / "runs" / "voiceprint_erlan.json"
RATE = 16000


def to_16k(audio, rate):
    audio = np.asarray(audio, np.float32)
    if rate == RATE:
        return audio
    t = np.arange(0, len(audio) / rate, 1 / RATE)
    return np.interp(t, np.arange(len(audio)) / rate, audio).astype(np.float32)


class Speakers:
    def __init__(self, model=MODEL):
        import sherpa_onnx
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model), num_threads=2))
        self.erlan = self.lunes = None
        self.threshold = 0.5
        if VOICEPRINT.exists():
            saved = json.loads(VOICEPRINT.read_text(encoding="utf-8"))
            self.erlan = np.array(saved["erlan"], np.float32)
            self.lunes = np.array(saved["lunes"], np.float32) if saved.get("lunes") else None
            self.threshold = saved["threshold"]

    @property
    def enrolled(self):
        return self.erlan is not None

    def embed(self, audio, rate=RATE):
        stream = self.extractor.create_stream()
        stream.accept_waveform(RATE, to_16k(audio, rate))
        stream.input_finished()
        e = np.array(self.extractor.compute(stream), np.float32)
        return e / (np.linalg.norm(e) + 1e-9)

    def who(self, audio):
        """("erlan" | "lunes" | "otra persona", similarity to Erlan). Anyone counts as Erlan before enrolling."""
        if not self.enrolled:
            return "erlan", 1.0
        e = self.embed(audio)
        sim = float(e @ self.erlan)
        if self.lunes is not None and float(e @ self.lunes) > sim:
            return "lunes", sim
        return ("erlan" if sim >= self.threshold else "otra persona"), sim

    def enroll(self, samples, lunes_samples=()):
        """samples: Erlan's utterances (16 kHz float32); lunes_samples: (audio, rate) of Lunes' own voice.
        Saves the voiceprint and an acceptance threshold set between Erlan's own spread and Lunes' voice."""
        es = np.array([self.embed(a) for a in samples])
        mean = es.mean(axis=0)
        mean /= np.linalg.norm(mean)
        own = [float(e @ ((mean * len(es) - e) / np.linalg.norm(mean * len(es) - e))) for e in es]  # leave-one-out
        lunes = None
        other = 0.3
        if len(lunes_samples):
            ls = np.array([self.embed(a, r) for a, r in lunes_samples])
            lunes = ls.mean(axis=0)
            lunes /= np.linalg.norm(lunes)
            other = float(lunes @ mean)
        # Enrolment phrases come from one spot and one moment, so they resemble each other more than Erlan's speech
        # does over a session (0.77 already happened against 0.91 to 0.96 here): stay well below his worst sample.
        # Lunes' own voice is told apart by its own voiceprint, not by this threshold.
        threshold = float(min(0.6, min(own) - 0.15))
        threshold = max(threshold, 0.4)
        self.erlan, self.lunes, self.threshold = mean, lunes, threshold
        VOICEPRINT.parent.mkdir(exist_ok=True)
        VOICEPRINT.write_text(json.dumps({"erlan": mean.tolist(), "lunes": None if lunes is None else lunes.tolist(),
                                          "threshold": threshold, "own_similarity": own,
                                          "lunes_similarity": other}), encoding="utf-8")
        return {"own_similarity": [round(s, 2) for s in own], "lunes_similarity": round(other, 2),
                "threshold": round(threshold, 2)}
