"""Meta Model API client: Muse Spark with images, and a log of every call for provenance.

The API speaks the OpenAI chat-completions protocol at https://api.meta.ai/v1. The key comes from MODEL_API_KEY in
the environment or in the repository's .env file (which git ignores). Every call appends one line to
runs/api_calls.jsonl with the model, tokens and time, so judges can see which Meta model did what.

Muse Spark reasons before it answers, and by default that reasoning is most of the bill (323 of 334 tokens for a
one-word answer). So every call says how hard to think (reasoning_effort: minimal, low, medium, high...), caps its
answer, and the day's tokens are capped too: past DAILY_TOKENS the client refuses to call.
"""
import base64
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import cv2
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "https://api.meta.ai/v1"
DEFAULT_MODEL = "muse-spark-1.3"
LOG = ROOT / "runs" / "api_calls.jsonl"
DAILY_TOKENS = 2_000_000  # all calls of one day together (at most about US$4); raise it knowingly. Erlan set it on
# 2026-10-28: 300 000 ran out in an afternoon of tests and left the conversation to slow, guessing Glimmer


class OverBudget(RuntimeError):
    pass


def tokens_today():
    if not LOG.exists():
        return 0
    today = datetime.now().strftime("%Y-%m-%d")
    total = 0
    for line in LOG.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("at", "").startswith(today):
            total += (row.get("usage") or {}).get("total_tokens") or 0
    return total


def api_key():
    key = os.environ.get("MODEL_API_KEY")
    env = ROOT / ".env"
    if not key and env.exists():
        lines = [line.strip() for line in env.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        for line in lines:
            name, _, value = line.partition("=")
            if name.strip() == "MODEL_API_KEY":
                key = value.strip().strip('"').strip("'")
        if not key and len(lines) == 1 and "=" not in lines[0]:  # a file with just the key in it
            key = lines[0]
    if not key:
        raise RuntimeError(f"no MODEL_API_KEY: put a line MODEL_API_KEY=... in {env}")
    return key


def image_part(image, quality=90):
    """An OpenAI-style image content part from a BGR array (OpenCV) or an image file path."""
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
    ok, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("could not encode the image")
    url = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()
    return {"type": "image_url", "image_url": {"url": url}}


class MetaModels:
    def __init__(self, model=DEFAULT_MODEL):
        self.client = OpenAI(base_url=BASE_URL, api_key=api_key())
        self.model = model

    def ask(self, question, images=(), system=None, purpose="", effort="low", max_tokens=1500):
        """One question with optional images; returns the answer text. Images go in the user message.
        effort: how much Muse Spark reasons first (minimal, low, medium, high); max_tokens caps reasoning plus answer."""
        used = tokens_today()
        if used >= DAILY_TOKENS:
            raise OverBudget(f"today's Muse Spark budget is spent ({used} of {DAILY_TOKENS} tokens)")
        content = [{"type": "text", "text": question}] + [image_part(im) for im in images]
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        t0 = time.time()
        r = self.client.chat.completions.create(model=self.model, messages=messages, reasoning_effort=effort,
                                                max_completion_tokens=max_tokens)
        took = time.time() - t0
        usage = r.usage.model_dump() if r.usage else {}
        LOG.parent.mkdir(exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"at": datetime.now().isoformat(timespec="seconds"), "model": r.model or self.model,
                                "where": BASE_URL, "purpose": purpose, "images": len(images), "effort": effort,
                                "seconds": round(took, 2), "usage": usage}) + "\n")
        return r.choices[0].message.content

    def transcribe(self, audio, rate=16000, languages=("Spanish",), keywords=(), purpose="Lunes ear"):
        """Speech to text with Muse Voice Transcribe: float32 mono audio in, the transcript out. keywords bias
        its vocabulary (names like "Lunes" that it would otherwise hear as "luna")."""
        import io
        import urllib.request
        import uuid
        import wave

        import numpy as np
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
        request = {"model": "muse-voice-transcribe-1.0", "audioEncoding": "WAV", "mode": "PUSH_TO_TALK",
                   "languageBias": list(languages)}
        if keywords:
            request["keywords"] = list(keywords)
        # multipart/form-data by hand: a JSON "request" part and the WAV "audio" part
        boundary = uuid.uuid4().hex
        data = (f'--{boundary}\r\nContent-Disposition: form-data; name="request"\r\n'
                f'Content-Type: application/json\r\n\r\n{json.dumps(request)}\r\n'
                f'--{boundary}\r\nContent-Disposition: form-data; name="audio"; filename="speech.wav"\r\n'
                f'Content-Type: audio/wav\r\n\r\n').encode() + buf.getvalue() + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(BASE_URL + "/asr/transcribe", data, {
            "Authorization": f"Bearer {api_key()}", "Accept": "application/json",
            "Content-Type": f"multipart/form-data; boundary={boundary}"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.loads(r.read())
        LOG.parent.mkdir(exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"at": datetime.now().isoformat(timespec="seconds"), "model": "muse-voice-transcribe-1.0",
                                "where": BASE_URL + "/asr/transcribe", "purpose": purpose,
                                "seconds": round(time.time() - t0, 2),
                                "audio_ms": body.get("audioDurationMs")}) + "\n")
        text = (body.get("transcript") or "").strip()
        # Now and then an early guess comes glued in front of the final words ("Olá, lunes.Hola, Lunes."): keep
        # what follows the last place where a new sentence starts without a space.
        return re.split(r"(?<=[.?!a-záéíóúñ])(?=[¿¡]?[A-ZÁÉÍÓÚÑ])", text)[-1].strip()

    def chat(self, messages, tools=None, purpose="", effort="low", max_tokens=4000, timeout=120):
        """One agent turn in the OpenAI chat format (tool calls included); returns the response message. timeout is
        in seconds (the conversation gives up early and answers locally)."""
        used = tokens_today()
        if used >= DAILY_TOKENS:
            raise OverBudget(f"today's Muse Spark budget is spent ({used} of {DAILY_TOKENS} tokens)")
        t0 = time.time()
        kw = {"tools": tools} if tools else {}
        client = self.client if timeout >= 30 else self.client.with_options(max_retries=0)  # no 3 x 8 s of waiting
        r = client.chat.completions.create(model=self.model, messages=messages, reasoning_effort=effort,
                                           max_completion_tokens=max_tokens, timeout=timeout, **kw)
        usage = r.usage.model_dump() if r.usage else {}
        images = sum(1 for m in messages if isinstance(m.get("content"), list)
                     for part in m["content"] if part.get("type") == "image_url")
        LOG.parent.mkdir(exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"at": datetime.now().isoformat(timespec="seconds"), "model": r.model or self.model,
                                "where": BASE_URL, "purpose": purpose, "images": images, "effort": effort,
                                "seconds": round(time.time() - t0, 2), "usage": usage}) + "\n")
        return r.choices[0].message

    def segment(self, image, concept, purpose="find components", max_side=2048):
        """SAM 3.1 finds every instance of a concept (a short English noun phrase, one concept per call) in a BGR
        image. Returns [(x1, y1, x2, y2, confidence)] in the image's own pixels. It costs US$2.50 per 1000 images.
        On microscope mosaics it finds chips, capacitors and heat sinks well, and connectors or fuses not at all."""
        k = min(1.0, max_side / max(image.shape[:2]))
        small = cv2.resize(image, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else image
        ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 90])
        url = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()
        t0 = time.time()
        r = self.client.responses.create(
            model="sam-3.1", metadata={"include_confidence": "true", "mask_encoding": "one_bit"}, timeout=60,
            input=[{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": concept}, {"type": "input_image", "image_url": url}]}])
        found = [(int(x1) / k, int(y1) / k, int(x2) / k, int(y2) / k, float(c) if c else 1.0)
                 for x1, y1, x2, y2, _, _, c in re.findall(
                     r"<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)(?:;c=([\d.]+))?\|>",
                     r.output_text or "")]
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"at": datetime.now().isoformat(timespec="seconds"), "model": "sam-3.1",
                                "where": BASE_URL + "/responses", "purpose": purpose, "images": 1, "concept": concept,
                                "found": len(found), "seconds": round(time.time() - t0, 2), "usage": {}}) + "\n")
        return found

    def segment_video(self, path, concept, purpose="track in video"):
        """SAM 3.1 follows every instance of a concept through an MP4 (uploaded with the Files API first). Returns
        {frame: [(object id, x1, y1, x2, y2, confidence)]} in the video's pixels. Video costs US$0.20 per 1000 frames,
        at least 50 frames a request."""
        t0 = time.time()
        with open(path, "rb") as f:
            uploaded = self.client.files.create(purpose="user_data", file=f)
        r = self.client.responses.create(
            model="sam-3.1", metadata={"include_confidence": "true", "mask_encoding": "one_bit"}, timeout=600,
            input=[{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": concept}, {"type": "input_video", "file_id": uploaded.id}]}])
        frames = {}
        for line in (r.output_text or "").splitlines():
            marker = re.match(r"<(\d+)f>", line)
            if not marker:
                continue
            frames[int(marker.group(1))] = [
                (int(i), int(x1), int(y1), int(x2), int(y2), float(c) if c else 1.0)
                for i, x1, y1, x2, y2, c in re.findall(
                    r"(?:f>|,)(\d+)<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=\d+;h=\d+(?:;c=([\d.]+))?\|>", line)]
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"at": datetime.now().isoformat(timespec="seconds"), "model": "sam-3.1",
                                "where": BASE_URL + "/responses", "purpose": purpose, "video_frames": len(frames),
                                "concept": concept, "seconds": round(time.time() - t0, 2), "usage": {}}) + "\n")
        return frames

    def search(self, question, effort="low", purpose="web search", max_tokens=6000):
        """Muse Spark answers with web search (the Responses API's web_search tool; it costs US$2.50 per 1000
        searches on top of the tokens). Returns (answer, list of source URLs)."""
        used = tokens_today()
        if used >= DAILY_TOKENS:
            raise OverBudget(f"today's Muse Spark budget is spent ({used} of {DAILY_TOKENS} tokens)")
        t0 = time.time()
        r = self.client.responses.create(
            model=self.model, reasoning={"effort": effort}, max_output_tokens=max_tokens, tools=[{"type": "web_search"}],
            input=[{"role": "system", "content": "Responde en español, corto y concreto, con lo que encuentres en "
                                                  "internet; di de qué fuente sale cada dato y si no encontraste algo."},
                   {"role": "user", "content": question}], timeout=120)
        sources, searches = [], 0
        for item in r.output or []:
            searches += item.type == "web_search_call"
            for part in getattr(item, "content", None) or []:
                for a in getattr(part, "annotations", None) or []:
                    if getattr(a, "url", None) and a.url not in sources:
                        sources.append(a.url)
        usage = r.usage.model_dump() if r.usage else {}
        usage["total_tokens"] = usage.get("total_tokens") or (usage.get("input_tokens", 0) + usage.get("output_tokens", 0))
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"at": datetime.now().isoformat(timespec="seconds"), "model": self.model,
                                "where": BASE_URL + "/responses", "purpose": purpose, "effort": effort,
                                "web_searches": searches, "seconds": round(time.time() - t0, 2),
                                "usage": usage}) + "\n")
        return (r.output_text or "").strip(), sources
