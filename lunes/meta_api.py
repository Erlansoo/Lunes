"""Meta Model API client: Muse Spark with images, and a log of every call for provenance.

The API speaks the OpenAI chat-completions protocol at https://api.meta.ai/v1. The key comes from MODEL_API_KEY in
the environment or in the repository's .env file (which git ignores). Every call appends one line to
runs/api_calls.jsonl with the model, tokens and time, so judges can see which Meta model did what.
"""
import base64
import json
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "https://api.meta.ai/v1"
DEFAULT_MODEL = "muse-spark-1.3"
LOG = ROOT / "runs" / "api_calls.jsonl"


def api_key():
    key = os.environ.get("MODEL_API_KEY")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "MODEL_API_KEY":
                key = value.strip().strip('"').strip("'")
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

    def ask(self, question, images=(), system=None, purpose=""):
        """One question with optional images; returns the answer text. Images go in the user message."""
        content = [{"type": "text", "text": question}] + [image_part(im) for im in images]
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        t0 = time.time()
        r = self.client.chat.completions.create(model=self.model, messages=messages)
        took = time.time() - t0
        usage = r.usage.model_dump() if r.usage else {}
        LOG.parent.mkdir(exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"at": datetime.now().isoformat(timespec="seconds"), "model": r.model or self.model,
                                "where": BASE_URL, "purpose": purpose, "images": len(images),
                                "seconds": round(took, 2), "usage": usage}) + "\n")
        return r.choices[0].message.content
