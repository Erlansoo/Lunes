"""The conductor: Lunes' side of the call with Erlan.

It decides by judgment, not by a list of rules: each turn it gets who Lunes is, what its senses and hands are doing
right now, what each action costs, and how a good colleague decides; and it answers in a fixed form: what to say,
and whether to do nothing, take a quick look, or hand a job to the worker (Hume with the bench tools). The brain is
Muse Spark at minimal reasoning through the Meta Model API: as fast as local Muse Glimmer (1.9 s against 1.8 s on
average) and with better judgment; Glimmer, with the same prompt, claimed "te veo" without looking. Glimmer on the
PC takes over when there is no internet, and in private mode. The conductor never does the work and never invents
results: it only knows what it is given.
"""
import json
import re
import threading
import time
import urllib.request

OLLAMA = "http://127.0.0.1:11434/api/chat"
MODEL = "muse-glimmer:30b"
NUM_CTX = 8192  # the same as Hume's: Ollama reloads Glimmer (22 s) whenever two callers ask for different sizes
SPARK_TIMEOUT_S = 8  # past this, Glimmer answers instead
SOUL = """Eres Lunes, el compañero de banco de electrónica de Erlan, en Cochabamba. Hablas con él como en una
llamada: frases cortas y naturales, en español, sin listas, sin formato y sin emojis. Eres atento, curioso y
honesto.

Lo que puedes hacer y cuánto cuesta:
- contestar con lo que ya sabes: al instante.
- mirar: una foto del microscopio, de la cámara del banco (toda el área de trabajo) o del Kinect y decir qué se ve,
  sin mover la máquina: unos 10 segundos.
- trabajo: algo de varios pasos que hace el equipo de Lunes con la CNC, el microscopio, la cámara del banco, el
  Kinect y Muse Spark (medir la placa, recorrerla entera, acercarse a un componente y leerlo, diagnosticar, buscar
  datos en internet): de un minuto a varios. Si ya hay uno en curso, el nuevo queda en cola.
- comprobar dónde está de verdad el cabezal con el Kinect (un trabajo de segundos): compara dónde lo ve el Kinect con
  dónde cree estar la máquina. Necesita el Kinect calibrado; calibrarlo es un trabajo de un minuto que mueve la
  máquina hasta 30 mm, y solo se hace si Erlan lo pide. Una foto del Kinect no comprueba la posición.

Cómo decides, como lo haría un buen colega:
- Primero entiende qué quiere Erlan, no qué palabras usó. "¿Qué ves?" puede ser "¿funcionan tus ojos?" o "dime qué
  hay en la placa": son cosas distintas.
- Haz lo mínimo que responde a lo que quiere. Nunca lo grande si alcanza lo chico.
- Si puede querer dos cosas que cuestan muy distinto, pregúntale cuál, en una frase. Si lo pidió claro, hazlo sin
  preguntar.
- Lo que ya sabes por tus sentidos, por el trabajo o en general, contéstalo sin hacer nada; lo general, dilo como
  general.
- No afirmes lo que no viste ni mediste: para decir que ves algo, mira.
- El oído a veces escribe ruido como si fueran palabras. Si lo que oíste es un pedazo suelto sin sentido ("daadanara"),
  no digas nada: no era para ti.
- Erlan hace pausas a media frase y el oído la corta ahí. Si la dejó a medias ("quisiera saber", "quiero que la"), no
  digas nada y espera el resto (accion "esperar"), como haría alguien que escucha. Si se cortó y no se entiende, pide
  que la repita. Lo que te parece ruido no se espera: "esperar" es solo para una frase de Erlan que va a seguir.
- Si es claramente para otra persona, no digas nada; ante la duda, contestas.

Responde solo con JSON: {"entiendo": "en pocas palabras, qué quiere Erlan y qué necesitas para responderle", "decir":
"lo que le dices", "accion": "ninguna" | "mirar" | "trabajo" | "esperar", "pedido": "qué hay que hacer, con sus
detalles, si hay acción"}"""
# "entiendo" comes first so the model says what it understood before it decides (with Spark at minimal reasoning,
# skipping it gave 18 or 19 of 20 test phrases right instead of 20)
TURN = {"type": "object", "properties": {
    "entiendo": {"type": "string"}, "decir": {"type": "string"},
    "accion": {"type": "string", "enum": ["ninguna", "mirar", "trabajo", "esperar"]}, "pedido": {"type": "string"}},
    "required": ["entiendo", "decir", "accion", "pedido"]}
LINE = {"type": "object", "properties": {"decir": {"type": "string"}}, "required": ["decir"]}


def parse(text):
    """The JSON object in a reply; when it is cut off, what it managed to say."""
    text = re.sub(r"<\|[a-z_]+\|>", "", text or "").strip()  # end-of-turn marks that slip through
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        m = re.search(r'"decir"\s*:\s*"([^"]*)', text)
        return {"decir": m.group(1) if m else ""}


class Conductor:
    def __init__(self, spark=None, model=MODEL, about="", keep_turns=8):
        """spark: a lunes.meta_api.MetaModels, or None to use only Glimmer on the PC (private mode)."""
        self.spark, self.model, self.about, self.keep_turns = spark, model, about, keep_turns
        self.context = lambda: ("", "")  # set by the caller: (its senses now, the work and the bench)
        self.turns = []  # the last exchanges, as chat messages
        self.seconds, self.brain, self.understood = 0.0, "", ""  # of the last call
        self.lock = threading.Lock()  # the call, the narrator and the worker all ask it

    def _messages(self, history, last):
        senses, state = self.context()
        return ([{"role": "system", "content": f"{SOUL}\n\n{self.about}"}] + history +
                [{"role": "user", "content": f"[Tus sentidos ahora]\n{senses}\n[El trabajo y el banco]\n{state}\n\n"
                                             f"{last}"}])

    def _glimmer(self, messages, schema, max_tokens):
        body = {"model": self.model, "messages": messages, "stream": False, "think": False, "format": schema,
                "keep_alive": -1, "options": {"num_ctx": NUM_CTX, "temperature": 0.2, "num_predict": max_tokens}}
        req = urllib.request.Request(OLLAMA, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return parse(json.loads(r.read())["message"].get("content"))

    def _ask(self, messages, schema, max_tokens=300):
        with self.lock:
            t0 = time.time()
            out = None
            if self.spark is not None:
                try:
                    msg = self.spark.chat(messages, purpose="Lunes conductor", effort="minimal", max_tokens=1200,
                                          timeout=SPARK_TIMEOUT_S)
                    out, self.brain = parse(msg.content), "spark"
                except Exception:  # no internet, the API failing or slow, or today's budget spent
                    out = None
            if out is None:
                out, self.brain = self._glimmer(messages, schema, max_tokens), "glimmer"
            self.seconds = time.time() - t0
            return out

    def warm(self):
        """Loads Glimmer on the GPU and keeps it there (the first load takes most of a minute), for when Spark
        cannot answer."""
        self._glimmer([{"role": "user", "content": "Hola"}], LINE, 10)

    def reply(self, heard, called=False):
        """(what to say, "ninguna" | "mirar" | "trabajo" | "esperar", what to do) for what Erlan said; called: he said
        "Lunes". "esperar": he left the sentence half-way; say nothing and let the rest join it."""
        # "te llamó por tu nombre" made it call Erlan "Lunes" now and then; say plainly who talks to whom
        who = "Erlan te habla a ti" if called else "Erlan"
        out = self._ask(self._messages(self.turns, f"{who}: {heard}"), TURN)
        say = (out.get("decir") or "").strip()
        if re.fullmatch(r"\(.*\)|\[.*\]|nada\.?", say, re.IGNORECASE):  # a stage note like "(nada)" is silence
            say = ""
        self.understood = (out.get("entiendo") or "").strip()  # for the session log: what it understood
        action = out.get("accion") if out.get("accion") in ("mirar", "trabajo", "esperar") else "ninguna"
        request = (out.get("pedido") or "").strip() or heard
        if action == "esperar":  # the half sentence is not a turn yet: the whole one is, once the rest arrives
            return "", action, heard
        if action != "ninguna" and not say:  # it acted but forgot to tell him
            say = "Ya lo hago." if action == "trabajo" else "A ver."
        # A silent turn leaves no answer in the history: with "(nada)" there, the model began to say "(nada)" aloud.
        self.turns += [{"role": "user", "content": f"Erlan: {heard}"}] + (
            [{"role": "assistant", "content": say}] if say else [])
        self.turns = self.turns[-2 * self.keep_turns:]
        return say, action, request

    def tell(self, what, history=True, sentences=3):
        """What to say about the work, at most a few sentences and no questions: an update after a long silence, or a
        long result made short (history=False, so the talk around it does not leak into the result)."""
        messages = self._messages(self.turns[-4:] if history else [], f"(Sistema, no lo dijo Erlan: {what} "
                                                                       f"Responde solo con JSON: {{\"decir\": \"...\"}})")
        say = (self._ask(messages, LINE).get("decir") or "").strip()
        kept = [s for s in re.split(r"(?<=[.!?])\s+", say) if s and not s.endswith("?")][:sentences]
        return " ".join(kept)

    def heard_from_lunes(self, text):
        """Something said to Erlan outside reply(), so later turns know it was said."""
        self.turns += [{"role": "assistant", "content": text}]
        self.turns = self.turns[-2 * self.keep_turns:]
