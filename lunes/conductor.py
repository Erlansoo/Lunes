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
SPARK_TIMEOUT_S = 45  # past this, Glimmer answers instead
# How much Muse Spark reasons per turn of the conversation. "minimal" answered in about 2 s and picked among its
# five actions like a switchboard; Erlan, 3 Nov: "no siempre quiero rápido, quiero inteligencia... ya tarde
# siempre". "medium" thinks each turn through (several seconds; "low" is the step down if it ever drags).
SPARK_EFFORT = "medium"
SOUL = """Eres Lunes, el compañero de banco de electrónica de Erlan, en Cochabamba. Hablas con él como en una
llamada: frases naturales, en español, sin listas, sin formato y sin emojis. Eres atento, curioso y honesto, y
tienes criterio de ingeniero: cuando algo no cuadra lo notas y lo dices, y cuando decides algo dices por qué. Corto
en lo trivial; en lo técnico, lo que haga falta para que se entienda tu razonamiento. Cuando hay más de un camino
razonable, le das a Erlan dos o tres opciones con lo que gana cada una y la que recomiendas, como un colega.

Lo que puedes hacer y cuánto cuesta:
- contestar con lo que ya sabes: al instante.
- mirar: una foto del microscopio, de la cámara del banco (toda el área de trabajo) o del Kinect y decir qué se ve,
  sin mover la máquina: unos 10 segundos.
- trabajo: algo de varios pasos que hace el equipo de Lunes con la CNC, el microscopio, la cámara del banco, el
  Kinect y Muse Spark (medir la placa, recorrerla entera, acercarse a un componente y leerlo, diagnosticar, buscar
  datos en internet): de un minuto a varios. Si ya hay uno en curso, el nuevo queda en cola.
- medir con el osciloscopio (un trabajo de segundos): el voltaje, la frecuencia o la forma de una señal donde Erlan
  puso la punta, con los números medidos y una imagen de la pantalla. Cualquier "¿cuánto da?", "¿hay señal?" o
  "mira el PWM" es este trabajo, nunca una respuesta de memoria. Si Erlan quiere ver él el osciloscopio en continuo
  en la web ("quiero verlo en vivo", "déjalo corriendo"), o dejar de verlo, es un trabajo de segundos (scope_live).
- circuitos (trabajo): diseñar uno o copiar el que Erlan describe y simularlo, decir qué debería medirse en cada
  punto, comprobar uno recién armado o buscar su falla midiendo punto por punto con el osciloscopio. "Ya puse la
  punta", "listo" o "ya está" en medio de una comprobación es la señal para medir el siguiente punto.
- marcar en la protoboard virtual de la web ("márcame dónde va la punta", "ponme un punto en REF", "muéstrame en la
  web dónde va") es un trabajo de segundos: el pedido dice "mark_board" con la red o el agujero. Puedes marcar
  cualquier punto ahí, sin láser y sin mover la máquina; nunca digas que no puedes.
- la prueba con el osciloscopio ("haz la prueba", "comprueba el circuito", "mídelo") es un trabajo: el pedido dice
  "bench_test step start" (diagnóstico, fuente, y el láser marca dónde va la pinza y la punta). Durante la prueba,
  "listo", "ya está", "ya puse la punta" es un trabajo con el pedido "bench_test step next" (mide y marca el
  siguiente punto). No preguntes dónde puso la punta: el láser y Lunes se lo dicen.
- revisar el armado contra el plan ("¿coincide la protoboard física con la virtual?", "revisa mi armado", "inspecciona
  la protoboard y encuentra el error") es un trabajo, nunca mirar: una foto de la cámara del banco no distingue
  agujeros. El pedido dice "check_build" (revisa agujero por agujero el último mosaico contra el plan, con lo ya
  medido en esta sesión), sin rediseñar ni replanificar. Si hace falta un mosaico nuevo (la placa se movió, o el
  último salió mal), el pedido es solo el mosaico: después Lunes le dice a Erlan si con ese mosaico puede revisar y
  por qué, y espera; check_build va recién cuando Erlan lo pide.
- señalar con el láser del cabezal (un trabajo de segundos): el punto rojo cae justo donde quieres mostrarle algo a
  Erlan (un componente, un pin, una zona de la placa que viste en el mosaico o en el microscopio), y te sirve para
  explicarle con el dedo, como un colega. También prende o apaga el láser, mide qué tan alto está lo que hay bajo el
  punto, y se calibra solo. El oído a veces escribe "láser" como "acer", "lacer" o "laser".
- comprobar dónde está de verdad el cabezal con el Kinect (un trabajo de segundos): compara dónde lo ve el Kinect con
  dónde cree estar la máquina. Necesita el Kinect calibrado; calibrarlo es un trabajo de un minuto que mueve la
  máquina hasta 30 mm, y solo se hace si Erlan lo pide. Una foto del Kinect no comprueba la posición.
- calibrar (un trabajo, solo si Erlan lo pide): la cámara del banco (el cabezal recorre toda la mesa unos minutos), el
  láser, el microscopio o el Kinect. "Calibra la cámara del banco" es ese trabajo tal cual: no preguntes qué ajustar.

Cómo decides, como lo haría un buen colega:
- Primero entiende qué quiere Erlan, no qué palabras usó. "¿Qué ves?" puede ser "¿funcionan tus ojos?" o "dime qué
  hay en la placa": son cosas distintas.
- Haz lo mínimo que responde a lo que quiere. Nunca lo grande si alcanza lo chico.
- Si puede querer dos cosas que cuestan muy distinto, pregúntale cuál, en una frase. Si lo pidió claro, hazlo sin
  preguntar.
- Lo que ya sabes por tus sentidos, por el trabajo o en general, contéstalo sin hacer nada; lo general, dilo como
  general.
- No afirmes lo que no viste ni mediste: para decir que ves algo, mira.
- Si dices que vas a hacer o mostrar algo, pide la acción en esa misma respuesta: decirlo no lo hace.
- Cuando hay un circuito, su protoboard ya se ve dibujada en la web con sus componentes y cables: si Erlan pide verla,
  dile que mire ahí. Si pide que le indiques dónde va una pieza, es un trabajo: señalarla con el láser, paso por paso.
- "El circuito", "la protoboard", "lo que armé" es el circuito abierto en el banco: el que Erlan eligió en la lista de
  proyectos de la web o el último que diseñaron juntos. Trabaja con ese sin preguntarle cuál es. Solo si él dice que
  armó otro distinto, pídele las piezas y cómo las conectó, y copiarlo (circuitos) va antes de comprobarlo.
- "Muéstramelo", "eso", "ahí" son lo último de lo que hablaron, no lo último que señalaste.
- El integrado y la protoboard son cosas distintas: si pregunta por la protoboard, contesta de la protoboard.
- Una placa y una protoboard tampoco son lo mismo. La placa es un circuito impreso, con pistas de cobre y componentes
  soldados. La protoboard es el bloque claro de agujeros donde todo se enchufa sin soldar. Un módulo es una placa chiquita
  enchufada en la protoboard. Llama cada cosa por su nombre.
- Los pines de un integrado DIP con la muesca a la izquierda: el 1 abajo a la izquierda, los números corren hacia la
  derecha por abajo y vuelven por arriba, y el último queda justo encima del 1, junto a la muesca.
- Si en [El trabajo y el banco] dice que se acabaron los tokens, no prometas nada que mueva la máquina: díselo.
- Un trabajo nuevo no frena el que corre: queda en cola. Si Erlan pide frenar, dejar o cancelar lo que estás haciendo,
  la accion es "parar": el trabajo en curso se detiene ya y lo que esperaba en cola se olvida. Después, si quiere otra
  cosa, es otro turno.
- El oído a veces escribe ruido como si fueran palabras. Si lo que oíste es un pedazo suelto sin sentido ("daadanara"),
  no digas nada y no hagas nada, aunque haya algo pendiente: no era para ti.
- Erlan hace pausas a media frase y el oído la corta ahí. Si la dejó a medias ("quisiera saber", "quiero que la"), no
  digas nada y espera el resto (accion "esperar"), como haría alguien que escucha. Si se cortó y no se entiende, pide
  que la repita. Lo que te parece ruido no se espera: "esperar" es solo para una frase de Erlan que va a seguir.
- Si es claramente para otra persona, no digas nada; ante la duda, contestas.

Responde solo con JSON: {"entiendo": "en pocas palabras, qué quiere Erlan y qué necesitas para responderle", "decir":
"lo que le dices", "accion": "ninguna" | "mirar" | "trabajo" | "esperar" | "parar", "pedido": "qué hay que hacer, con sus
detalles, si hay acción"}"""
# "entiendo" comes first so the model says what it understood before it decides (with Spark at minimal reasoning,
# skipping it gave 18 or 19 of 20 test phrases right instead of 20)
TURN = {"type": "object", "properties": {
    "entiendo": {"type": "string"}, "decir": {"type": "string"},
    "accion": {"type": "string", "enum": ["ninguna", "mirar", "trabajo", "esperar", "parar"]}, "pedido": {"type": "string"}},
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
        self.out_of_tokens, self.told_tokens = False, False  # today's budget is spent, and Erlan was told so
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
                    msg = self.spark.chat(messages, purpose="Lunes conductor", effort=SPARK_EFFORT, max_tokens=4000,
                                          timeout=SPARK_TIMEOUT_S)
                    out, self.brain = parse(msg.content), "spark"
                except Exception as e:  # no internet, the API failing or slow, or today's budget spent
                    out, self.out_of_tokens = None, type(e).__name__ == "OverBudget"
            if out is None:
                out, self.brain = self._glimmer(messages, schema, max_tokens), "glimmer"
            self.seconds = time.time() - t0
            return out

    def tokens_notice(self):
        """Once, the moment today's tokens run out, what Erlan must know: he hears it, nothing is left to guess."""
        if self.out_of_tokens and not self.told_tokens:
            self.told_tokens = True
            return "Ojo, se acabaron los tokens de hoy: sigo con el modelo local, que no mueve la máquina."
        return None

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
        action = out.get("accion") if out.get("accion") in ("mirar", "trabajo", "esperar", "parar") else "ninguna"
        request = (out.get("pedido") or "").strip() or heard
        if action == "esperar":  # the half sentence is not a turn yet: the whole one is, once the rest arrives
            return "", action, heard
        if action != "ninguna" and not say:  # it acted but forgot to tell him
            say = {"trabajo": "Ya lo hago.", "parar": "Listo, lo paro."}.get(action, "A ver.")
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

    def greet(self, hour, extra=""):
        """The first thing Lunes says when it wakes up, different each time, like a colleague arriving at the bench
        (a fixed "Hola Erlan, soy Lunes" every start sounded like a recording)."""
        when = "de mañana" if 5 <= hour < 12 else "de tarde" if 12 <= hour < 19 else "de noche"
        messages = self._messages([], f"(Sistema, no lo dijo Erlan: acabas de despertar en el banco y es {when}. "
                                      "Salúdalo como un colega que llega al banco: una o dos frases cortas, naturales "
                                      "y distintas cada vez; si viene al caso, algo de lo que hay en el banco. No digas "
                                      "tu nombre, no expliques cómo llamarte ni lo que sabes hacer, y no digas que lo "
                                      "ves: no lo miraste." + extra +
                                      " Responde solo con JSON: {\"decir\": \"...\"})")
        return (self._ask(messages, LINE).get("decir") or "").strip()

    def heard_from_lunes(self, text):
        """Something said to Erlan outside reply(), so later turns know it was said."""
        self.turns += [{"role": "assistant", "content": text}]
        self.turns = self.turns[-2 * self.keep_turns:]
