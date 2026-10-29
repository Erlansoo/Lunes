"""Lunes' local brain and its memory, Hume.

Lunes' brain is Muse Spark through Meta's API when Erlan allows the bench pictures to go out, and Muse Glimmer
(Meta's open weights) on the bench PC through Ollama otherwise: for private boards, without internet, or when the
day's Spark budget is spent (then Glimmer takes the turn over). Lunes plans with tools that move the CNC and look
through the microscope, the bench camera (the whole work area) and the Kinect (depth), and answers in short spoken
Spanish.

Its memory is called Hume, after David Hume: Lunes only believes what it observed. Every fact it remembers keeps the
observation behind it, and it says so when it is only guessing.

Lunes works as an agent with whole skills rather than single steps: measure_board walks the board's outline,
survey_board photographs exactly that area, inspect_component photographs one part entirely and close up to read it.

Safety lives in the tools, not in the prompt: moves stay inside the bench's allowed area (the travel Erlan measured,
in config/machine.json, minus a margin), X/Y moves lift Z first, and Z never goes below the board's focus height (or
below the bench's Z origin before the first autofocus).
"""
import base64
import json
import math
import re
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from microscope import autofocus, calibrate, sharpness  # noqa: E402
from survey import (Interrupted, board_corner_in_picture, closeup_for_reading, component_box,  # noqa: E402
                    distance_in_picture, find_perimeter, geometry, grid_for, hole_in_picture, hop, locate_in_mosaic,
                    mosaic_point_to_mm, pixels_to_mm, survey)

OLLAMA = "http://127.0.0.1:11434/api/chat"
MODEL = "muse-glimmer:30b"
THINK = "medium"  # Erlan prefers a slower answer that is right over a fast guess
LOG = ROOT / "runs" / "hume_calls.jsonl"
MEMORY = ROOT / "runs" / "hume_memory.jsonl"
SURVEYS = ROOT / "runs" / "hume_surveys.json"  # the board mosaics taken so far, kept across restarts
CONFIG = ROOT / "config" / "machine.json"
MARGIN_MM = {"X": 10.0, "Y": 10.0, "Z": 3.0}  # kept clear of the travel ends Erlan measured


def _workspace():
    """Allowed area in bench mm: the travel Erlan measured (config/machine.json) minus a margin."""
    travel = json.loads(CONFIG.read_text(encoding="utf-8")).get("workspace_mm")
    if not travel:
        return {"X": (-80.0, 80.0), "Y": (-80.0, 80.0), "Z": (-3.0, 12.0)}
    return {a: (travel[a][0] + MARGIN_MM[a], travel[a][1] - MARGIN_MM[a]) for a in "XYZ"}


WORKSPACE = _workspace()
CIRCUIT = Path(__file__).resolve().parent.parent / "runs" / "circuit.json"  # the circuit on the bench and its simulation
GUIDE = Path(__file__).resolve().parent / "knowledge" / "circuits.md"  # Lunes' circuit guide, read by circuit_guide
LIFT_MM = 3.0
MAX_ROUNDS = 10
MAX_TILES = 80  # one board survey: about 9 x 9 pictures, some 75 x 95 mm
MAX_PART_TILES = 12
BOARD_MARGIN_MM = 2.0
PART_MARGIN_MM = 1.5
PART_RUNS_OFF_MM = 8.0  # how much further to look where a part runs off the edge of the mosaic
SENTENCE_END = re.compile(r"[.!?¡¿]\s|[.!?]$")
PROMISE = re.compile(r"\b(voy a|vamos a|d[ée]jame|en un (momento|segundo|instante)|ahora mismo|intentar[ée]|usar[ée])\b",
                     re.IGNORECASE)
BOOKKEEPING = ("origin_px", "size_px", "shift_px_per_mm", "home_mm", "fov_mm", "step_mm")
SPARK_HELP = """- Muse Spark es un modelo de Meta más grande que tú, por internet y gasta créditos: inspect_component ya le pide
  leer la marca. Usa ask_spark solo para una pregunta difícil sobre la última imagen que viste, cuando tu propia
  mirada no alcance, y di que fue Muse Spark quien lo dijo.
"""
SPARK_EFFORT = "low"  # how much Muse Spark reasons per step when it is the brain ("medium" took 5 to 20 s a step)
SPARK_SYSTEM = ("You help Lunes, an electronics bench assistant, see through a USB microscope. Answer in Spanish, in at "
                "most three short sentences. Separate what you can read or see from what you infer, and mark "
                "characters you cannot read with ?.")
SPARK_READ = ("This picture shows one electronic component through a microscope, twice: the second copy is turned "
              "upside down, so the marking is upright in one of them. Read the printed marking character by "
              "character, then say what the part probably is.")

SYSTEM = """Eres Lunes, el asistente del banco de electrónica de Erlan, un ingeniero electrónico. Te llamas Lunes
como el asistente de Iron Man se llama Viernes. Tu memoria se llama Hume, por el filósofo David Hume: solo crees lo que
observaste, y cada cosa que recuerdas guarda la evidencia que la respalda.
Tienes una CNC con un microscopio montado en el eje Z, una cámara del banco y un Kinect que miran la máquina. El
microscopio ve unos 12 x 15 mm, así que moverte menos de 5 mm casi no cambia lo que ves. Coordenadas en milímetros
sobre el origen del banco; Z positivo sube.
La cámara del banco (una Sony) ve toda el área de trabajo, con mucho mejor imagen que el color del Kinect: úsala para
ver qué hay en el banco, dónde está la placa, los cables y las herramientas (look_bench, find_in_bench).
El Kinect te da la profundidad: la distancia en milímetros desde el Kinect hasta cada punto, con unos milímetros de
error a un metro (kinect_distance), y comprueba dónde está el cabezal (check_position).
Ni la cámara del banco ni el Kinect están calibrados a las coordenadas del banco: no des posiciones del banco ni
tamaños de la placa a partir de ellos; para eso están el microscopio y measure_board.

Cómo trabajas: eres un agente, haces el trabajo completo con tus herramientas, no pasitos.
- Para ver o fotografiar una placa entera: measure_board encuentra su perímetro desde donde estás (si no ve placa
  debajo, la busca alrededor) y te dice cuánto mide; después survey_board fotografía exactamente esa área. Dile a
  Erlan cuánto mide la placa.
- Si Erlan puso otra placa, la movió o le dio la vuelta, llama measure_board con new_board=true: olvida la placa y
  la altura de foco anteriores y busca la superficie nueva desde más arriba. Si la placa no cambió, no la vuelvas a
  medir ni a recorrer: usa lo que ya tienes.
- Para volver a ver un mosaico que ya tomaste usa look_survey; no saques las fotos de nuevo. Cada survey_board te
  dice su número y su cara. look_survey con number 0 te muestra juntas la última foto de cada cara (frente y
  atrás), para explicar de qué trata la placa.
- Si Erlan te pide explicar o describir, explica con lo que ya viste; no repitas el trabajo.
- Un mosaico de look_survey es de antes: dilo así ("en el mosaico de las 19:44 vi..."), nunca "veo". Si Erlan pide
  revisar la placa que está ahora en la máquina, mírala primero con look_microscope: si no se parece al mosaico, la
  placa se movió o se dio vuelta, y entonces mídela de nuevo con measure_board y new_board=true.
- Solo actúas cuando Erlan te lo pide. Un saludo, una pregunta sobre ti o una charla se responden hablando, sin
  usar herramientas ni mover la máquina.
- Si measure_board no encuentra la placa, mira con look_bench o pregúntale a Erlan dónde está. Si Erlan te dice
  dónde está o cuánto mide, úsalo: survey_board con width_mm y height_mm.
- Para apuntar el microscopio a algo que ves en su foto: center_on_point con la posición de eso en la foto; a una
  esquina de la placa: center_on_corner. Nunca calcules milímetros a ojo con move_by: la geometría la hace el código.
- Para medir en la foto del microscopio: measure_hole para un agujero (y su anillo de cobre), measure_distance entre
  dos puntos. No des medidas a ojo; si no puedes medir con una herramienta, di que no lo sabes.
- Tienes un osciloscopio USB (Hantek 6022BE, dos canales, 20 MHz): para cualquier voltaje o señal, mide con
  scope_measure; nunca des valores de una señal sin medirla. Deja que el código elija rango y velocidad (auto). Antes,
  Erlan pone el clip de masa en la masa del circuito y la punta en el punto. Las notas de scope_measure dicen cuánto
  confiar en la medición: léelas antes de juzgar el circuito. Juzga comparando con lo que el circuito debería dar (una
  fuente de 3.3 V, un cristal de 16 MHz, un PWM al 50 %) y dilo con los números medidos.
- La masa del osciloscopio es la masa de la PC por el USB: nunca le pidas medir la red de 220 V ni algo conectado a
  ella sin transformador de aislamiento. Si no sabes si la punta está en x1 o x10, pregúntale a Erlan o usa
  scope_selftest con la punta en la salida de calibración del osciloscopio.
- Para trabajar con un circuito (armarlo en la protoboard, comprobarlo o encontrar su falla): si es uno de los
  usuales (LED, 555, transistor como interruptor, divisor, filtro RC, amplificador), diséñalo con circuit_design: el
  código calcula los valores con fórmulas y los verifica en la simulación. Si no, escríbelo con circuit_set y dale
  sus metas (spec). Solo propónselo a Erlan cuando todas las metas pasen y la revisión no avise nada serio; si
  avisa, corrígelo. Para el criterio (patillajes, fallas típicas, límites) consulta circuit_guide. Si el simulador
  rechaza un circuito, su mensaje dice por qué: corrige eso, no repitas lo mismo. Para comprobarlo o diagnosticarlo: diagnose_start, y en cada paso dile a Erlan dónde poner la punta en
  palabras de banco ("el pin 3 del 555", "la pata de R2 que va al capacitor") y, cuando diga que está, llama
  diagnose_measure. El motor decide qué medir y qué falla es; tú lo explicas con los números. Solo nombra una falla
  cuando el motor la da como clara; si varias leen igual, pídele mirar esas partes. Nunca inventes una falla.
- Si una herramienta falla o responde raro (el simulador, el osciloscopio, el Kinect), usa system_check antes de
  culpar al circuito, y system_restart para la parte que falle; dile a Erlan qué pasó.
- Para identificar o leer un integrado u otro componente: inspect_component con un punto sobre él en el mosaico. Lo
  fotografía entero, de cerca y enfocado en su parte de arriba, y te devuelve la imagen para leer su marca. No
  intentes leerlo moviéndote de a poco.
- Si te falta algo que Erlan sabe (dónde está la placa, qué es, qué quiere), pregúntale en una frase.

Reglas:
- Responde en español, corto y natural, para decirlo en voz alta: una a tres frases, sin listas, sin markdown.
- Antes de afirmar algo sobre la placa o el banco, míralo con una herramienta.
- Nunca anuncies una acción ("voy a mirar", "en un momento"): hazla llamando la herramienta en ese mismo turno.
- Lo que puedes hacer es lo que hacen tus herramientas: mirar por el microscopio, la cámara del banco y el Kinect, mover la CNC y recordar. No puedes recibir videos, archivos ni fotos del teléfono, ni entrar a internet: si te ofrecen algo así, dilo.
- Solo crees lo que observaste. Distingue "vi" de "supongo", y di cuando la imagen no alcanza para saberlo. Al leer
  una marca, di qué caracteres no se ven claros en vez de inventarlos.
- Sabes mucha electrónica (cómo funciona un integrado conocido, sus valores típicos, cómo se mide): úsalo, pero dilo como conocimiento general ("en general el TP4056..."), nunca como algo que viste ni como una hoja de datos que leíste; y si la placa podría ser distinta, dilo.
- Antes de responder, revisa cada cosa que vas a decir: si no la viste con una herramienta en esta conversación,
  dila como suposición o no la digas. Prefiere "no lo sé" o "no estoy seguro" antes que adivinar. Erlan prefiere
  que tardes más y digas la verdad clara.
- Tu memoria viene de sesiones anteriores y puede ser de otra placa: dila como recuerdo ("recuerdo que...") y
  compruébala mirando antes de usarla. Si lo que ves ahora la contradice, cree lo que ves.
- Para moverte en X o Y no tienes que subir Z: la herramienta lo hace sola. No intentes bajar Z más allá del foco.
- Nunca inventes coordenadas. Para ir a algo que viste en el mosaico usa go_to_mosaic_point con su posición en la
  imagen (fracciones de 0 a 1).
- Solo atiendes a Erlan. Si lo que oyes claramente no es para ti (Erlan conversando con otra persona), no hagas
  nada y responde solo "(no era para mí)".
- Lo que Erlan dice te llega transcrito de su voz y a veces sale mal oído. Si parece dirigido a ti pero no tiene
  sentido, le falta algo, o viene marcado como transcripción dudosa, no adivines: pide en una frase que lo repita
  ("¿Puedes repetir, por favor?"), o di lo que entendiste y pregunta si es eso.
- Tu trabajo es la electrónica del banco. No describas a las personas que veas con la cámara del banco o el Kinect
  (qué hacen, qué comen, cómo son) salvo que Erlan te lo pida, y úsalos para mirar la máquina, no la habitación.
- Si Erlan pide algo peligroso o fuera del área permitida, dilo y no lo hagas.
- Guarda con remember los hechos importantes que descubras, con la evidencia que los respalda.
"""

TOOLS = [
    {"type": "function", "function": {
        "name": "where", "description": "Current machine position (mm), the focus height and the allowed area.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "find_in_mosaic",
        "description": "Find every instance of one kind of component in the mosaic of the face that is up, with SAM 3.1 "
                       "(a Meta segmentation model): returns where each one is (x, y from 0 to 1, as "
                       "go_to_mosaic_point and inspect_component take them) and how sure it is. Use it before going "
                       "to a component instead of guessing points. The concept is a short English noun phrase, one "
                       "kind per call: 'chip' and 'integrated circuit', 'capacitor' and 'heat sink' work well; "
                       "'connector', 'fuse' and 'screw terminal' find nothing on these pictures, so for those look "
                       "at the mosaic yourself. Stepper drivers are usually under heat sinks.",
        "parameters": {"type": "object", "properties": {"concept": {"type": "string"}}, "required": ["concept"]}}},
    {"type": "function", "function": {
        "name": "find_in_bench",
        "description": "Find every instance of one thing in a fresh picture of the bench camera with SAM 3.1: where it "
                       "is in the picture (x, y from 0 to 1) and how sure. For what is on the bench: 'circuit board', "
                       "'cable', 'screwdriver', 'multimeter'. It gives no machine millimetres.",
        "parameters": {"type": "object", "properties": {"concept": {"type": "string"}}, "required": ["concept"]}}},
    {"type": "function", "function": {
        "name": "find_in_kinect",
        "description": "Find every instance of one thing in a fresh Kinect picture with SAM 3.1: where it is in the "
                       "picture and how sure, and whether the microscope's ring is inside it. 'circuit board' finds "
                       "the board on the bed well (is it under the microscope?); 'cable' works; 'microscope', 'stepper "
                       "motor' and 'hand' did not. Its picture is poorer than the bench camera's. It gives no machine "
                       "millimetres: the Kinect's calibration only covers the head's height, and for the bed it was "
                       "40-50 mm off. On microscope pictures SAM finds almost nothing.",
        "parameters": {"type": "object", "properties": {"concept": {"type": "string"}}, "required": ["concept"]}}},
    {"type": "function", "function": {
        "name": "record_video",
        "description": "Record a video from the microscope, the bench camera or the Kinect into an MP4 for Erlan (and "
                       "for track_in_video), up to 60 s. The machine does not move unless you move it meanwhile.",
        "parameters": {"type": "object", "properties": {
            "seconds": {"type": "number"},
            "source": {"type": "string", "enum": ["microscopio", "banco", "kinect"]}},
            "required": ["seconds", "source"]}}},
    {"type": "function", "function": {
        "name": "track_in_video",
        "description": "Follow one thing through the last recorded video with SAM 3.1 and write a copy with the boxes "
                       "drawn. Works on bench camera and Kinect videos ('circuit board', 'cable'); on microscope "
                       "videos SAM finds almost nothing.",
        "parameters": {"type": "object", "properties": {"concept": {"type": "string"}}, "required": ["concept"]}}},
    {"type": "function", "function": {
        "name": "search_web",
        "description": "Search the internet with Muse Spark: datasheets, application notes, part markings, prices, "
                       "the weather, news. Returns an answer with its sources; say where each fact comes from.",
        "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}}},
    {"type": "function", "function": {
        "name": "check_position",
        "description": "Where the Kinect sees the microscope head against where you think the machine is. Use it when "
                       "the motors were released, when Erlan says it moved wrong, or before trusting the position "
                       "for something that matters. Moves already check it by themselves.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "calibrate_kinect",
        "description": "Teaches the Kinect where the microscope head is: moves the machine to nine positions up to "
                       "30 mm around where it is (Z only up) and back, in about a minute. Only when Erlan asks, and "
                       "only if the machine is where you think (he did not move it by hand).",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "learn_direction",
        "description": "Save which way an axis goes as Erlan sees it, when he tells you or corrects you (\"it went "
                       "forward, not back\"). It replaces what you knew about that axis and applies from now on.",
        "parameters": {"type": "object", "properties": {
            "axis": {"type": "string", "enum": ["X", "Y", "Z"]},
            "positive_is": {"type": "string", "description": "where + goes, in Erlan's words: adelante, atrás, "
                                                             "izquierda, derecha, arriba, abajo"},
            "evidence": {"type": "string", "description": "what happened that shows it"}},
            "required": ["axis", "positive_is", "evidence"]}}},
    {"type": "function", "function": {
        "name": "motors", "description": "Release the motors (on=false) so Erlan can move the carriage by hand, or "
                                         "hold them again (on=true). Once released, you no longer know for sure where "
                                         "the machine is: before moving it again, tell Erlan, and relocate with "
                                         "measure_board or ask him.",
        "parameters": {"type": "object", "properties": {"on": {"type": "boolean"}}, "required": ["on"]}}},
    {"type": "function", "function": {
        "name": "move_to", "description": "Move to absolute machine coordinates in mm; leave out an axis to keep it.",
        "parameters": {"type": "object", "properties": {
            "x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}}}}},
    {"type": "function", "function": {
        "name": "move_by", "description": "Move relative to the current position, in mm.",
        "parameters": {"type": "object", "properties": {
            "dx": {"type": "number"}, "dy": {"type": "number"}, "dz": {"type": "number"}}}}},
    {"type": "function", "function": {
        "name": "look_microscope", "description": "Take a picture through the microscope at the current position.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "center_on_point",
        "description": "Move the microscope so that a point of its last picture (from look_microscope or a center_on_ "
                       "tool) comes to the middle of the view. Give the point as fractions of that picture: x from "
                       "the left edge, y from the top edge (0 to 1). The code turns pixels into mm with the "
                       "microscope's calibration; never estimate mm by eye with move_by. Returns the new picture with "
                       "a cross in the middle.",
        "parameters": {"type": "object", "properties": {
            "x": {"type": "number"}, "y": {"type": "number"}}, "required": ["x", "y"]}}},
    {"type": "function", "function": {
        "name": "center_on_corner",
        "description": "Put one corner of the board exactly in the middle of the microscope's view: the code finds the "
                       "board's two edges in the picture, moves until they cross under the cross, and checks it with "
                       "a last picture. Give the corner by its sides along the machine axes: x_side and y_side, '+' "
                       "or '-' ('+' is the board's largest X or Y). Left out, it takes the measured board's corner "
                       "nearest to the microscope. Needs a measured board, or the corner already in view.",
        "parameters": {"type": "object", "properties": {
            "x_side": {"type": "string", "enum": ["+", "-"]}, "y_side": {"type": "string", "enum": ["+", "-"]}}}}},
    {"type": "function", "function": {
        "name": "measure_hole",
        "description": "Measure a round hole (a mounting hole, a via, a through-hole pad) in the last microscope "
                       "picture: give a point inside it as fractions of that picture (x from the left, y from the "
                       "top, 0 to 1). The code finds its edge all around and fits a circle in mm with the "
                       "calibration: the hole's diameter and its copper ring's outer diameter. Returns the picture "
                       "with the edges drawn, to check they sit where they should.",
        "parameters": {"type": "object", "properties": {
            "x": {"type": "number"}, "y": {"type": "number"}}, "required": ["x", "y"]}}},
    {"type": "function", "function": {
        "name": "measure_distance",
        "description": "Distance in mm between two points of the last microscope picture, given as fractions of it "
                       "(x from the left, y from the top, 0 to 1), with the calibration: a pin pitch, a part's "
                       "length, a gap. Points you place by eye are only as exact as where you put them.",
        "parameters": {"type": "object", "properties": {
            "x1": {"type": "number"}, "y1": {"type": "number"}, "x2": {"type": "number"}, "y2": {"type": "number"}},
            "required": ["x1", "y1", "x2", "y2"]}}},
    {"type": "function", "function": {
        "name": "scope_measure",
        "description": "Measure with the oscilloscope (a Hantek 6022BE, two channels) what is on its probe(s) now. By "
                       "default the code picks the range and the sample rate itself (auto) and measures: min, max, "
                       "peak to peak, mean, frequency, duty cycle, rise time, the low and high levels, the shape (dc, "
                       "sine, square, pwm, digital, ...), and for serial data the shortest pulse and the likely baud "
                       "rate. Returns the numbers, notes on how far to trust them, and a picture of the screen. "
                       "Give range_v (volts at the tip, either way of 0) and rate (samples per second) only to "
                       "repeat a setting on purpose.",
        "parameters": {"type": "object", "properties": {
            "channel": {"type": "string", "enum": ["CH1", "CH2", "both"]},
            "auto": {"type": "boolean"}, "range_v": {"type": "number"}, "rate": {"type": "number"}}}}},
    {"type": "function", "function": {
        "name": "scope_selftest",
        "description": "Check the oscilloscope and a probe on the scope's own calibration output (a 1 kHz, about 2 V "
                       "square wave): first ask Erlan to put that channel's tip on the calibration pin and its "
                       "ground clip on the ground pin beside it. It finds out whether the probe's x1/x10 switch "
                       "matches what Lunes believes.",
        "parameters": {"type": "object", "properties": {"channel": {"type": "string", "enum": ["CH1", "CH2"]}}}}},
    {"type": "function", "function": {
        "name": "scope_probe",
        "description": "Note the factor on a probe's switch (1 or 10) after Erlan says it or the self-test shows it. "
                       "Every volt the scope reports depends on it.",
        "parameters": {"type": "object", "properties": {
            "channel": {"type": "string", "enum": ["CH1", "CH2"]}, "factor": {"type": "number", "enum": [1, 10]}},
            "required": ["channel", "factor"]}}},
    {"type": "function", "function": {
        "name": "circuit_set",
        "description": "Set the circuit on the bench (one Erlan describes, or one you design) and simulate it with "
                       "ngspice: returns what the scope should read at each net (DC volts, or frequency, peak to "
                       "peak and duty for nets that move) and which pins and parts touch each net. Format: {name, "
                       "supply_v, parts: [{ref, kind, value or model or color, pins}]}. kind R, C or L: value "
                       "('10k', '68n', '4k7'), pins [netA, netB]. D: model 1N4148, 1N4007 or 1N5819, pins {A, K}. "
                       "LED: color red/yellow/green/blue/white, pins {A, K}. NPN or PNP: model 2N2222, 2N3904, "
                       "BC547, 2N3906 or BC557, pins {C, B, E}. NE555: pins {GND, TRIG, OUT, RESET, CTRL, THRES, "
                       "DISCH, VCC}. OPAMP: pins {OUT, IN-, IN+, V+, V-}. V (an extra source): value '3.3' or "
                       "{sine: [offset, amplitude, hz]} or {square: [low, high, hz]}, pins {+, -}. Nets are names; "
                       "GND is ground and VCC the supply. A review checks the wiring first (nets with no DC path "
                       "to ground, a shorted supply, parts in one row) and the parts after (LED current, resistor "
                       "power, transistor ratings). Give spec, the goals it must meet, and the simulation says "
                       "whether each one passes: [{net, frequency_hz | duty | mean_v, tolerance}], [{current_of: "
                       "ref, target_a, tolerance}], [{net, of: other net, ratio_vpp, tolerance}]. For the usual "
                       "circuits prefer circuit_design, which computes the values by formula.",
        "parameters": {"type": "object", "properties": {"circuit": {"type": "object", "properties": {
            "name": {"type": "string"}, "supply_v": {"type": "number"},
            "parts": {"type": "array", "items": {"type": "object"}}}, "required": ["parts"]},
            "spec": {"type": "array", "items": {"type": "object"}}},
            "required": ["circuit"]}}},
    {"type": "function", "function": {
        "name": "circuit_design",
        "description": "Design a usual circuit by formula, with standard values (E12 resistors, E6 capacitors), "
                       "simulate it and check it meets its goals; it becomes the circuit on the bench, as with "
                       "circuit_set. Templates and their params: led_resistor {supply_v, color, current_ma}; "
                       "astable_555 {frequency_hz, duty (0.51 to 0.95), supply_v, led_color}; npn_switch "
                       "{load_current_ma, supply_v, logic_v, input_hz, color, model}; divider {vout, supply_v, "
                       "current_ma}; rc_filter {cutoff_hz, kind lowpass|highpass, supply_v}; opamp_noninverting "
                       "{gain, supply_v, input_offset_v, input_amplitude_v, input_hz}. Returns the parts, the "
                       "formulas used, each goal with what the simulation gave, and the review.",
        "parameters": {"type": "object", "properties": {
            "template": {"type": "string", "enum": ["led_resistor", "astable_555", "npn_switch", "divider",
                                                    "rc_filter", "opamp_noninverting"]},
            "params": {"type": "object"}}, "required": ["template"]}}},
    {"type": "function", "function": {
        "name": "circuit_guide",
        "description": "Read a section of Lunes' circuit guide, what a senior bench engineer keeps in mind: method, "
                       "555, led, transistor, opamp, rc, divider, diodes, supply, breadboard, simulation, diagnosis, "
                       "measurement, arduino. For judgement (pinouts, typical faults, limits), not for arithmetic.",
        "parameters": {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]}}},
    {"type": "function", "function": {
        "name": "system_check",
        "description": "How Lunes' own parts are doing: the simulator (a self-test on two known circuits, and any "
                       "ngspice left hanging), the oscilloscope, the Kinect, the cameras and the CNC (its alarm). "
                       "Use it when something fails or answers oddly, before blaming the circuit.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "system_restart",
        "description": "Restart one of Lunes' parts after system_check found it failing: simulator (ends hanging "
                       "ngspice runs and tests it again), scope (closes and opens the oscilloscope again, loading "
                       "its firmware), kinect (opens it again), cnc_alarm (clears the motion board's alarm once "
                       "Erlan knows why it stopped; it never moves the machine).",
        "parameters": {"type": "object", "properties": {
            "part": {"type": "string", "enum": ["simulator", "scope", "kinect", "cnc_alarm"]}},
            "required": ["part"]}}},
    {"type": "function", "function": {
        "name": "diagnose_start",
        "description": "Start checking the circuit set with circuit_set, to confirm one just built works or to find "
                       "the fault in one that does not. The simulator works out what every typical fault of every "
                       "part would read (about half a minute the first time), and the engine says which net to "
                       "measure first. Then, for each step: tell Erlan where to put the probe (the pins and parts "
                       "that touch that net), and when he says it is there, diagnose_measure.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "diagnose_measure",
        "description": "Measure the net where Erlan put the CH1 probe now, with the scope, and update the diagnosis: "
                       "returns the reading, the likeliest explanations with their probability, and the next net "
                       "to measure, or the verdict once one explanation is clear (90 %). If Erlan measured with "
                       "his multimeter instead, give mean_v and source 'multimeter'.",
        "parameters": {"type": "object", "properties": {
            "net": {"type": "string"}, "source": {"type": "string", "enum": ["scope", "multimeter"]},
            "mean_v": {"type": "number"}}, "required": ["net"]}}},
    {"type": "function", "function": {
        "name": "scope_test_signal",
        "description": "Put a square wave of about 2 V on the scope's calibration pin, from 32 Hz to 100 kHz, to feed "
                       "a circuit a known signal (an RC filter, an amplifier's input) and measure what comes out. "
                       "Returns the frequency it really puts out.",
        "parameters": {"type": "object", "properties": {"hz": {"type": "number"}}, "required": ["hz"]}}},
    {"type": "function", "function": {
        "name": "look_bench",
        "description": "Take a picture of the whole work area with the bench camera: wider than the microscope and "
                       "much sharper than the Kinect. For what is on the bench and where the board is; not for "
                       "reading markings (that is inspect_component).",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "look_kinect", "description": "Take a picture of the whole machine and bench with the Kinect, colour "
                                              "and depth side by side. For distances; to see the bench, look_bench is "
                                              "sharper.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "kinect_distance",
        "description": "Distance in mm from the Kinect to a point of its picture, given as fractions of the colour "
                       "picture (x from the left, y from the top, 0 to 1). The depth camera sits a few centimetres "
                       "beside the colour one, so near edges the point can land on what is just behind.",
        "parameters": {"type": "object", "properties": {
            "x": {"type": "number"}, "y": {"type": "number"}}, "required": ["x", "y"]}}},
    {"type": "function", "function": {
        "name": "autofocus",
        "description": "Focus the microscope. The board's surface is in focus near the focus height; the top of a "
                       "tall part is higher, so give up_mm to also search above (about 3 to 4 mm for a DIP chip). "
                       "Z never goes below the board's focus height.",
        "parameters": {"type": "object", "properties": {
            "up_mm": {"type": "number", "description": "how far above the current height to search, 0.3 to 5"}}}}},
    {"type": "function", "function": {
        "name": "measure_board",
        "description": "Find the outline of the circuit board from the current position: walk to each side until "
                       "its edges show up (searching around first if there is no board under the microscope). "
                       "Returns the board's size and position in mm. Takes some 20 to 90 seconds.",
        "parameters": {"type": "object", "properties": {
            "new_board": {"type": "boolean",
                          "description": "true when Erlan changed, moved or flipped the board since the last one"}}}}},
    {"type": "function", "function": {
        "name": "survey_board",
        "description": "Photograph the whole board that measure_board found (it measures first if needed) and "
                       "stitch one mosaic. With width_mm and height_mm, photograph an area of that size centred on "
                       "the current position instead.",
        "parameters": {"type": "object", "properties": {
            "width_mm": {"type": "number", "description": "size along X, only when Erlan gave it"},
            "height_mm": {"type": "number", "description": "size along Y, only when Erlan gave it"}}}}},
    {"type": "function", "function": {
        "name": "look_survey",
        "description": "See again a mosaic taken earlier in this session, without moving: number 1 is the first "
                       "survey_board, -1 the last one, -2 the one before it. Lists them all when no number is given.",
        "parameters": {"type": "object", "properties": {
            "number": {"type": "integer", "description": "0 shows the last mosaic of each face side by side"}}}}},
    {"type": "function", "function": {
        "name": "inspect_component",
        "description": "Photograph one component (a chip, a connector, a module) entirely, close up and focused on "
                       "its top, to read its marking. Give a point on it in the last survey_board mosaic, as "
                       "fractions of that image: x from the left edge, y from the top edge (0 to 1).",
        "parameters": {"type": "object", "properties": {
            "x": {"type": "number"}, "y": {"type": "number"}}, "required": ["x", "y"]}}},
    {"type": "function", "function": {
        "name": "ask_spark",
        "description": "Ask Muse Spark, Meta's larger model (over the internet; it spends credits), a hard question "
                       "about the last image you saw: a close-up, a mosaic or both faces. Only when your own look "
                       "is not enough.",
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string"}}, "required": ["question"]}}},
    {"type": "function", "function": {
        "name": "go_to_mosaic_point",
        "description": "Move the microscope over a point of the last survey_board mosaic, given as fractions of "
                       "that mosaic image: x from the left edge (0 to 1), y from the top edge (0 to 1).",
        "parameters": {"type": "object", "properties": {
            "x": {"type": "number"}, "y": {"type": "number"}}, "required": ["x", "y"]}}},
    {"type": "function", "function": {
        "name": "remember", "description": "Save a fact for later, with the observation that supports it.",
        "parameters": {"type": "object", "properties": {
            "fact": {"type": "string"}, "evidence": {"type": "string"}}, "required": ["fact", "evidence"]}}},
]


def jpeg_b64(image, max_side=1280, quality=88):
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


def to_openai(messages):
    """Lunes' conversation (kept in Ollama's shape) in the OpenAI chat format Meta's API speaks. Pictures become
    image parts, and a picture taken by a tool waits until all the tool results of that step are in, since the
    API wants the results right after the call."""
    out, waiting = [], []
    for m in messages:
        role = m.get("role")
        if role != "tool" and waiting and not (role == "user" and m.get("images")):
            out += waiting
            waiting = []
        if role == "assistant":
            msg = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("tool_calls"):
                msg["tool_calls"] = [{"id": c.get("id", f"call_{k}"), "type": "function",
                                      "function": {"name": c["function"]["name"],
                                                   "arguments": json.dumps(c["function"].get("arguments") or {})}}
                                     for k, c in enumerate(m["tool_calls"])]
            out.append(msg)
        elif role == "tool":
            out.append({"role": "tool", "tool_call_id": m.get("tool_call_id", "call_0"), "content": m["content"]})
        elif role == "user" and m.get("images"):
            parts = [{"type": "text", "text": m.get("content") or ""}] + [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b}} for b in m["images"]]
            waiting.append({"role": "user", "content": parts})
        else:
            out.append({"role": role, "content": m.get("content") or ""})
    return out + waiting


def kinect_view(color, depth):
    """Colour and depth side by side; the depth half has the distance (mm) written at nine points."""
    from lunes.kinect import depth_colormap
    good = depth[depth > 0]
    near, far = (int(np.percentile(good, 2)), int(np.percentile(good, 98))) if good.size else (800, 2000)
    dview = depth_colormap(depth, near, max(far, near + 100))
    h, w = depth.shape
    for fy in (0.25, 0.5, 0.75):
        for fx in (0.25, 0.5, 0.75):
            x, y = int(fx * w), int(fy * h)
            patch = depth[y - 4:y + 5, x - 4:x + 5]
            text = f"{int(np.median(patch[patch > 0]))}" if (patch > 0).any() else "?"
            cv2.circle(dview, (x, y), 4, (255, 255, 255), -1)
            for ink, width in (((0, 0, 0), 4), ((255, 255, 255), 1)):
                cv2.putText(dview, text, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ink, width)
    return np.hstack([color, dview])


def _mm(d):
    return {k: round(float(v), 3) for k, v in d.items()}


def with_cross(frame, mark=None):
    """A microscope picture with a thin cross at its middle (where the microscope points) and, if given, a ring on
    another pixel (what was found)."""
    view = frame.copy()
    h, w = view.shape[:2]
    for ink, width in (((0, 0, 0), 3), ((0, 255, 255), 1)):
        cv2.line(view, (w // 2 - 25, h // 2), (w // 2 + 25, h // 2), ink, width)
        cv2.line(view, (w // 2, h // 2 - 25), (w // 2, h // 2 + 25), ink, width)
    if mark is not None:
        cv2.circle(view, (int(round(mark[0])), int(round(mark[1]))), 9, (255, 0, 255), 2)
    return view


class Hume:
    def __init__(self, machine, microscope, kinect=None, show=None, model=MODEL, think=THINK, on_think=None,
                 spark=False, about="", bench=None, scope=None):
        """think: False, "low", "medium" or "high", how much Glimmer reasons before it answers (it is not spoken).
        on_think(text) gets the reasoning of each step, for the session log. spark=True lets Lunes send pictures
        to Muse Spark over the internet (with free credits Meta may use them), so only when Erlan allows it.
        bench: the camera that sees the whole work area (the Sony), if it is plugged in; scope: the oscilloscope
        (lunes.hantek), if it is plugged in."""
        self.m, self.cam, self.kinect, self.bench, self.scope = machine, microscope, kinect, bench, scope
        self.last_scope = None  # {"at", "result"} of the last scope_measure
        self.circuit = self.expected = self.diagnosis = None  # the circuit on the bench, its simulation, a diagnosis
        self.fault_table = None  # the circuit's fault dictionary, simulated in the background after circuit_set
        self._table_thread = None
        self.show = show or (lambda frame, text: None)
        self.model, self.think, self.on_think = model, think, on_think
        self.on_image = None  # on_image(picture, what) for each picture shown to the brain (the dashboard)
        self.on_result = None  # on_result(tool name, result) after each tool, for the progress Lunes tells
        self.status, self.request = "", ""  # what Lunes is doing now, and what it was asked
        self.brain_seconds, self.brain_steps = 0.0, 0
        self.focus_z = None
        self.board = None  # the last measure_board result
        self.last_survey, self.last_mosaic = None, None
        self.surveys = []  # every board mosaic: number, face, size, file
        self.side = 1  # which face of the board is up; a new_board measure after a flip counts the next one
        if SURVEYS.exists():
            saved = json.loads(SURVEYS.read_text(encoding="utf-8"))
            self.surveys, self.side = saved["surveys"], saved["side"]
        self.stop = threading.Event()  # set by the ear when Erlan says "para"
        self.last_image = None  # the last picture shown to the model, for ask_spark
        self.last_look = None  # (position, picture) of the last microscope picture, for center_on_ and measure_
        self.mosaic_doubtful = False  # the microscope did not find itself in the last mosaic at start
        # the motors were released (now or in an earlier session): someone may have moved the carriage by hand
        from lunes.machine import load_position
        self.position_doubtful = load_position("by") == "motors released"
        self.last_kinect_check = None  # {"at", "agrees", "off_mm"} of the last check_position
        self.spark = None
        if spark:
            try:
                from lunes.meta_api import MetaModels
                self.spark = MetaModels()
            except Exception:  # no key or no network: Lunes works locally without Muse Spark
                self.spark = None
        # Muse Spark is the brain whenever it is allowed and reachable; Glimmer otherwise
        self.brain = "spark" if self.spark else "glimmer"
        brain = ("Muse Spark, el modelo de Meta, por su API (si no hay internet, Muse Glimmer en esta PC)"
                 if self.brain == "spark" else f"Muse Glimmer, el modelo abierto de Meta, en la RTX 3090 de esta PC")
        # what Lunes runs on, so it can say it when asked instead of guessing
        made_of = f"\nCon qué funcionas: tu cerebro es {brain}. {about}\n"
        self.messages = [{"role": "system", "content": SYSTEM + made_of + (SPARK_HELP if self.spark else "") +
                          self._memory_text()}]

    # Focus ------------------------------------------------------------------------------------------------
    def _scan_focus(self, top, bottom):
        """Focus search from top down to bottom: coarse steps, then fine; ends at the sharpest height."""
        def scan(hi, lo, step):
            best = (-1.0, hi)
            z = hi
            while z >= lo - 1e-9:  # top to bottom, so backlash is always taken the same way
                self.m.move_to(Z=z)
                s = sharpness(self.cam.fresh_gray(2))
                self.show(self.cam.frame(), f"enfocando  Z {z:+.2f}  nitidez {s:.0f}")
                best = max(best, (s, z))
                z -= step
            return best

        s1, z1 = scan(top, bottom, 0.25)
        s2, z2 = scan(min(z1 + 0.25, top), max(bottom, z1 - 0.25), 0.05)
        self.m.move_to(Z=min(z2 + 0.3, top))
        self.m.move_to(Z=z2)
        return z2, s2

    def _focus_up(self, up_mm):
        """Focus on something up to up_mm above the current height (the top of a tall part), down to the floor."""
        here = self.m.position()["Z"]
        return self._scan_focus(min(here + up_mm, WORKSPACE["Z"][1]), max(self._z_floor(), here - 0.3))

    def _focus_board(self, above_mm=0.5):
        """Puts the board's surface in focus: at the known focus height, or found by searching from above_mm over
        the current height down to the floor."""
        if self.focus_z is not None:
            self.m.move_to(Z=self.focus_z)
            return self.focus_z
        # The bench's Z origin was set by eye: the board may sit a little lower, so look 1.5 mm below it too.
        here = self.m.position()["Z"]
        z, _ = self._scan_focus(min(here + above_mm, WORKSPACE["Z"][1]), max(self._z_floor() - 1.5, WORKSPACE["Z"][0]))
        self.focus_z = z
        return z

    # Memory ------------------------------------------------------------------------------------------------
    def _memory_text(self, last=15):
        if not MEMORY.exists():
            return ""
        notes = [json.loads(line) for line in MEMORY.read_text(encoding="utf-8").splitlines() if line.strip()][-last:]
        if not notes:
            return ""
        lines = "\n".join(f"- {n['fact']} (evidencia: {n['evidence']}; {n['at']})" for n in notes)
        return ("\nLo que recuerdas de sesiones anteriores. Puede ser de otra placa que ya no está en la máquina: "
                "compruébalo mirando antes de decirlo como cierto.\n" + lines + "\n")

    # Moves -------------------------------------------------------------------------------------------------
    def _z_floor(self):
        return self.focus_z if self.focus_z is not None else 0.0

    def _check(self, target):
        for axis, v in target.items():
            lo, hi = WORKSPACE[axis]
            if axis == "Z":
                lo = max(lo, self._z_floor())
            if not lo - 0.05 <= v <= hi + 0.05:  # positions come back rounded to whole steps (Y: 0.025 mm)
                return f"{axis}={v:.2f} mm está fuera del área permitida ({lo:.2f} a {hi:.2f} mm)"
        return None

    def _move(self, x=None, y=None, z=None):
        here = self.m.position()
        target = {"X": here["X"] if x is None else float(x), "Y": here["Y"] if y is None else float(y),
                  "Z": here["Z"] if z is None else float(z)}
        problem = self._check(target)
        if problem:
            return {"ok": False, "error": problem}
        if abs(target["X"] - here["X"]) > 1e-3 or abs(target["Y"] - here["Y"]) > 1e-3:
            safe_z = max(here["Z"], target["Z"], self._z_floor() + LIFT_MM)
            self.m.move_to(Z=min(safe_z, WORKSPACE["Z"][1]))
            self.m.move_to(X=target["X"], Y=target["Y"])
        self.m.move_to(Z=target["Z"])
        return {"ok": True, "position_mm": _mm(self.m.position())}

    def _look(self):
        """A fresh microscope picture, kept with where it was taken so a point of it can be turned into mm."""
        self.cam.fresh_gray(1)
        frame = self.cam.frame()
        self.last_look = (self.m.position(), frame)
        return frame

    def _center_on_point(self, x, y):
        """Moves the microscope so a point of its last picture is in the middle of the view. The code turns the
        picture's pixels into mm with the calibration, so the model never guesses distances."""
        if self.last_look is None:
            return {"ok": False, "error": "look_microscope first, then give a point of that picture"}, None
        if not (0 <= x <= 1 and 0 <= y <= 1):
            return {"ok": False, "error": "x and y are fractions of the picture, from 0 to 1"}, None
        at, last = self.last_look
        h, w = last.shape[:2]
        vx, vy, _, _ = geometry(self.m.config["microscope"])
        tx, ty = (float(v[0]) for v in pixels_to_mm(at, vx, vy, [x * (w - 1)], [y * (h - 1)]))
        result = self._move(tx, ty)
        if not result.get("ok"):
            return result, None
        frame = self._look()
        result.update(moved_mm={"X": round(tx - at["X"], 3), "Y": round(ty - at["Y"], 3)},
                      note="that point is now in the middle of the picture, under the cross")
        return result, with_cross(frame)

    def _last_picture_point(self, *fractions):
        """(position, picture, pixels) for points of the last microscope picture given as fractions, or an error."""
        if self.last_look is None:
            return None, "look_microscope first, then give points of that picture"
        if not all(0 <= f <= 1 for f in fractions):
            return None, "points are fractions of the picture, from 0 to 1"
        at, frame = self.last_look
        h, w = frame.shape[:2]
        pixels = [(fx * (w - 1), fy * (h - 1)) for fx, fy in zip(fractions[::2], fractions[1::2])]
        return (at, frame, pixels), None

    def _measure_hole(self, x, y):
        """Diameter of a hole of the last microscope picture (and of its copper ring), measured by the code."""
        got, problem = self._last_picture_point(x, y)
        if problem:
            return {"ok": False, "error": problem}, None
        at, frame, [(px, py)] = got
        vx, vy, _, _ = geometry(self.m.config["microscope"])
        hole, why = hole_in_picture(frame, at, vx, vy, px, py)
        if hole is None:
            return {"ok": False, "error": why}, with_cross(frame, (px, py))
        view = frame.copy()
        for edge, ink in zip(hole.pop("edges_px"), ((0, 255, 0), (255, 0, 255))):
            for ex, ey in edge.astype(int):
                cv2.circle(view, (ex, ey), 1, ink, -1)
        hole.update(ok=True, note="measured by the code from the edge all around, with the microscope's calibration, "
                                  "at the board's surface (green: the hole's edge; pink: the ring's). Say it as "
                                  "measured, to a tenth of a millimetre")
        return hole, view

    def _measure_distance(self, x1, y1, x2, y2):
        """Distance between two points of the last microscope picture, with the calibration."""
        got, problem = self._last_picture_point(x1, y1, x2, y2)
        if problem:
            return {"ok": False, "error": problem}, None
        at, frame, (p1, p2) = got
        vx, vy, _, _ = geometry(self.m.config["microscope"])
        d, dx, dy = distance_in_picture(at, vx, vy, p1, p2)
        view = frame.copy()
        cv2.line(view, tuple(int(v) for v in p1), tuple(int(v) for v in p2), (0, 255, 255), 1)
        return {"ok": True, "distance_mm": round(d, 3), "along_X_mm": round(dx, 3), "along_Y_mm": round(dy, 3),
                "note": "as exact as where the two points were put; at the board's surface"}, view

    # The oscilloscope ---------------------------------------------------------------------------------------------
    def _scope_measure(self, channel="CH1", auto=True, range_v=None, rate=None):
        """Measure what is on the probe(s): the code sets the scope up (auto) or uses the given setting."""
        from lunes.hantek import ranges
        from lunes.signals import autoset, fit_range, measure, notes, picture, step_v
        if self.scope is None:
            return {"ok": False, "error": "the oscilloscope is not connected"}, None
        names = ["CH1", "CH2"] if channel == "both" else [channel]
        try:
            if auto:
                cap, _, _ = autoset(self.scope, names[0], alone=channel == "CH1")
                if len(names) == 2:  # CH2 gets its own range at the sample rate CH1 needs
                    fit_range(self.scope, "CH2")
                    cap = self.scope.capture(20000)
            else:
                given = {names[0]: float(range_v)} if range_v else None
                self.scope.configure(rate=float(rate) if rate else None, range_v=given,
                                     channels=2 if "CH2" in names else None)
                cap = self.scope.capture(20000)
        except (RuntimeError, ValueError) as e:
            return {"ok": False, "error": str(e)}, None
        result, shown = {"ok": True, "sample_rate": cap.rate, "probe": dict(self.scope.probe)}, {}
        for ch in cap.volts:
            m = measure(cap.volts[ch], cap.rate, step_v(cap.range_v[ch]))
            shown[ch] = m
            if ch in names:
                allowed = ranges(self.scope.probe[ch])
                result[ch] = dict(m, range_v=cap.range_v[ch],
                                  notes=notes(m, cap.range_v[ch], cap.clipped[ch], allowed[0], allowed[-1]))
        if getattr(self.scope, "signals", None) is not None:
            result["simulated"] = "made-up signals from the simulated scope, not the bench: say so if you report them"
        self.last_scope = {"at": time.time(), "result": result}
        return result, picture(cap, shown, title=" + ".join(names))

    def _scope_selftest(self, channel="CH1"):
        """The probe on the scope's 1 kHz calibration output: frequency, shape and amplitude, and what a wrong x1/x10
        belief looks like."""
        if self.scope is None:
            return {"ok": False, "error": "the oscilloscope is not connected"}, None
        hz = self.scope.calibration_output(1000)
        result, view = self._scope_measure(channel)
        if not result.get("ok"):
            return result, view
        m = result[channel]
        amp = m.get("amplitude_v") or m["vpp_v"]
        factor = self.scope.probe[channel]
        problems = []
        if m["shape"] not in ("square", "pwm"):
            problems.append(f"it sees {m['shape']}, not a square wave: is the tip on the calibration pin and the "
                            f"ground clip on its ground?")
        else:
            if abs((m.get("frequency_hz") or 0) / hz - 1) > 0.03:
                problems.append(f"frequency {m.get('frequency_hz')} Hz instead of {hz:g} Hz")
            if 0.12 < amp < 0.3:
                problems.append(f"amplitude {amp:g} V, ten times too small: the probe is probably at x10 while Lunes "
                                f"counts x{factor:g}; use scope_probe {channel} 10 once Erlan confirms")
            elif 12 < amp < 30:
                problems.append(f"amplitude {amp:g} V, ten times too big: the probe is probably at x1 while Lunes "
                                f"counts x{factor:g}; use scope_probe {channel} 1 once Erlan confirms")
            elif not 1.4 < amp < 2.6:
                problems.append(f"amplitude {amp:g} V, expected about 2 V")
        return {"ok": not problems, "channel": channel, "expected": f"square {hz:g} Hz, about 2 V",
                "measured": {k: m.get(k) for k in ("shape", "frequency_hz", "amplitude_v", "vpp_v", "duty")},
                "probe_factor_believed": factor, "problems": problems}, view

    # The circuit on the bench: simulation and diagnosis -------------------------------------------------------------
    def _circuit_set(self, circuit, spec=None):
        from lunes import review
        from lunes.circuit import check
        from lunes.designs import as_spec, verify
        from lunes.simulator import SimulationError, expected, fault_dictionary
        problems = check(circuit)
        if problems:
            return {"ok": False, "problems": problems, "note": "fix these and call again"}, None
        errors, warnings = review.before(circuit)
        if errors:
            return {"ok": False, "wiring_errors": errors, "warnings": warnings,
                    "note": "simulating this would give meaningless numbers: fix the wiring first"}, None
        try:
            sim = expected(circuit)
        except SimulationError as e:
            return {"ok": False, "simulation_problems": e.problems, "warnings": warnings}, None
        except (RuntimeError, OSError) as e:
            return {"ok": False, "error": f"the simulator itself failed ({e}): use system_check"}, None
        warnings += review.after(circuit, sim) + sim["warnings"]
        goals = verify(sim, as_spec(spec)) if spec else []
        self.circuit, self.expected, self.diagnosis, self.fault_table = circuit, sim, None, None
        CIRCUIT.parent.mkdir(exist_ok=True)
        CIRCUIT.write_text(json.dumps({"circuit": circuit, "expected": sim}, ensure_ascii=False, indent=1),
                           encoding="utf-8")

        def table():  # ready by the time Erlan has built it or picked up the probe
            try:
                self.fault_table = fault_dictionary(circuit)
            except RuntimeError:
                self.fault_table = None
        self._table_thread = threading.Thread(target=table, daemon=True)
        self._table_thread.start()
        out = {"ok": True, "name": circuit.get("name"), "supply_v": circuit.get("supply_v", 5),
               "expected_by_net": sim["nets"], "currents": sim["currents"], "review": warnings,
               "note": "simulated with ngspice; what the scope should read at each net"}
        if goals:
            out["goals"] = goals
            out["all_goals_met"] = all(g["ok"] for g in goals)
        return out, None

    def _circuit_design(self, template, params):
        from lunes.designs import TEMPLATES
        if template not in TEMPLATES:
            return {"ok": False, "error": f"template must be one of {sorted(TEMPLATES)}"}, None
        try:
            circuit, spec, notes = TEMPLATES[template](**(params or {}))
        except (TypeError, ValueError) as e:
            return {"ok": False, "error": f"cannot design it: {e}"}, None
        result, view = self._circuit_set(circuit, spec)
        result["design_notes"] = notes
        result["parts"] = [{k: v for k, v in part.items() if k in ("ref", "kind", "value", "model", "color", "pins")}
                           for part in circuit["parts"]]
        return result, view

    def _circuit_guide(self, topic):
        text = GUIDE.read_text(encoding="utf-8")
        sections = {}
        for block in text.split("\n## ")[1:]:
            name, _, body = block.partition("\n")
            sections[name.strip().lower()] = body.strip()
        key = str(topic).strip().lower()
        if key not in sections:
            return {"ok": False, "error": f"topics: {sorted(sections)}"}, None
        return {"ok": True, "topic": key, "guide": sections[key]}, None

    def _system_check(self):
        from lunes import simulator
        out = {"simulator": simulator.selftest()}
        out["simulator"]["hanging_runs_ended"] = simulator.kill_strays()
        if self.scope is None:
            out["scope"] = {"ok": False, "state": "not connected"}
        else:
            try:
                cap = self.scope.capture(2000)
                out["scope"] = {"ok": True, "state": "answering", "info": self.scope.info(),
                                "samples": cap.samples}
            except Exception as e:  # a pulled cable shows up as any USB error
                out["scope"] = {"ok": False, "state": f"not answering ({e})"}
        if self.kinect is None:
            out["kinect"] = {"ok": False, "state": "not connected"}
        else:
            try:
                self.kinect.color()
                out["kinect"] = {"ok": True, "state": "sending pictures", **self.kinect.info()}
            except RuntimeError as e:
                out["kinect"] = {"ok": False, "state": str(e)}
        try:
            st = self.m.status()
            out["cnc"] = {"ok": not st.get("alarm"), "state": st.get("st"), "alarm": st.get("alarm") or None}
        except Exception as e:
            out["cnc"] = {"ok": False, "state": f"not answering ({e})"}
        for name, cam in (("microscope", self.cam), ("bench_camera", self.bench)):
            if cam is None:
                out[name] = {"ok": False, "state": "not connected"}
                continue
            try:
                frame = cam.frame()
                out[name] = {"ok": frame is not None and float(frame.std()) > 2, "state":
                             "giving pictures" if frame is not None and float(frame.std()) > 2 else "black picture"}
            except Exception as e:
                out[name] = {"ok": False, "state": str(e)}
        out["all_ok"] = all(v.get("ok") for v in out.values() if isinstance(v, dict))
        return out, None

    def _system_restart(self, part):
        from lunes import simulator
        if part == "simulator":
            ended = simulator.kill_strays()
            return {"ok": True, "hanging_runs_ended": ended, "selftest": simulator.selftest()}, None
        if part == "scope":
            from lunes.hantek import Scope, SimulatedScope
            if isinstance(self.scope, SimulatedScope):
                return {"ok": True, "note": "the simulated scope needs no restart"}, None
            probe = dict(self.scope.probe) if self.scope else self.m.config.get("scope", {}).get("probe", {})
            try:
                if self.scope is not None:
                    self.scope.close()
            except Exception:
                pass
            try:
                self.scope = Scope(probe=probe)
                return {"ok": True, "state": "open again", "info": self.scope.info()}, None
            except RuntimeError as e:
                self.scope = None
                return {"ok": False, "error": str(e)}, None
        if part == "kinect":
            if self.kinect is None:
                from lunes.kinect import Kinect
                try:
                    self.kinect = Kinect()
                except RuntimeError as e:
                    return {"ok": False, "error": str(e)}, None
            else:
                try:
                    self.kinect.reconnect()
                except RuntimeError as e:
                    return {"ok": False, "error": str(e)}, None
            return {"ok": True, "state": "open again", **self.kinect.info()}, None
        if part == "cnc_alarm":
            try:
                self.m.reset_alarm()
                return {"ok": True, "state": "alarm cleared; the machine did not move",
                        "position_doubtful": True,
                        "note": "after an alarm the position may be off: check_position before fine work"}, None
            except Exception as e:
                return {"ok": False, "error": str(e)}, None
        return {"ok": False, "error": "part must be simulator, scope, kinect or cnc_alarm"}, None

    def _diagnose_start(self):
        from lunes.diagnosis import Diagnosis
        if self.circuit is None:
            return {"ok": False, "error": "no circuit yet: circuit_set first"}, None
        if self._table_thread is not None:
            self._table_thread.join()
        if self.fault_table is None:
            return {"ok": False, "error": "the simulator could not work out the faults of this circuit"}, None
        self.diagnosis = Diagnosis(self.circuit, self.fault_table)
        out = {"ok": True, "hypotheses": len(self.fault_table), **self.diagnosis.state()}
        blind = {h: e.get("problem", "") for h, e in self.fault_table.items() if e.get("nets") is None}
        if blind:
            out["not_predicted"] = {h: f"{self.fault_table[h]['says']}: {why}" for h, why in blind.items()}
            out["not_predicted_note"] = ("the simulator could not work these faults out, so the engine cannot "
                                         "confirm or rule them out by measuring")
        return out, None

    def _diagnose_measure(self, net, source="scope", mean_v=None):
        if self.diagnosis is None:
            return {"ok": False, "error": "no diagnosis running: diagnose_start first"}, None
        view = None
        if source == "multimeter":
            if mean_v is None:
                return {"ok": False, "error": "give mean_v, the volts the multimeter shows"}, None
            reading = {"mean_v": float(mean_v)}
        else:
            result, view = self._scope_measure("CH1")
            if not result.get("ok"):
                return result, view
            m = result["CH1"]
            reading = {"mean_v": m["mean_v"], "vpp_v": m["vpp_v"], "frequency_hz": m.get("frequency_hz"),
                       "duty": m.get("duty")}
            if m.get("notes"):
                reading["scope_notes"] = m["notes"]
        notes = reading.pop("scope_notes", None)
        try:
            self.diagnosis.update(net, reading)
        except ValueError as e:
            return {"ok": False, "error": str(e)}, view
        out = {"ok": True, "net": net, "reading": reading, **self.diagnosis.state()}
        if notes:
            out["scope_notes"] = notes
        if getattr(self.scope, "signals", None) is not None and source == "scope":
            out["simulated"] = "the reading came from the simulated scope, not the bench"
        return out, view

    def _scope_probe(self, channel, factor):
        """What Erlan set on a probe's switch, kept in config/machine.json for the next session."""
        from lunes.hantek import ranges
        if self.scope is None:
            return {"ok": False, "error": "the oscilloscope is not connected"}, None
        if channel not in ("CH1", "CH2") or float(factor) not in (1.0, 10.0):
            return {"ok": False, "error": "channel CH1 or CH2, factor 1 or 10"}, None
        self.scope.probe[channel] = float(factor)
        self.scope.range_v[channel] = ranges(float(factor))[0]
        self.m.config.setdefault("scope", {}).setdefault("probe", {})[channel] = float(factor)
        self.m.config_path.write_text(json.dumps(self.m.config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"ok": True, "probe": dict(self.scope.probe)}, None

    def _center_on_corner(self, x_side=None, y_side=None, tolerance_mm=0.05, tries=4):
        """Puts one corner of the board in the middle of the microscope's view. The corner is found by the code
        from the board's edges in each picture, moved to, and checked with a last picture."""
        vx, vy, _, _ = geometry(self.m.config["microscope"])
        here = self.m.position()
        box = self.board["box_mm"] if self.board else None
        if x_side in ("+", "-") and y_side in ("+", "-"):
            sx, sy = (1 if x_side == "+" else -1), (1 if y_side == "+" else -1)
        elif box:  # the measured board's corner nearest to the microscope
            sx = 1 if abs(here["X"] - box["X"][1]) < abs(here["X"] - box["X"][0]) else -1
            sy = 1 if abs(here["Y"] - box["Y"][1]) < abs(here["Y"] - box["Y"][0]) else -1
        else:
            return {"ok": False, "error": "say which corner (x_side and y_side), or measure_board first"}, None
        which = f"X{'+' if sx > 0 else '-'} Y{'+' if sy > 0 else '-'}"
        if box:  # start over that corner of the measured board, unless the microscope is already near it
            near = (box["X"][0 if sx < 0 else 1], box["Y"][0 if sy < 0 else 1])
            if math.hypot(near[0] - here["X"], near[1] - here["Y"]) > 4:
                moved = self._move(*near)
                if not moved["ok"]:
                    return moved, None
        if self.focus_z is not None:  # crisp edges
            self.m.move_to(Z=self.focus_z)
        corrections = []
        for attempt in range(tries + 1):
            frame = self._look()
            at = self.m.position()
            corner, why = board_corner_in_picture(frame, at, vx, vy, sx, sy)
            if corner is None:
                return {"ok": False, "corner": which, "error": f"I cannot place the corner: {why}",
                        "position_mm": _mm(at), "corrections_mm": corrections}, with_cross(frame)
            d = np.array([corner["X"] - at["X"], corner["Y"] - at["Y"]])
            off = float(np.hypot(*d))
            mark = (frame.shape[1] / 2, frame.shape[0] / 2) - (vx * d[0] + vy * d[1])  # the corner's pixel
            if off <= tolerance_mm or attempt == tries:
                return {"ok": off <= 3 * tolerance_mm, "corner": which,
                        "corner_mm": {"X": round(corner["X"], 2), "Y": round(corner["Y"], 2)},
                        "off_centre_mm": round(off, 3), "position_mm": _mm(at), "corrections_mm": corrections,
                        "edges_seen_mm": corner["edges_seen_mm"], "out_of_square_deg": corner["out_of_square_deg"],
                        "note": "checked with this last picture: the cross is on the board's corner (the ring is "
                                "where the code found it)"}, with_cross(frame, mark)
            moved = self._move(corner["X"], corner["Y"])
            if not moved["ok"]:
                return moved, with_cross(frame, mark)
            corrections.append(round(off, 3))

    def find_in_mosaic(self, concept):
        """Every instance of a concept in the mosaic of the face that is up, found by SAM 3.1; returns (result, the
        mosaic with the boxes drawn, for Erlan's live page)."""
        if self.spark is None:
            return {"ok": False, "error": "SAM 3.1 is a Meta API model: not in private mode (without --spark)"}, None
        if not concept:
            return {"ok": False, "error": "say what to find, as a short English noun phrase"}, None
        face_up = [v for v in self.surveys if v["face"] == self.side]
        if self.last_mosaic is None and face_up:  # the latest mosaic of this face, as go_to_mosaic_point uses
            self.last_survey, self.last_mosaic = face_up[-1]["summary"], cv2.imread(face_up[-1]["summary"]["mosaic"])
        if self.last_mosaic is None:
            return {"ok": False, "error": "no mosaic of the face that is up yet: survey_board first"}, None
        mosaic = self.last_mosaic
        h, w = mosaic.shape[:2]
        found = sorted(self.spark.segment(mosaic, concept), key=lambda b: -b[4])[:20]
        view = mosaic.copy()
        items = []
        for n, (x1, y1, x2, y2, c) in enumerate(found, 1):
            items.append({"n": n, "x": round((x1 + x2) / 2 / w, 3), "y": round((y1 + y2) / 2 / h, 3),
                          "width": round((x2 - x1) / w, 3), "height": round((y2 - y1) / h, 3),
                          "confidence": round(c, 2)})
            cv2.rectangle(view, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), max(3, w // 400))
            cv2.putText(view, str(n), (int(x1), max(30, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, w / 900,
                        (0, 255, 255), max(2, w // 500))
        result = {"ok": True, "concept": concept, "found": items,
                  "note": "" if items else "SAM found none: try another word, or look at the mosaic yourself"}
        return result, view

    def _find_in_picture(self, color, concept, where, head=None):
        """Every instance of a concept in one picture, found by SAM 3.1, where it is in the picture (0 to 1); head:
        where the microscope's ring light is in it, to say whether a thing is under it."""
        h, w = color.shape[:2]
        view, items = color.copy(), []
        thick, letters = max(2, w // 640), w / 1100
        for n, (x1, y1, x2, y2, c) in enumerate(sorted(self.spark.segment(color, concept), key=lambda b: -b[4])[:12], 1):
            item = {"n": n, "x": round((x1 + x2) / 2 / w, 2), "y": round((y1 + y2) / 2 / h, 2),
                    "confidence": round(c, 2)}
            if head:
                item["microscope_ring_inside_box"] = bool(x1 <= head[0] <= x2 and y1 <= head[1] <= y2 + 40)
            items.append(item)
            cv2.rectangle(view, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), thick)
            cv2.putText(view, str(n), (int(x1), max(14, int(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, letters,
                        (0, 255, 255), thick)
        return {"ok": True, "concept": concept, "found": items,
                "note": f"x, y are where it is in the {where} picture (0 to 1), not machine positions"
                if items else "SAM found none: try another word"}, view

    def bench_picture(self):
        """(a fresh picture of the bench camera, None) or (None, why there is none). A flat picture is checked here,
        not left to the model: the Sony sends black frames with its lens cap on or while it shuts down."""
        if self.bench is None:
            return None, "the bench camera is not connected (the Sony, in USB Streaming mode)"
        try:
            frame = self.bench.frame()
        except RuntimeError:
            return None, "the bench camera stopped sending pictures: the Sony may have turned off; tell Erlan"
        if frame.std() < 2:
            return None, ("the bench camera sends a flat black picture: its lens cap may be on, or the camera turned "
                          "off; tell Erlan")
        return frame, None

    def find_in_bench(self, concept):
        """Every instance of a concept in a fresh picture of the bench camera, found by SAM 3.1."""
        if self.spark is None:
            return {"ok": False, "error": "SAM 3.1 needs the Meta API (--spark)"}, None
        if not concept:
            return {"ok": False, "error": "say what to find, as a short English noun phrase"}, None
        frame, problem = self.bench_picture()
        if problem:
            return {"ok": False, "error": problem}, None
        return self._find_in_picture(frame, concept, "bench camera")

    def find_in_kinect(self, concept):
        """Every instance of a concept in a fresh Kinect picture, found by SAM 3.1, where it is in the picture (no
        machine mm: the Kinect calibration only holds at the head's height; on the bed it was 40-50 mm off)."""
        if self.spark is None or self.kinect is None:
            return {"ok": False, "error": "needs the Kinect and the Meta API (--spark)"}, None
        if not concept:
            return {"ok": False, "error": "say what to find, as a short English noun phrase"}, None
        color = self.kinect.color()
        return self._find_in_picture(color, concept, "Kinect", head=self.kinect_head())

    def record_video(self, seconds, source="microscopio"):
        """Records the microscope (15 frames a second), the bench camera (15) or the Kinect (5) into
        captures/videos/."""
        seconds = max(1.0, min(float(seconds), 60.0))
        if source == "kinect" and self.kinect is None:
            return {"ok": False, "error": "the Kinect is not connected"}
        if source == "banco" and self.bench is None:
            return {"ok": False, "error": "the bench camera is not connected"}
        if source == "kinect":
            grab, fps = self.kinect.color, 5
        else:
            grab, fps = (self.bench if source == "banco" else self.cam).frame, 15
        folder = ROOT / "captures" / "videos"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{datetime.now():%Y%m%d_%H%M%S}_{source}.mp4"
        first = grab()
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (first.shape[1], first.shape[0]))
        t0, frames = time.time(), 0
        try:
            while time.time() - t0 < seconds and not self.stop.is_set():
                time.sleep(max(0.0, t0 + frames / fps - time.time()))
                writer.write(first if frames == 0 else grab())
                frames += 1
        finally:
            writer.release()
        self.last_video = path
        return {"ok": True, "file": str(path), "seconds": round(time.time() - t0, 1), "frames": frames, "fps": fps}

    def track_in_video(self, concept):
        """SAM 3.1 follows a concept through the last recorded video; writes <video>_sam.mp4 with the boxes."""
        path = getattr(self, "last_video", None)
        if self.spark is None:
            return {"ok": False, "error": "SAM 3.1 needs the Meta API (--spark)"}, None
        if path is None or not Path(path).exists():
            return {"ok": False, "error": "no video recorded yet: record_video first"}, None
        tracks = self.spark.segment_video(path, concept)
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 10
        out_path = Path(path).with_name(Path(path).stem + "_sam.mp4")
        writer, n, objects, still = None, 0, set(), None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if writer is None:
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (frame.shape[1], frame.shape[0]))
            for i, x1, y1, x2, y2, c in tracks.get(n, []):
                objects.add(i)
                color = [(0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 128, 0)][i % 4]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{concept} {i}", (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            if tracks.get(n) and still is None:
                still = frame.copy()
            writer.write(frame)
            n += 1
        cap.release()
        if writer is not None:
            writer.release()
        return {"ok": True, "concept": concept, "frames": n, "frames_with_it": len(tracks), "objects": len(objects),
                "file": str(out_path)}, still

    # Position seen by the Kinect -------------------------------------------------------------------------------
    def kinect_head(self):
        """(column, row, distance) of the microscope head in a fresh Kinect picture, or None."""
        from lunes.head import find_ring
        color = self.kinect.color()
        return find_ring(color, self.kinect.depth())

    def check_position(self):
        """Where the Kinect sees the head against where Lunes thinks the machine is. Agreeing clears the doubt left by
        released motors; disagreeing sets it, and Lunes tells Erlan instead of moving on its own."""
        from lunes.head import predict
        cal = self.m.config.get("kinect_head")
        if self.kinect is None:
            return {"ok": False, "error": "the Kinect is not connected"}
        if not cal:
            return {"ok": False, "error": "the Kinect is not calibrated yet: calibrate_kinect, when Erlan asks"}
        seen = self.kinect_head()
        if seen is None:
            return {"ok": False, "error": "the Kinect does not find the microscope's ring light (on, and in view?)"}
        seen_mm, here = predict(cal["map"], seen), self.m.position()
        if any(not (WORKSPACE[a][0] - 20 <= seen_mm[a] <= WORKSPACE[a][1] + 20) for a in "XYZ"):
            # Low over the board the ring's reflection on it can pass for the ring: once it read Z -34 at Z 1.9.
            # An impossible reading neither confirms nor contradicts anything.
            return {"ok": False, "error": "the Kinect reading is impossible (outside the machine): it probably took "
                                          "the ring light's reflection for the ring; position neither confirmed nor "
                                          "contradicted", "kinect_mm": _mm(seen_mm)}
        difference = {k: round(seen_mm[k] - here[k], 1) for k in "XYZ"}
        tolerance = max(5.0, 3 * cal["typical_error_mm"])
        agrees = all(abs(v) <= tolerance for v in difference.values())
        if not agrees:  # one look can catch a reflection or a hand in the way: look again before saying so
            time.sleep(0.5)
            again = self.kinect_head()
            if again is not None:
                second = predict(cal["map"], again)
                difference = {k: round(second[k] - here[k], 1) for k in "XYZ"}
                agrees, seen_mm = all(abs(v) <= tolerance for v in difference.values()), second
        self.position_doubtful = not agrees
        self.last_kinect_check = {"at": time.time(), "agrees": agrees,
                                  "off_mm": max(abs(v) for v in difference.values())}
        return {"ok": True, "agrees": agrees, "kinect_mm": _mm(seen_mm), "believed_mm": _mm(here),
                "difference_mm": difference, "tolerance_mm": tolerance,
                "note": "" if agrees else "the Kinect sees the head elsewhere: tell Erlan and do not move on your "
                                          "own. Either the carriage moved (measure_board with new_board=true finds "
                                          "where the board really is) or the Kinect itself was moved (then, if Erlan "
                                          "says the carriage did not move, calibrate_kinect). He decides which"}

    def calibrate_kinect(self):
        """Moves around where the machine is, finds the head with the Kinect at each place, and fits the map from
        what the Kinect sees to machine millimetres. Comes back to where it started."""
        from lunes.head import CALIBRATION_OFFSETS, fit
        if self.kinect is None:
            return {"ok": False, "error": "the Kinect is not connected"}
        start, points, skipped = self.m.position(), [], []
        try:
            for dx, dy, dz in CALIBRATION_OFFSETS:
                if self.stop.is_set():
                    raise Interrupted()
                moved = self._move(start["X"] + dx, start["Y"] + dy, start["Z"] + dz)
                if not moved["ok"]:
                    skipped.append(moved["error"])
                    continue
                time.sleep(0.5)  # the carriage settles and the Kinect sends a picture taken after the move
                seen = self.kinect_head()
                if seen is None:
                    skipped.append(f"no ring at {dx:+} {dy:+} {dz:+} mm")
                    continue
                points.append((seen, self.m.position()))
                self.show(self.cam.frame(), f"calibrando el Kinect {len(points)} de {len(CALIBRATION_OFFSETS)}")
        finally:
            self._move(start["X"], start["Y"], start["Z"])
        if len(points) < 6:
            return {"ok": False, "error": f"the Kinect found the head in {len(points)} of {len(CALIBRATION_OFFSETS)} "
                                          f"positions", "skipped": skipped}
        mapping, errors = fit(points)
        typical = round(float(np.median(errors)), 1)
        self.m.config["kinect_head"] = {
            "map": mapping, "typical_error_mm": typical, "points": len(points),
            "at": datetime.now().isoformat(timespec="minutes"),
            "note": "linear map from where the Kinect sees the ring light (column, row, distance mm) to machine mm; "
                    "made by calibrate_kinect"}
        self.m.config_path.write_text(json.dumps(self.m.config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"ok": True, "points": len(points), "error_mm": errors, "typical_error_mm": typical,
                "skipped": skipped}

    # Skills ------------------------------------------------------------------------------------------------
    def _calibrate_here(self, z):
        """Millimetres per pixel at this board's height: the lens magnifies a few per cent more or less with its
        distance to the board, which would make every size and mosaic that much off."""
        def lifted(axis, mm):
            hop(self.m, z, **{axis: self.m.position()[axis] + mm})

        try:
            calibrate(self.m, self.cam, lifted)
            self.m.move_to(Z=z)
        except Exception:  # bare solder mask has nothing to track: keep the last calibration
            pass

    def relocate(self):
        """Finds where the microscope really is by looking for what it sees inside the last mosaic of the face of
        the board that is up, and corrects the machine's position. Returns the correction, or None."""
        face_up = [v for v in self.surveys if v["face"] == self.side]
        if not face_up:
            return None
        summary = face_up[-1]["summary"]
        mosaic = cv2.imread(summary["mosaic"])
        if mosaic is None:
            return None
        self.cam.fresh_gray(6)  # let the exposure settle first
        found = locate_in_mosaic(self.cam.frame(), mosaic, summary)
        if not found:
            self.board_may_have_changed()
            return None
        (x, y), score, second = found
        here = self.m.position()
        error = math.hypot(x - here["X"], y - here["Y"])
        # Only small corrections of a clear match: a machine does not jump centimetres by itself, a wrong match
        # on a repeating board does (it once moved the position 71 mm while nothing had moved).
        clear = score >= 0.5 and score - second >= 0.08
        correct = clear and 1.0 < error <= 15.0
        if correct:
            self.m.set_position(X=x, Y=y)
        if not correct and error > 1.0:
            self.board_may_have_changed()
        return {"was_mm": [round(here["X"], 2), round(here["Y"], 2)], "seen_mm": [round(x, 2), round(y, 2)],
                "corrected": correct, "score": round(score, 2), "next_best": round(second, 2),
                "note": "" if correct or error <= 1.0 else "not corrected: the match is not clear or the jump too big"}

    def board_may_have_changed(self):
        """Remembers that what the microscope saw at start did not fit the last mosaic; look_survey says so. Nothing
        is started because of it: Lunes only acts when Erlan asks."""
        self.mosaic_doubtful = True

    def _measure(self, new_board=False):
        above = 0.5
        if not new_board and self.board:
            return dict(self.board, note="already measured and the board has not changed; if Erlan changed, moved "
                                         "or flipped it, call measure_board with new_board=true")
        if new_board:  # another board, or this one flipped: its surface may be higher or lower than the last one
            if self.surveys:
                self.side += 1
            self.focus_z, self.board, self.last_survey, self.last_mosaic = None, None, None, None
            above = 8.0
            self.m.move_to(Z=min(self.m.position()["Z"] + above, WORKSPACE["Z"][1]))
            above = 0.0
        z = self._focus_board(above)
        self._calibrate_here(z)
        found = find_perimeter(self.m, self.cam, z, self.m.config["microscope"], WORKSPACE, self.show,
                               stop=self.stop.is_set)
        if found["found"]:
            self.board = found
            found["note"] = "the microscope is now over the board's centre; tell Erlan the size"
            self.position_doubtful = False  # the board's edges place the machine again
        else:
            found["note"] = "no board under the microscope nor around it: look with the bench camera or ask Erlan where it is"
        return found

    def _survey_board(self, width_mm=None, height_mm=None):
        cal = self.m.config["microscope"]
        if width_mm and height_mm:
            here = self.m.position()
            center, size = (here["X"], here["Y"]), (float(width_mm), float(height_mm))
        else:
            if not self.board:
                found = self._measure()
                if not found["found"]:
                    return {"ok": False, "error": "no board found under or around the microscope", "measure": found}, None
            center = tuple(self.board["center_mm"])
            size = tuple(s + 2 * BOARD_MARGIN_MM for s in self.board["size_mm"])
        cols, rows = grid_for(*size, cal)
        note = None
        if cols * rows > MAX_TILES:
            k = math.sqrt(MAX_TILES / (cols * rows))
            cols, rows = max(1, int(cols * k)), max(1, int(rows * k))
            note = "the board is bigger than one survey can cover: only its central part was photographed"
        moved = self._move(*center)
        if not moved["ok"]:
            return moved, None
        self._focus_board()
        mosaic, _, summary = survey(self.m, self.cam, span=0.3, show=self.show, recalibrate=False, cols=cols,
                                    rows=rows, stop=self.stop.is_set)
        self.focus_z = summary["focus_z_mm"] if self.focus_z is None else min(self.focus_z, summary["focus_z_mm"])
        self.last_survey, self.last_mosaic = summary, mosaic
        result = {k: v for k, v in summary.items() if k not in BOOKKEEPING}
        result["pictures"] = cols * rows
        result["survey_number"] = len(self.surveys) + 1
        result["face"] = f"face {self.side} of this board (face 2 is the other side after Erlan flipped it)"
        self.surveys.append({"number": result["survey_number"], "face": self.side, "summary": summary,
                             "board_size_mm": self.board["size_mm"] if self.board else None,
                             "at": datetime.now().strftime("%Y-%m-%d %H:%M")})
        SURVEYS.write_text(json.dumps({"surveys": self.surveys, "side": self.side}, default=float), encoding="utf-8")
        if self.board and not (width_mm and height_mm):
            result["board_size_mm"] = self.board["size_mm"]
        if note:
            result["note"] = note
        return result, mosaic

    def _inspect(self, x, y):
        if self.last_mosaic is None:
            return {"ok": False, "error": "no mosaic yet: call survey_board first"}, None
        s, mosaic = self.last_survey, self.last_mosaic
        H, W = mosaic.shape[:2]
        ppm = s["px_per_mm"]
        ppm = float(np.mean(list(ppm.values()) if isinstance(ppm, dict) else ppm))
        box = component_box(mosaic, x, y)
        if box is None:  # no part body found there: take 12 mm around the point
            half = 6.0 * ppm
            x0, y0, x1, y1 = x - half / W, y - half / H, x + half / W, y + half / H
        else:
            x0, y0, x1, y1 = box
            run_x, run_y = PART_RUNS_OFF_MM * ppm / W, PART_RUNS_OFF_MM * ppm / H
            x0, y0 = x0 - (run_x if x0 <= 0.01 else 0), y0 - (run_y if y0 <= 0.01 else 0)
            x1, y1 = x1 + (run_x if x1 >= 0.99 else 0), y1 + (run_y if y1 >= 0.99 else 0)
        mx, my = PART_MARGIN_MM * ppm / W, PART_MARGIN_MM * ppm / H
        corners = [mosaic_point_to_mm(s, a, b) for a in (x0 - mx, x1 + mx) for b in (y0 - my, y1 + my)]
        xs, ys = [c[0] for c in corners], [c[1] for c in corners]
        center = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
        size = (max(xs) - min(xs), max(ys) - min(ys))
        cols, rows = grid_for(*size, self.m.config["microscope"])
        while cols * rows > MAX_PART_TILES:
            cols, rows = (cols - 1, rows) if cols >= rows else (cols, rows - 1)
        moved = self._move(*center, z=self.focus_z)
        if not moved["ok"]:
            return moved, None
        top_z, _ = self._focus_up(5.0)
        close, folder, _ = survey(self.m, self.cam, span=0.15, show=self.show, recalibrate=False, cols=cols,
                                  rows=rows, stop=self.stop.is_set, register=True)
        image = closeup_for_reading(close)
        cv2.imwrite(str(folder / "closeup.png"), image)
        self.m.move_to(Z=max(top_z, self._z_floor()) + LIFT_MM)  # stay clear of the part afterwards
        result = {}
        if self.brain == "glimmer" and self.spark:  # Glimmer gets a second pair of eyes; Spark reads it itself
            result["muse_spark_reading"] = self._spark(SPARK_READ, image, "read a part's marking", effort="low")
        return {**result, "part_found_in_mosaic": box is not None,
                "part_size_mm": [round(size[0] - 2 * PART_MARGIN_MM, 1), round(size[1] - 2 * PART_MARGIN_MM, 1)],
                "center_mm": [round(center[0], 2), round(center[1], 2)],
                "top_height_above_board_mm": round(top_z - self._z_floor(), 1), "pictures": cols * rows,
                "note": "two copies of the part, the second turned upside down: read the marking in whichever is "
                        "upright, character by character, and say which characters are not clear"}, image

    def _faces_side_by_side(self, listing):
        """The last mosaic of each face of the board, next to each other at the same height."""
        last = {}
        for v in self.surveys:
            last[v["face"]] = v
        pictures = []
        for face in sorted(last):
            img = cv2.imread(last[face]["summary"]["mosaic"])
            img = cv2.resize(img, None, fx=900 / img.shape[0], fy=900 / img.shape[0], interpolation=cv2.INTER_AREA)
            label = np.full((50, img.shape[1], 3), 255, np.uint8)
            cv2.putText(label, f"cara {face}", (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
            pictures.append(np.vstack([label, img]))
            pictures.append(np.full((950, 20, 3), 255, np.uint8))
        result = {"faces": [{"face": f, "survey": last[f]["number"], "board_size_mm": last[f]["board_size_mm"],
                             "taken_at": last[f]["at"]} for f in sorted(last)], "face_up_now": self.side,
                  "surveys": listing}
        if self.mosaic_doubtful:
            result["warning"] = ("at start the microscope did not find itself in the last mosaic: the board may have "
                                 "been moved or changed since")
        return result, np.hstack(pictures[:-1])

    def _spark(self, question, image, purpose, effort="low"):
        """Muse Spark's answer about an image, or why there is none (no key, no network, budget spent)."""
        if self.spark is None:
            return "(Muse Spark is off: Erlan has not allowed sending pictures, or there is no key or network)"
        k = min(1.0, 1280 / max(image.shape[:2]))
        small = cv2.resize(image, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        try:
            return self.spark.ask(question, [small], system=SPARK_SYSTEM, purpose=purpose, effort=effort,
                                  max_tokens=4000)
        except Exception as e:
            return f"(Muse Spark could not answer: {e})"

    # Tools -------------------------------------------------------------------------------------------------
    def run_tool(self, name, args):
        """Runs one tool; returns (result for the model, image to show the model or None)."""
        if name == "where":
            return {"position_mm": _mm(self.m.position()), "focus_z_mm": self.focus_z, "area_mm": WORKSPACE,
                    "board": self.board and {k: self.board[k] for k in ("size_mm", "center_mm")}}, None
        if name == "learn_direction":
            axis = str(args.get("axis", "")).upper()
            if axis not in ("X", "Y", "Z"):
                return {"ok": False, "error": "axis must be X, Y or Z"}, None
            sides = {"adelante": "atrás", "atrás": "adelante", "izquierda": "derecha", "derecha": "izquierda",
                     "arriba": "abajo", "abajo": "arriba"}
            plus = str(args.get("positive_is", "")).strip().lower()
            minus = sides.get(plus, "el lado contrario")
            self.m.config.setdefault("como_lo_ve_erlan", {})[axis] = (
                f"+{axis} va hacia {plus} y -{axis} hacia {minus}, como lo dice Erlan "
                f"({datetime.now():%d/%m}: {args.get('evidence', '')})")
            self.m.config_path.write_text(json.dumps(self.m.config, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
            return {"ok": True, "saved": self.m.config["como_lo_ve_erlan"][axis]}, None
        if name == "motors":
            on = bool(args.get("on"))
            self.m.enable(on)
            self.position_doubtful = self.position_doubtful or not on
            if not on:  # kept in the position file too, so a restart does not forget it
                from lunes.machine import save_position
                save_position(self.m.at_mm, "motors released", self.m.direction)
            return {"ok": True, "motors": "holding" if on else "released",
                    "note": "" if on else "the carriage can be moved by hand now; its position is no longer certain"}, None
        if name == "find_in_mosaic":
            return self.find_in_mosaic(str(args.get("concept", "")).strip())
        if name == "find_in_bench":
            return self.find_in_bench(str(args.get("concept", "")).strip())
        if name == "find_in_kinect":
            return self.find_in_kinect(str(args.get("concept", "")).strip())
        if name == "record_video":
            return self.record_video(float(args.get("seconds", 10)), str(args.get("source", "microscopio"))), None
        if name == "track_in_video":
            return self.track_in_video(str(args.get("concept", "")).strip())
        if name == "search_web":
            if self.spark is None:
                return {"ok": False, "error": "no internet in private mode (without --spark)"}, None
            answer, sources = self.spark.search(str(args.get("question", "")))
            return {"ok": True, "answer": answer, "sources": sources[:6]}, None
        if name == "check_position":
            return self.check_position(), None
        if name == "calibrate_kinect":
            return self.calibrate_kinect(), None
        if name in ("move_to", "move_by"):
            here = self.m.position()
            if name == "move_to":
                result = self._move(args.get("x"), args.get("y"), args.get("z"))
            else:
                result = self._move(here["X"] + float(args.get("dx", 0)), here["Y"] + float(args.get("dy", 0)),
                                    here["Z"] + float(args.get("dz", 0)))
            # every move is checked once with the Kinect; a disagreement is reported, never acted on by itself
            if result.get("ok") and self.kinect is not None and self.m.config.get("kinect_head"):
                result["kinect_check"] = self.check_position()
            return result, None
        if name == "measure_hole":
            return self._measure_hole(float(args.get("x", -1)), float(args.get("y", -1)))
        if name == "measure_distance":
            return self._measure_distance(*(float(args.get(k, -1)) for k in ("x1", "y1", "x2", "y2")))
        if name == "scope_measure":
            return self._scope_measure(str(args.get("channel", "CH1")), bool(args.get("auto", True)),
                                       args.get("range_v"), args.get("rate"))
        if name == "scope_selftest":
            return self._scope_selftest(str(args.get("channel", "CH1")))
        if name == "scope_probe":
            return self._scope_probe(str(args.get("channel", "")), args.get("factor", 0))
        if name == "circuit_set":
            return self._circuit_set(args.get("circuit") or {}, args.get("spec"))
        if name == "circuit_design":
            return self._circuit_design(str(args.get("template", "")), args.get("params") or {})
        if name == "circuit_guide":
            return self._circuit_guide(args.get("topic", ""))
        if name == "system_check":
            return self._system_check()
        if name == "system_restart":
            return self._system_restart(str(args.get("part", "")))
        if name == "diagnose_start":
            return self._diagnose_start()
        if name == "diagnose_measure":
            return self._diagnose_measure(str(args.get("net", "")), str(args.get("source", "scope")),
                                          args.get("mean_v"))
        if name == "scope_test_signal":
            if self.scope is None:
                return {"ok": False, "error": "the oscilloscope is not connected"}, None
            try:
                return {"ok": True, "hz": self.scope.calibration_output(float(args.get("hz", 1000))),
                        "note": "a square wave of about 2 V on the scope's calibration pin"}, None
            except ValueError as e:
                return {"ok": False, "error": str(e)}, None
        if name in ("center_on_point", "center_on_corner"):
            if name == "center_on_point":
                result, picture = self._center_on_point(float(args.get("x", -1)), float(args.get("y", -1)))
            else:
                result, picture = self._center_on_corner(args.get("x_side"), args.get("y_side"))
            if "position_mm" in result and self.kinect is not None and self.m.config.get("kinect_head"):
                result["kinect_check"] = self.check_position()  # like every move: checked once, never acted on
            return result, picture
        if name == "look_microscope":
            frame = self._look()
            s = sharpness(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
            self.show(frame, f"microscopio  nitidez {s:.0f}")
            return {"position_mm": _mm(self.m.position()), "sharpness": round(s),
                    "note": "below about 300 the image is blurry; it shows about 12 x 15 mm"}, frame
        if name == "look_bench":
            frame, problem = self.bench_picture()
            if problem:
                return {"ok": False, "error": problem}, None
            return {"note": "the whole work area, now"}, frame
        if name == "look_kinect":
            if self.kinect is None:
                return {"ok": False, "error": "the Kinect is not connected"}, None
            frame, depth = self.kinect.color(), self.kinect.depth()
            return {"note": "left: colour; right: depth, with the distance from the Kinect in mm written at nine "
                            "points (black where it could not measure). The microscope is the black tube with a lit "
                            "ring at its tip."}, kinect_view(frame, depth)
        if name == "kinect_distance":
            if self.kinect is None:
                return {"ok": False, "error": "the Kinect is not connected"}, None
            depth = self.kinect.depth()
            h, w = depth.shape
            x, y = int(float(args["x"]) * (w - 1)), int(float(args["y"]) * (h - 1))
            patch = depth[max(0, y - 4):y + 5, max(0, x - 4):x + 5]
            good = patch[patch > 0]
            if not good.size:
                return {"ok": False, "error": "the Kinect could not measure that point (too near, shiny or dark)"}, None
            return {"distance_mm": int(np.median(good)), "spread_mm": int(good.max() - good.min()),
                    "note": "straight-line distance from the Kinect, not a bench coordinate"}, None
        if name == "autofocus":
            up = float(min(max(args.get("up_mm", 0.3), 0.3), 5.0))
            if up <= 0.3:
                z, s = autofocus(self.m, self.cam, 0.3, lambda t: self.show(self.cam.frame(), t))
            else:
                z, s = self._focus_up(up)
            # The floor stays at the board's focus height: focusing on the top of a tall part never lowers it.
            self.focus_z = z if self.focus_z is None else min(self.focus_z, z)
            return {"focused_z_mm": round(z, 3), "sharpness": round(s), "z_floor_mm": round(self.focus_z, 3)}, None
        if name == "measure_board":
            return self._measure(bool(args.get("new_board"))), None
        if name == "survey_board":
            return self._survey_board(args.get("width_mm"), args.get("height_mm"))
        if name == "inspect_component":
            return self._inspect(float(args["x"]), float(args["y"]))
        if name == "look_survey":
            listing = [{"number": v["number"], "face": v["face"], "board_size_mm": v["board_size_mm"], "at": v["at"]}
                       for v in self.surveys]
            if "number" not in args or not self.surveys:
                return {"surveys": listing}, None
            n = int(args["number"])
            if n == 0:
                return self._faces_side_by_side(listing)
            try:
                v = self.surveys[n - 1 if n > 0 else n]
            except IndexError:
                return {"ok": False, "error": f"there is no survey {n}", "surveys": listing}, None
            mosaic = cv2.imread(v["summary"]["mosaic"])
            # pointing at it (go_to_mosaic_point, inspect_component) only makes sense while that face is up
            if v["face"] == self.side:
                self.last_survey, self.last_mosaic = v["summary"], mosaic
            result = {"number": v["number"], "face": v["face"], "board_size_mm": v["board_size_mm"],
                      "face_up_now": v["face"] == self.side, "taken_at": v["at"]}
            if self.mosaic_doubtful:
                result["warning"] = ("at start the microscope did not find itself in this mosaic: the board may have "
                                     "been moved or changed since")
            return result, mosaic
        if name == "ask_spark":
            if self.last_image is None:
                return {"ok": False, "error": "no picture yet to ask about"}, None
            return {"muse_spark_answer": self._spark(args["question"], self.last_image, "question from Lunes"),
                    "note": "this is what Muse Spark says, not something you observed yourself"}, None
        if name == "go_to_mosaic_point":
            if not self.last_survey:
                return {"ok": False, "error": "no mosaic yet: call survey_board first"}, None
            x, y = mosaic_point_to_mm(self.last_survey, float(args["x"]), float(args["y"]))
            result = self._move(x, y)
            if result.get("ok") and self.kinect is not None and self.m.config.get("kinect_head"):
                result["kinect_check"] = self.check_position()  # like every move: checked once, never acted on
            return result, None
        if name == "remember":
            MEMORY.parent.mkdir(exist_ok=True)
            note = {"at": datetime.now().isoformat(timespec="seconds"), "fact": args["fact"],
                    "evidence": args["evidence"]}
            with MEMORY.open("a", encoding="utf-8") as f:
                f.write(json.dumps(note, ensure_ascii=False) + "\n")
            return {"ok": True}, None
        return {"ok": False, "error": f"unknown tool {name}"}, None

    # Brain -------------------------------------------------------------------------------------------------
    def tools(self):
        """The tools the current brain gets: Spark does not need to ask itself."""
        return [t for t in TOOLS if not (self.brain == "spark" and t["function"]["name"] == "ask_spark")]

    def _chat(self, on_text=None):
        t0 = time.time()
        try:
            return self._chat_any(on_text)
        finally:
            self.brain_seconds += time.time() - t0
            self.brain_steps += 1

    def brief(self):
        """What Lunes knows about the bench right now, in a few words, for the conductor."""
        parts = []
        if self.board:
            w, h = self.board["size_mm"]
            parts.append(f"la placa de la CNC está medida: {w:.0f} por {h:.0f} mm")
        if self.surveys:
            parts.append(f"hay {len(self.surveys)} mosaicos guardados; el último es de la cara {self.surveys[-1]['face']}")
        if self.mosaic_doubtful:
            parts.append("al arrancar el microscopio no se ubicó en el último mosaico: la placa pudo cambiar")
        if self.position_doubtful:
            parts.append("los motores se liberaron: si alguien movió la máquina a mano, no sabes bien dónde está")
        return "; ".join(parts) or "todavía no se midió ni se recorrió ninguna placa en esta sesión"

    def _chat_any(self, on_text=None):
        if self.brain == "spark":
            try:
                return self._chat_spark(on_text)
            except Exception as e:  # no internet, the API failing, or today's budget spent: Glimmer takes over
                if self.on_think:
                    self.on_think(f"Muse Spark no respondió ({type(e).__name__}: {str(e)[:120]}); contesta Glimmer")
        return self._chat_glimmer(on_text)

    def _chat_spark(self, on_text=None):
        """One model turn on Muse Spark through Meta's API."""
        system = self.messages[0]["content"]
        messages = [{"role": "system", "content": system}] + to_openai(self.messages[1:])
        tools = [{"type": "function", "function": t["function"]} for t in self.tools()]
        t0 = time.time()
        msg = self.spark.chat(messages, tools, purpose="Lunes brain", effort=SPARK_EFFORT)
        content = msg.content or ""
        if on_text and not self.stop.is_set():
            for sentence in re.split(r"(?<=[.!?])\s+", content.strip()):
                if sentence.strip():
                    on_text(sentence.strip())
        LOG.parent.mkdir(exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"at": datetime.now().isoformat(timespec="seconds"), "model": self.spark.model,
                                "where": "Meta Model API (api.meta.ai)", "seconds": round(time.time() - t0, 2),
                                "effort": SPARK_EFFORT}) + "\n")
        out = {"role": "assistant", "content": content}
        if msg.tool_calls:
            out["tool_calls"] = [{"id": c.id, "function": {"name": c.function.name,
                                                           "arguments": json.loads(c.function.arguments or "{}")}}
                                 for c in msg.tool_calls]
        return out

    def _chat_glimmer(self, on_text=None):
        """One model turn on local Muse Glimmer, streamed. on_text gets each finished sentence as soon as it
        exists. Stops reading when Erlan interrupts."""
        # the same context size and keep-alive as the conductor's, so Ollama never reloads Glimmer between them
        body = json.dumps({"model": self.model, "messages": self.messages, "tools": self.tools(), "stream": True,
                           "think": self.think, "keep_alive": -1,
                           "options": {"num_ctx": 8192, "temperature": 0.3}}).encode()
        req = urllib.request.Request(OLLAMA, body, {"Content-Type": "application/json"})
        t0 = time.time()
        content, calls, pending, last, thought = "", [], "", {}, ""
        with urllib.request.urlopen(req, timeout=300) as r:
            for line in r:
                if self.stop.is_set():
                    break
                if not line.strip():
                    continue
                last = json.loads(line)
                msg = last.get("message", {})
                calls += msg.get("tool_calls") or []
                thought += msg.get("thinking") or ""
                piece = msg.get("content") or ""
                content += piece
                pending += piece
                while on_text:
                    m = SENTENCE_END.search(pending)
                    if not m:
                        break
                    on_text(pending[:m.end()].strip())
                    pending = pending[m.end():]
        if on_text and pending.strip() and not self.stop.is_set():
            on_text(pending.strip())
        if self.on_think and thought.strip():
            self.on_think(thought.strip())
        LOG.parent.mkdir(exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:  # provenance: which model ran where, and how much
            f.write(json.dumps({"at": datetime.now().isoformat(timespec="seconds"), "model": last.get("model"),
                                "where": "local RTX 3090 through Ollama", "seconds": round(time.time() - t0, 2),
                                "think": self.think, "thinking_chars": len(thought),
                                "prompt_tokens": last.get("prompt_eval_count"),
                                "output_tokens": last.get("eval_count")}) + "\n")
        out = {"role": "assistant", "content": content}
        if calls:
            out["tool_calls"] = calls
        return out

    def _trim(self, keep=24):
        """Keeps the conversation small: the system prompt, the last messages, and only the newest image."""
        head, tail = self.messages[:1], self.messages[1:][-keep:]
        while tail and tail[0].get("role") == "tool":  # never start with an orphan tool result
            tail = tail[1:]
        seen_image = False
        for msg in reversed(tail):
            if msg.get("images"):
                if seen_image:
                    msg.pop("images")
                    msg["content"] += " (imagen ya no disponible)"
                seen_image = True
        self.messages = head + tail

    def handle(self, text, on_tool=None, on_text=None):
        """One request from Erlan; runs tools until there is an answer, and returns the whole answer text.
        Setting self.stop (Erlan said "para") ends it at the next step."""
        self.stop.clear()
        self.request, self.status = text, "pensando"
        self.brain_seconds, self.brain_steps = 0.0, 0
        # which way each axis goes as Erlan sees it, read now: a correction he made a minute ago already counts
        sides = "; ".join(f"{a}: {t}" for a, t in self.m.config.get("como_lo_ve_erlan", {}).items())
        self.messages.append({"role": "user", "content": text + (f"\n(Hacia dónde va cada eje, visto desde Erlan: "
                                                                  f"{sides}.)" if sides else "")})
        used_tools, nudged, tiny_moves, said = False, set(), 0, []
        speak = (lambda t: (said.append(t), on_text(t))) if on_text else None
        for _ in range(MAX_ROUNDS):
            if self.stop.is_set():
                break
            self._trim()
            msg = self._chat(speak)
            self.messages.append(msg)
            if self.stop.is_set():
                break
            calls = msg.get("tool_calls") or []
            if not calls:
                answer = (msg.get("content") or "").strip()
                # Models like to announce an action ("voy a tomar una foto") and stop there. That is a promise,
                # not an observation: make it act now, once.
                if not used_tools and "promise" not in nudged and PROMISE.search(answer):
                    nudged.add("promise")
                    self.messages.append({"role": "user", "content": "(Sistema: no anuncies lo que vas a hacer. "
                                          "Llama ahora la herramienta que corresponde y responde con lo que veas.)"})
                    continue
                return " ".join(said) if on_text else answer
            used_tools = True
            for n, call in enumerate(calls):
                call.setdefault("id", f"call_{len(self.messages)}_{n}")
                name = call["function"]["name"]
                args = call["function"].get("arguments") or {}
                if isinstance(args, str):
                    args = json.loads(args or "{}")
                if name == "move_by" and math.hypot(float(args.get("dx", 0)), float(args.get("dy", 0))) < 3:
                    tiny_moves += 1
                if on_tool:
                    on_tool(name, args)
                self.status = name
                try:
                    result, image = self.run_tool(name, args)
                except Interrupted:
                    self.stop.set()
                    result, image = {"ok": False, "error": "Erlan stopped it"}, None
                except Exception as e:  # a failed tool is information for the model, not a crash
                    result, image = {"ok": False, "error": str(e)}, None
                if self.on_result:
                    self.on_result(name, result)
                self.messages.append({"role": "tool", "tool_name": name, "tool_call_id": call["id"],
                                      "content": json.dumps(result, ensure_ascii=False, default=float)})
                if image is not None:
                    self.last_image = image
                    if self.on_image:
                        self.on_image(image, name)
                    self.messages.append({"role": "user", "content": f"(imagen tomada por {name})",
                                          "images": [jpeg_b64(image)]})
                if self.stop.is_set():
                    break
            # Inching along a few tenths of a millimetre never shows a whole part: point to the real skills.
            if tiny_moves >= 2 and "tiny" not in nudged:
                nudged.add("tiny")
                self.messages.append({"role": "user", "content": "(Sistema: te estás moviendo de a poco y el "
                                      "microscopio ve unos 12 x 15 mm. Para ver un componente entero usa "
                                      "inspect_component; para toda la placa, measure_board y survey_board.)"})
        if self.stop.is_set():
            self.messages.append({"role": "user", "content": "(Sistema: Erlan te interrumpió; quedó a medias.)"})
            return "Listo, me detengo."
        return "No alcancé a terminar eso con los pasos que tengo. Pídemelo de otra forma."
