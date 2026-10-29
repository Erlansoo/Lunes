"""The conductor against the phrases in tests/conductor_frases.json: what a good colleague would do with each one.

Run it before trying a change with Erlan; a change is good when it does not break phrases that passed. It uses a fixed
picture of the bench (all senses working, a board not yet measured) and, for some phrases, a job in progress or just
finished.

    python tools/test_conductor.py            (Muse Spark, as Lunes runs with --spark)
    python tools/test_conductor.py glimmer    (local Muse Glimmer only)
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lunes.conductor import Conductor  # noqa: E402

SENSES = ("oído: micrófono AB13X funcionando, te oye bien. posición: el Kinect todavía no está calibrado para confirmarla. microscopio: conectado, imagen nítida (nitidez 2100), mira la "
          "placa que está en la CNC. cámara del banco (una Sony): conectada y apuntando a toda el área de trabajo. Kinect: "
          "conectado, ve el banco desde arriba. CNC: conectada y quieta en X -27, Y 17, "
          "Z 7 mm. Muse Spark: disponible por internet.")
BENCH = ("la placa de la CNC todavía no se midió en esta sesión; lo que ve el microscopio no coincide con el último "
         "mosaico, así que puede ser otra placa")
STATES = {
    None: f"nada en curso; {BENCH}.",
    "trabajando": f"trabajo en curso desde hace 40 s: «recorrer la placa con el microscopio»; paso actual: recorro la "
                  f"placa (foto 14 de 36); lo que ya encontró el equipo: Ya la medí: 43 por 48 milímetros.; la placa de "
                  f"la CNC está medida: 43 por 48 mm.",
    "terminado": "nada en curso; el último trabajo fue «revisar la placa» y su resultado: Es un panel de cuatro módulos "
                 "cargadores TP4056 con protección; cada módulo tiene un CSM4056, un DW01A y un doble MOSFET 8205A; no "
                 "vi soldaduras dañadas. La placa de la CNC está medida: 43 por 48 mm.",
}
ABOUT = ("Con qué funcionas: tu oído es Omnilingual ASR de Meta y tu voz es MMS-TTS de Meta, los dos en esta PC; tú "
         "piensas con Muse Spark por la API de Meta (Muse Glimmer en esta PC si no hay internet); el equipo que trabaja "
         "usa Muse Spark. Tu memoria, Hume, guarda en esta PC lo que observaste.")


def check(case, say, action):
    """The reasons this answer is not what a good colleague would do (empty when it is)."""
    wrong = []
    if action not in case["acciones"]:
        wrong.append(f"hizo «{action}», valía {'/'.join(case['acciones'])}")
    if case.get("pregunta") and "?" not in say:
        wrong.append("tenía que preguntar")
    if case.get("callar") and say:
        wrong.append("tenía que callarse")
    if not case.get("callar") and not say and action != "esperar":
        wrong.append("no dijo nada")
    if action == "esperar" and say:
        wrong.append("esperaba el resto pero habló")
    if action == "ninguna":
        for claim in case.get("no_afirmar", []):
            if claim in say.lower():
                wrong.append(f"afirmó «{claim}» sin mirar")
    return wrong


def main():
    spark = None
    if not (len(sys.argv) > 1 and sys.argv[1] == "glimmer"):
        from lunes.meta_api import MetaModels
        spark = MetaModels()
    cases = json.loads((ROOT / "tests" / "conductor_frases.json").read_text(encoding="utf-8"))["frases"]
    conductor = Conductor(spark, about=ABOUT)
    passed, seconds = 0, []
    for case in cases:
        conductor.turns = []
        conductor.context = lambda case=case: (SENSES, STATES[case.get("estado")])
        t0 = time.time()
        say, action, request = conductor.reply(case["frase"], called=case["llamado"])
        seconds.append(time.time() - t0)
        wrong = check(case, say, action)
        passed += not wrong
        print(f"{'ok ' if not wrong else 'MAL'} {seconds[-1]:4.1f} s {conductor.brain:7s} {case['frase']!r}\n"
              f"      [{action}] {say}" + (f"  (pedido: {request})" if action != "ninguna" else "") +
              ("".join(f"\n      -> {w}" for w in wrong)))
    print(f"\n{passed} de {len(cases)} bien; {sum(seconds) / len(seconds):.1f} s en promedio, el más lento "
          f"{max(seconds):.1f} s")


if __name__ == "__main__":
    main()
