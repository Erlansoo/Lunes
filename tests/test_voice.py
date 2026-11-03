"""Telling Erlan from Lunes' own voice coming back from the speakers, no hardware: python tests/test_voice.py

Over Lunes' voice the ear hears both. Its own voice comes back garbled, and answering it would start a loop; but Erlan
often starts his request as Lunes ends its answer, and on Nov 2 three of his requests were dropped as echo. The phrases
here are the ones the ear wrote that night.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lunes.voice import sounds_like_speech  # noqa: E402

FAILED = []
ERLAN = [
    "en el protovo alarme un circuito con un integrado ne ccc por favor quería los revisios completos y las conexiones "
    "está bien",
    "hice una conexión del ne555 con componentes y bueno quería revises toda la configuración por favor del circuito y "
    "me digas tipo de frecuencia voy sacar en el let",
    "quisiera revises el protoboar porque he armado un circuito y quería los revises cuál es la configuración el "
    "integrado es un n 55",
]
ECHOES = [
    "sevea laes tierda luse inseiter min puerna deldanco",
    "bediso protoguarla dinectme ocho mivíletros menaspelviso tra claro",
    "yalabedi se ceita y dos ciento setenta y tresvidido pros",
    "sid plotomoal poriplosco les perlendon tento epocaso",
    "reponnisdo prosongaal crídito stondo mejor cirte sint",
    "resolvindo protocoais comincaspode doy lanta",
    "atrosodo alto me lostoyel boy mcrina",
    "redumire a descoder otogual",
    "puenta me cue",
    "tfe didnomasgeas",
    "boelas",
    "l acns insedido paradententanredicar",
]


def ok(condition, what):
    print(("  ok   " if condition else "  FAIL ") + what)
    if not condition:
        FAILED.append(what)


def over_lunes():
    for text in ERLAN:
        ok(sounds_like_speech(text), f"Erlan over Lunes' voice is heard: «{text[:50]}...»")
    for text in ECHOES:
        ok(not sounds_like_speech(text), f"Lunes' own voice coming back is not: «{text}»")
    ok(not sounds_like_speech("sí te doy mi permiso"), "a short phrase still needs the call, as before")


if __name__ == "__main__":
    over_lunes()
    print(f"\n{'all passed' if not FAILED else f'{len(FAILED)} FAILED'}")
    sys.exit(1 if FAILED else 0)
