"""Text as it should be said: Meta's MMS voice only knows letters, so numbers, units and acronyms become words.

    "La placa mide 76 x 64 mm" -> "La placa mide setenta y seis por sesenta y cuatro milímetros"
    "el LA7770 y el HDMI"      -> "el ele a siete siete siete cero y el hache de eme i"
"""
import re

UNITS = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez", "once", "doce",
         "trece", "catorce", "quince", "dieciséis", "diecisiete", "dieciocho", "diecinueve", "veinte", "veintiuno",
         "veintidós", "veintitrés", "veinticuatro", "veinticinco", "veintiséis", "veintisiete", "veintiocho",
         "veintinueve"]
TENS = {3: "treinta", 4: "cuarenta", 5: "cincuenta", 6: "sesenta", 7: "setenta", 8: "ochenta", 9: "noventa"}
HUNDREDS = {1: "ciento", 2: "doscientos", 3: "trescientos", 4: "cuatrocientos", 5: "quinientos", 6: "seiscientos",
            7: "setecientos", 8: "ochocientos", 9: "novecientos"}
LETTERS = {"A": "a", "B": "be", "C": "ce", "D": "de", "E": "e", "F": "efe", "G": "ge", "H": "hache", "I": "i",
           "J": "jota", "K": "ka", "L": "ele", "M": "eme", "N": "ene", "Ñ": "eñe", "O": "o", "P": "pe", "Q": "cu",
           "R": "erre", "S": "ese", "T": "te", "U": "u", "V": "uve", "W": "uve doble", "X": "equis", "Y": "ye",
           "Z": "zeta"}
# after a number: the unit's name (plural; "un milímetro" sounds odd only for 1, and that is rare on a bench)
UNIT_WORDS = {"mm": "milímetros", "cm": "centímetros", "m": "metros", "µm": "micras", "um": "micras",
              "mV": "milivoltios", "V": "voltios", "mA": "miliamperios", "A": "amperios", "W": "vatios",
              "kΩ": "kiloohmios", "Ω": "ohmios", "ohm": "ohmios", "kHz": "kilohercios", "MHz": "megahercios",
              "Hz": "hercios", "%": "por ciento", "°C": "grados", "°": "grados", "s": "segundos", "ms": "milisegundos",
              "pines": "pines"}


def number_words(n):
    """Spanish words for a whole number from 0 to 999 999."""
    if n < 30:
        return UNITS[n]
    if n < 100:
        tens, unit = divmod(n, 10)
        return TENS[tens] + ("" if unit == 0 else " y " + UNITS[unit])
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        if n == 100:
            return "cien"
        return HUNDREDS[hundreds] + ("" if rest == 0 else " " + number_words(rest))
    if n < 1_000_000:
        thousands, rest = divmod(n, 1000)
        head = "mil" if thousands == 1 else number_words(thousands).replace("uno", "un") + " mil"
        return head + ("" if rest == 0 else " " + number_words(rest))
    return " ".join(UNITS[int(d)] for d in str(n))


def _number(match):
    whole, frac = match.group(1), match.group(3)
    words = number_words(int(whole)) if len(whole) <= 6 else " ".join(UNITS[int(d)] for d in whole)
    if frac:
        words += " coma " + (number_words(int(frac)) if not frac.startswith("0") else " ".join(UNITS[int(d)] for d in frac))
    return words


def _acronym(match):
    word = match.group(0)
    return " ".join(LETTERS.get(c, UNITS[int(c)] if c.isdigit() else c.lower()) for c in word)


def speakable(text):
    """Spanish text with every number, unit and acronym written as it is said."""
    units = "|".join(sorted((re.escape(u) for u in UNIT_WORDS), key=len, reverse=True))
    text = re.sub(rf"(\d)\s*({units})(?![\wΩ])", lambda m: m.group(1) + " " + UNIT_WORDS[m.group(2)], text)
    text = re.sub(r"(\d)\s*[x×]\s*(\d)", r"\1 por \2", text)
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", lambda m: f"{m.group(1)} y {m.group(2)}", text)  # 19:44
    # part numbers and acronyms: two or more capitals, maybe with digits (HDMI, LA7770, TP4056, BGA)
    text = re.sub(r"\b(?=[A-ZÑ0-9]*[A-ZÑ][A-ZÑ0-9]*\b)(?=\w*[A-ZÑ]\w*[A-ZÑ0-9])[A-ZÑ][A-ZÑ0-9]+\b", _acronym, text)
    text = re.sub(r"(\d+)([.,](\d+))?", _number, text)
    text = text.replace("&", " y ").replace("/", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


if __name__ == "__main__":
    for t in ["La placa mide 76 x 64 mm y pesa 120 g.", "Veo el LA7770 y el conector HDMI.", "Está a 2,5 mm, 12 V y 1,05 A.",
              "Hay 16 pines, un TP4056 y 1000 µm.", "Son las 19:44.", "Muse Spark leyó LA 7770 y P?"]:
        print(f"{t}\n   -> {speakable(t)}")
