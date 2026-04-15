"""
Gujarati number-to-words converter and text normalizer for TTS output.

Converts digit sequences in text to Gujarati words so that TTS engines
pronounce them correctly instead of garbling compound numbers.

Examples:
    136      → એકસો છત્રીસ
    3.56     → ત્રણ પોઈન્ટ પાંચ છ
    15       → પંદર
    tag 1062853187210 → એક શૂન્ય છ બે આઠ પાંચ ત્રણ એક આઠ સાત બે એક શૂન્ય
"""

import re

# --- Lookup tables ---

_ONES = {
    0: "શૂન્ય",
    1: "એક",
    2: "બે",
    3: "ત્રણ",
    4: "ચાર",
    5: "પાંચ",
    6: "છ",
    7: "સાત",
    8: "આઠ",
    9: "નવ",
}

# Gujarati has unique words for 1-99 (Indian numbering irregularity)
_1_TO_99 = {
    1: "એક", 2: "બે", 3: "ત્રણ", 4: "ચાર", 5: "પાંચ",
    6: "છ", 7: "સાત", 8: "આઠ", 9: "નવ", 10: "દસ",
    11: "અગિયાર", 12: "બાર", 13: "તેર", 14: "ચૌદ", 15: "પંદર",
    16: "સોળ", 17: "સત્તર", 18: "અઢાર", 19: "ઓગણીસ",
    20: "વીસ", 21: "એકવીસ", 22: "બાવીસ", 23: "ત્રેવીસ",
    24: "ચોવીસ", 25: "પચ્ચીસ", 26: "છવ્વીસ", 27: "સત્તાવીસ",
    28: "અઠ્ઠાવીસ", 29: "ઓગણત્રીસ",
    30: "ત્રીસ", 31: "એકત્રીસ", 32: "બત્રીસ", 33: "તેત્રીસ",
    34: "ચોત્રીસ", 35: "પાંત્રીસ", 36: "છત્રીસ", 37: "સાડત્રીસ",
    38: "આડત્રીસ", 39: "ઓગણચાલીસ",
    40: "ચાલીસ", 41: "એકતાલીસ", 42: "બેતાલીસ", 43: "ત્રેતાલીસ",
    44: "ચુંમાલીસ", 45: "પિસ્તાલીસ", 46: "છેતાલીસ", 47: "સુડતાલીસ",
    48: "અડતાલીસ", 49: "ઓગણપચાસ",
    50: "પચાસ", 51: "એકાવન", 52: "બાવન", 53: "ત્રેપન",
    54: "ચોપન", 55: "પંચાવન", 56: "છપ્પન", 57: "સત્તાવન",
    58: "અઠ્ઠાવન", 59: "ઓગણસાઠ",
    60: "સાઠ", 61: "એકસઠ", 62: "બાસઠ", 63: "ત્રેસઠ",
    64: "ચોસઠ", 65: "પાંસઠ", 66: "છાસઠ", 67: "સડસઠ",
    68: "અડસઠ", 69: "અગણોતેર",
    70: "સિત્તેર", 71: "એકોતેર", 72: "બોતેર", 73: "તોતેર",
    74: "ચુમોતેર", 75: "પંચોતેર", 76: "છોતેર", 77: "સિત્યોતેર",
    78: "ઇઠ્યોતેર", 79: "ઓગણાએંસી",
    80: "એંસી", 81: "એક્યાસી", 82: "બ્યાસી", 83: "ત્યાસી",
    84: "ચોર્યાસી", 85: "પંચાસી", 86: "છ્યાસી", 87: "સત્યાસી",
    88: "અઠ્ઠ્યાસી", 89: "નેવ્યાસી",
    90: "નેવું", 91: "એકાણું", 92: "બાણું", 93: "ત્રાણું",
    94: "ચોરાણું", 95: "પંચાણું", 96: "છન્નું", 97: "સત્તાણું",
    98: "અઠ્ઠાણું", 99: "નવ્વાણું",
}

_HUNDREDS = {
    1: "એકસો", 2: "બસો", 3: "ત્રણસો", 4: "ચારસો", 5: "પાંચસો",
    6: "છસો", 7: "સાતસો", 8: "આઠસો", 9: "નવસો",
}


def _int_to_gujarati(n: int) -> str:
    """Convert a non-negative integer to Gujarati words.

    Uses the Indian numbering system (lakh, crore etc.).
    """
    if n < 0:
        return "ઋણ " + _int_to_gujarati(-n)
    if n == 0:
        return _ONES[0]
    if n <= 99:
        return _1_TO_99[n]

    parts: list[str] = []

    # Crore (1,00,00,000)
    if n >= 10_000_000:
        crore = n // 10_000_000
        parts.append(_int_to_gujarati(crore) + " કરોડ")
        n %= 10_000_000

    # Lakh (1,00,000)
    if n >= 100_000:
        lakh = n // 100_000
        parts.append(_int_to_gujarati(lakh) + " લાખ")
        n %= 100_000

    # Thousand (1,000)
    if n >= 1000:
        thousand = n // 1000
        parts.append(_int_to_gujarati(thousand) + " હજાર")
        n %= 1000

    # Hundred (100)
    if n >= 100:
        h = n // 100
        parts.append(_HUNDREDS[h])
        n %= 100

    # Remainder 1-99
    if n > 0:
        parts.append(_1_TO_99[n])

    return " ".join(parts)


def number_to_gujarati(value: float | int) -> str:
    """Convert a number (int or float) to Gujarati words.

    Integers: 136 → "એકસો છત્રીસ"
    Floats:   3.56 → "ત્રણ પોઈન્ટ પાંચ છ"  (decimals read digit-by-digit)
    """
    if isinstance(value, int) or (isinstance(value, float) and value == int(value)):
        return _int_to_gujarati(int(value))

    text = f"{value:g}"
    if "." not in text:
        return _int_to_gujarati(int(value))

    int_part_str, dec_part_str = text.split(".", 1)
    int_part = int(int_part_str)
    int_word = _int_to_gujarati(int_part)

    # Decimal digits read one at a time: 3.56 → "ત્રણ પોઈન્ટ પાંચ છ"
    dec_digits = " ".join(_ONES[int(d)] for d in dec_part_str)
    return f"{int_word} પોઈન્ટ {dec_digits}"


def tag_to_gujarati(tag: str) -> str:
    """Convert a tag number to digit-by-digit Gujarati words.

    Tag numbers are identifiers, not quantities, so they are read digit by digit.
    "1062853187210" → "એક શૂન્ય છ બે આઠ પાંચ ત્રણ એક આઠ સાત બે એક શૂન્ય"
    """
    return " ".join(_ONES[int(d)] for d in tag if d.isdigit())


# --- Text normalizer for TTS output ---

# Matches sequences of digits, optionally with one decimal point
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

# Matches digit sequences of 10+ digits (tag numbers, phone numbers)
_LONG_DIGIT_RE = re.compile(r"\d{10,}")


def normalize_numbers_for_tts(text: str) -> str:
    """Replace digit sequences in Gujarati text with Gujarati words.

    - Long digit sequences (10+ digits, e.g. tag numbers) → digit-by-digit
    - Regular numbers → natural Gujarati words (e.g. 136 → એકસો છત્રીસ)
    - Decimals → "X પોઈન્ટ Y Z" (e.g. 3.56 → ત્રણ પોઈન્ટ પાંચ છ)
    """
    if not text:
        return text

    # First pass: convert long digit sequences (tags, phone numbers) to digit-by-digit
    def _replace_long(m: re.Match) -> str:
        return tag_to_gujarati(m.group())

    text = _LONG_DIGIT_RE.sub(_replace_long, text)

    # Second pass: convert remaining number sequences to words
    def _replace_number(m: re.Match) -> str:
        s = m.group()
        if "." in s:
            return number_to_gujarati(float(s))
        return number_to_gujarati(int(s))

    text = _NUMBER_RE.sub(_replace_number, text)

    return text
