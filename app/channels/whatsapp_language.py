"""Which language a WhatsApp turn runs in.

The WhatsApp integration tags Gujarati-script messages "gu", Devanagari ones
"hi" and everything else "en". So Bengali and Punjabi messages were answered in
English and Marathi ones in Hindi (prod Langfuse, 15-24 Nov 2026). The web app
sends the language the farmer picked and is never second-guessed; a WhatsApp
message is read for its script instead.
"""
from __future__ import annotations

import re

_DEVANAGARI = "devanagari"

#: Unicode block -> language, for the scripts of the languages the app offers
#: (en, gu, hi, mr, bn, pa). Gujarati, Gurmukhi and Bengali each belong to one of
#: them; Devanagari is shared by Hindi and Marathi.
_SCRIPT_BLOCKS: tuple[tuple[range, str], ...] = (
    (range(0x0900, 0x0980), _DEVANAGARI),
    (range(0x0980, 0x0A00), "bn"),
    (range(0x0A00, 0x0A80), "pa"),
    (range(0x0A80, 0x0B00), "gu"),
)

#: A run of Devanagari letters and signs; the dandas (U+0964, U+0965) end a word.
_DEVANAGARI_WORD = re.compile(
    "[" + chr(0x0900) + "-" + chr(0x0963) + chr(0x0966) + "-" + chr(0x097F) + "]+"
)

#: Words Marathi uses and Hindi does not (Hindi: है, नहीं, क्या, कितना, से ... तक).
_MARATHI_WORDS = frozenset({
    "आहे", "आहेत", "आहात", "आहेस", "नाही", "नाहीत", "काय", "कसे", "कशी", "कसा",
    "करावे", "करावा", "करावी", "करायचे", "करायची", "सांगा", "सांग", "माझा", "माझी",
    "माझे", "माझ्या", "माझ", "मला", "आम्हाला", "तुम्ही", "आणि", "किती", "झाले",
    "झाली", "झाला", "होईल", "द्या", "ते", "चे", "ची", "चा", "च्या", "असते", "असतो",
    "असतात", "गेल्या", "केलेल्या", "मिळेल", "पाहिजे", "कोणते", "कोणती", "आमच्या",
    "आमचे", "तुमच्या", "तुमचे", "वाढले", "नोव्हेंबर", "म्हैस", "म्हशी", "रक्कम",
})

#: Marathi genitive and postposition endings: दुधाचे, पगाराची, उवांचे,
#: वाढवण्याचे, नागरिकांसाठी, तारखेपर्यंत.
_MARATHI_SUFFIXES = (
    "ाचे", "ाची", "ाचा", "ाच्या", "ांचे", "ांची", "ांचा", "ांच्या",
    "ण्याचे", "ण्याची", "ण्यासाठी", "साठी", "पर्यंत", "मध्ये",
)

#: Hindi words that happen to end like a Marathi genitive (चाचा, नाचे, जांचा).
_HINDI_SUFFIX_LOOKALIKES = ("चाच", "नाच", "जांच", "जाँच")

#: Marathi writes ळ (दिवाळी, मिळेल, वेळ); Hindi does not.
_MARATHI_LETTER = "ळ"

_HINDI_WORDS = frozenset({
    "है", "हैं", "हो", "था", "थी", "थे", "क्या", "कैसे", "कैसा", "कैसी", "नहीं",
    "नही", "मेरी", "मेरा", "मेरे", "मुझे", "करें", "करे", "करना", "का", "की", "के",
    "में", "मे", "को", "से", "तक", "और", "भी", "कितना", "कितनी", "कितने", "बताइए",
    "बताओ", "बताएं", "बताए", "बताये", "दीजिए", "गया", "गई", "गए", "रहा", "रही",
    "रहे", "आप", "हम", "हमें", "कब", "क्यों", "कौन", "कोनसी", "चाहिए", "होगा",
    "होगी", "आयेगा", "आएगा", "वाला", "वाली", "लिए", "हुआ", "हुई", "हू", "हूं",
    "हूँ", "सकता", "सकती", "सकते", "जाता", "जाती", "नवंबर",
})

_HINDI_OR_MARATHI = frozenset({"hi", "hindi", "mr", "marathi"})


def _dominant_script(text: str) -> str | None:
    """The script of most of the message's letters, or None (Latin, mixed, none)."""
    counts: dict[str, int] = {}
    letters = 0
    for ch in text:
        point = ord(ch)
        script = next((name for block, name in _SCRIPT_BLOCKS if point in block), None)
        if script is not None:
            counts[script] = counts.get(script, 0) + 1
            letters += 1
        elif ch.isalpha():
            letters += 1
    if not counts:
        return None
    best = max(counts, key=counts.__getitem__)
    return best if counts[best] * 2 >= letters else None


def _is_marathi_word(word: str) -> bool:
    if word in _MARATHI_WORDS or _MARATHI_LETTER in word:
        return True
    return word.endswith(_MARATHI_SUFFIXES) and not word.startswith(_HINDI_SUFFIX_LOOKALIKES)


def _reads_as_marathi(text: str) -> bool:
    """Devanagari text with more Marathi-only words than Hindi-only ones."""
    words = _DEVANAGARI_WORD.findall(text)
    marathi = sum(1 for word in words if _is_marathi_word(word))
    hindi = sum(1 for word in words if word in _HINDI_WORDS)
    return marathi > hindi


def whatsapp_languages(query: str, source_lang: str, target_lang: str) -> tuple[str, str]:
    """The (source, target) languages a WhatsApp turn runs in.

    A message written mostly in Gujarati, Gurmukhi or Bengali script runs in
    that language. A Devanagari one runs in Marathi when its wording is Marathi,
    otherwise in the caller's Hindi/Marathi, else Hindi. Anything else (English,
    romanized, mostly-Latin mixed text, scripts the app does not offer) keeps the
    caller's languages.
    """
    script = _dominant_script(query or "")
    if script is None:
        return source_lang, target_lang
    if script != _DEVANAGARI:
        return script, script
    if _reads_as_marathi(query):
        return "mr", "mr"
    if (target_lang or "").lower() in _HINDI_OR_MARATHI:
        return source_lang, target_lang
    return "hi", "hi"
