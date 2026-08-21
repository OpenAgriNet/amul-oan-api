"""Central (Bharat Vistaar) agriculture scheme codes and farmer-phrasing aliases.

ONE place that knows the 15 codes the Bharat Vistaar BPP answers to, and how a
farmer's actual words map onto them. Both consumers import from here:

  - `agents/tools/vistaar.py`   — the direct `get_vistaar_scheme_info` tool
  - `agents/tools/beckn_network.py` — the merged union+central scheme discovery

Why an alias map at all: the BV BPP matches `item.descriptor.name` against the
scheme CODE and nothing else. It answers "kcc" and returns an empty catalogue
for "KCC ", "Kisan Credit Card", "crop insurance", "પાક વીમો". Production chat
is mostly Gujarati, so Gujarati and Hindi phrasings are first-class here, not an
afterthought.

`SchemeCode` is the Literal used in the tool signature so the model picks from a
JSON-schema enum instead of inventing a code (this is what
`bharat-oan-api/agents/tools/scheme_info.py` does). The literal members are
spelled out because `Literal[*tuple]` is not valid on this Python; a test pins
`get_args(SchemeCode) == SCHEME_CODES` so the two cannot drift.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Literal, Optional

SchemeCode = Literal[
    "kcc", "pmkisan", "pmfby", "shc", "pmksy", "sathi", "pmasha", "aif",
    "smam", "pdmc", "pkvy", "nfsm", "rad", "ffs", "nbhm",
]

SCHEME_CODES: tuple[str, ...] = (
    "kcc", "pmkisan", "pmfby", "shc", "pmksy", "sathi", "pmasha", "aif",
    "smam", "pdmc", "pkvy", "nfsm", "rad", "ffs", "nbhm",
)

# Farmer-facing English names. Used for the "I can't look that up, but I can
# look up these" message — we name SCHEMES, never internal codes, because
# `assets/prompts/agrinet_system.md` forbids exposing tool mechanics.
SCHEME_LABELS: dict[str, str] = {
    "kcc": "Kisan Credit Card",
    "pmkisan": "PM-KISAN (Kisan Samman Nidhi)",
    "pmfby": "Pradhan Mantri Fasal Bima Yojana (crop insurance)",
    "shc": "Soil Health Card",
    "pmksy": "Pradhan Mantri Krishi Sinchayee Yojana (irrigation)",
    "sathi": "SATHI (seed authentication and traceability)",
    "pmasha": "PM-AASHA (Annadata Aay Sanrakshan Abhiyan)",
    "aif": "Agriculture Infrastructure Fund",
    "smam": "Sub-Mission on Agricultural Mechanization",
    "pdmc": "Per Drop More Crop (micro-irrigation)",
    "pkvy": "Paramparagat Krishi Vikas Yojana (organic farming)",
    "nfsm": "National Food Security Mission",
    "rad": "Rainfed Area Development",
    "ffs": "Framework for Fertilizer Sales",
    "nbhm": "National Beekeeping and Honey Mission",
}

# code -> the phrasings a farmer (or the model, paraphrasing one) actually uses.
# English + Gujarati + Hindi. The code itself is always an alias of itself.
SCHEME_ALIASES: dict[str, tuple[str, ...]] = {
    "kcc": (
        "kcc", "kisan credit card", "kisan card", "farmer credit card",
        "કિસાન ક્રેડિટ કાર્ડ", "કિસાન કાર્ડ", "કેસીસી",
        "किसान क्रेडिट कार्ड", "किसान कार्ड", "केसीसी",
    ),
    "pmkisan": (
        "pmkisan", "pm kisan", "pm kisan samman nidhi",
        "pradhan mantri kisan samman nidhi", "kisan samman nidhi", "samman nidhi",
        "પીએમ કિસાન", "પ્રધાનમંત્રી કિસાન સન્માન નિધિ", "કિસાન સન્માન નિધિ", "સન્માન નિધિ",
        "पीएम किसान", "प्रधानमंत्री किसान सम्मान निधि", "किसान सम्मान निधि", "सम्मान निधि",
    ),
    "pmfby": (
        "pmfby", "pm fby", "crop insurance", "fasal bima", "fasal bima yojana",
        "pradhan mantri fasal bima yojana", "pmfby crop insurance",
        "પાક વીમો", "પાક વીમા", "પાક વીમા યોજના", "ફસલ બીમા",
        "પ્રધાનમંત્રી ફસલ બીમા યોજના", "પ્રધાનમંત્રી પાક વીમા યોજના",
        "फसल बीमा", "फ़सल बीमा", "फसल बीमा योजना", "प्रधानमंत्री फसल बीमा योजना",
    ),
    "shc": (
        "shc", "soil health card", "soil card", "soil health", "soil testing",
        "જમીન આરોગ્ય કાર્ડ", "જમીન આરોગ્ય", "માટી આરોગ્ય કાર્ડ", "જમીન ચકાસણી",
        "मृदा स्वास्थ्य कार्ड", "मिट्टी स्वास्थ्य कार्ड", "मृदा कार्ड", "मिट्टी जांच",
    ),
    "pmksy": (
        "pmksy", "pm ksy", "krishi sinchayee", "krishi sinchai",
        "pradhan mantri krishi sinchayee yojana", "irrigation scheme",
        "કૃષિ સિંચાઈ યોજના", "પ્રધાનમંત્રી કૃષિ સિંચાઈ યોજના", "સિંચાઈ યોજના",
        "कृषि सिंचाई योजना", "प्रधानमंत्री कृषि सिंचाई योजना", "सिंचाई योजना",
    ),
    "sathi": (
        "sathi", "seed authentication", "seed traceability", "seed certification",
        "બિયારણ પ્રમાણન", "બિયારણ ટ્રેસેબિલિટી", "બીજ પ્રમાણન",
        "बीज प्रमाणीकरण", "बीज ट्रेसेबिलिटी",
    ),
    "pmasha": (
        "pmasha", "pm asha", "pm aasha", "annadata aay sanrakshan",
        "pradhan mantri annadata aay sanrakshan abhiyan", "price support scheme",
        "minimum support price scheme",
        "અન્નદાતા આય સંરક્ષણ", "ટેકાના ભાવ યોજના", "ટેકાના ભાવ",
        "अन्नदाता आय संरक्षण", "न्यूनतम समर्थन मूल्य योजना", "समर्थन मूल्य योजना",
    ),
    "aif": (
        "aif", "agriculture infrastructure fund", "agri infrastructure fund",
        "agricultural infrastructure fund",
        "કૃષિ ઇન્ફ્રાસ્ટ્રક્ચર ફંડ", "કૃષિ માળખાકીય ભંડોળ",
        "कृषि अवसंरचना निधि", "कृषि इंफ्रास्ट्रक्चर फंड",
    ),
    "smam": (
        "smam", "sub mission on agricultural mechanization",
        "agricultural mechanization", "agricultural mechanisation",
        "farm mechanization", "farm mechanisation", "farm machinery subsidy",
        "ખેત યાંત્રિકીકરણ", "કૃષિ યાંત્રિકીકરણ", "ખેત ઓજાર સહાય",
        "कृषि यंत्रीकरण", "खेत मशीनीकरण", "कृषि मशीनीकरण", "कृषि यंत्र सब्सिडी",
    ),
    "pdmc": (
        "pdmc", "per drop more crop", "micro irrigation", "drip irrigation",
        "sprinkler irrigation",
        "ટપક સિંચાઈ", "સૂક્ષ્મ સિંચાઈ", "ફુવારા સિંચાઈ", "પર ડ્રોપ મોર ક્રોપ",
        "ड्रिप सिंचाई", "सूक्ष्म सिंचाई", "प्रति बूंद अधिक फसल", "फव्वारा सिंचाई",
    ),
    "pkvy": (
        "pkvy", "paramparagat krishi vikas yojana", "organic farming",
        "organic farming scheme",
        "સજીવ ખેતી", "જૈવિક ખેતી", "પરંપરાગત કૃષિ વિકાસ યોજના",
        "जैविक खेती", "परंपरागत कृषि विकास योजना",
    ),
    "nfsm": (
        "nfsm", "national food security mission", "food security mission",
        "રાષ્ટ્રીય અન્ન સુરક્ષા મિશન", "અન્ન સુરક્ષા મિશન",
        "राष्ट्रीय खाद्य सुरक्षा मिशन", "खाद्य सुरक्षा मिशन",
    ),
    "rad": (
        "rad", "rainfed area development", "rainfed farming", "rainfed area",
        "વરસાદ આધારિત વિસ્તાર વિકાસ", "બિનપિયત ખેતી", "વરસાદ આધારિત ખેતી",
        "वर्षा आधारित क्षेत्र विकास", "बारानी खेती", "वर्षा आधारित खेती",
    ),
    "ffs": (
        "ffs", "framework for fertilizer sales", "fertilizer sales",
        "fertiliser sales", "fertilizer subsidy",
        "ખાતર વેચાણ", "રાસાયણિક ખાતર વેચાણ", "ખાતર સહાય",
        "उर्वरक बिक्री", "खाद बिक्री", "उर्वरक सब्सिडी",
    ),
    "nbhm": (
        "nbhm", "national beekeeping and honey mission",
        "national beekeeping honey mission", "beekeeping", "bee keeping",
        "honey mission",
        "મધમાખી ઉછેર", "મધ મિશન", "રાષ્ટ્રીય મધમાખી ઉછેર અને મધ મિશન",
        "मधुमक्खी पालन", "शहद मिशन", "राष्ट्रीय मधुमक्खी एवं शहद मिशन",
    ),
}

_ALIAS_TO_CODE: dict[str, str] = {}


def _normalize(text: str) -> str:
    """Casefold, strip punctuation that farmers/models scatter around scheme
    names ("PM-KISAN", "PM_Kisan", "pmfby."), and collapse whitespace."""
    norm = unicodedata.normalize("NFKC", text or "").casefold()
    norm = re.sub(r"[-_/.,:;()\[\]'\"]+", " ", norm)
    return re.sub(r"\s+", " ", norm).strip()


for _code, _aliases in SCHEME_ALIASES.items():
    for _alias in _aliases:
        _ALIAS_TO_CODE[_normalize(_alias)] = _code

# Longest alias first, so "kisan samman nidhi" wins over a bare "kisan" style
# fragment and "soil health card" over "soil health".
_ALIASES_BY_LENGTH: tuple[str, ...] = tuple(
    sorted(_ALIAS_TO_CODE, key=len, reverse=True)
)
# `(?<!\w)…(?!\w)` rather than `\b`: it behaves the same for Latin text but does
# not require the alias itself to start/end with a word character, and Python's
# \w already covers Gujarati/Devanagari letters so Indic aliases match too.
_ALIAS_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(rf"(?<!\w){re.escape(a)}(?!\w)"), _ALIAS_TO_CODE[a])
    for a in _ALIASES_BY_LENGTH
)


def resolve_scheme_code(text: Optional[str]) -> Optional[str]:
    """Map free farmer/model phrasing onto a Bharat Vistaar scheme code.

    Returns None when nothing matches — callers MUST treat that as "this is not
    a central-scheme question" and skip the BV leg entirely rather than sending
    a word the BPP will never match (a guaranteed-empty ~2.2s round trip).

    Matching is exact-after-normalization first, then a longest-alias substring
    scan with word boundaries, so "tell me about the Kisan Credit Card please"
    and "પાક વીમો શું છે" both resolve.
    """
    norm = _normalize(text or "")
    if not norm:
        return None
    direct = _ALIAS_TO_CODE.get(norm)
    if direct:
        return direct
    for pattern, code in _ALIAS_PATTERNS:
        if pattern.search(norm):
            return code
    return None


def scheme_names_sentence(limit: int = 4) -> str:
    """A few farmer-facing scheme NAMES for the 'ask me about one of these'
    fallback. Deliberately not the code list — codes are internal."""
    names = [SCHEME_LABELS[c].split(" (")[0] for c in SCHEME_CODES[:limit]]
    return ", ".join(names[:-1]) + f", or {names[-1]}" if len(names) > 1 else names[0]
