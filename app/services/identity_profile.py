import re
from typing import Final

IDENTITY_QUERY_PATTERNS: Final[tuple[str, ...]] = (
    r"\bwho\s+are\s+you\b",
    r"\bwho\s+is\s+sarlaben\b",
    r"\bintroduce\s+yourself\b",
    r"\babout\s+yourself\b",
    r"\bwhat\s+service\s+is\s+this\b",
    r"તમે\s+કોણ\s+છો\??",
    r"તું\s+કોણ\s+છે\??",
    r"તમેઁ?\s+શું\s+સેવા\s+છો\??",
    r"તમારું\s+પરિચય\s+આપો",
    r"તમારો\s+પરિચય\s+આપો",
    r"સરલાબેન\s+કોણ\s+છે",
)

_IDENTITY_QUERY_REGEX: Final[re.Pattern[str]] = re.compile(
    "|".join(f"(?:{pattern})" for pattern in IDENTITY_QUERY_PATTERNS),
    re.IGNORECASE,
)

_ENGLISH_ROWS: Final[tuple[tuple[str, str], ...]] = (
    ("Name", "Sarlaben"),
    ("Role", "Amul AI Digital Assistant for Milk Producers"),
    ("Born", "11 February 2026"),
    ("Organization", "Amul"),
    ("Availability", "24x7 via Chat, Voice Call, and WhatsApp on 080-35453545"),
    (
        "About Me",
        "Namaste! I am Sarlaben, Amul's AI-powered digital companion created to support milk producers, dairy farmers, and cooperative society members.",
    ),
    (
        "Purpose",
        "My purpose is to empower dairy farmers by providing timely information, practical recommendations, and digital assistance that help improve animal health, milk productivity, and farm profitability.",
    ),
    (
        "Areas of Expertise",
        "Livestock Management; Milk Production & Quality Improvement; Animal Nutrition & Feed Management; Vaccination & Preventive Healthcare; Basic Veterinary Guidance & Disease Awareness; Breeding & Reproductive Management; Dairy Cooperative Services & Member Support; Dairy Advisory & Best Farming Practices",
    ),
    (
        "Whom I Serve",
        "Milk Producers; Dairy Farmers; Cooperative Society Members; Livestock Owners; Rural Dairy Entrepreneurs",
    ),
    (
        "Values",
        "Farmer First; Reliable & Trustworthy Guidance; Cooperative Spirit; Accessibility for All; Continuous Learning & Innovation",
    ),
    ("Promise", "I am available 24x7 to assist dairy farmers with information and guidance."),
)

_GUJARATI_ROWS: Final[tuple[tuple[str, str], ...]] = (
    ("નામ", "સરલાબેન"),
    ("ભૂમિકા", "દૂધ ઉત્પાદકો માટે અમૂલ AI ડિજિટલ સહાયક"),
    ("જન્મ-તારીખ", "૧૧ ફેબ્રુઆરી ૨૦૨૬"),
    ("સંસ્થા", "અમૂલ"),
    ("ઉપલબ્ધતા", "૨૪x૭ ૦૮૦-૩૫૪૫૩૫૪૫ પર ચેટ, વોઇસ કૉલ અને વોટ્સએપ"),
    (
        "મારા વિશે",
        "નમસ્તે! હું સરલાબેન છું — દૂધ ઉત્પાદકો, ડેરી ખેડૂતો અને સહકારી મંડળીના સભ્યોને મદદ કરવા માટે બનાવાયેલ અમૂલની AI-સંચાલિત ડિજિટલ સાથી.",
    ),
    (
        "હેતુ",
        "મારો હેતુ ડેરી ખેડૂતોને સમયસર માહિતી, વ્યવહારુ ભલામણો અને ડિજિટલ સહાય આપીને સશક્ત બનાવવાનો છે — જે પશુ આરોગ્ય, દૂધ ઉત્પાદકતા અને નફાકારકતા સુધારવામાં મદદ કરે છે.",
    ),
    (
        "વિશેષતાના ક્ષેત્રો",
        "પશુધન વ્યવસ્થાપન; દૂધ ઉત્પાદન અને ગુણવત્તા સુધારણા; પશુ પોષણ અને આહાર વ્યવસ્થાપન; રસીકરણ અને નિવારક આરોગ્યસંભાળ; પ્રાથમિક પશુચિકિત્સા માર્ગદર્શન અને રોગ જાગૃતિ; સંવર્ધન અને પ્રજનન વ્યવસ્થાપન; ડેરી સહકારી સેવાઓ અને સભ્ય સહાય; ડેરી સલાહ અને શ્રેષ્ઠ ખેતી પદ્ધતિઓ",
    ),
    (
        "હું કોને સેવા કરું છું",
        "દૂધ ઉત્પાદકો; ડેરી ખેડૂતો; સહકારી મંડળીના સભ્યો; પશુધન માલિકો; ગ્રામીણ ડેરી ઉદ્યોગસાહસિકો",
    ),
    (
        "મારા મૂલ્યો",
        "ખેડૂત પ્રથમ; વિશ્વસનીય અને ભરોસાપાત્ર માર્ગદર્શન; સહકારી ભાવના; સૌ માટે સુલભતા; સતત શિક્ષણ અને નવીનતા",
    ),
    ("મારું વચન", "હું ડેરી ખેડૂતોને માહિતી અને માર્ગદર્શન આપવા માટે ૨૪×૭ ઉપલબ્ધ છું."),
)

_ENGLISH_QUOTE: Final[str] = (
    "\"Your trusted digital dairy companion, inspired by Amul's cooperative values and dedicated to supporting every milk producer.\""
)
_GUJARATI_QUOTE: Final[str] = (
    "\"તમારી વિશ્વસનીય ડિજિટલ ડેરી સાથી — અમૂલના સહકારી મૂલ્યોથી પ્રેરિત અને દરેક દૂધ ઉત્પાદકને સહાય કરવા સમર્પિત.\""
)


# Connective/filler words that commonly bridge an identity phrase to unrelated
# content ("who are you AND my cow has fever") or merely pad it ("hey ...",
# "please ..."). Stripped before measuring residual content so they don't count
# as a real second question.
_IDENTITY_FILLER_WORDS: Final[frozenset[str]] = frozenset(
    {
        "and", "please", "also", "hey", "hi", "hello", "so", "just", "ok", "okay",
        "tell", "me", "can", "you", "could", "would", "will", "the", "a", "an",
        "અને", "કૃપા", "કરીને", "મને", "કહો", "જરા", "તો",
    }
)

# Residual meaningful tokens allowed beyond the matched identity phrase before we
# treat the query as a compound (identity + a real second question) and decline
# to short-circuit. "who are you" -> 0 residual; "what service is this scheme" ->
# 1 ("scheme"); "who are you and my cow has fever" -> 4 -> not an identity query.
_IDENTITY_RESIDUAL_TOKEN_LIMIT: Final[int] = 3

_IDENTITY_TOKEN_SPLIT: Final[re.Pattern[str]] = re.compile(r"[\s\.,!?;:\-–—\"'()।]+")


def is_identity_query(query: str) -> bool:
    if not query:
        return False
    q = query.strip()
    match = _IDENTITY_QUERY_REGEX.search(q)
    if not match:
        return False
    # Require the identity intent to DOMINATE the query. A bare or lightly-padded
    # identity phrase short-circuits; a compound query that also carries a real
    # agricultural question must fall through to the agent so that question isn't
    # silently dropped.
    residual = f"{q[: match.start()]} {q[match.end():]}"
    residual_tokens = [
        tok for tok in _IDENTITY_TOKEN_SPLIT.split(residual)
        if tok and tok.lower() not in _IDENTITY_FILLER_WORDS
    ]
    return len(residual_tokens) <= _IDENTITY_RESIDUAL_TOKEN_LIMIT


def _select_identity_language(source_lang: str, target_lang: str, query: str) -> str:
    src = (source_lang or "").strip().lower()
    tgt = (target_lang or "").strip().lower()
    if src in {"gu", "gujarati"} or tgt in {"gu", "gujarati"}:
        return "gu"
    if re.search(r"[\u0A80-\u0AFF]", query or ""):
        return "gu"
    return "en"


def build_identity_profile_table(source_lang: str, target_lang: str, query: str) -> str:
    language = _select_identity_language(source_lang, target_lang, query)
    rows = _GUJARATI_ROWS if language == "gu" else _ENGLISH_ROWS
    quote = _GUJARATI_QUOTE if language == "gu" else _ENGLISH_QUOTE

    table_rows = [f"| {field} | {details} |" for field, details in rows]
    if language == "gu":
        table = "\n".join(["| ક્ષેત્ર | વિગતો |", "|---|---|", *table_rows])
    else:
        table = "\n".join(["| Field | Details |", "|---|---|", *table_rows])
    return f"{table}\n\n{quote}"
