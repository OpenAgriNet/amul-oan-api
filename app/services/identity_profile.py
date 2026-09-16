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
    # Bengali. ``কে`` ("who") is also the start of কেমন/কেন ("how"/"why"), so it
    # must not run on into another Bengali letter. ``পরিচ\S*`` accepts both
    # encodings of য় (U+09DF, or U+09AF + nukta) in পরিচয়.
    r"আপনি\s+কে(?![\u0980-\u09FF])",
    r"তুমি\s+কে(?![\u0980-\u09FF])",
    r"সরলাবেন\s+কে(?![\u0980-\u09FF])",
    r"আপনার\s+পরিচ\S*\s+দিন",
    r"তোমার\s+পরিচ\S*\s+দাও",
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

_BENGALI_ROWS: Final[tuple[tuple[str, str], ...]] = (
    ("নাম", "সরলাবেন"),
    ("ভূমিকা", "দুধ উৎপাদকদের জন্য আমুলের AI ডিজিটাল সহায়ক"),
    ("জন্ম তারিখ", "11 ফেব্রুয়ারি 2026"),
    ("প্রতিষ্ঠান", "আমুল"),
    ("উপলব্ধতা", "080-35453545 নম্বরে চ্যাট, ভয়েস কল ও হোয়াটসঅ্যাপে 24x7"),
    (
        "আমার সম্পর্কে",
        "নমস্কার! আমি সরলাবেন — দুধ উৎপাদক, ডেয়ারি কৃষক এবং সমবায় সমিতির সদস্যদের সাহায্য করার জন্য তৈরি আমুলের AI-চালিত ডিজিটাল সঙ্গী।",
    ),
    (
        "উদ্দেশ্য",
        "আমার উদ্দেশ্য হল সময়মতো তথ্য, কাজের পরামর্শ এবং ডিজিটাল সহায়তা দিয়ে ডেয়ারি কৃষকদের এগিয়ে নিয়ে যাওয়া — যা পশুর স্বাস্থ্য, দুধের উৎপাদন এবং খামারের লাভ বাড়াতে সাহায্য করে।",
    ),
    (
        "দক্ষতার ক্ষেত্র",
        "পশুপালন ব্যবস্থাপনা; দুধ উৎপাদন ও মান উন্নয়ন; পশুর পুষ্টি ও খাদ্য ব্যবস্থাপনা; টিকাকরণ ও রোগ প্রতিরোধ; প্রাথমিক পশুচিকিৎসা পরামর্শ ও রোগ সচেতনতা; প্রজনন ও গর্ভধারণ ব্যবস্থাপনা; ডেয়ারি সমবায় পরিষেবা ও সদস্য সহায়তা; ডেয়ারি পরামর্শ ও উন্নত খামার পদ্ধতি",
    ),
    (
        "আমি কাদের সেবা করি",
        "দুধ উৎপাদক; ডেয়ারি কৃষক; সমবায় সমিতির সদস্য; পশুপালক; গ্রামীণ ডেয়ারি উদ্যোক্তা",
    ),
    (
        "আমার মূল্যবোধ",
        "কৃষক প্রথম; নির্ভরযোগ্য ও বিশ্বস্ত পরামর্শ; সমবায়ের চেতনা; সবার জন্য সহজলভ্যতা; নিরন্তর শেখা ও উদ্ভাবন",
    ),
    ("আমার প্রতিশ্রুতি", "আমি ডেয়ারি কৃষকদের তথ্য ও পরামর্শ দিতে 24x7 উপলব্ধ।"),
)

_ENGLISH_QUOTE: Final[str] = (
    "\"Your trusted digital dairy companion, inspired by Amul's cooperative values and dedicated to supporting every milk producer.\""
)
_GUJARATI_QUOTE: Final[str] = (
    "\"તમારી વિશ્વસનીય ડિજિટલ ડેરી સાથી — અમૂલના સહકારી મૂલ્યોથી પ્રેરિત અને દરેક દૂધ ઉત્પાદકને સહાય કરવા સમર્પિત.\""
)
_BENGALI_QUOTE: Final[str] = (
    "\"আপনার বিশ্বস্ত ডিজিটাল ডেয়ারি সঙ্গী — আমুলের সমবায় মূল্যবোধে অনুপ্রাণিত এবং প্রত্যেক দুধ উৎপাদকের পাশে থাকতে নিবেদিত।\""
)

_ROWS_BY_LANGUAGE: Final[dict[str, tuple[tuple[str, str], ...]]] = {
    "en": _ENGLISH_ROWS,
    "gu": _GUJARATI_ROWS,
    "bn": _BENGALI_ROWS,
}
_QUOTE_BY_LANGUAGE: Final[dict[str, str]] = {
    "en": _ENGLISH_QUOTE,
    "gu": _GUJARATI_QUOTE,
    "bn": _BENGALI_QUOTE,
}
_TABLE_HEADER_BY_LANGUAGE: Final[dict[str, str]] = {
    "en": "| Field | Details |",
    "gu": "| ક્ષેત્ર | વિગતો |",
    "bn": "| ক্ষেত্র | বিবরণ |",
}
_DOCTOR_IDENTITY_BY_LANGUAGE: Final[dict[str, str]] = {
    "en": (
        "I am Amul Veterinary Assistant, an AI clinical decision-support assistant "
        "for veterinary doctors working with cattle, buffalo, and calves."
    ),
    "gu": (
        "હું અમૂલ વેટરનરી આસિસ્ટન્ટ છું—ગાય, ભેંસ અને વાછરડાં માટે "
        "પશુચિકિત્સકોને દસ્તાવેજ-આધારિત ક્લિનિકલ નિર્ણય સહાય આપતો AI સહાયક."
    ),
    "bn": (
        "আমি আমুল ভেটেরিনারি অ্যাসিস্ট্যান্ট—গরু, মহিষ ও বাছুরের চিকিৎসায় "
        "পশুচিকিৎসকদের নথি-ভিত্তিক ক্লিনিক্যাল সিদ্ধান্তে সাহায্যকারী একটি AI সহায়ক।"
    ),
}


# Connective/filler words that commonly bridge an identity phrase to unrelated
# content ("who are you AND my cow has fever") or merely pad it ("hey ...",
# "please ..."). Stripped before measuring residual content so they don't count
# as a real second question.
_IDENTITY_FILLER_WORDS: Final[frozenset[str]] = frozenset(
    {
        "and", "please", "also", "hey", "hi", "hello", "so", "just", "ok", "okay",
        "tell", "me", "can", "you", "could", "would", "will", "the", "a", "an",
        "અને", "કૃપા", "કરીને", "મને", "કહો", "જરા", "તો",
        "এবং", "আর", "অনুগ্রহ", "করে", "আমাকে", "বলুন", "বলো", "একটু", "তো", "নমস্কার",
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
    if tgt in {"bn", "bengali"}:
        return "bn"
    if src in {"gu", "gujarati"} or tgt in {"gu", "gujarati"}:
        return "gu"
    if src in {"bn", "bengali"}:
        return "bn"
    if re.search(r"[\u0A80-\u0AFF]", query or ""):
        return "gu"
    if re.search(r"[\u0980-\u09FF]", query or ""):
        return "bn"
    return "en"


def build_identity_profile_table(source_lang: str, target_lang: str, query: str) -> str:
    language = _select_identity_language(source_lang, target_lang, query)
    table_rows = [f"| {field} | {details} |" for field, details in _ROWS_BY_LANGUAGE[language]]
    table = "\n".join([_TABLE_HEADER_BY_LANGUAGE[language], "|---|---|", *table_rows])
    return f"{table}\n\n{_QUOTE_BY_LANGUAGE[language]}"


def build_doctor_identity_response(source_lang: str, target_lang: str, query: str) -> str:
    """Return a deterministic Doctor identity without invoking SarlaBen/RAG."""
    return _DOCTOR_IDENTITY_BY_LANGUAGE[_select_identity_language(source_lang, target_lang, query)]
