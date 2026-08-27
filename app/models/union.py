from enum import Enum


class UnionName(str, Enum):
    BANAS = "banas"
    SABAR = "sabar"
    KAIRA = "kaira"
    SUMUL = "sumul"
    PANCHMAHAL = "panchmahal"
    BARODA = "baroda"
    VALSAD = "valsad"
    RAJKOT = "rajkot"
    BHAVNAGAR = "bhavnagar"
    MEHSANA = "mehsana"
    SURENDRANAGAR = "surendranagar"
    JAMNAGAR = "jamnagar"
    GANDHINAGAR = "gandhinagar"
    BHARUCH = "bharuch"
    KUTCH = "kutch"
    BOTAD = "botad"
    JUNAGADH = "junagadh"
    AMRELI = "amreli"
    MORBI = "morbi"
    PORBANDAR = "porbandar"
    AHMEDABAD = "ahmedabad"


# Brand / spelling variants that farmer-source APIs return for a union, mapped to
# the canonical UnionName value. A union is often returned by its dairy brand
# (Kutch -> "Sarhad", Mehsana -> "Dudhsagar") or an alternate spelling
# ("Kachchh", "Banaskantha"). Normalizing through this map lets union-scoped
# features (e.g. scheme lookup) resolve a farmer's union regardless of which
# name the source returns. Keys are lowercase; values equal a UnionName value.
UNION_NAME_ALIASES: dict[str, str] = {
    "sarhad": UnionName.KUTCH.value,
    "kachchh": UnionName.KUTCH.value,
    "kutchh": UnionName.KUTCH.value,
    "banaskantha": UnionName.BANAS.value,
    "dudhsagar": UnionName.MEHSANA.value,
    "sursagar": UnionName.SURENDRANAGAR.value,
}


def canonical_union_name(name: str | None) -> str:
    """Normalize a raw union name to its canonical ``UnionName`` value.

    Trims and lowercases the input, then maps known brand/spelling variants
    (see :data:`UNION_NAME_ALIASES`) to the canonical union. Returns the cleaned
    input unchanged when no alias applies, and ``""`` for ``None``/blank.
    """
    if not name:
        return ""
    key = name.strip().lower()
    return UNION_NAME_ALIASES.get(key, key)


# Canonical union names that must not book artificial-insemination calls.
# Compare against ``canonical_union_name`` output so brand/spelling aliases
# (e.g. "sarhad" for Kutch) hit the same entry.
AI_CALL_BANNED_UNIONS: frozenset[str] = frozenset({UnionName.KUTCH.value})
# Agent-facing English line (farmer context, translation-pipeline prompt, tool
# on lang_code=en). Gujarati/Hindi are selected by union_banned_message() and
# by the post-translation short-circuit so TranslateGemma cannot paraphrase.
UNION_BANNED_MESSAGE = "Kindly contact your Milk Society to book the service."
UNION_BANNED_MESSAGE_GU = "કૃપા કરીને આપની દૂધ મંડળીનો સંપર્ક કરશો."
UNION_BANNED_MESSAGE_HI = "कृपया सेवा बुक करने के लिए अपनी दूध मंडली से संपर्क करें।"
_UNION_BANNED_BY_LANG: dict[str, str] = {
    "en": UNION_BANNED_MESSAGE,
    "english": UNION_BANNED_MESSAGE,
    "gu": UNION_BANNED_MESSAGE_GU,
    "gujarati": UNION_BANNED_MESSAGE_GU,
    "hi": UNION_BANNED_MESSAGE_HI,
    "hindi": UNION_BANNED_MESSAGE_HI,
}
UNION_BANNED_MESSAGE_VARIANTS: frozenset[str] = frozenset(_UNION_BANNED_BY_LANG.values())


def union_banned_message(lang: str | None) -> str:
    """Canned AI-call ban line for ``lang`` (en/gu/hi). Unknown/missing → English."""
    key = (lang or "").strip().lower()
    return _UNION_BANNED_BY_LANG.get(key, UNION_BANNED_MESSAGE)


def is_ai_call_banned_union(name: str | None) -> bool:
    """True when ``name`` canonicalizes to a union banned from AI-call booking."""
    canonical = canonical_union_name(name)
    return bool(canonical) and canonical in AI_CALL_BANNED_UNIONS


def any_union_banned_from_ai_calls(names: list[str] | None) -> bool:
    """True when any entry in ``names`` is banned from AI-call booking.

    Farmer context stores unions as ``strip().lower()`` without alias mapping,
    so this canonicalizes each name before checking :data:`AI_CALL_BANNED_UNIONS`.
    An empty/missing list is not banned.
    """
    return any(is_ai_call_banned_union(name) for name in (names or []))


def resolve_supported_unions(union_names: list[str] | None, supported_unions: set[str]) -> list[str]:
    """Canonicalize raw union names and keep only supported values.

    Preserves first-seen order and removes duplicates, so callers can safely pick
    index 0 as the preferred supported union.
    """
    if not union_names:
        return []

    resolved: list[str] = []
    seen: set[str] = set()
    for union_name in union_names:
        canonical = canonical_union_name(union_name)
        if not canonical or canonical not in supported_unions or canonical in seen:
            continue
        seen.add(canonical)
        resolved.append(canonical)
    return resolved
