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
