"""Nearest government veterinary office lookup, from the AH Department sheet.

Deliberately not a Marqo index. The source is a 3.2k-row lookup table whose rows
read almost identically ("Veterinary Dispensary, <village>, <taluka>"), so they
embed to nearly the same vector, and location matching is a filter over
village/taluka, not a semantic similarity. It is a flat JSON asset plus this
tool instead — see scripts/build_vet_offices.py for how the asset is produced.

Location comes from the user's profile (FarmerContext), not from the model.
The sheet is keyed on taluka, which the profile does not carry, so when village
and district are not enough the tool asks for the taluka or village name and
matches it as text — there are no coordinates in the sheet to measure with.

The asset is a faithful transcription: the department's spellings, variants and
typos are all carried across untouched. Every judgement about which of them mean
the same place is made here instead, so the data stays theirs and the reading of
it stays ours, in one file that can be corrected without a re-import.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic_ai import RunContext
from rapidfuzz import fuzz, process

from agents.deps import FarmerContext
from helpers.utils import get_logger

logger = get_logger(__name__)

ASSET_NAME = "vet_offices.json"

# Offices a farmer can actually take an animal to, best first. The other 27
# categories in the sheet (150 rows: ICDP Office, Frozen Semen Bank, Disease
# Investigation Office, breeding farms) are administrative and are never
# offered. Drop a line to stop offering that tier.
CATEGORY_RANK = {
    "Veterinary Polyclinic": 0,
    "Veterinary Dispensary": 1,
    "First Aid Veterinary Centre": 2,
    "ICDP Group Centre": 3,
    "ICDP Sub-Centre": 4,
}

# The sheet spells some districts more than one way, and we do not rewrite the
# sheet — it is the department's data, and a transcription that quietly renames
# places is worse than one that keeps the mess visible. So the variants are
# reconciled here, at lookup time, where the list is short enough to read and
# argue with. Casing and punctuation already fold on their own (BHAVNAGAR,
# "Sabar Kantha", "Vav-Tharad"); only genuinely different spellings need a line.
#
# Fuzzy matching cannot do this job: Dahod/Dohad score 60, but so do Anand/Patan
# and Botad/Dohad, and Banaskantha/Sabarkantha — two real, separate districts —
# score 72. Kutch/Kachchh scores below 55. There is no threshold that merges the
# variants without also merging real districts, so the list is explicit.
DISTRICT_ALIASES = {
    "kutch": "kachchh",
    "dohad": "dahod",
    "narmda": "narmada",
    "vadpdara": "vadodara",
    "valasad": "valsad",
    "panchamhal": "panchmahal",
    "panchmahals": "panchmahal",
    "porbandarhightech": "porbandar",
}

MAX_RESULTS = 3
# Place names reach us transliterated by the chat translation pipeline, so
# "Vijapur"/"Vijaypur" and "Bavla"/"Bavala" have to match. 80 accepts those and
# still separates real neighbours like Mehsana and Mehmadabad.
_FUZZY_CUTOFF = 80
# Taluka spellings vary the same way, but there are 360 of them — too many to
# list by hand, and unlike districts they are always compared inside one
# district, which makes fuzziness affordable. Names this close are searched
# together: it catches Mehsana/Mahesana (80) and Shihor/Sihor (91), and where it
# overreaches — Halol/Kalol also score 80 — the extra office is ranked under the
# exact match and names its own taluka, so it reads as a real nearby office
# rather than a wrong one. Unjha/Unza (67) is below this and stays split.
_SIBLING_TALUKA_RATIO = 75


def _norm(value: Any) -> str:
    """Comparison key for a place name: lowercase, letters and digits only."""
    return re.sub(r"[^a-z0-9]", "", str(value or "").strip().lower())


def _district_key(value: Any) -> str:
    """Comparison key for a district, folding the sheet's spelling variants."""
    key = _norm(value)
    return DISTRICT_ALIASES.get(key, key)


def _asset_path() -> Optional[Path]:
    candidates = [
        Path.cwd() / "assets" / ASSET_NAME,
        Path(__file__).resolve().parents[2] / "assets" / ASSET_NAME,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


@lru_cache(maxsize=1)
def _load_offices() -> Tuple[Dict[str, Any], ...]:
    """Every treatable office from the asset, as an immutable tuple.

    Cached for the process: the asset is static and read-only, so the parse
    happens once rather than on every call.
    """
    path = _asset_path()
    if path is None:
        logger.error("Vet office asset %s not found", ASSET_NAME)
        return ()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("Could not read vet office asset %s: %s", path, e)
        return ()

    offices = [
        office
        for office in data.get("offices", [])
        if office.get("category") in CATEGORY_RANK
    ]
    logger.info("Loaded %s treatable vet offices from %s", len(offices), path)
    return tuple(offices)


@lru_cache(maxsize=1)
def _taluka_index() -> Dict[str, List[Dict[str, Any]]]:
    index: Dict[str, List[Dict[str, Any]]] = {}
    for office in _load_offices():
        index.setdefault(_norm(office.get("taluka")), []).append(office)
    index.pop("", None)
    return index


@lru_cache(maxsize=1)
def _village_index() -> Dict[str, List[Dict[str, Any]]]:
    index: Dict[str, List[Dict[str, Any]]] = {}
    for office in _load_offices():
        index.setdefault(_norm(office.get("village")), []).append(office)
    index.pop("", None)
    return index


def _match_place(
    spoken: str, index: Dict[str, List[Dict[str, Any]]], district: str
) -> Optional[str]:
    """Resolve a spoken place name to a key in `index`, within the district.

    Gujarat reuses village names across districts freely, so a fuzzy match over
    all 3k of them lands a Kachchh user at a Surat dispensary. When we know the
    district the search stays inside it and returns nothing rather than reaching
    outside — being asked for a taluka beats being sent 400km away. The whole
    index is only searched when the district is unknown, or is a spelling the
    sheet does not carry at all.
    """
    key = _norm(spoken)
    if not key:
        return None

    names = list(index.keys())
    district_key = _district_key(district)
    if district_key:
        scoped = [
            name
            for name, offices in index.items()
            if any(_district_key(o.get("district")) == district_key for o in offices)
        ]
        if scoped:
            # An exact name elsewhere in Gujarat must not override a district we
            # can resolve. The directory does not list every village, and place
            # names repeat freely, so doing the global exact check first can turn
            # a missing local village into a confident result hundreds of
            # kilometres away.
            if key in scoped:
                return key
            if key in index:
                return None
            hit = process.extractOne(
                key, scoped, scorer=fuzz.WRatio, score_cutoff=_FUZZY_CUTOFF
            )
            return hit[0] if hit else None

    if key in index:
        return key

    hit = process.extractOne(
        key, names, scorer=fuzz.WRatio, score_cutoff=_FUZZY_CUTOFF
    )
    return hit[0] if hit else None


def _resolve_spoken_place(spoken: str, district: str) -> Tuple[Optional[str], Optional[str]]:
    """Resolve the user's place name into (taluka key, village key).

    We ask for "your taluka or village" and the user answers with whichever
    they know, so the name has to be tried against both. Order matters: an exact
    hit in EITHER index beats a fuzzy hit in the other. Without that rule a real
    village name is captured by a same-sounding taluka elsewhere in the state —
    "Bagodara", a village in Bavla, scored 80 against Amreli's Bagsara taluka and
    sent the user 300km away, because the taluka index was simply tried first.

    A name that is both (Wankaner is a taluka and a village in it) returns both
    keys: the taluka scopes the search, the village floats their own office up.
    """
    key = _norm(spoken)
    if not key:
        return None, None

    in_taluka = key in _taluka_index()
    in_village = key in _village_index()
    if in_taluka or in_village:
        return (key if in_taluka else None), (key if in_village else None)

    taluka_key = _match_place(spoken, _taluka_index(), district)
    if taluka_key is not None:
        return taluka_key, None
    return None, _match_place(spoken, _village_index(), district)


def _in_district(offices: List[Dict[str, Any]], district: str) -> List[Dict[str, Any]]:
    """Narrow a candidate list to the user's district.

    Place names repeat across Gujarat — 105 village names do, and the taluka
    "City" covers seven districts — so an unnarrowed match names the right
    village in the wrong part of the state. Falls back to the unnarrowed list
    when the district is unknown or the profile spells it in a way the sheet
    does not carry, since a wider answer still names each office's own district.
    """
    district_key = _district_key(district)
    if not district_key:
        return offices
    scoped = [o for o in offices if _district_key(o.get("district")) == district_key]
    return scoped or offices


def _sibling_taluka_keys(taluka_key: str, district: str) -> List[str]:
    """The matched taluka plus its unmerged spelling variants in that district."""
    index = _taluka_index()
    keys = [taluka_key]
    district_key = _district_key(district) or _district_key(
        index.get(taluka_key, [{}])[0].get("district")
    )
    for name, offices in index.items():
        if name == taluka_key:
            continue
        if district_key and not any(
            _district_key(o.get("district")) == district_key for o in offices
        ):
            continue
        if fuzz.ratio(name, taluka_key) >= _SIBLING_TALUKA_RATIO:
            keys.append(name)
    return keys


def _rank(
    offices: List[Dict[str, Any]], village_key: str, taluka_key: str = ""
) -> List[Dict[str, Any]]:
    """Order candidates: own village, then exact taluka, then by office tier.

    Also drops repeats. The sheet lists a handful of offices twice and the asset
    keeps every row it has, so the de-duplication belongs here rather than in a
    transcription that is meant to stay faithful.
    """
    ordered = sorted(
        offices,
        key=lambda o: (
            0 if village_key and _norm(o.get("village")) == village_key else 1,
            0 if not taluka_key or _norm(o.get("taluka")) == taluka_key else 1,
            CATEGORY_RANK.get(o.get("category"), 99),
            str(o.get("village") or ""),
        ),
    )

    seen = set()
    unique = []
    for office in ordered:
        key = (
            _norm(office.get("village")),
            _norm(office.get("taluka")),
            _district_key(office.get("district")),
            office.get("category"),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(office)
    return unique


def _format(offices: List[Dict[str, Any]]) -> str:
    """Flat labelled list that is easy for the chat model to relay accurately."""
    lines = [f"Veterinary offices listed for the user's location ({len(offices)}):"]
    for i, office in enumerate(offices, 1):
        lines.append(f"{i}. Official Gujarati office type: {office.get('office_type_gu')}")
        lines.append(f"   Office type in English: {office.get('category')}")
        lines.append(f"   Village: {office.get('village')}")
        lines.append(f"   Taluka: {office.get('taluka')}, District: {office.get('district')}")
    lines.append(
        "The directory holds no phone number or street address for these offices. "
        "Give the user the office type, village and taluka. The directory has no "
        "coordinates or distances: describe these as offices in their village or "
        "taluka, never as the nearest or closest office."
    )
    return "\n".join(lines)


async def find_nearby_vet_offices(
    ctx: RunContext[FarmerContext],
    taluka: str = "",
) -> str:
    """Find government veterinary offices listed for the user's location.

    Use this whenever the user asks where the nearest veterinary hospital,
    animal dispensary, pashu dawakhanu or first-aid centre is. An unambiguous
    village and district are read from the user's profile. When the
    profile is missing or carries several locations, ask for their taluka or
    village as directed by the tool result.

    Args:
        ctx: Tool context.
        taluka: A place name the user provided — their taluka (subdistrict) or
            their village, whichever they gave. Spell it in English letters.
            Leave empty on the first call: the tool asks for a place name itself
            when the profile is not enough to locate them.
    """
    if not _load_offices():
        return "The veterinary office directory is unavailable right now."

    village = getattr(ctx.deps, "farmer_village", None) or ""
    district = getattr(ctx.deps, "farmer_district", None) or ""

    spoken = (taluka or "").strip()
    if spoken and not _norm(spoken):
        # Nothing Latin in it, so every key in the index scores zero against it.
        # Say so instead of asking the same question again: the sheet is written
        # in English and the user is answering in Gujarati, which is a request
        # to transliterate, not a reason to give up on them.
        logger.info("Vet office lookup got a non-Latin place name: %r", spoken)
        return (
            "That place name did not come through in English letters. Ask again "
            "and call this tool with the name spelled in English."
        )

    # What the user said wins over the profile: they are either answering our
    # question or correcting what we had.
    taluka_key, spoken_village_key = (
        _resolve_spoken_place(spoken, district) if spoken else (None, None)
    )
    village_key = spoken_village_key or (
        _match_place(village, _village_index(), district) if village else None
    )

    if taluka_key is None and village_key is None:
        logger.info(
            "Vet office lookup needs a spoken taluka: village=%r district=%r taluka_said=%r",
            village,
            district,
            taluka,
        )
        known = f" They are in {district} district." if district else ""
        return (
            "Could not place the user from their profile."
            + known
            + " Ask the user which taluka or village they live in, then call this "
            "tool again with that name."
        )

    if taluka_key is None:
        # Located by village: the sheet row for their own village carries the
        # taluka the profile never had, so widen the search to that taluka.
        village_offices = _in_district(_village_index()[village_key], district)
        taluka_key = _norm(village_offices[0].get("taluka"))

    index = _taluka_index()
    candidates = _in_district(
        [
            office
            for key in _sibling_taluka_keys(taluka_key, district)
            for office in index.get(key, [])
        ],
        district,
    )
    if not candidates:
        return (
            "No veterinary office is listed for that taluka. Ask the user for a "
            "nearby larger town or taluka name and try again."
        )

    results = _rank(candidates, village_key or "", taluka_key)[:MAX_RESULTS]
    logger.info(
        "Vet office lookup: village=%r district=%r taluka_said=%r matched_taluka=%r hits=%s",
        village,
        district,
        taluka,
        taluka_key,
        len(results),
    )
    return _format(results)
