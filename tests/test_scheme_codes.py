"""Tests for the shared central-scheme code/alias map (agents/tools/scheme_codes.py).

The Bharat Vistaar BPP matches `item.descriptor.name` against the scheme CODE
and nothing else — it answers "kcc" and returns an empty catalogue for "KCC ",
"Kisan Credit Card", "crop insurance", "પાક વીમો". Every one of those is
something a farmer (or the model paraphrasing one) actually says, and
production chat is mostly Gujarati, so the alias map is load-bearing rather than
cosmetic.
"""
import sys
from pathlib import Path
from typing import get_args

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.tools.scheme_codes import (  # noqa: E402
    SCHEME_ALIASES,
    SCHEME_CODES,
    SCHEME_LABELS,
    SchemeCode,
    resolve_scheme_code,
    scheme_names_sentence,
)


def test_literal_and_code_tuple_cannot_drift():
    """The Literal is spelled out by hand (Literal[*tuple] is not valid here),
    so pin it against the tuple — otherwise adding a 16th code to one and not
    the other silently un-constrains or over-constrains the tool schema."""
    assert get_args(SchemeCode) == SCHEME_CODES


def test_every_code_has_aliases_and_a_farmer_facing_label():
    assert set(SCHEME_ALIASES) == set(SCHEME_CODES)
    assert set(SCHEME_LABELS) == set(SCHEME_CODES)
    for code in SCHEME_CODES:
        assert code in {a.lower() for a in SCHEME_ALIASES[code]}, code


def test_no_alias_is_claimed_by_two_codes():
    seen: dict[str, str] = {}
    for code, aliases in SCHEME_ALIASES.items():
        for alias in aliases:
            key = alias.casefold()
            assert key not in seen, f"{alias!r} claimed by {seen.get(key)} and {code}"
            seen[key] = code


@pytest.mark.parametrize(
    "phrase,expected",
    [
        # The exact cases the brief calls out.
        ("Kisan Credit Card", "kcc"),
        ("KCC", "kcc"),
        ("kcc", "kcc"),
        ("PM-KISAN", "pmkisan"),
        ("PM Kisan", "pmkisan"),
        ("pmkisan", "pmkisan"),
        ("crop insurance", "pmfby"),
        ("fasal bima", "pmfby"),
        ("પાક વીમો", "pmfby"),
        ("soil health card", "shc"),
        ("જમીન આરોગ્ય કાર્ડ", "shc"),
        # Hindi, since Hindi chat is live in production.
        ("किसान क्रेडिट कार्ड", "kcc"),
        ("फसल बीमा", "pmfby"),
        ("मृदा स्वास्थ्य कार्ड", "shc"),
        # Sensible equivalents for the rest of the fifteen.
        ("drip irrigation", "pdmc"),
        ("organic farming", "pkvy"),
        ("beekeeping", "nbhm"),
        ("farm mechanisation", "smam"),
        ("agriculture infrastructure fund", "aif"),
        ("rainfed area development", "rad"),
        ("national food security mission", "nfsm"),
        ("seed authentication", "sathi"),
        ("price support scheme", "pmasha"),
        ("fertilizer sales", "ffs"),
        ("irrigation scheme", "pmksy"),
    ],
)
def test_farmer_phrasing_resolves_to_the_right_code(phrase, expected):
    assert resolve_scheme_code(phrase) == expected


@pytest.mark.parametrize(
    "sentence,expected",
    [
        ("tell me about the Kisan Credit Card please", "kcc"),
        ("પાક વીમો શું છે?", "pmfby"),
        ("मुझे PM-KISAN की जानकारी चाहिए", "pmkisan"),
        ("how do I apply for a soil health card", "shc"),
    ],
)
def test_codes_are_found_inside_a_whole_question(sentence, expected):
    assert resolve_scheme_code(sentence) == expected


@pytest.mark.parametrize(
    "phrase",
    ["", None, "   ", "schemes", "what schemes do I get", "cattle insurance",
     "banas", "milk price", "મારી યોજનાઓ"],
)
def test_non_central_questions_resolve_to_nothing(phrase):
    """None is the signal to SKIP the Bharat Vistaar leg. A false positive here
    would send a union question down a leg that cannot answer it; a false
    negative costs a guaranteed-empty ~2.2s round trip."""
    assert resolve_scheme_code(phrase) is None


def test_short_codes_do_not_match_inside_other_words():
    """`rad`, `aif`, `ffs`, `shc` are three letters. Without word boundaries
    they would fire on "radish", "waif", "caffs"..."""
    for phrase in ("radish crop", "my waif", "shocking", "traffic"):
        assert resolve_scheme_code(phrase) is None, phrase


def test_longest_alias_wins():
    """"kisan samman nidhi" must not be shadowed by a shorter overlapping
    alias of a different code."""
    assert resolve_scheme_code("pradhan mantri kisan samman nidhi") == "pmkisan"
    assert resolve_scheme_code("pradhan mantri fasal bima yojana") == "pmfby"


def test_scheme_names_sentence_leaks_no_internal_codes():
    """agrinet_system.md: "Do not mention internal tool mechanics". The
    farmer-visible fallback names schemes, never codes."""
    sentence = scheme_names_sentence()
    assert "Kisan Credit Card" in sentence
    for code in SCHEME_CODES:
        assert code not in sentence.lower().split(), code
