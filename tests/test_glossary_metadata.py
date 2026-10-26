from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from agents.tools.terms import (
    ALIAS_TO_CANONICAL_EN,
    TermPair,
    get_mini_glossary_for_text,
)
from helpers.glossary_validation import (
    GlossaryValidationError,
    validate_glossary_assets,
)


ROOT = Path(__file__).resolve().parents[1]
GLOSSARY = json.loads(
    (ROOT / "assets/glossary_terms.json").read_text(encoding="utf-8")
)
POLICY = json.loads(
    (ROOT / "assets/gu_term_policy.json").read_text(encoding="utf-8")
)


def _row(english: str) -> dict:
    return next(row for row in GLOSSARY if row["en"].casefold() == english.casefold())


def _assert_invalid(glossary: list[dict], policy: dict, match: str) -> None:
    with pytest.raises(GlossaryValidationError, match=match):
        validate_glossary_assets(glossary, policy)


def test_glossary_has_expected_unique_concepts() -> None:
    concepts = {" ".join(row["en"].strip().casefold().split()) for row in GLOSSARY}
    assert len(GLOSSARY) == 752
    assert len(concepts) == 752


def test_policy_owns_only_forbidden_outputs() -> None:
    assert set(POLICY) == {"forbidden"}
    assert len(POLICY["forbidden"]) == 85
    for removed_global_rule in ("બળદ", "ગર્ભાધાન", "તણાવ", "પોટેશિયમ પરમેંગેનેટ"):
        assert removed_global_rule not in POLICY["forbidden"]


def test_voice_only_and_former_virtual_concepts_are_materialized() -> None:
    assert _row("Silage")["gu"] == "સાયલેજ"
    assert _row("Feed supplements")["gu"] == "ફીડ સપ્લીમેન્ટ્સ"
    assert _row("Appetite and milk production support nutrients")["gu"] == (
        "ભૂખ અને દૂધ ઉત્પાદન સુધારતા પોષક તત્વો / ફીડ એડિટિવ્સ"
    )


def test_meaning_first_canonical_values() -> None:
    expected = {
        "Bull": "બુલ",
        "Bullock": "બળદ",
        "Conception/Pregnancy": "ગર્ભાધાન",
        "ARTIFICIAL INSEMINATION": "કૃત્રિમ બીજદાન",
        "Stress": "તણાવ",
        "Potassium Permanganate": "પોટેશિયમ પરમેંગેનેટ",
        "Potassium Permanganate Solution": "પોટેશિયમ પરમેંગેનેટના દ્રાવણ",
        "Mastitis": "આંચળનો સોજો",
        "udder infection": "આંચળનો સોજો",
        "TDN — Total Digestible Nutrients": "કુલ પાચ્ય પોષક તત્વ (ટીડીએન)",
    }
    assert {english: _row(english)["gu"] for english in expected} == expected


def test_policy_metadata_moved_to_owning_rows() -> None:
    assert "fat (dairy)" in _row("fat")["en_input_aliases"]
    potassium_aliases = _row("Potassium Permanganate Solution")[
        "en_input_aliases"
    ]
    assert "potassium permanganate antiseptic solution" in potassium_aliases
    assert "pp solution" in potassium_aliases
    assert "feed supplement" in _row("Feed supplements")["en_input_aliases"]
    assert "mammary gland" in _row("Udder")["en_input_aliases"]
    assert any(
        "આઉનો/બાવલાનો સોજો" in row.get("gu_input_aliases", [])
        for row in GLOSSARY
    )


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("fat (dairy)", "fat"),
        ("pp solution", "potassium permanganate solution"),
        ("feed supplement", "feed supplements"),
        ("mammary gland", "udder"),
        ("dewormer", "deworming"),
    ],
)
def test_english_aliases_resolve_to_one_canonical_concept(
    alias: str, canonical: str
) -> None:
    assert ALIAS_TO_CANONICAL_EN[alias] == canonical


def test_aliases_feed_mini_glossary() -> None:
    assert "Silage -> સાયલેજ" in get_mini_glossary_for_text("silage")
    assert "Feed supplements -> ફીડ સપ્લીમેન્ટ્સ" in get_mini_glossary_for_text(
        "feed supplement"
    )
    assert "Potassium Permanganate Solution -> પોટેશિયમ પરમેંગેનેટના દ્રાવણ" in (
        get_mini_glossary_for_text("pp solution")
    )


def test_optional_alias_fields_default_to_empty_lists() -> None:
    pair = TermPair(en="Example", gu="ઉદાહરણ", transliteration="udāharaṇ")
    assert pair.en_input_aliases == []
    assert pair.gu_input_aliases == []
    assert pair.transliteration_input_aliases == []
    assert pair.gu_output_aliases == []


@pytest.mark.parametrize("legacy_field", ["preferred", "input_aliases", "allowed_aliases"])
def test_policy_rejects_removed_sections(legacy_field: str) -> None:
    policy = copy.deepcopy(POLICY)
    policy[legacy_field] = {}
    _assert_invalid(GLOSSARY, policy, "policy has unknown fields")


def test_validator_rejects_duplicate_concept() -> None:
    glossary = copy.deepcopy(GLOSSARY)
    glossary.append(copy.deepcopy(glossary[0]))
    _assert_invalid(glossary, POLICY, "duplicate English concept")


def test_validator_rejects_alias_with_multiple_owners() -> None:
    glossary = copy.deepcopy(GLOSSARY)
    _row_one = next(row for row in glossary if row["en"] == "Udder")
    _row_two = next(row for row in glossary if row["en"] == "Deworming")
    _row_one["en_input_aliases"].append("shared alias")
    _row_two["en_input_aliases"].append("shared alias")
    _assert_invalid(glossary, POLICY, "belongs to multiple concepts")


def test_validator_rejects_output_alias_with_multiple_owners() -> None:
    glossary = copy.deepcopy(GLOSSARY)
    glossary[0].setdefault("gu_output_aliases", []).append("સાંઝો વિકલ્પ")
    glossary[1].setdefault("gu_output_aliases", []).append("સાંઝો વિકલ્પ")
    _assert_invalid(glossary, POLICY, "belongs to multiple concepts")


def test_validator_rejects_canonical_forbidden_conflict() -> None:
    glossary = copy.deepcopy(GLOSSARY)
    glossary[0]["gu"] = next(iter(POLICY["forbidden"]))
    _assert_invalid(glossary, POLICY, "canonical Gujarati value.*is forbidden")


def test_current_assets_pass_validation_without_exceptions() -> None:
    validate_glossary_assets(GLOSSARY, POLICY)
