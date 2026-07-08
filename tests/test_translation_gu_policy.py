"""Tests for the Gujarati post-translation normalization + gu_term_policy union
(§14, chat-facing — the part already merged). Pins that the policy decision is
actually applied (e.g. વોડકી → પાડી), the base script/term fixups work, and
non-Gujarati text passes through untouched.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import re

import pytest

from app.services.translation import (
    _post_normalize_gu_translation,
    _build_gu_policy_replacements,
    GU_TERM_POLICY,
    GU_POLICY_REPLACEMENTS,
)


def test_non_gujarati_passthrough_unchanged():
    assert _post_normalize_gu_translation("She is pregnant", "english") == "She is pregnant"
    assert _post_normalize_gu_translation("unchanged", "hi") == "unchanged"


def test_base_pregnant_term_normalized():
    # ગર્ભવતી -> ગાભણ (base replacement)
    out = _post_normalize_gu_translation("આ ગાય ગર્ભવતી છે", "gujarati")
    assert "ગાભણ" in out and "ગર્ભવતી" not in out


def test_base_paho_latin_to_bavlu():
    # \bpaho\b (latin) -> બાવલું (base replacement)
    out = _post_normalize_gu_translation("the paho is swollen", "gu")
    assert "બાવલું" in out and "paho" not in out


def test_base_red_colour_scaffolding_removed():
    out = _post_normalize_gu_translation("ગાય red colour દૂધ", "gujarati")
    assert "red colour" not in out.lower()


def test_policy_loaded_nonempty():
    assert len(GU_POLICY_REPLACEMENTS) > 0
    assert isinstance(GU_TERM_POLICY.get("forbidden"), dict)


def test_policy_forbidden_term_vodki_replaced():
    forbidden = GU_TERM_POLICY.get("forbidden", {})
    if "વોડકી" not in forbidden:
        pytest.skip("policy term 'વોડકી' no longer present")
    expected = forbidden["વોડકી"]  # 'પાડી'
    out = _post_normalize_gu_translation("આ વોડકી છે", "gujarati")
    assert expected in out and "વોડકી" not in out


def test_build_replacements_orders_longer_keys_first():
    # phrase-level replacements must win before single-word ones
    reps = _build_gu_policy_replacements({"forbidden": {"aa": "X", "aaaa": "Y"}})
    patterns = [p for p, _ in reps]
    assert patterns[0] == re.escape("aaaa")


def test_strip_outer_trims_whitespace():
    out = _post_normalize_gu_translation("  આ ગાય ગાભણ છે  ", "gujarati", strip_outer=True)
    assert out == out.strip() and not out.startswith(" ")


def test_collapses_extra_spaces_after_removal():
    # red-colour removal leaves double spaces; they should collapse to one
    out = _post_normalize_gu_translation("ગાય  red colour  દૂધ", "gujarati")
    assert "  " not in out
