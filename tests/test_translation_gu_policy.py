"""Tests for the Gujarati post-translation normalization + gu_term_policy union
(§14, chat-facing — the part already merged). Pins that the policy decision is
actually applied (e.g. વોડકી → પાડી), the base script/term fixups work, and
non-Gujarati text passes through untouched.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

from app.services.translation import (
    VOICE_GU_PREFERRED_TRANSLATION_RULES,
    _apply_protected_output,
    _buffered_protected_stream,
    _protected_output_triggers,
    _post_normalize_gu_translation,
    GU_TERM_POLICY,
    GU_TERM_REPLACEMENTS,
    translation_channel,
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
    assert len(GU_TERM_REPLACEMENTS) > 0
    assert isinstance(GU_TERM_POLICY.get("forbidden"), dict)


def test_policy_forbidden_term_vodki_replaced():
    forbidden = GU_TERM_POLICY.get("forbidden", {})
    if "વોડકી" not in forbidden:
        pytest.skip("policy term 'વોડકી' no longer present")
    expected = forbidden["વોડકી"]  # 'પાડી'
    out = _post_normalize_gu_translation("આ વોડકી છે", "gujarati")
    assert expected in out and "વોડકી" not in out


@pytest.mark.parametrize("raw", [
    "ઓર્ગેનિક ખાતર વાપરો",
    "જવિૈ ક ખાતર વાપરો",
    "ઓર્ગેનિર્ગે ક ખાતર વાપરો",
    "organic manure વાપરો",
])
def test_chat_organic_variants_are_canonicalized(raw):
    out = _post_normalize_gu_translation(raw, "gujarati")
    assert "જૈવિક" in out
    assert "ઓર્ગેનિક" not in out
    assert "જવિૈ ક" not in out
    assert "ઓર્ગેનિર્ગે ક" not in out
    assert "organic" not in out.lower()


def test_voice_channel_does_not_force_chat_only_organic_normalization():
    with translation_channel("voice"):
        out = _post_normalize_gu_translation("ઓર્ગેનિક ખાતર વાપરો", "gujarati")
    assert "ઓર્ગેનિક" in out
    assert "જૈવિક" not in out


def test_term_replacements_are_ordered_longest_first():
    lengths = [len(source) for source, _, _, _ in GU_TERM_REPLACEMENTS]
    assert lengths == sorted(lengths, reverse=True)


def test_strip_outer_trims_whitespace():
    out = _post_normalize_gu_translation("  આ ગાય ગાભણ છે  ", "gujarati", strip_outer=True)
    assert out == out.strip() and not out.startswith(" ")


def test_collapses_extra_spaces_after_removal():
    # red-colour removal leaves double spaces; they should collapse to one
    out = _post_normalize_gu_translation("ગાય  red colour  દૂધ", "gujarati")
    assert "  " not in out


def _apply_source_gated(source: str, output: str) -> str:
    triggers = _protected_output_triggers(source, "gu")
    return _apply_protected_output(output, triggers)


def test_bull_correction_is_source_gated_without_rewriting_bullock():
    assert _apply_source_gated("The bull needs treatment", "બળદને સારવાર જોઈએ") == (
        "બુલને સારવાર જોઈએ"
    )
    assert _apply_source_gated("The bullock needs treatment", "બળદને સારવાર જોઈએ") == (
        "બળદને સારવાર જોઈએ"
    )
    assert _apply_source_gated("The bull and bullock", "બળદ અને બળદ") == "બળદ અને બળદ"


def test_insemination_correction_is_source_gated_and_bare_ai_is_ignored():
    assert _apply_source_gated("Artificial insemination service", "ગર્ભાધાન સેવા") == (
        "બીજદાન સેવા"
    )
    assert _apply_source_gated("Conception after insemination", "ગર્ભાધાન") == "ગર્ભાધાન"
    assert _apply_source_gated("Amul AI helpline", "અમૂલ એ.આઈ. હેલ્પલાઇન") == (
        "અમૂલ એ.આઈ. હેલ્પલાઇન"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "input_chunks", "expected"),
    [
        ("The bull needs treatment", ["બ", "ળદને સારવાર"], "બુલને સારવાર"),
        ("The bullock needs treatment", ["બ", "ળદને સારવાર"], "બળદને સારવાર"),
        ("Artificial insemination service", ["ગર્ભા", "ધાન સેવા"], "બીજદાન સેવા"),
        ("Amul AI helpline", ["અમૂલ એ.", "આઈ. હેલ્પલાઇન"], "અમૂલ એ.આઈ. હેલ્પલાઇન"),
    ],
)
async def test_source_gated_corrections_are_stream_boundary_safe(
    source, input_chunks, expected
):
    async def source_chunks():
        for chunk in input_chunks:
            yield chunk

    stream = _buffered_protected_stream(
        source_chunks(), _protected_output_triggers(source, "gu")
    )
    assert "".join([chunk async for chunk in stream]) == expected


def test_meaning_first_terms_are_not_globally_rewritten():
    text = "બળદ ગર્ભાધાન પછી તણાવમાં છે. પોટેશિયમ પરમેંગેનેટ વાપરો."
    assert _post_normalize_gu_translation(text, "gu") == text


def test_mastitis_and_tdn_outputs_are_still_canonicalized():
    out = _post_normalize_gu_translation(
        "આઉનો/બાવલાનો સોજો માટે ટીડીએન આપો", "gu"
    )
    assert "આંચળનો સોજો" in out
    assert "કુલ પાચ્ય પોષક તત્વ (ટીડીએન)" in out


def test_voice_prompt_distinguishes_stress_from_trauma():
    joined = "\n".join(VOICE_GU_PREFERRED_TRANSLATION_RULES)
    assert "'તણાવ' for stress" in joined
    assert "'માનસિક આઘાત' only for explicit mental trauma" in joined
