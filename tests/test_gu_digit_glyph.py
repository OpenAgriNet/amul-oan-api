"""Gujarati/Hindi digit post-translation fixes.

1. TranslateGemma confuses the digit ૫ with the letter પ (glyph repair).
2. TranslateGemma often leaves ASCII digits in Indic output — chat forces
   target-script digits (ASCII→Gujarati / ASCII→Devanagari). Voice is skipped
   so TTS can still wordify ASCII via normalize_numbers_for_tts.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

from app.services.translation import (
    _normalize_digit_script_for_target,
    _post_normalize_gu_translation,
    translation_channel,
)


def _norm(text: str, target_lang: str = "gu") -> str:
    return _post_normalize_gu_translation(text, target_lang=target_lang, strip_outer=True)


# ── Glyph repair (૫ ↔ પ) ──────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("૧પ", "૧૫"),      # 15, letter after digit
    ("પ૦", "૫૦"),      # 50, letter before digit
    ("૨પ લિટર", "૨૫ લિટર"),
])
def test_pa_adjacent_to_gujarati_digit_becomes_five(raw, expected):
    assert expected in _norm(raw)


@pytest.mark.parametrize("word", ["પાણી", "પશુ", "પીઠ"])
def test_standalone_pa_is_untouched(word):
    """No digit adjacency means no repair — these are ordinary words."""
    assert word in _norm(word)


# ── Digit-script helper (ASCII → target) ──────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("1. Wait 17.83 days. Range 90-100.", "૧. Wait ૧૭.૮૩ days. Range ૯૦-૧૦૦."),
    ("step 2 then 4", "step ૨ then ૪"),
    ("already ૧૫ ok", "already ૧૫ ok"),
    ("50%, 2/3, 080-3545", "૫૦%, ૨/૩, ૦૮૦-૩૫૪૫"),
])
def test_helper_ascii_to_gujarati(raw, expected):
    assert _normalize_digit_script_for_target(raw, "gu") == expected
    assert _normalize_digit_script_for_target(raw, "gujarati") == expected


@pytest.mark.parametrize("raw,expected", [
    ("1. Wait 17.83 days. Range 90-100.", "१. Wait १७.८३ days. Range ९०-१००."),
    ("step 2 then 4", "step २ then ४"),
    ("already १५ ok", "already १५ ok"),
    ("50%, 2/3, 080-3545", "५०%, २/३, ०८०-३५४५"),
])
def test_helper_ascii_to_devanagari(raw, expected):
    assert _normalize_digit_script_for_target(raw, "hi") == expected
    assert _normalize_digit_script_for_target(raw, "hindi") == expected


def test_helper_no_hindi_gujarati_cross_mapping():
    # Devanagari digits are not rewritten for a Gujarati target.
    assert _normalize_digit_script_for_target("१५ दिन", "gu") == "१५ दिन"
    # Gujarati digits are not rewritten for a Hindi target.
    assert _normalize_digit_script_for_target("૧૫ દિવસ", "hi") == "૧૫ દિવસ"


def test_helper_non_target_passthrough():
    assert _normalize_digit_script_for_target("Wait 15 days", "en") == "Wait 15 days"
    assert _normalize_digit_script_for_target("", "gu") == ""


def test_helper_preserves_markdown_link_destinations():
    raw = "યોજના [જુઓ](https://example.com/schemes/2026/form-1.pdf) 2 દિવસ"
    out = _normalize_digit_script_for_target(raw, "gu")
    assert out == "યોજના [જુઓ](https://example.com/schemes/2026/form-1.pdf) ૨ દિવસ"


def test_helper_preserves_bare_urls():
    raw = "Visit https://example.com/schemes/2026/form-1.pdf in 2 days"
    out = _normalize_digit_script_for_target(raw, "hi")
    assert out == "Visit https://example.com/schemes/2026/form-1.pdf in २ days"


def test_helper_preserves_inline_code_spans():
    raw = "Use `scheme-2026-v1` and wait 3 days"
    out = _normalize_digit_script_for_target(raw, "gu")
    assert out == "Use `scheme-2026-v1` and wait ૩ days"


# ── Wired through _post_normalize_gu_translation (chat) ───────────────────────

def test_chat_gujarati_post_normalize_converts_ascii_digits():
    out = _norm("1. Mix. Wait 17.83 days. Range 90-100.")
    assert out == "૧. Mix. Wait ૧૭.૮૩ days. Range ૯૦-૧૦૦."
    assert not any(ch.isdigit() and ord(ch) < 128 for ch in out)


def test_chat_hindi_post_normalize_converts_ascii_digits_keeps_danda():
    out = _norm("1. Wait 17.83 days। next", target_lang="hi")
    assert out == "१. Wait १७.८३ days। next"
    assert "।" in out
    assert not any(ch.isdigit() and ord(ch) < 128 for ch in out)


def test_voice_channel_does_not_force_gujarati_digit_script():
    """Voice must keep ASCII so normalize_numbers_for_tts can wordify them."""
    with translation_channel("voice"):
        out = _post_normalize_gu_translation("Wait 15 days", "gu", strip_outer=True)
    assert "પંદર" in out
    assert "૧૫" not in out
