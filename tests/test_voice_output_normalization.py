"""Voice-output normalization regressions recovered during the migration audit.

Two voice-only behaviors that voice-prod had inside _post_normalize_gu_translation
but were dropped when the function was split during the merge:

  G2 — deterministic gendered caller-address stripping (a safety net beyond the
       prompt rule), voice-channel-gated.
  G3 — TranslateGemma ૫↔પ glyph-confusion repair, restored in the shared TTS
       number normalizer (normalize_numbers_for_tts).
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from app.services.translation import _post_normalize_gu_translation, translation_channel
from helpers.gujarati_numbers import normalize_numbers_for_tts


def _norm(text: str, channel: str) -> str:
    with translation_channel(channel):
        return _post_normalize_gu_translation(text, target_lang="gu", strip_outer=True)


# ── G2: gendered caller-address stripping (voice only) ───────────────────────

def test_voice_strips_gendered_caller_address():
    # "ભાઈ," as a caller address is stripped before TTS on the voice channel.
    out = _norm("ભાઈ, તમારી ગાય ને તાવ છે.", "voice")
    assert "ભાઈ" not in out
    assert "તાવ" in out  # the rest of the message is preserved


def test_chat_keeps_gendered_address_unchanged():
    # Chat must NOT run the deterministic stripper (chat is channel-isolated).
    out = _norm("ભાઈ, તમારી ગાય ને તાવ છે.", "chat")
    assert "ભાઈ" in out


def test_voice_does_not_overstrip_midword():
    # "ભાઈબંધ" (friend) embeds "ભાઈ" but is not a caller address — must survive.
    out = _norm("તમારા ભાઈબંધ ને પૂછો.", "voice")
    assert "ભાઈબંધ" in out


# ── G3: ૫↔પ glyph-confusion repair in the shared TTS number normalizer ───────

def test_digit_glyph_repair_pa_to_5_adjacent_to_digits():
    # Model emitted letter પ where digit ૫ was meant ("૧પ" = 15) -> verbalized.
    assert normalize_numbers_for_tts("૧પ") == "પંદર"


def test_digit_glyph_repair_leading_pa_before_digit():
    # "પ૦" -> "૫૦" -> પચાસ (50)
    assert normalize_numbers_for_tts("પ૦") == "પચાસ"


def test_standalone_pa_not_adjacent_to_digit_is_untouched():
    # A normal word starting with પ (not next to any digit) must not be altered.
    assert normalize_numbers_for_tts("પાણી") == "પાણી"


def test_voice_strips_placeholder_dash_after_label():
    out = _norm("લીલો ચારો: – કિ.ગ્રા.", "voice")
    assert "–" not in out
    assert ": " in out


def test_chat_keeps_placeholder_dash_after_label():
    out = _norm("લીલો ચારો: – કિ.ગ્રા.", "chat")
    assert "–" in out


def test_voice_collapses_label_scaffold_to_spoken_flow():
    out = _norm("ચારો: 10 કિ.ગ્રા.\nદાણ: 2 કિ.ગ્રા.", "voice")
    # Scaffold collapse removes label prefixes; normalize_voice_output then
    # verbalizes digits/units for TTS (10 -> દસ, 2 -> બે, કિ.ગ્રા. -> કિલોગ્રામ).
    assert "\nદાણ:" not in out
    assert "ચારો:" not in out
    assert "દાણ:" not in out
    assert out == "દસ કિલોગ્રામ, બે કિલોગ્રામ"


def test_chat_keeps_label_scaffold_structure():
    out = _norm("ચારો: 10 કિ.ગ્રા.\nદાણ: 2 કિ.ગ્રા.", "chat")
    assert "\nદાણ:" in out


def test_voice_removes_nbsp_and_zwj_noise():
    noisy = "તમારી\u00A0ગાય\u200D સારું છે"
    out = _norm(noisy, "voice")
    assert "\u00A0" not in out
    assert "\u200D" not in out


def test_chat_leaves_nbsp_and_zwj_noise_unchanged():
    noisy = "તમારી\u00A0ગાય\u200D સારું છે"
    out = _norm(noisy, "chat")
    assert "\u00A0" in out
    assert "\u200D" in out


def test_chat_enforces_feminine_self_reference_for_shakto():
    out = _norm("હું કોઈ ચોક્કસ દવાના નામ અથવા ડોઝ જણાવી શકતો નથી.", "chat")
    assert "શકતી નથી" in out
    assert "શકતો નથી" not in out


def test_voice_enforces_feminine_self_reference_for_shakto():
    out = _norm("હું કોઈ ચોક્કસ દવાના નામ અથવા ડોઝ જણાવી શકતો નથી.", "voice")
    assert "શકતી નથી" in out
    assert "શકતો નથી" not in out
