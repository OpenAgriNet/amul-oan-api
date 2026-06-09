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
