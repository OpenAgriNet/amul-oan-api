"""§14 — channel-aware translation. Voice's richer, telephony-tuned translation
data applies only under translation_channel("voice"); chat's path is unchanged.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from agents.tools.terms import get_mini_glossary_for_text
from app.services.translation import (
    _format_translation_prompt,
    translation_channel,
    _get_glossary_hints_for_gu_query,
    GU_PREFERRED_TRANSLATION_RULES,
    VOICE_GU_PREFERRED_TRANSLATION_RULES,
)

_FARMER_PROFILE_SOCIETY_LINE = (
    "- Society: Suchit Bajrang D.U.S.M.Ltd (Code: 2239)"
)

_VOICE_ONLY = "professional, cordial, and detached"  # a voice-only rule phrase


def test_chat_channel_uses_chat_rules_by_default():
    prompt = _format_translation_prompt("hi", "english", "gujarati")
    assert "farmer-preferred Gujarati livestock terms" in prompt  # chat rule present
    assert _VOICE_ONLY not in prompt                              # voice rule absent


def test_voice_channel_uses_voice_rules():
    with translation_channel("voice"):
        prompt = _format_translation_prompt("hi", "english", "gujarati")
    assert _VOICE_ONLY in prompt
    assert "do not call the caller બહેન" in prompt   # voice telephony addressing rule


def test_channel_resets_after_context_block():
    with translation_channel("voice"):
        pass
    prompt = _format_translation_prompt("hi", "english", "gujarati")
    assert _VOICE_ONLY not in prompt   # back to chat default


def test_chat_rules_constant_unchanged():
    assert len(GU_PREFERRED_TRANSLATION_RULES) == 5
    assert _VOICE_ONLY not in "\n".join(GU_PREFERRED_TRANSLATION_RULES)
    assert _VOICE_ONLY in "\n".join(VOICE_GU_PREFERRED_TRANSLATION_RULES)


def test_voice_glossary_hint_for_asr_buffalo_variant():
    # ભંચ is a voice-only ASR spelling variant of buffalo (chat's glossary lacks it)
    hints = _get_glossary_hints_for_gu_query("ભંચ")
    assert "Buffalo" in hints


def test_standalone_society_in_mini_glossary_for_compact_profile_line():
    """Agent often emits 'Society:' (not 'Society name') on farmer profile lines."""
    glossary = get_mini_glossary_for_text(
        _FARMER_PROFILE_SOCIETY_LINE, threshold=0.90, max_terms=40
    )
    assert "Society -> સોસાયટી" in glossary


def test_standalone_society_injected_into_translation_prompt():
    glossary = get_mini_glossary_for_text(
        _FARMER_PROFILE_SOCIETY_LINE, threshold=0.90, max_terms=40
    )
    prompt = _format_translation_prompt(
        _FARMER_PROFILE_SOCIETY_LINE, "english", "gujarati", mini_glossary=glossary
    )
    assert "'Society' must be translated as 'સોસાયટી'" in prompt
