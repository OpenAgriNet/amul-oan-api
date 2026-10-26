"""Voice answers mandi (APMC) prices with chat's own tool (AMUL-113).

Before this, every voice prompt variant sent the market intent to
search_documents, which holds no live prices, so a caller asking a tomato or
onion rate heard "I don't know based on the provided documents."
"""
import os
import re

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LLM_MODEL_NAME", "gpt-test")

from helpers.utils import get_prompt  # noqa: E402

VARIANTS = [
    "voice_system_translation_pipeline_en",
    "voice_system_translation_pipeline_gpt5_1_en",
    "voice_system_translation_pipeline_gemma4_en",
]
# gpt-5.1 sits at its 4500-word cap, so it carries only the routing line.
FULL_RULE_VARIANTS = {
    "voice_system_translation_pipeline_en": "## Mandi Price Rules",
    "voice_system_translation_pipeline_gemma4_en": "## get_vistaar_mandi_prices",
}
CONTEXT = {
    "creation_date_words": "eleventh February two thousand twenty six",
    "service_channels_words": "chat, voice call, and WhatsApp",
    "helpline_number_words": "zero eight zero three five four five three five four five",
}


def _section(text, heading):
    lines = text.splitlines()
    start = lines.index(heading)
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("#")), len(lines))
    return "\n".join(lines[start:end])


def test_voice_registers_chats_mandi_tool_itself():
    from agents.tools.vistaar import get_vistaar_mandi_prices
    from agents.voice.tools import BASE_TOOLS

    tool = {t.name: t for t in BASE_TOOLS}["get_vistaar_mandi_prices"]
    # The same function chat registers, not a copy; wrapped for the hold line.
    assert tool.function.__wrapped__ is get_vistaar_mandi_prices
    assert tool.takes_ctx is True
    assert tool.prepare is None


@pytest.mark.parametrize("name", VARIANTS)
def test_market_goes_to_the_mandi_tool_not_document_search(name):
    text = get_prompt(name, context=CONTEXT)
    assert "get_vistaar_mandi_prices" in text
    routed = 0
    for line in text.splitlines():
        before = line.split("search_documents")[0]
        if "search_documents" in line and re.search(r"\bcrop\b", before) and "weather" in before:
            routed += 1
            assert not re.search(r"\bmarket\b", before), line
    assert routed, "the clinical/crop/weather routing line was not found"


@pytest.mark.parametrize("name", VARIANTS)
def test_the_spoken_price_names_a_market_yard_not_apmc(name):
    # Latin letters are dropped from the Gujarati speech, so "APMC" would
    # leave the caller hearing a gap where the market should be.
    text = get_prompt(name, context=CONTEXT)
    assert any("market yard" in line and 'never "APMC"' in line for line in text.splitlines())


@pytest.mark.parametrize("name,heading", FULL_RULE_VARIANTS.items())
def test_an_empty_or_failed_lookup_never_gets_the_documents_line(name, heading):
    section = _section(get_prompt(name, context=CONTEXT), heading)
    assert "documents line" in section or "provided documents" in section
    assert "rupees per quintal" in get_prompt(name, context=CONTEXT)


def test_voice_moderation_lets_crop_prices_through():
    text = get_prompt("voice_moderation_en")
    assert "Market prices for milk, fodder, livestock" in text
    assert "ટામેટાનો ભાવ શું છે?" in text
    assert "not a recipe or cooking question" in text


def test_the_voice_milk_price_hint_does_not_capture_crop_prices():
    # "price" triggers voice's milk-price rule on every price question; without
    # the guard the agent asks which dairy union instead of pricing.
    from agents.voice.tools.terms import get_ambiguity_hints_for_query

    hints = get_ambiguity_hints_for_query("What is the price of tomato in the market?")
    assert "Which dairy union" in hints
    assert "This applies only to milk" in hints
