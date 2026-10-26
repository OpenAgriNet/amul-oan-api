"""Bharat Vistaar weather and mandi prices on a call (AMUL-81).

Chat had these tools and voice did not: a caller asking for the forecast was
routed to `search_documents` (the vet documents) and got no weather. These
tests pin the three things that fix it: the tools are on both voice agents,
they answer from a voice-shaped FarmerContext, and every voice prompt variant
routes weather and market questions to them instead of the documents.
"""
import asyncio
import time
from unittest.mock import patch

import pytest

from agents.deps import FarmerContext
from agents.tools import vistaar as chat_vistaar
from agents.voice import agent as voice_agent_module
from agents.voice.tools import vistaar as voice_vistaar
from app.config import settings
from helpers.utils import get_prompt

VARIANTS = ["voice_system_translation_pipeline_en", "voice_system_translation_pipeline_gemma4_en",
            "voice_system_translation_pipeline_gpt5_1_en"]
PROMPT_CONTEXT = {
    "loan_max_amount": "10,000",
    "loan_interest_rate_pct": "7",
    "creation_date_words": "eleventh February two thousand twenty six",
    "service_channels_words": "chat, voice call, and WhatsApp",
    "helpline_number_words": "zero eight zero",
}

_WEATHER_ITEM = {
    "descriptor": {"name": "Day 1"},
    "tags": [{"descriptor": {"code": "forecast"}, "list": [
        {"descriptor": {"code": "Rainfall"}, "value": "2 mm"},
        {"descriptor": {"code": "Max Temp"}, "value": "33"},
    ]}],
}


class _Ctx:
    def __init__(self, deps):
        self.deps = deps


def _voice_deps(district="banaskantha"):
    # The fields app/voice/agent_input.py fills that these tools read.
    return FarmerContext(query="weather", lang_code="gu", session_id=None, farmer_district=district)


@pytest.mark.parametrize("name", ["get_vistaar_weather", "get_vistaar_mandi_prices"])
def test_both_voice_agents_offer_the_tool(name):
    for agent in (voice_agent_module.voice_agent, voice_agent_module.voice_agent_signed_in):
        assert name in agent._function_toolset.tools


def test_the_voice_tool_keeps_chats_name_signature_and_docstring():
    for voice_fn, chat_fn in (
        (voice_vistaar.get_vistaar_weather, chat_vistaar.get_vistaar_weather),
        (voice_vistaar.get_vistaar_mandi_prices, chat_vistaar.get_vistaar_mandi_prices),
    ):
        assert voice_fn.__name__ == chat_fn.__name__
        assert voice_fn.__doc__ == chat_fn.__doc__
        assert voice_fn.__wrapped__ is chat_fn


def test_weather_answers_for_the_callers_own_district():
    async def fake_search(intent):
        return [_WEATHER_ITEM]

    with patch.object(chat_vistaar, "_vistaar_search", fake_search):
        out = asyncio.run(voice_vistaar.get_vistaar_weather(_Ctx(_voice_deps("banaskantha"))))

    assert out.startswith("Weather forecast for ")
    assert "Banaskantha" in out
    assert "Rainfall: 2 mm" in out
    assert "do not have your district" not in out


def test_weather_without_a_district_says_the_location_was_assumed():
    async def fake_search(intent):
        return [_WEATHER_ITEM]

    with patch.object(chat_vistaar, "_vistaar_search", fake_search):
        out = asyncio.run(voice_vistaar.get_vistaar_weather(_Ctx(_voice_deps(None))))

    assert "Anand" in out
    assert "do not have your district on file" in out


@pytest.mark.parametrize("call, unavailable", [
    (lambda ctx: voice_vistaar.get_vistaar_weather(ctx), "Weather is temporarily unavailable"),
    (lambda ctx: voice_vistaar.get_vistaar_mandi_prices(ctx, "Onion"), "Mandi prices are temporarily unavailable"),
])
def test_a_stalled_vistaar_call_is_cut_off_and_says_unavailable(monkeypatch, call, unavailable):
    async def stalled_search(intent):
        await asyncio.sleep(5)
        return [_WEATHER_ITEM]

    monkeypatch.setattr(settings, "voice_vistaar_budget_s", 0.2)
    started = time.monotonic()
    with patch.object(chat_vistaar, "_vistaar_search", stalled_search):
        out = asyncio.run(call(_Ctx(_voice_deps())))

    assert out.startswith(unavailable)
    assert time.monotonic() - started < 2


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_voice_prompt_routes_weather_and_market_to_vistaar(variant):
    text = get_prompt(variant, context=PROMPT_CONTEXT)
    routing = [line for line in text.splitlines() if "search_documents" in line and "clinical" in line]
    assert routing, f"{variant}: no search_documents routing line found"
    for line in routing:
        # The AMUL-81 bug: weather and market were among the intents sent to the
        # vet documents. Only the intent list, before the tool name, is checked.
        intents = line.split("search_documents")[0]
        assert "weather" not in intents and "market" not in intents, f"{variant} still routes to search_documents: {line}"
    vistaar_lines = [line for line in text.splitlines()
                     if "get_vistaar_weather" in line and "get_vistaar_mandi_prices" in line and "market" in line]
    assert vistaar_lines, f"{variant}: no routing line sends weather and market to the Vistaar tools"
    assert "rupees per quintal" in text
    assert "Speak only figures the tool returned" in text
