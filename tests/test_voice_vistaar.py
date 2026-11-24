"""Bharat Vistaar weather on a call (AMUL-81).

Chat had the weather tool and voice did not: a caller asking for the forecast
was routed to `search_documents` (the vet documents) and got no weather. These
tests pin the three things that fix it: the tool is on both voice agents, it
answers from a voice-shaped FarmerContext under the voice time cap, and every
voice prompt variant routes weather to it instead of the documents. Mandi
prices reached voice separately (AMUL-113, #448).
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


def test_both_voice_agents_offer_the_weather_tool():
    for agent in (voice_agent_module.voice_agent, voice_agent_module.voice_agent_signed_in):
        assert "get_vistaar_weather" in agent._function_toolset.tools


def test_the_voice_tool_keeps_chats_name_signature_and_docstring():
    voice_fn, chat_fn = voice_vistaar.get_vistaar_weather, chat_vistaar.get_vistaar_weather
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


def test_a_stalled_weather_call_is_cut_off_and_says_unavailable(monkeypatch):
    async def stalled_search(intent):
        await asyncio.sleep(5)
        return [_WEATHER_ITEM]

    monkeypatch.setattr(settings, "voice_vistaar_budget_s", 0.2)
    started = time.monotonic()
    with patch.object(chat_vistaar, "_vistaar_search", stalled_search):
        out = asyncio.run(voice_vistaar.get_vistaar_weather(_Ctx(_voice_deps())))

    assert out.startswith("Weather is temporarily unavailable")
    assert time.monotonic() - started < 2


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_voice_prompt_routes_weather_to_vistaar(variant):
    text = get_prompt(variant, context=PROMPT_CONTEXT)
    routing = [line for line in text.splitlines() if "search_documents" in line and "clinical" in line]
    assert routing, f"{variant}: no search_documents routing line found"
    for line in routing:
        # The AMUL-81 bug: weather was among the intents sent to the vet
        # documents. Only the intent list, before the tool name, is checked.
        intents = line.split("search_documents")[0]
        assert "weather" not in intents, f"{variant} still routes weather to search_documents: {line}"
    weather_lines = [line for line in text.splitlines()
                     if "weather" in line and "get_vistaar_weather" in line and "never `search_documents`" in line.replace("Do not call `search_documents`", "never `search_documents`")]
    assert weather_lines, f"{variant}: no routing line sends weather to get_vistaar_weather"
    assert "minimum and maximum temperature" in text
