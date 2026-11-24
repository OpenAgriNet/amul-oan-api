"""WhatsApp Bengali: a Bengali message is answered in Bengali.

The WhatsApp integration only sends gu/hi/en, so Bengali messages arrived as
en/en and were answered in English (prod Langfuse, 21-24 Nov 2026). The inputs
below are those farmers' messages.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from unittest.mock import AsyncMock

import pytest
from fastapi import BackgroundTasks

from app.channels.chat import whatsapp_languages
from app.models.requests import ChatRequest
from app.routers import chat as chat_router


@pytest.mark.parametrize("query", [
    "দুধ উৎপাদন কীভাবে বাড়ানো যায়?",
    "৭ দিন দুধ",
    "বাংল বলুন",
    "মোসকে কি খবর দিলে ফাট বাড়বে",
    "৭",
])
@pytest.mark.parametrize("sent", [("en", "en"), ("gu", "gu"), ("hi", "hi")])
def test_bengali_script_is_answered_in_bengali(query, sent):
    assert whatsapp_languages(query, *sent) == ("bn", "bn")


@pytest.mark.parametrize(("query", "sent"), [
    ("Hello", ("en", "en")),
    ("7 days milk", ("en", "en")),
    ("મારી ગાય દૂધ ઓછું આપે છે", ("gu", "gu")),
    ("गाय का दूध कैसे बढ़ाएं", ("hi", "hi")),
    ("Amul cattle feed price for দুধ cows please", ("en", "en")),
    ("", ("gu", "gu")),
    ("123 ?", ("en", "en")),
])
def test_other_messages_keep_the_callers_languages(query, sent):
    assert whatsapp_languages(query, *sent) == sent


async def _langs_seen(monkeypatch, *, channel, query, source_lang="en", target_lang="en"):
    seen = {}

    async def _stream(**kwargs):
        seen.update(kwargs)
        yield "ok"

    monkeypatch.setattr(chat_router, "stream_chat_messages", _stream)
    monkeypatch.setattr(chat_router, "_get_message_history", AsyncMock(return_value=[]))
    await chat_router.chat_endpoint(
        BackgroundTasks(),
        request=ChatRequest(
            query=query, session_id="s1", stream=False, channel=channel,
            source_lang=source_lang, target_lang=target_lang,
        ),
        user_info={"phone": "9876543210", "auth_type": "api_key"},
    )
    return seen["source_lang"], seen["target_lang"]


@pytest.mark.asyncio
async def test_whatsapp_bengali_turn_runs_in_bengali(monkeypatch):
    langs = await _langs_seen(monkeypatch, channel="whatsapp", query="দুধ উৎপাদন কীভাবে বাড়ানো যায়?")
    assert langs == ("bn", "bn")


@pytest.mark.asyncio
async def test_whatsapp_english_turn_is_unchanged(monkeypatch):
    assert await _langs_seen(monkeypatch, channel="whatsapp", query="Ho") == ("en", "en")


@pytest.mark.asyncio
async def test_web_keeps_the_language_the_app_sent(monkeypatch):
    """The web app sends the farmer's chosen language; it is never overridden."""
    langs = await _langs_seen(monkeypatch, channel="web", query="দুধ উৎপাদন কীভাবে বাড়ানো যায়?")
    assert langs == ("en", "en")
