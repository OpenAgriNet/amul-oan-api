import asyncio

from fastapi import HTTPException
from starlette.requests import Request

from app.auth.jwt_auth import get_chat_user
from app.config import settings
from app.models.requests import ChatRequest
from app.services.chat import (
    WHATSAPP_RESPONSE_MAX_CHARS,
    _response_max_chars_for_channel,
)
from app.services.translation import _format_translation_prompt
from helpers.utils import get_prompt


def _request_with_headers(headers: dict[str, str]) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/chat/",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
    }
    return Request(scope)


def test_get_chat_user_accepts_api_key_phone_auth(monkeypatch):
    monkeypatch.setattr(settings, "chat_api_key", "secret-key")
    request = _request_with_headers(
        {
            "X-API-Key": "secret-key",
            "X-User-Phone": "+91 93750 28676",
        }
    )

    user = asyncio.run(get_chat_user(request, token=None))

    assert user["auth_type"] == "api_key"
    assert user["phone"] == "9375028676"
    assert user["sub"] == "9375028676"


def test_get_chat_user_rejects_invalid_api_key(monkeypatch):
    monkeypatch.setattr(settings, "chat_api_key", "secret-key")
    request = _request_with_headers(
        {
            "X-API-Key": "wrong-key",
            "X-User-Phone": "9375028676",
        }
    )

    try:
        asyncio.run(get_chat_user(request, token=None))
        assert False, "Expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 401
        assert exc.detail == "Invalid API key"


def test_get_chat_user_requires_phone_with_api_key(monkeypatch):
    monkeypatch.setattr(settings, "chat_api_key", "secret-key")
    request = _request_with_headers({"X-API-Key": "secret-key"})

    try:
        asyncio.run(get_chat_user(request, token=None))
        assert False, "Expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "X-User-Phone" in exc.detail


def test_chat_request_defaults_channel_to_web():
    request = ChatRequest(query="hello")

    assert request.channel == "web"


def test_response_max_chars_is_set_only_for_whatsapp_channel():
    assert _response_max_chars_for_channel("whatsapp") == WHATSAPP_RESPONSE_MAX_CHARS
    assert _response_max_chars_for_channel("WhatsApp") == WHATSAPP_RESPONSE_MAX_CHARS
    assert _response_max_chars_for_channel("web") is None
    assert _response_max_chars_for_channel(None) is None


def test_default_agrinet_prompt_includes_whatsapp_limit_when_provided():
    prompt = get_prompt(
        "agrinet_system.md",
        context={
            "today_date": "Monday, 04 May 2026",
            "today_datetime": "Monday, 04 May 2026 12:00 PM IST",
            "farmer_context": None,
            "ambiguity_hints": None,
            "response_max_chars": 1600,
        },
    )

    assert "WhatsApp Response Limit" in prompt
    assert "no more than 1600 characters" in prompt


def test_translation_pipeline_prompt_includes_whatsapp_limit_when_provided():
    prompt = get_prompt(
        "agrinet_system_translation_pipeline.md",
        context={
            "today_date": "Monday, 04 May 2026",
            "farmer_context": None,
            "ambiguity_hints": None,
            "response_max_chars": 1600,
        },
    )

    assert "WhatsApp Response Limit" in prompt
    assert "final translated user-facing answer" in prompt
    assert "no more than 1600 characters" in prompt


def test_translation_prompt_length_rule_is_optional():
    default_prompt = _format_translation_prompt(
        text="Keep the animal hydrated.",
        source_lang="english",
        target_lang="gujarati",
    )
    whatsapp_prompt = _format_translation_prompt(
        text="Keep the animal hydrated.",
        source_lang="english",
        target_lang="gujarati",
        max_output_chars=1600,
    )

    assert "Length Rule" not in default_prompt
    assert "no more than 1600 characters" in whatsapp_prompt
