import asyncio

from fastapi import HTTPException
from starlette.requests import Request

from app.auth.jwt_auth import get_chat_user
from app.config import settings
from app.models.requests import ChatRequest


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
