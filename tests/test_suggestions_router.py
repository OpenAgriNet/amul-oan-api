"""Tests for GET /api/suggest — fresh generation every request, no suggestion caching."""

import asyncio
import json

import pytest
from fastapi import HTTPException

from app.models.requests import SuggestionsRequest
from app.routers import suggestions as suggestions_router


def test_suggest_always_calls_create_suggestions(monkeypatch):
    calls = []

    async def fake_create_suggestions(session_id, target_lang):
        calls.append((session_id, target_lang))
        return ["q1", "q2"]

    monkeypatch.setattr(suggestions_router, "create_suggestions", fake_create_suggestions)

    asyncio.run(
        suggestions_router.suggest(
            request=SuggestionsRequest(session_id="session-a", target_lang="gu"),
            user_info={"sub": "user-1"},
        )
    )
    asyncio.run(
        suggestions_router.suggest(
            request=SuggestionsRequest(session_id="session-a", target_lang="gu"),
            user_info={"sub": "user-1"},
        )
    )

    assert calls == [
        ("session-a", "gu"),
        ("session-a", "gu"),
    ]


def test_suggest_returns_generated_json(monkeypatch):
    async def fake_create_suggestions(session_id, target_lang):
        assert session_id == "session-b"
        assert target_lang == "mr"
        return ["one", "two"]

    monkeypatch.setattr(suggestions_router, "create_suggestions", fake_create_suggestions)

    response = asyncio.run(
        suggestions_router.suggest(
            request=SuggestionsRequest(session_id="session-b", target_lang="mr"),
            user_info={"sub": "user-2"},
        )
    )

    assert json.loads(response.body) == ["one", "two"]


def test_suggest_create_suggestions_raises_http_500(monkeypatch):
    async def fake_create_suggestions(session_id, target_lang):
        raise RuntimeError("agent failed")

    monkeypatch.setattr(suggestions_router, "create_suggestions", fake_create_suggestions)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            suggestions_router.suggest(
                request=SuggestionsRequest(session_id="session-c", target_lang="gu"),
                user_info={"sub": "user-3"},
            )
        )
    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to generate suggestions"
