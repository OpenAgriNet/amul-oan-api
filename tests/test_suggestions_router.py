import asyncio
import json

import pytest
from fastapi import HTTPException

from app.models.requests import SuggestionsRequest
from app.redis.config import SUGGESTIONS_TTL_SECONDS, key as redis_key
from app.routers import suggestions as suggestions_router


def _cache_key(session_id: str, target_lang: str) -> str:
    return redis_key("suggestions", f"{session_id}:{target_lang}")


def test_suggest_returns_cached_suggestions_without_regeneration_or_write(monkeypatch):
    calls = {"create": 0, "set": []}

    async def fake_get_cache(key):
        assert key == _cache_key("session-1", "gu")
        return ["cached question"]

    async def fake_create_suggestions(session_id, target_lang):
        calls["create"] += 1
        return ["generated question"]

    async def fake_set_cache(key, value, *, ttl=None, namespace=None):
        calls["set"].append((key, value, ttl, namespace))
        return True

    monkeypatch.setattr(suggestions_router, "get_cache", fake_get_cache)
    monkeypatch.setattr(suggestions_router, "create_suggestions", fake_create_suggestions)
    monkeypatch.setattr(suggestions_router, "set_cache", fake_set_cache)

    response = asyncio.run(
        suggestions_router.suggest(
            request=SuggestionsRequest(session_id="session-1", target_lang="gu"),
            user_info={"sub": "user-1"},
        )
    )

    assert json.loads(response.body) == ["cached question"]
    assert calls["create"] == 0
    assert calls["set"] == []


def test_suggest_on_cache_miss_generates_and_set_cache_called_once(monkeypatch):
    calls = {"set": []}

    async def fake_get_cache(key):
        assert key == _cache_key("session-2", "mr")
        return None

    async def fake_create_suggestions(session_id, target_lang):
        assert session_id == "session-2"
        assert target_lang == "mr"
        return ["generated question"]

    async def fake_set_cache(key, value, *, ttl=None, namespace=None):
        calls["set"].append((key, value, ttl, namespace))
        return True

    monkeypatch.setattr(suggestions_router, "get_cache", fake_get_cache)
    monkeypatch.setattr(suggestions_router, "create_suggestions", fake_create_suggestions)
    monkeypatch.setattr(suggestions_router, "set_cache", fake_set_cache)

    response = asyncio.run(
        suggestions_router.suggest(
            request=SuggestionsRequest(session_id="session-2", target_lang="mr"),
            user_info={"sub": "user-2"},
        )
    )

    assert json.loads(response.body) == ["generated question"]
    assert len(calls["set"]) == 1
    key, value, ttl, namespace = calls["set"][0]
    assert key == _cache_key("session-2", "mr")
    assert value == ["generated question"]
    assert ttl == SUGGESTIONS_TTL_SECONDS
    assert namespace is None


def test_suggest_get_cache_raises_still_generates_and_persists(monkeypatch):
    calls = {"set": []}

    async def fake_get_cache(key):
        raise ConnectionError("redis read failed")

    async def fake_create_suggestions(session_id, target_lang):
        return ["after error"]

    async def fake_set_cache(key, value, *, ttl=None, namespace=None):
        calls["set"].append((key, value))
        return True

    monkeypatch.setattr(suggestions_router, "get_cache", fake_get_cache)
    monkeypatch.setattr(suggestions_router, "create_suggestions", fake_create_suggestions)
    monkeypatch.setattr(suggestions_router, "set_cache", fake_set_cache)

    response = asyncio.run(
        suggestions_router.suggest(
            request=SuggestionsRequest(session_id="session-3", target_lang="gu"),
            user_info={"sub": "user-3"},
        )
    )

    assert json.loads(response.body) == ["after error"]
    assert len(calls["set"]) == 1
    assert calls["set"][0][0] == _cache_key("session-3", "gu")


def test_suggest_create_suggestions_raises_http_500(monkeypatch):
    async def fake_get_cache(key):
        return None

    async def fake_create_suggestions(session_id, target_lang):
        raise RuntimeError("agent failed")

    async def fake_set_cache(*args, **kwargs):
        pytest.fail("set_cache must not run when generation fails")

    monkeypatch.setattr(suggestions_router, "get_cache", fake_get_cache)
    monkeypatch.setattr(suggestions_router, "create_suggestions", fake_create_suggestions)
    monkeypatch.setattr(suggestions_router, "set_cache", fake_set_cache)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            suggestions_router.suggest(
                request=SuggestionsRequest(session_id="session-4", target_lang="gu"),
                user_info={"sub": "user-4"},
            )
        )
    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to generate suggestions"


def test_suggest_set_cache_raises_http_503(monkeypatch):
    async def fake_get_cache(key):
        return None

    async def fake_create_suggestions(session_id, target_lang):
        return ["ok"]

    async def fake_set_cache(key, value, *, ttl=None, namespace=None):
        raise OSError("redis write failed")

    monkeypatch.setattr(suggestions_router, "get_cache", fake_get_cache)
    monkeypatch.setattr(suggestions_router, "create_suggestions", fake_create_suggestions)
    monkeypatch.setattr(suggestions_router, "set_cache", fake_set_cache)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            suggestions_router.suggest(
                request=SuggestionsRequest(session_id="session-5", target_lang="mr"),
                user_info={"sub": "user-5"},
            )
        )
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Failed to persist suggestions"
