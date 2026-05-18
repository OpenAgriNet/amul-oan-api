import asyncio
import json

from app.routers import suggestions as suggestions_router
from app.models.requests import SuggestionsRequest


def test_suggest_returns_cached_suggestions_without_regeneration(monkeypatch):
    async def fake_get_cache(key):
        assert key == "suggestions_session-1_gu"
        return ["cached question"]

    monkeypatch.setattr(suggestions_router, "get_cache", fake_get_cache)

    response = asyncio.run(
        suggestions_router.suggest(
            request=SuggestionsRequest(session_id="session-1", target_lang="gu"),
            user_info={"sub": "user-1"},
        )
    )

    assert json.loads(response.body) == ["cached question"]


def test_suggest_returns_empty_on_cache_miss_without_pending(monkeypatch):
    async def fake_get_cache(key):
        assert key in {"suggestions_session-2_mr", "suggestions_session-2_mr:pending"}
        return None

    monkeypatch.setattr(suggestions_router, "get_cache", fake_get_cache)

    response = asyncio.run(
        suggestions_router.suggest(
            request=SuggestionsRequest(session_id="session-2", target_lang="mr"),
            user_info={"sub": "user-2"},
        )
    )

    assert json.loads(response.body) == []


def test_suggest_waits_for_pending_cache_fill(monkeypatch):
    calls = {"suggestions_reads": 0}
    monkeypatch.setattr(suggestions_router, "SUGGESTIONS_WAIT_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(suggestions_router, "SUGGESTIONS_WAIT_INTERVAL_SECONDS", 0.0)

    async def fake_get_cache(key):
        if key == "suggestions_session-3_hi":
            calls["suggestions_reads"] += 1
            if calls["suggestions_reads"] >= 2:
                return ["fresh question"]
            return None
        if key == "suggestions_session-3_hi:pending":
            return True
        return None

    monkeypatch.setattr(suggestions_router, "get_cache", fake_get_cache)

    response = asyncio.run(
        suggestions_router.suggest(
            request=SuggestionsRequest(session_id="session-3", target_lang="hi"),
            user_info={"sub": "user-3"},
        )
    )

    assert json.loads(response.body) == ["fresh question"]
