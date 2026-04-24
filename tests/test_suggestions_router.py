import asyncio
import json

from app.routers import suggestions as suggestions_router
from app.models.requests import SuggestionsRequest


def test_suggest_returns_cached_suggestions_without_regeneration(monkeypatch):
    calls = {"create": 0}

    async def fake_get_cache(key):
        assert key == "suggestions_session-1_gu"
        return ["cached question"]

    async def fake_create_suggestions(session_id, target_lang):
        calls["create"] += 1
        return ["generated question"]

    monkeypatch.setattr(suggestions_router, "get_cache", fake_get_cache)
    monkeypatch.setattr(suggestions_router, "create_suggestions", fake_create_suggestions)

    response = asyncio.run(
        suggestions_router.suggest(
            request=SuggestionsRequest(session_id="session-1", target_lang="gu"),
            user_info={"sub": "user-1"},
        )
    )

    assert json.loads(response.body) == ["cached question"]
    assert calls["create"] == 0


def test_suggest_generates_suggestions_on_cache_miss(monkeypatch):
    async def fake_get_cache(key):
        assert key == "suggestions_session-2_mr"
        return None

    async def fake_create_suggestions(session_id, target_lang):
        assert session_id == "session-2"
        assert target_lang == "mr"
        return ["generated question"]

    monkeypatch.setattr(suggestions_router, "get_cache", fake_get_cache)
    monkeypatch.setattr(suggestions_router, "create_suggestions", fake_create_suggestions)

    response = asyncio.run(
        suggestions_router.suggest(
            request=SuggestionsRequest(session_id="session-2", target_lang="mr"),
            user_info={"sub": "user-2"},
        )
    )

    assert json.loads(response.body) == ["generated question"]
