"""Suggestions delegates all model execution to llm_core."""

import asyncio
from types import SimpleNamespace

import pytest

import app.tasks.suggestions as suggestions


@pytest.fixture
def isolated(monkeypatch):
    async def fake_history(session_id):
        return []

    async def fake_set_cache(*args, **kwargs):
        return True

    async def fake_delete(*args, **kwargs):
        return None

    async def fake_profile(session_id):
        return "oss"

    monkeypatch.setattr(suggestions, "_get_message_history", fake_history)
    monkeypatch.setattr(suggestions, "set_cache", fake_set_cache)
    monkeypatch.setattr(suggestions.cache, "delete", fake_delete)
    monkeypatch.setattr(suggestions.llm_core, "profile", fake_profile)
    monkeypatch.setattr(
        suggestions.llm_core,
        "primary_info",
        lambda *args, **kwargs: SimpleNamespace(model_name="test-model"),
    )


def test_suggestions_use_llm_core(isolated, monkeypatch):
    async def fake_run(*args, **kwargs):
        assert args[0] is suggestions._LlmStep.SUGGESTIONS
        assert args[1] == "s1"
        assert args[2] is suggestions.suggestions_agent
        return SimpleNamespace(output=["q1", "q2", "q3"])

    monkeypatch.setattr(suggestions.llm_core, "run", fake_run)

    result = asyncio.run(suggestions.create_suggestions("s1", "gu"))
    assert result == ["q1", "q2", "q3"]


def test_suggestions_degrade_to_empty(isolated, monkeypatch):
    async def fake_run(*args, **kwargs):
        raise ConnectionError("all configured tiers are down")

    monkeypatch.setattr(suggestions.llm_core, "run", fake_run)

    result = asyncio.run(suggestions.create_suggestions("s1", "gu"))
    assert result == []
