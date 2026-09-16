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

    monkeypatch.setattr(suggestions, "_get_message_history", fake_history)
    monkeypatch.setattr(suggestions, "set_cache", fake_set_cache)
    monkeypatch.setattr(suggestions.cache, "delete", fake_delete)


class _Execution:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    def info(self, step):
        return SimpleNamespace(model_name="test-model")

    async def run(self, *args, **kwargs):
        if self.error:
            raise self.error
        return self.result


def test_suggestions_use_turn_execution_snapshot(isolated):
    execution = _Execution(SimpleNamespace(output=["q1", "q2", "q3"]))
    result = asyncio.run(suggestions.create_suggestions("s1", "gu", execution))
    assert result == ["q1", "q2", "q3"]


def test_suggestions_degrade_to_empty(isolated):
    execution = _Execution(error=ConnectionError("all configured tiers are down"))
    result = asyncio.run(suggestions.create_suggestions("s1", "gu", execution))
    assert result == []
