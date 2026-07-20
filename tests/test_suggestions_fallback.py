"""Wiring test: create_suggestions routes through execute_with_fallback when
FALLBACK_ENABLED, falling back OSS->managed, and degrades to [] when all tiers fail."""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import asyncio
from types import SimpleNamespace

import pytest

import app.tasks.suggestions as sug
from app.services import fallback as fb


@pytest.fixture
def oss_on(monkeypatch):
    monkeypatch.setattr(fb.settings, "fallback_enabled", True)
    # Legacy attempt_chain walker path (split defaults are now ON at boot).
    monkeypatch.setattr(fb.settings, "profiles_enabled", False)
    monkeypatch.setattr(fb.settings, "llm_core_enabled", False)
    monkeypatch.setattr(fb, "oss_model_available", lambda: True)
    monkeypatch.setattr(fb, "OSS_LLM_MODEL", "OSS")
    monkeypatch.setattr(fb, "OSS_LLM_MODEL_NAME", "gemma")
    monkeypatch.setattr(fb, "OSS_INFERENCE_ENDPOINT_URL", "http://oss")
    monkeypatch.setattr(fb, "LLM_MODEL", "MANAGED")
    events = []
    monkeypatch.setattr(fb, "emit", events.append)

    # neutralize history / cache I/O so we isolate the fallback wiring
    async def fake_hist(session_id):
        return []

    async def fake_set_cache(*a, **k):
        return True

    async def fake_delete(*a, **k):
        return None

    monkeypatch.setattr(sug, "_get_message_history", fake_hist)
    monkeypatch.setattr(sug, "set_cache", fake_set_cache)
    monkeypatch.setattr(sug.cache, "delete", fake_delete)
    return events


def test_suggestions_fall_back_to_managed(oss_on, monkeypatch):
    async def fake_run(message, model=None):
        if model == "OSS":
            raise ConnectionError("vllm down")
        return SimpleNamespace(output=["q1", "q2", "q3"])

    monkeypatch.setattr(sug.suggestions_agent, "run", fake_run)

    out = asyncio.run(sug.create_suggestions("s1", "gu", "oss"))
    assert out == ["q1", "q2", "q3"]
    assert any(e.fell_back for e in oss_on)  # fell back OSS -> managed


def test_suggestions_empty_when_all_tiers_fail(oss_on, monkeypatch):
    async def fake_run(message, model=None):
        raise ConnectionError("down")

    monkeypatch.setattr(sug.suggestions_agent, "run", fake_run)

    out = asyncio.run(sug.create_suggestions("s1", "gu", "oss"))
    assert out == []  # both tiers fail -> create_suggestions degrades to []


def test_suggestions_success_on_oss_no_fallback(oss_on, monkeypatch):
    async def fake_run(message, model=None):
        assert model == "OSS"  # OSS session tries OSS first
        return SimpleNamespace(output=["a", "b"])

    monkeypatch.setattr(sug.suggestions_agent, "run", fake_run)

    out = asyncio.run(sug.create_suggestions("s1", "gu", "oss"))
    assert out == ["a", "b"]
    assert oss_on == []  # no fallback event on success
