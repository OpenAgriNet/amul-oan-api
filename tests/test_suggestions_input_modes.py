import asyncio
from pathlib import Path
from types import SimpleNamespace

import app.tasks.suggestions as sug


def _valid_shadow_payload(turn_id: str) -> dict:
    long_snippet = ("milk production guidance and cooperative payment details " * 8).strip()
    return {
        "request_turn_id": turn_id,
        "search_return_count": 2,
        "distilled_calls": [
            {
                "query": "milk payment guidance",
                "no_results": False,
                "snippets": [long_snippet, long_snippet],
            }
        ],
    }


def _setup_common_mocks(monkeypatch, *, hybrid_enabled: bool):
    monkeypatch.setattr(sug.settings, "fallback_enabled", False)
    monkeypatch.setattr(sug.settings, "suggestions_hybrid_enabled", hybrid_enabled)

    async def fake_hist(session_id):
        return []

    async def fake_set_cache(*a, **k):
        return True

    async def fake_delete(*a, **k):
        return None

    async def fake_publish(*a, **k):
        return True

    async def fake_clear(*a, **k):
        return False

    monkeypatch.setattr(sug, "_get_message_history", fake_hist)
    monkeypatch.setattr(sug, "set_cache", fake_set_cache)
    monkeypatch.setattr(sug.cache, "delete", fake_delete)
    monkeypatch.setattr(sug, "_publish_suggestions_if_latest", fake_publish)
    monkeypatch.setattr(sug, "_clear_suggestions_turn_if_owned", fake_clear)


def test_hybrid_mode_uses_retrieval_on_matching_turn(monkeypatch):
    _setup_common_mocks(monkeypatch, hybrid_enabled=True)
    payload = _valid_shadow_payload("turn-1")

    async def fake_get_cache(*a, **k):
        return payload

    captured = {}

    async def fake_run(message, model=None):
        captured["message"] = message
        return SimpleNamespace(output=["q1", "q2"])

    monkeypatch.setattr(sug, "get_cache", fake_get_cache)
    monkeypatch.setattr(sug.suggestions_agent, "run", fake_run)

    out = asyncio.run(
        sug.create_suggestions(
            "s1",
            "gu",
            "managed",
            request_turn_id="turn-1",
        )
    )
    assert out == ["q1", "q2"]
    assert "**Retrieved Evidence**" in captured["message"]
    assert "Answerability guardrails:" not in captured["message"]


def test_turn_mismatch_falls_back_to_conversation_only(monkeypatch):
    _setup_common_mocks(monkeypatch, hybrid_enabled=True)
    payload = _valid_shadow_payload("turn-1")

    async def fake_get_cache(*a, **k):
        return payload

    captured = {}

    async def fake_run(message, model=None):
        captured["message"] = message
        return SimpleNamespace(output=["q1"])

    monkeypatch.setattr(sug, "get_cache", fake_get_cache)
    monkeypatch.setattr(sug.suggestions_agent, "run", fake_run)

    out = asyncio.run(
        sug.create_suggestions(
            "s1",
            "gu",
            "managed",
            request_turn_id="turn-2",
        )
    )
    assert out == ["q1"]
    assert "**Retrieved Evidence**" not in captured["message"]
    assert captured["message"].startswith("**Conversation**")


def test_hybrid_on_unsupported_language_skips_shadow_cache(monkeypatch):
    _setup_common_mocks(monkeypatch, hybrid_enabled=True)

    async def raising_get_cache(*a, **k):
        raise AssertionError("get_cache should not be called for unsupported hybrid language")

    captured = {}

    async def fake_run(message, model=None):
        captured["message"] = message
        return SimpleNamespace(output=["q1"])

    monkeypatch.setattr(sug, "get_cache", raising_get_cache)
    monkeypatch.setattr(sug.suggestions_agent, "run", fake_run)

    out = asyncio.run(sug.create_suggestions("s1", "mr", "managed", request_turn_id="turn-1"))
    assert out == ["q1"]
    assert "**Retrieved Evidence**" not in captured["message"]


def test_prompt_does_not_reintroduce_bank_suppression():
    prompt = Path("assets/prompts/suggestions_system.md").read_text(encoding="utf-8")
    assert "Do not generate questions about bank accounts or non-milk financial transactions." not in prompt


def test_common_prompt_applies_answerability_guardrails_to_all_input_modes():
    prompt = Path("assets/prompts/suggestions_system.md").read_text(encoding="utf-8")
    assert "## Answerability guardrails" in prompt
    assert "Do not suggest personal account lookup actions the agent cannot perform." in prompt
    assert "anything other than English or Gujarati" in prompt
