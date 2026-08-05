"""Every turn records how it ended.

Before this guard an outcome was recorded only on normal completion. A client
disconnect or an exception left the trace with no output and no signal at all —
#179's B2. These pin the three exits the guard can distinguish without touching
the turn body.
"""
import asyncio
import os
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

from app.services import chat as chat_service


def _capture(monkeypatch):
    """Replace the Langfuse client with a permissive recorder of scores.

    It must tolerate every call the turn makes (set_current_trace_io,
    start_as_current_observation as a context manager, ...) — a thin stub sends
    the turn down the moderation fail-closed path and the test measures nothing.
    """
    from unittest.mock import MagicMock

    scored = []
    client = MagicMock()
    client.score_current_trace.side_effect = lambda **kw: scored.append(kw)
    monkeypatch.setattr(chat_service, "get_langfuse_client", lambda: client)
    return scored


def _outcomes(scored):
    return [s["value"] for s in scored if s.get("name") == "turn_outcome"]


def test_records_success_on_normal_completion(monkeypatch):
    scored = _capture(monkeypatch)
    chat_service._record_turn_outcome("success", "sess")
    assert _outcomes(scored) == ["success"]


def test_records_cancelled_and_error(monkeypatch):
    scored = _capture(monkeypatch)
    chat_service._record_turn_outcome("cancelled", "sess")
    chat_service._record_turn_outcome("error", "sess")
    assert _outcomes(scored) == ["cancelled", "error"]


def test_never_raises_when_langfuse_is_broken(monkeypatch):
    """It runs in a finally during GeneratorExit — it must never mask the real exit."""
    class _Broken:
        def score_current_trace(self, **kw):
            raise RuntimeError("langfuse down")

    monkeypatch.setattr(chat_service, "get_langfuse_client", lambda: _Broken())
    chat_service._record_turn_outcome("success", "sess")  # must not raise


def test_no_langfuse_client_is_a_noop(monkeypatch):
    monkeypatch.setattr(chat_service, "get_langfuse_client", None)
    chat_service._record_turn_outcome("success", "sess")  # must not raise


def _drive(monkeypatch, *, agent_raises=False, stop_after=None):
    """Drive a real turn through stream_chat_messages and return emitted outcomes."""
    from tests.test_chat_turn_sequence import (
        _Cache, _Run, _delta,
    )
    from fastapi import BackgroundTasks

    scored = _capture(monkeypatch)
    monkeypatch.setattr(chat_service.settings, "fallback_enabled", False)
    monkeypatch.setattr(chat_service, "propagate_attributes", None)
    monkeypatch.setattr(chat_service, "cache", _Cache())
    monkeypatch.setattr(chat_service, "trim_history", lambda *_a, **_k: [])
    monkeypatch.setattr(chat_service, "format_message_pairs", lambda *_a, **_k: "")

    async def _pre(text, *_a, **_k):
        return "How much water?"

    async def _mod(user_message, model=None):
        return SimpleNamespace(output=SimpleNamespace(category="valid_agricultural", action="allow"))

    def _iter(**_k):
        if agent_raises:
            raise RuntimeError("agent exploded")
        return _Run(["Give clean water daily."])

    async def _tr(text, *_a, **_k):
        yield "રોજ સ્વચ્છ પાણી આપો."

    async def _noop(*_a, **_k):
        return None

    monkeypatch.setattr(chat_service, "translate_to_english_pretranslation", _pre)
    monkeypatch.setattr(chat_service.moderation_agent, "run", _mod)
    monkeypatch.setattr(chat_service.agrinet_agent, "iter", _iter)
    monkeypatch.setattr(chat_service, "translate_text_stream_fast", _tr)
    monkeypatch.setattr(chat_service, "update_message_history", _noop)
    monkeypatch.setattr(chat_service, "set_cache", _noop)
    monkeypatch.setattr(chat_service, "create_suggestions", lambda *_a, **_k: None)

    async def _go():
        gen = chat_service.stream_chat_messages(
            query="મારી ગાયને કેટલું પાણી આપવું?", session_id="outcome",
            source_lang="gu", target_lang="gu", channel="web",
            user_id="+919876543210", history=[], user_info={},
            background_tasks=BackgroundTasks(), use_translation_pipeline=True,
            pipeline_profile="managed",
        )
        n = 0
        async for _ in gen:
            n += 1
            if stop_after is not None and n >= stop_after:
                await gen.aclose()      # client disconnect
                break

    return _go, scored


def test_a_completed_turn_emits_success(monkeypatch):
    go, scored = _drive(monkeypatch)
    asyncio.run(go())
    assert _outcomes(scored) == ["success"]


def test_a_failing_turn_emits_error(monkeypatch):
    go, scored = _drive(monkeypatch, agent_raises=True)
    with pytest.raises(RuntimeError):
        asyncio.run(go())
    assert _outcomes(scored) == ["error"]


def test_a_disconnected_turn_emits_cancelled(monkeypatch):
    go, scored = _drive(monkeypatch, stop_after=1)
    asyncio.run(go())
    assert _outcomes(scored) == ["cancelled"]


def test_is_synchronous(monkeypatch):
    """Awaiting is unsafe in a finally reached via GeneratorExit."""
    import inspect
    assert not inspect.iscoroutinefunction(chat_service._record_turn_outcome)
