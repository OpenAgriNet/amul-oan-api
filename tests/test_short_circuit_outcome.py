"""The short-circuit exits record what actually happened.

Two exits answer the farmer and then return early: the identity short-circuit
and the moderation decline. Both used to leave `_turn_outcome` at its "error"
default, so every identity query and every moderated query inflated the error
rate on the `turn_outcome` metric. The decline additionally never wrote the
trace output, so the chat export saw it as a blank answer.

The third early exit — moderation raising — is asserted to STAY "error": the
farmer got a placeholder, not an answer, and that belongs in the error rate.
It does now record its output so the export is not blank.
"""
import asyncio
import os
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest
from fastapi import BackgroundTasks

from app.services import chat as chat_service


def _capture(monkeypatch):
    """Permissive MagicMock Langfuse client; records scores and trace I/O.

    A THIN stub would blow up inside the turn and silently send it down the
    moderation fail-closed path, where these assertions would measure nothing.
    """
    from unittest.mock import MagicMock

    scored: list[dict] = []
    trace_io: list[dict] = []
    client = MagicMock()
    client.score_current_trace.side_effect = lambda **kw: scored.append(kw)
    client.set_current_trace_io.side_effect = lambda **kw: trace_io.append(kw)
    monkeypatch.setattr(chat_service, "get_langfuse_client", lambda: client)
    return scored, trace_io


def _outcomes(scored):
    return [s["value"] for s in scored if s.get("name") == "turn_outcome"]


def _outputs(trace_io):
    return [io["output"] for io in trace_io if "output" in io]


def _drive(monkeypatch, *, query, moderation):
    """Run a real turn end to end and return (emitted chunks, outcomes, outputs).

    `moderation` is either a category string or an Exception instance to raise.
    """
    from tests.test_chat_turn_sequence import _Cache, _Run

    scored, trace_io = _capture(monkeypatch)
    monkeypatch.setattr(chat_service.settings, "fallback_enabled", False)
    monkeypatch.setattr(chat_service, "propagate_attributes", None)
    monkeypatch.setattr(chat_service, "cache", _Cache())
    monkeypatch.setattr(chat_service, "trim_history", lambda *_a, **_k: [])
    monkeypatch.setattr(chat_service, "format_message_pairs", lambda *_a, **_k: "")

    async def _pre(text, *_a, **_k):
        return "translated query"

    async def _mod(user_message, model=None):
        if isinstance(moderation, Exception):
            raise moderation
        return SimpleNamespace(
            output=SimpleNamespace(category=moderation, action="I only answer farming questions.")
        )

    async def _tr(text, *_a, **_k):
        yield text

    async def _noop(*_a, **_k):
        return None

    monkeypatch.setattr(chat_service, "translate_to_english_pretranslation", _pre)
    monkeypatch.setattr(chat_service.moderation_agent, "run", _mod)
    monkeypatch.setattr(chat_service.agrinet_agent, "iter", lambda **_k: _Run(["Answer."]))
    monkeypatch.setattr(chat_service, "translate_text_stream_fast", _tr)
    monkeypatch.setattr(chat_service, "translate_text", _pre)
    monkeypatch.setattr(chat_service, "update_message_history", _noop)
    monkeypatch.setattr(chat_service, "set_cache", _noop)
    monkeypatch.setattr(chat_service, "create_suggestions", lambda *_a, **_k: None)

    emitted: list[str] = []

    async def _go():
        gen = chat_service.stream_chat_messages(
            query=query, session_id="short-circuit",
            source_lang="gu", target_lang="gu", channel="web",
            user_id="+919876543210", history=[], user_info={},
            background_tasks=BackgroundTasks(), use_translation_pipeline=True,
            pipeline_profile="managed",
        )
        async for chunk in gen:
            emitted.append(chunk)

    asyncio.run(_go())
    return emitted, _outcomes(scored), _outputs(trace_io)


IDENTITY_QUERY = "who are you?"


def test_identity_short_circuit_is_a_success_not_an_error(monkeypatch):
    emitted, outcomes, _ = _drive(monkeypatch, query=IDENTITY_QUERY, moderation="valid_agricultural")
    assert emitted, "the identity path must still answer the farmer"
    assert outcomes == ["success"]


def test_moderation_decline_is_a_success_not_an_error(monkeypatch):
    emitted, outcomes, _ = _drive(monkeypatch, query="who won the cricket match?", moderation="off_topic")
    assert emitted, "the decline text must still reach the farmer"
    assert outcomes == ["success"]


def test_moderation_decline_records_its_text_as_the_trace_output(monkeypatch):
    """Otherwise the trace has no output and the chat export logs a blank answer."""
    emitted, _, outputs = _drive(monkeypatch, query="who won the cricket match?", moderation="off_topic")
    decline_text = "".join(emitted)
    assert decline_text.strip()
    assert decline_text in outputs


def test_moderation_failure_stays_an_error_but_records_its_output(monkeypatch):
    emitted, outcomes, outputs = _drive(
        monkeypatch, query="how much water for my cow?", moderation=RuntimeError("moderation down")
    )
    assert emitted, "the fail-closed message must still reach the farmer"
    assert outcomes == ["error"]
    assert "".join(emitted) in outputs


def test_a_normal_turn_is_unaffected(monkeypatch):
    """Guards against the short-circuit edits leaking into the main path."""
    emitted, outcomes, _ = _drive(
        monkeypatch, query="how much water for my cow?", moderation="valid_agricultural"
    )
    assert emitted
    assert outcomes == ["success"]
