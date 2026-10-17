"""Pins what a chat turn delivers, end to end, before `run_turn` moves it.

Written ahead of the #307 seam extraction so "behaviour-preserving" is checked
rather than asserted. Everything here drives ``stream_chat_messages`` — the
public entry point the router calls — not the internals, so the same tests hold
before and after the body moves behind ``run_turn``.

Pinned here because nothing else did:

  * the private-artifact frame: after the answer, before the history write, only
    when streaming, and always into the caller's sink;
  * suggestions are DEFERRED to after the response (they read the history this
    turn writes), never run inline;
  * one root span per turn;
  * the whole answer reaches the caller and the trace (ported from the sink
    gap tests on ``feat/run-turn-classifier-chain``: dropping the residual batch
    or the trace output left the suite green there).

The agent output in the sink tests is deliberately TWO SHORT SENTENCES: neither
meets the batch threshold, so the first survives only via the residual-batch
flush and the second only via the tail-fragment flush.
"""
import asyncio
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from fastapi import BackgroundTasks

from app.chat_artifacts import CHAT_ARTIFACTS_END, CHAT_ARTIFACTS_START, encode_chat_artifacts
from app.services import chat as chat_service
from tests.test_chat_turn_sequence import _Cache, _Node

_ARTIFACT = {"id": "shc-1", "kind": "soil_health_card", "content": "<p>private</p>"}
_NEW_MESSAGE = SimpleNamespace(kind="new-message-from-agent")


class _Run:
    """An agent run that can attach an artifact and report new messages."""

    def __init__(self, chunks, deps=None, artifact=None):
        self._chunks = chunks
        self.ctx = object()
        self.result = SimpleNamespace(new_messages=lambda: [_NEW_MESSAGE])
        if artifact is not None:
            deps.add_chat_artifact(artifact)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_e):
        return False

    async def __aiter__(self):
        yield _Node(self._chunks)


def patch_turn(monkeypatch, *, agent_text="Give clean water daily.", artifact=None):
    """Stub every external stage of a chat turn; return the dict it records into.

    Shared with tests/test_run_turn_seam.py, which drives the same stubbed turn
    through ``run_turn`` directly and through the real router.
    """
    seen = {
        "events": [],
        "history_writes": [],
        "trace_io": [],
        "translated_inputs": [],
        "suggestions_ran_inline": 0,
        "deps": [],
        "spans": [],
    }

    class _Span:
        """Records when an observation is actually ENTERED and EXITED, not just built."""

        def __init__(self, name):
            self.name = name

        def __enter__(self):
            seen["spans"].append(("enter", self.name))
            return MagicMock()

        def __exit__(self, *_exc):
            seen["spans"].append(("exit", self.name))
            return False

    client = MagicMock()
    client.set_current_trace_io.side_effect = lambda **kw: seen["trace_io"].append(kw)
    client.start_as_current_observation.side_effect = lambda **kw: _Span(kw.get("name"))
    monkeypatch.setattr(chat_service, "get_langfuse_client", lambda: client)
    seen["langfuse"] = client
    monkeypatch.setattr(chat_service, "propagate_attributes", None)
    monkeypatch.setattr(chat_service, "cache", _Cache())
    monkeypatch.setattr(chat_service, "trim_history", lambda *_a, **_k: [])
    monkeypatch.setattr(chat_service, "format_message_pairs", lambda *_a, **_k: "")

    async def _pretranslate(_tier, *, text, **_k):
        return "How much water should I give my cow?"

    async def _moderate(user_message, model=None):
        return SimpleNamespace(
            output=SimpleNamespace(category="valid_agricultural", action="allow")
        )

    def _agent_iter(**kw):
        seen["deps"].append(kw["deps"])
        return _Run([agent_text], deps=kw["deps"], artifact=artifact)

    async def _translate_stream(text, *_a, **_k):
        # Echo the input so a test can see WHICH pieces were translated.
        seen["translated_inputs"].append(text)
        yield f"[{text}]"

    async def _history(key, messages):
        seen["events"].append("history_write")
        seen["history_writes"].append((key, list(messages)))

    async def _noop(*_a, **_k):
        return None

    def _suggestions(*_a, **_k):
        seen["events"].append("suggestions")
        seen["suggestions_ran_inline"] += 1

    monkeypatch.setattr(chat_service, "pretranslate_with_tier", _pretranslate)
    monkeypatch.setattr(chat_service.moderation_agent, "run", _moderate)
    monkeypatch.setattr(chat_service.agrinet_agent, "iter", _agent_iter)
    monkeypatch.setattr(chat_service, "translate_text_stream_fast", _translate_stream)
    monkeypatch.setattr(chat_service, "update_message_history", _history)
    monkeypatch.setattr(chat_service, "set_cache", _noop)
    monkeypatch.setattr(chat_service, "create_suggestions", _suggestions)
    return seen


def _drive(
    monkeypatch,
    *,
    agent_text="Give clean water daily.",
    source_lang="gu",
    target_lang="gu",
    artifact=None,
    emit_artifact_frames=True,
    history=(),
    history_session_id=None,
):
    """Run one turn. Returns a dict of everything observable about it."""
    seen = patch_turn(monkeypatch, agent_text=agent_text, artifact=artifact)
    seen["sink"] = []
    seen["background_tasks"] = BackgroundTasks()

    async def _go():
        out = []
        async for chunk in chat_service.stream_chat_messages(
            query="મારી ગાયને કેટલું પાણી આપવું?",
            session_id="contract",
            source_lang=source_lang,
            target_lang=target_lang,
            channel="web",
            user_id="+919876543210",
            history=list(history),
            user_info={},
            background_tasks=seen["background_tasks"],
            history_session_id=history_session_id,
            artifact_sink=seen["sink"],
            emit_artifact_frames=emit_artifact_frames,
        ):
            seen["events"].append(("chunk", chunk))
            out.append(chunk)
        return out

    seen["chunks"] = asyncio.run(_go())
    seen["output"] = "".join(seen["chunks"])
    return seen


def _trace_outputs(seen):
    return [kw["output"] for kw in seen["trace_io"] if "output" in kw]


# ── private artifacts ───────────────────────────────────────────────────────


def test_artifact_frame_is_the_last_chunk_after_the_answer(monkeypatch):
    seen = _drive(monkeypatch, artifact=_ARTIFACT)

    frame = encode_chat_artifacts([_ARTIFACT])
    assert seen["chunks"][-1] == frame
    assert seen["chunks"][-1].startswith(CHAT_ARTIFACTS_START)
    assert seen["chunks"][-1].endswith(CHAT_ARTIFACTS_END)
    # The answer came first, and none of it is inside the frame.
    assert "".join(seen["chunks"][:-1]).strip()
    assert seen["sink"] == [_ARTIFACT]


def test_artifact_frame_precedes_the_history_write(monkeypatch):
    """A disconnect at the frame must not have persisted the turn yet."""
    seen = _drive(monkeypatch, artifact=_ARTIFACT)

    frame = encode_chat_artifacts([_ARTIFACT])
    assert seen["events"].index(("chunk", frame)) < seen["events"].index("history_write")


def test_artifacts_never_enter_history_or_trace_output(monkeypatch):
    seen = _drive(monkeypatch, artifact=_ARTIFACT)

    for _key, messages in seen["history_writes"]:
        assert "private" not in repr(messages)
    for output in _trace_outputs(seen):
        assert "private" not in output
        assert CHAT_ARTIFACTS_START not in output


def test_non_streaming_turn_gets_artifacts_in_the_sink_without_a_frame(monkeypatch):
    seen = _drive(monkeypatch, artifact=_ARTIFACT, emit_artifact_frames=False)

    assert CHAT_ARTIFACTS_START not in seen["output"]
    assert seen["sink"] == [_ARTIFACT]


def test_turn_without_artifacts_emits_no_frame(monkeypatch):
    seen = _drive(monkeypatch)

    assert CHAT_ARTIFACTS_START not in seen["output"]
    assert "" not in seen["chunks"], "an empty chunk leaked to the caller"
    assert seen["sink"] == []


# ── deferred work ───────────────────────────────────────────────────────────


def test_suggestions_are_deferred_not_run_inline(monkeypatch):
    """Suggestions read the history this turn writes, so they must run after it."""
    seen = _drive(monkeypatch)

    assert seen["suggestions_ran_inline"] == 0
    tasks = seen["background_tasks"].tasks
    assert len(tasks) == 1
    assert tasks[0].func is chat_service.create_suggestions
    session_id, target_lang, execution = tasks[0].args
    assert (session_id, target_lang) == ("contract", "gu")
    assert execution is not None


# ── history ─────────────────────────────────────────────────────────────────


def test_history_is_written_once_with_prior_and_new_messages(monkeypatch):
    prior = (SimpleNamespace(kind="prior-1"), SimpleNamespace(kind="prior-2"))
    seen = _drive(monkeypatch, history=prior, history_session_id="contract:ns")

    assert len(seen["history_writes"]) == 1
    key, messages = seen["history_writes"][0]
    assert key == "contract:ns"
    assert messages == [*prior, _NEW_MESSAGE]


def test_history_key_defaults_to_the_session_id(monkeypatch):
    seen = _drive(monkeypatch)
    assert [key for key, _ in seen["history_writes"]] == ["contract"]


# ── telemetry ───────────────────────────────────────────────────────────────


def test_one_root_span_wraps_the_whole_turn(monkeypatch):
    """Exactly one root span, entered first and exited last, children inside it.

    Without it every observation becomes its own top-level trace and trace-level
    writes are dropped (#200). Building the span is not enough — it must be
    ENTERED, which is why this records enter/exit rather than counting calls.
    """
    seen = _drive(monkeypatch)
    spans = seen["spans"]

    assert [s for s in spans if s[1] == "chat.translation"] == [
        ("enter", "chat.translation"),
        ("exit", "chat.translation"),
    ], spans
    assert spans[0] == ("enter", "chat.translation"), spans
    assert spans[-1] == ("exit", "chat.translation"), spans
    assert len(spans) > 2, f"no child observation was nested under the root: {spans}"


# ── the whole answer reaches the caller and the trace ───────────────────────


def test_no_part_of_the_answer_is_dropped(monkeypatch):
    seen = _drive(monkeypatch, agent_text="Yes. No.")

    joined = "".join(seen["translated_inputs"])
    assert "Yes." in joined, f"first sentence never translated: {seen['translated_inputs']}"
    assert "No." in joined, f"tail fragment never translated: {seen['translated_inputs']}"
    assert "[Yes. ]" in seen["output"] or "[Yes.]" in seen["output"], seen["output"]
    assert "[No.]" in seen["output"], seen["output"]


def test_trace_output_records_the_answer(monkeypatch):
    seen = _drive(monkeypatch, agent_text="Yes. No.")

    outputs = _trace_outputs(seen)
    assert outputs, f"no trace output recorded; calls={seen['trace_io']}"
    assert outputs[-1] == seen["output"]


def test_english_turn_records_untranslated_output(monkeypatch):
    seen = _drive(
        monkeypatch, source_lang="en", target_lang="en", agent_text="Give clean water."
    )

    assert seen["translated_inputs"] == []
    assert _trace_outputs(seen)[-1] == seen["output"] == "Give clean water."
