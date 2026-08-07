"""Pins what the SINK must deliver: the complete answer, and the trace output.

Written because a mutation sweep of app/turn/sinks.py found two changes the
existing 729 tests did not notice:

  * dropping the final unflushed translation batch -> the farmer's answer is
    silently TRUNCATED, suite stays green;
  * ``final_text()`` returning None -> the turn's trace output is lost, suite
    stays green (the same class of defect as the missing turn root span).

Both drive ``stream_chat_messages`` end to end rather than poking ChatSink
directly: testing the sink in isolation would prove the sink works, not that the
orchestrator uses it.

The agent output here is deliberately TWO SHORT SENTENCES. That shape is what
exercises both flush sites: neither sentence meets the batch threshold, so the
first leaves a residual batch flushed after the loop, and the second leaves a
tail fragment. The single-sentence output used elsewhere only reaches the tail.
"""
import asyncio
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from fastapi import BackgroundTasks

from app.services import chat as chat_service


def _drive(monkeypatch, *, agent_text="Yes. No.", target_lang="gu", source_lang="gu"):
    """Run one turn; return (client_output, trace_outputs, translated_inputs)."""
    from tests.test_chat_turn_sequence import _Cache, _Run

    trace_io = []
    translated_inputs = []

    client = MagicMock()
    client.set_current_trace_io.side_effect = lambda **kw: trace_io.append(kw)
    monkeypatch.setattr(chat_service, "get_langfuse_client", lambda: client)
    monkeypatch.setattr(chat_service, "propagate_attributes", None)
    monkeypatch.setattr(chat_service.settings, "fallback_enabled", False)
    monkeypatch.setattr(chat_service, "cache", _Cache())
    monkeypatch.setattr(chat_service, "trim_history", lambda *_a, **_kw: [])
    monkeypatch.setattr(chat_service, "format_message_pairs", lambda *_a, **_kw: "")

    async def _pretranslate(text, *_a, **_kw):
        return "How much water should I give my cow?"

    async def _moderate(user_message, model=None):
        return SimpleNamespace(
            output=SimpleNamespace(category="valid_agricultural", action="allow")
        )

    def _agent_iter(**_kw):
        return _Run([agent_text])

    async def _translate_stream(text, *_a, **_kw):
        # Echo the input so the test can see WHICH pieces were translated.
        translated_inputs.append(text)
        yield f"[{text}]"

    async def _noop(*_a, **_kw):
        return None

    monkeypatch.setattr(chat_service, "translate_to_english_pretranslation", _pretranslate)
    monkeypatch.setattr(chat_service.moderation_agent, "run", _moderate)
    monkeypatch.setattr(chat_service.agrinet_agent, "iter", _agent_iter)
    monkeypatch.setattr(chat_service, "translate_text_stream_fast", _translate_stream)
    monkeypatch.setattr(chat_service, "update_message_history", _noop)
    monkeypatch.setattr(chat_service, "set_cache", _noop)
    monkeypatch.setattr(chat_service, "create_suggestions", lambda *_a, **_kw: None)

    async def _go():
        out = []
        async for chunk in chat_service.stream_chat_messages(
            query="મારી ગાયને કેટલું પાણી આપવું?",
            session_id="sink",
            source_lang=source_lang,
            target_lang=target_lang,
            channel="web",
            user_id="+919876543210",
            history=[],
            user_info={},
            background_tasks=BackgroundTasks(),
            use_translation_pipeline=True,
            pipeline_profile="managed",
        ):
            out.append(chunk)
        return "".join(out)

    return asyncio.run(_go()), trace_io, translated_inputs


def test_no_part_of_the_answer_is_dropped(monkeypatch):
    """Every sentence the agent produced must reach the caller.

    Kills the mutation that drops the residual batch after the stream ends:
    "Yes." never meets the flush threshold, so it survives only via that path.
    """
    output, _trace_io, translated = _drive(monkeypatch)

    joined = "".join(translated)
    assert "Yes." in joined, f"first sentence never translated: {translated}"
    assert "No." in joined, f"tail fragment never translated: {translated}"
    assert "[Yes. ]" in output or "[Yes.]" in output, f"first sentence missing from output: {output!r}"
    assert "[No.]" in output, f"tail fragment missing from output: {output!r}"


def test_trace_output_records_the_answer(monkeypatch):
    """The turn's trace must carry the answer, not just its input.

    Kills the mutation that makes final_text() return None. Trace output going
    missing is invisible to the caller and shows up only as blank rows in the
    export, weeks later.
    """
    output, trace_io, _translated = _drive(monkeypatch)

    outputs = [kw["output"] for kw in trace_io if "output" in kw]
    assert outputs, f"no trace output recorded at all; calls={trace_io}"
    assert outputs[-1], "trace output recorded but empty"
    assert outputs[-1] == output, (
        f"trace output does not match what the caller received:\n"
        f"  trace={outputs[-1]!r}\n  client={output!r}"
    )


def test_english_turn_records_untranslated_output(monkeypatch):
    """The no-translation path must still populate the trace output."""
    output, trace_io, translated = _drive(
        monkeypatch, target_lang="en", source_lang="en", agent_text="Give clean water."
    )

    assert translated == [], f"english turn should not translate: {translated}"
    outputs = [kw["output"] for kw in trace_io if "output" in kw]
    assert outputs, "no trace output recorded on the english path"
    assert outputs[-1] == output == "Give clean water."
