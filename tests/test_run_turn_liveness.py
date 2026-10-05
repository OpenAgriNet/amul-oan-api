"""Liveness and the staleness check on the seam.

``run_turn`` starts a surface's liveness right after the classifiers when it
has a side channel to speak through, stops it just before the first thing the
caller hears or when the turn ends, and asks the staleness check before it
commits anything. Chat passes neither, so its turns are unchanged.
"""
import asyncio
import contextlib
import dataclasses
import os
import time
from types import MappingProxyType

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

from app.channels.chat import WEB
from app.config import settings
from app.llm_core.execution import ExecutionContext
from app.services import chat as chat_service
from app.turn.types import ClassifierResult, TextEmission, Turn
from app.voice import liveness as voice_liveness
from app.voice.history import history_pair
from tests.test_chat_turn_contract import patch_turn


def _turn(**overrides):
    values = dict(
        query="How much water?",
        session_id="live",
        source_lang="en",
        target_lang="en",
        user_id="anonymous",
        authenticated_user=MappingProxyType({}),
        history=(),
        history_session_id="live",
        channel=WEB,
        persona="farmer",
    )
    values.update(overrides)
    return Turn(**values)


class _Scheduler:
    def schedule(self, fn, /, *args):
        pass


class _Outcomes:
    seen = []

    def __init__(self, turn, *, pipeline_profile, pipeline_trace):
        pass

    @contextlib.contextmanager
    def root(self):
        yield

    def record_output(self, text, label):
        pass

    def record_outcome(self, outcome):
        _Outcomes.seen.append(outcome)


class _Liveness:
    """Records how run_turn builds and stops it."""

    log = []
    built_with = None

    def __init__(self, turn, *, started_at, send, is_stale):
        _Liveness.built_with = dict(started_at=started_at, send=send, is_stale=is_stale)
        _Liveness.log.append("built")

    async def stop(self):
        _Liveness.log.append("stopped")


class _Sender:
    def __init__(self):
        self.sent = []

    async def send(self, emission):
        self.sent.append(emission.text)


class _Staleness:
    def __init__(self, stale_at=None, outcome="stale_request"):
        self.asked = []
        self._stale_at = stale_at
        self._outcome = outcome

    async def __call__(self, reason):
        self.asked.append(reason)
        return self._outcome if reason == self._stale_at else None


def _surface(liveness=_Liveness, **fields):
    _Outcomes.seen = []
    _Liveness.log = []
    _Liveness.built_with = None
    return dataclasses.replace(
        chat_service.CHAT_SURFACE, telemetry=_Outcomes, liveness=liveness, **fields
    )


def _run(turn, surface, *, side_channel=None, is_stale=None):
    async def _drain():
        out = []
        async for emission in chat_service.run_turn(
            turn, surface, scheduler=_Scheduler(), side_channel=side_channel, is_stale=is_stale,
        ):
            _Liveness.log.append("emit")
            out.append(emission)
        return out

    return asyncio.run(asyncio.wait_for(_drain(), timeout=10))


# ── liveness ────────────────────────────────────────────────────────────────


def test_chat_has_no_liveness():
    assert chat_service.CHAT_SURFACE.liveness is None


def test_without_a_side_channel_there_is_no_liveness(monkeypatch):
    patch_turn(monkeypatch)

    _run(_turn(), _surface())

    assert _Liveness.log == ["emit"]


def test_liveness_starts_after_the_classifiers_and_stops_before_the_first_emission(monkeypatch):
    patch_turn(monkeypatch)
    sender, staleness = _Sender(), _Staleness()
    before = time.monotonic()

    _run(_turn(), _surface(), side_channel=sender, is_stale=staleness)

    assert _Liveness.log == ["built", "stopped", "emit"]
    assert _Liveness.built_with["send"] is sender
    assert _Liveness.built_with["is_stale"] is staleness
    assert before <= _Liveness.built_with["started_at"] <= time.monotonic()


def test_the_deadline_counts_from_the_start_of_the_turn_not_from_the_classifiers(monkeypatch):
    patch_turn(monkeypatch)

    async def _slow_miss(turn):
        await asyncio.sleep(0.1)
        return None

    before = time.monotonic()

    _run(_turn(), _surface(classifiers=(_slow_miss,)), side_channel=_Sender())

    assert _Liveness.built_with["started_at"] - before < 0.1


def test_a_classifier_answer_never_starts_liveness(monkeypatch):
    patch_turn(monkeypatch)

    _run(_turn(query="who are you"), _surface(), side_channel=_Sender())

    assert _Liveness.log == ["emit"]


def test_a_failing_turn_still_stops_liveness(monkeypatch):
    patch_turn(monkeypatch)

    def _explode(**_kw):
        raise RuntimeError("agent exploded")

    monkeypatch.setattr(chat_service.agrinet_agent, "iter", _explode)

    with pytest.raises(RuntimeError):
        _run(_turn(), _surface(), side_channel=_Sender())

    assert _Liveness.log == ["built", "stopped"]


def test_a_gate_answer_stops_liveness_before_it_goes_out(monkeypatch):
    patch_turn(monkeypatch)

    class _Refuse:
        def __init__(self, turn, *, execution):
            pass

        async def gate(self):
            return ClassifierResult(canned_text="Goodbye.", label="non_meaningful_hangup", raw=True)

        async def close(self):
            pass

    emissions = _run(_turn(), _surface(background=_Refuse), side_channel=_Sender())

    assert emissions == [TextEmission("Goodbye.", raw=True)]
    assert _Liveness.log == ["built", "stopped", "emit"]


# ── the staleness check ─────────────────────────────────────────────────────


def test_a_fresh_turn_is_asked_once_before_its_history_write(monkeypatch):
    seen = patch_turn(monkeypatch)
    staleness = _Staleness()

    _run(_turn(), _surface(), is_stale=staleness)

    assert staleness.asked == ["before_query_pretranslation", "before_history_write"]
    assert len(seen["history_writes"]) == 1
    assert _Outcomes.seen == ["success"]


def test_a_stale_classifier_answer_is_neither_sent_nor_saved(monkeypatch):
    seen = patch_turn(monkeypatch)
    staleness = _Staleness(stale_at="before_identity_response", outcome="client_disconnected")

    emissions = _run(_turn(query="who are you"), _surface(), is_stale=staleness)

    assert emissions == []
    assert seen["history_writes"] == []
    assert _Outcomes.seen == ["client_disconnected"]


def test_a_stale_gate_answer_is_neither_sent_nor_saved(monkeypatch):
    seen = patch_turn(monkeypatch)
    closed = []

    class _Decline:
        def __init__(self, turn, *, execution):
            pass

        async def gate(self):
            return ClassifierResult(
                canned_text="Please ask about your animals.",
                label="moderation_rejected",
                history_pair=history_pair("[moderation-rejected]", "Please ask about your animals."),
            )

        async def close(self):
            closed.append(True)

    staleness = _Staleness(stale_at="before_moderation_rejected_response")

    emissions = _run(_turn(), _surface(background=_Decline), side_channel=_Sender(), is_stale=staleness)

    assert emissions == []
    assert seen["history_writes"] == []
    assert _Outcomes.seen == ["stale_request"]
    assert closed == [True]
    assert _Liveness.log == ["built", "stopped"]


def test_a_turn_gone_stale_by_the_end_does_not_write_history(monkeypatch):
    seen = patch_turn(monkeypatch)
    staleness = _Staleness(stale_at="before_history_write")

    emissions = _run(_turn(), _surface(), is_stale=staleness)

    assert "".join(e.text for e in emissions if isinstance(e, TextEmission)) == "Give clean water daily."
    assert seen["history_writes"] == []
    assert _Outcomes.seen == ["stale_request"]


# ── voice's nudge through run_turn ──────────────────────────────────────────


def _agent_after(monkeypatch, delay, *, tool_call=False):
    """An agent whose first chunk arrives after ``delay``; optionally it calls a
    tool first, the way voice's tools signal the nudge."""

    async def _stream(self, agent, prompt, *, message_history, deps, new_messages):
        if tool_call:
            voice_liveness.fire_tool_call_nudge()
        await asyncio.sleep(delay)
        yield "Give clean water daily."

    monkeypatch.setattr(ExecutionContext, "stream", _stream)


@pytest.fixture
def nudge_after_20ms(monkeypatch):
    monkeypatch.setattr(settings, "nudge_timeout_seconds", 0.02)
    monkeypatch.setattr(settings, "enable_voice_nudges", True)


def test_a_slow_answer_gets_one_timeout_nudge_first(monkeypatch, nudge_after_20ms):
    patch_turn(monkeypatch)
    _agent_after(monkeypatch, 0.2)
    sender = _Sender()

    emissions = _run(_turn(), _surface(voice_liveness.VoiceLiveness), side_channel=sender)

    (nudge,) = sender.sent
    assert nudge in voice_liveness._TIMEOUT_NUDGE_MESSAGES["en"]
    assert emissions == [TextEmission("Give clean water daily.")]


def test_a_fast_answer_gets_no_nudge(monkeypatch):
    monkeypatch.setattr(settings, "nudge_timeout_seconds", 1.0)
    patch_turn(monkeypatch)
    _agent_after(monkeypatch, 0)
    sender = _Sender()

    _run(_turn(), _surface(voice_liveness.VoiceLiveness), side_channel=sender)

    assert sender.sent == []


def test_a_tool_call_nudges_once_before_the_first_chunk(monkeypatch):
    monkeypatch.setattr(settings, "nudge_timeout_seconds", 60)
    patch_turn(monkeypatch)
    _agent_after(monkeypatch, 0.05, tool_call=True)
    sender = _Sender()

    _run(_turn(), _surface(voice_liveness.VoiceLiveness), side_channel=sender)

    (nudge,) = sender.sent
    assert nudge in voice_liveness._TOOL_NUDGE_MESSAGES["en"]


def test_a_greeting_gets_no_nudge(monkeypatch, nudge_after_20ms):
    patch_turn(monkeypatch)
    sender = _Sender()
    from app.voice.classifiers import voice_classifiers

    async def _render(text_en, target_lang, *, execution):
        return text_en

    surface = _surface(voice_liveness.VoiceLiveness, classifiers=voice_classifiers(_render))

    _run(_turn(query="hello"), surface, side_channel=sender)

    assert sender.sent == []


def test_voice_nudges_can_be_disabled_by_config(monkeypatch, nudge_after_20ms):
    monkeypatch.setattr(settings, "enable_voice_nudges", False)
    patch_turn(monkeypatch)
    _agent_after(monkeypatch, 0.2, tool_call=True)
    sender = _Sender()

    _run(_turn(), _surface(voice_liveness.VoiceLiveness), side_channel=sender)

    assert sender.sent == []
