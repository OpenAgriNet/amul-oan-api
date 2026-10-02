"""The background set and the gate on the seam.

``run_turn`` builds a surface's background right after the classifiers, pulls
the agent's first chunk, asks the gate before anything is emitted, and closes
the background on every exit. Chat has no background, so its turns run exactly
as before; voice's is ``app.voice.background``.
"""
import asyncio
import contextlib
import dataclasses
import os
from types import MappingProxyType

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

from app.channels.chat import WEB
from app.llm_core.execution import ExecutionContext
from app.services import chat as chat_service
from app.turn.types import ClassifierResult, TextEmission, Turn
from app.voice import background as voice_bg
from app.voice.history import history_pair
from app.voice.moderation import ModerationVerdict
from app.voice.non_meaningful import NonMeaningfulVerdict
from tests.test_chat_turn_contract import patch_turn


def _turn(**overrides):
    values = dict(
        query="How much water?",
        session_id="gate",
        source_lang="en",
        target_lang="en",
        user_id="anonymous",
        authenticated_user=MappingProxyType({}),
        history=(),
        history_session_id="gate-history",
        channel=WEB,
        persona="farmer",
    )
    values.update(overrides)
    return Turn(**values)


class _Scheduler:
    def schedule(self, fn, /, *args):
        pass


class _Labels:
    """Records what run_turn hands the surface's telemetry."""

    seen = []

    def __init__(self, turn, *, pipeline_profile, pipeline_trace):
        pass

    @contextlib.contextmanager
    def root(self):
        yield

    def record_output(self, text, label):
        _Labels.seen.append(("output", label))

    def record_outcome(self, outcome):
        _Labels.seen.append(("outcome", outcome))


class _Background:
    """A background that records its lifecycle and answers the gate as told."""

    log = []
    decision = None
    seen = None

    def __init__(self, turn, *, execution):
        _Background.log.append("built")

    async def gate(self):
        # What had happened by the time the gate was asked.
        _Background.log.append(("gate", len(_Background.seen["deps"]), len(_Background.seen["emitted"])))
        return _Background.decision

    async def close(self):
        _Background.log.append("closed")


def _surface(background=_Background):
    _Labels.seen = []
    return dataclasses.replace(chat_service.CHAT_SURFACE, telemetry=_Labels, background=background)


def _prepare(monkeypatch, *, decision=None, agent_text="Give clean water daily."):
    seen = patch_turn(monkeypatch, agent_text=agent_text)
    seen["emitted"] = []
    _Background.log = []
    _Background.decision = decision
    _Background.seen = seen
    return seen


async def _drain(agen, seen):
    out = []
    async for emission in agen:
        seen["emitted"].append(emission)
        out.append(emission)
    return out


def _run(turn, surface, seen):
    # Bounded: some agents here never finish on their own, so a turn that stops
    # gating must fail the test, not hang it.
    turn_run = _drain(chat_service.run_turn(turn, surface, scheduler=_Scheduler()), seen)
    return asyncio.run(asyncio.wait_for(turn_run, timeout=10))


def _still_generating(monkeypatch, on_close):
    """An agent stream that has sent its first chunk and is still generating, as
    a real model is when the gate is asked. ``on_close`` runs when the stream the
    turn pulled from is closed."""

    async def _stream(self, agent, prompt, *, message_history, deps, new_messages):
        try:
            yield "Give clean water daily."
            await asyncio.Event().wait()
        finally:
            on_close()

    monkeypatch.setattr(ExecutionContext, "stream", _stream)


def _decision(*, raw=False, history=True):
    return ClassifierResult(
        canned_text="Goodbye." if raw else "Please ask about your animals.",
        label="non_meaningful_hangup" if raw else "moderation_rejected",
        history_pair=history_pair("[moderation-rejected]", "Please ask about your animals.") if history else None,
        raw=raw,
    )


# ── chat ────────────────────────────────────────────────────────────────────


def test_chat_has_no_background():
    assert chat_service.CHAT_SURFACE.background is None


def test_a_background_that_lets_the_turn_through_changes_nothing(monkeypatch):
    seen = _prepare(monkeypatch)
    plain = _run(_turn(), dataclasses.replace(chat_service.CHAT_SURFACE, telemetry=_Labels), seen)
    plain_history = seen["history_writes"]

    seen = _prepare(monkeypatch)
    gated = _run(_turn(), _surface(), seen)

    assert gated == plain
    assert seen["history_writes"] == plain_history
    assert _Background.log == ["built", ("gate", 1, 0), "closed"]


# ── the gate ────────────────────────────────────────────────────────────────


def test_the_gate_is_asked_after_the_agent_starts_and_before_anything_is_emitted(monkeypatch):
    seen = _prepare(monkeypatch)

    _run(_turn(), _surface(), seen)

    ((_, agents_started, emitted),) = [e for e in _Background.log if isinstance(e, tuple)]
    assert agents_started == 1
    assert emitted == 0


def test_a_gate_decision_replaces_the_agents_answer(monkeypatch):
    seen = _prepare(monkeypatch, decision=_decision())
    exits = []
    _still_generating(monkeypatch, lambda: exits.append(("agent stream closed", len(seen["emitted"]))))

    emissions = _run(_turn(source_lang="gu", target_lang="gu"), _surface(), seen)

    assert emissions == [TextEmission("Please ask about your animals.")]
    assert seen["translated_inputs"] == [], "the agent's answer reached the sink"
    # Closed by the gate, before the decline goes out, not later by the collector.
    assert exits == [("agent stream closed", 0)]
    ((key, messages),) = seen["history_writes"]
    assert key == "gate-history"
    assert [p.content for m in messages for p in m.parts] == ["[moderation-rejected]", "Please ask about your animals."]
    assert ("output", "moderation_rejected") in _Labels.seen
    assert _Labels.seen[-1] == ("outcome", "success")
    assert _Background.log[-1] == "closed"


def test_a_raw_gate_decision_stays_raw(monkeypatch):
    seen = _prepare(monkeypatch, decision=_decision(raw=True, history=False))

    emissions = _run(_turn(), _surface(), seen)

    assert emissions == [TextEmission("Goodbye.", raw=True)]
    assert seen["history_writes"] == []


def test_an_agent_failure_before_its_first_chunk_is_raised_after_the_gate(monkeypatch):
    seen = _prepare(monkeypatch)

    def _explode(**_kw):
        raise RuntimeError("agent exploded")

    monkeypatch.setattr(chat_service.agrinet_agent, "iter", _explode)

    with pytest.raises(RuntimeError, match="agent exploded"):
        _run(_turn(), _surface(), seen)

    assert [e if isinstance(e, str) else e[0] for e in _Background.log] == ["built", "gate", "closed"]
    assert _Labels.seen[-1] == ("outcome", "error")


def test_a_turn_the_gate_refuses_is_refused_even_if_the_agent_failed(monkeypatch):
    seen = _prepare(monkeypatch, decision=_decision())

    def _explode(**_kw):
        raise RuntimeError("agent exploded")

    monkeypatch.setattr(chat_service.agrinet_agent, "iter", _explode)

    emissions = _run(_turn(), _surface(), seen)

    assert emissions == [TextEmission("Please ask about your animals.")]
    assert _Labels.seen[-1] == ("outcome", "success")


def test_a_gate_that_fails_closes_the_agents_stream(monkeypatch):
    seen = _prepare(monkeypatch)
    _still_generating(monkeypatch, lambda: _Background.log.append("agent stream closed"))

    class _BrokenGate(_Background):
        async def gate(self):
            raise RuntimeError("gate broke")

    with pytest.raises(RuntimeError, match="gate broke"):
        _run(_turn(), _surface(_BrokenGate), seen)

    # Closed where the gate failed, before the turn unwinds, not later by the collector.
    assert _Background.log == ["built", "agent stream closed", "closed"]
    assert seen["emitted"] == []


# ── close on every exit ─────────────────────────────────────────────────────


def test_a_classifier_answer_never_builds_the_background(monkeypatch):
    seen = _prepare(monkeypatch)

    _run(_turn(query="who are you"), _surface(), seen)

    assert _Background.log == []


def test_a_failure_before_the_agent_still_closes_the_background(monkeypatch):
    seen = _prepare(monkeypatch)

    async def _broken(*_a, **_k):
        raise RuntimeError("farmer context down")

    monkeypatch.setattr(chat_service, "_load_farmer_context", _broken)

    with pytest.raises(RuntimeError, match="farmer context down"):
        _run(_turn(), _surface(), seen)

    assert _Background.log == ["built", "closed"]


def test_a_hang_up_mid_answer_still_closes_the_background(monkeypatch):
    _prepare(monkeypatch, agent_text="First sentence. Second sentence.")

    async def _first_only():
        agen = chat_service.run_turn(_turn(), _surface(), scheduler=_Scheduler())
        first = await agen.__anext__()
        await agen.aclose()
        return first

    first = asyncio.run(_first_only())

    assert isinstance(first, TextEmission)
    assert _Background.log[-1] == "closed"
    assert _Labels.seen[-1] == ("outcome", "cancelled")


# ── voice's background through run_turn ─────────────────────────────────────


async def _render(text_en, target_lang):
    return f"<{target_lang}> {text_en}"


def _voice_checks(monkeypatch, *, moderation, streak):
    async def _moderation(**_kw):
        return moderation

    async def _non_meaningful(**_kw):
        return streak

    monkeypatch.setattr(voice_bg, "check_moderation", _moderation)
    monkeypatch.setattr(voice_bg, "check_non_meaningful_streak", _non_meaningful)


def test_voice_declines_a_rejected_query_on_the_seam(monkeypatch):
    seen = _prepare(monkeypatch)
    _voice_checks(
        monkeypatch,
        moderation=ModerationVerdict(category="offensive", reason="abuse"),
        streak=NonMeaningfulVerdict(five_consecutive_non_meaningful=False, reason="fine"),
    )

    emissions = _run(_turn(target_lang="gu"), _surface(voice_bg.voice_background(_render)), seen)

    (emission,) = emissions
    assert emission.text.startswith("<gu> This is a service for farmers.")
    assert ("output", "moderation_rejected") in _Labels.seen


def test_voice_lets_a_clean_turn_through_on_the_seam(monkeypatch):
    seen = _prepare(monkeypatch)
    _voice_checks(
        monkeypatch,
        moderation=ModerationVerdict(category="in_scope", reason="ok"),
        streak=NonMeaningfulVerdict(five_consecutive_non_meaningful=False, reason="fine"),
    )

    emissions = _run(_turn(), _surface(voice_bg.voice_background(_render)), seen)

    assert "".join(e.text for e in emissions if isinstance(e, TextEmission)) == "Give clean water daily."
    assert ("output", "final") in _Labels.seen
