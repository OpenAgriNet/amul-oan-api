"""The agent input on the seam: run_turn runs the agent with what the surface
builds, and says an answer the surface gives instead like the gate's.

Chat's agent input is ``_chat_agent_input``: the farmer context, the
FarmerContext, and moderation, which decides before the agent starts.
"""
import asyncio
import dataclasses
import os
from contextlib import contextmanager
from types import MappingProxyType, SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest
from fastapi import BackgroundTasks
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from agents.deps import FarmerContext
from app import llm_core
from app.channels.chat import WEB
from app.services import chat as chat_service
from app.turn.types import AgentInput, ClassifierResult, Pretranslated, TextEmission, Turn
from tests.test_chat_turn_contract import _NEW_MESSAGE, _Run, patch_turn

_PRIOR = (ModelRequest(parts=[UserPromptPart(content="earlier")]),)
_KEPT = [ModelRequest(parts=[UserPromptPart(content="kept")])]
_PAIR = (
    ModelRequest(parts=[UserPromptPart(content="asked")]),
    ModelResponse(parts=[TextPart(content="answered")]),
)


def _turn(**overrides):
    values = dict(
        query="How much water?",
        session_id="agent-input",
        source_lang="gu",
        target_lang="gu",
        user_id="anonymous",
        authenticated_user=MappingProxyType({}),
        history=_PRIOR,
        history_session_id="agent-input",
        channel=WEB,
        persona="farmer",
    )
    values.update(overrides)
    return Turn(**values)


class _Scheduler:
    def schedule(self, fn, /, *args):
        pass


async def _collect(agen):
    return [item async for item in agen]


def _outcomes(seen):
    return [
        call.kwargs["value"]
        for call in seen["langfuse"].score_current_trace.call_args_list
        if call.kwargs.get("name") == "turn_outcome"
    ]


def _trace_outputs(seen):
    return [kw["output"] for kw in seen["trace_io"] if "output" in kw]


class _Agent:
    """Records what it is run with, and answers with fixed text."""

    def __init__(self):
        self.runs = []

    def iter(self, **kwargs):
        self.runs.append(kwargs)
        return _Run(["Built answer."])


class _Observation:
    def __init__(self):
        self.log = []

    @contextmanager
    def __call__(self):
        handle = SimpleNamespace(update=lambda **kw: self.log.append(("update", kw)))
        self.log.append("enter")
        yield handle
        self.log.append("exit")


def _built(agent, observe, calls):
    async def _agent_input(turn, pretranslated, *, execution, scheduler, translate_to, **_kw):
        calls.append(dict(turn=turn, pretranslated=pretranslated, scheduler=scheduler, translate_to=translate_to))
        return AgentInput(
            agent=agent,
            prompt="PROMPT",
            message_history=list(_KEPT),
            deps=FarmerContext(query="q"),
            history=("saved-before",),
            observe=observe,
        )

    return _agent_input


def _answering(result, calls=None):
    async def _agent_input(turn, pretranslated, **_kw):
        if calls is not None:
            calls.append(pretranslated)
        return result

    return _agent_input


def test_chat_surface_populates_its_agent_input():
    assert chat_service.CHAT_SURFACE.agent_input is chat_service._chat_agent_input


def test_the_agent_runs_with_what_the_surface_built(monkeypatch):
    seen = patch_turn(monkeypatch)
    agent, observe, calls = _Agent(), _Observation(), []
    surface = dataclasses.replace(chat_service.CHAT_SURFACE, agent_input=_built(agent, observe, calls))
    scheduler = _Scheduler()

    emissions = asyncio.run(_collect(chat_service.run_turn(_turn(), surface, scheduler=scheduler)))

    (run,) = agent.runs
    assert run["user_prompt"] == "PROMPT"
    assert run["message_history"] == _KEPT
    assert isinstance(run["deps"], FarmerContext) and run["deps"].query == "q"
    (call,) = calls
    assert call["pretranslated"] == Pretranslated(query="How much water should I give my cow?", lang="en")
    assert call["scheduler"] is scheduler
    assert call["translate_to"] == "gu"
    assert "".join(e.text for e in emissions if isinstance(e, TextEmission))
    # The new messages are saved after the history the surface named, not the turn's.
    assert seen["history_writes"] == [("agent-input", ["saved-before", _NEW_MESSAGE])]
    assert observe.log[0] == "enter" and observe.log[-1] == "exit"
    assert ("update", {"output": {"response": _trace_outputs(seen)[-1]}}) in observe.log
    assert _outcomes(seen) == ["success"]


def test_an_english_turn_asks_for_no_translation(monkeypatch):
    patch_turn(monkeypatch)
    calls = []
    surface = dataclasses.replace(
        chat_service.CHAT_SURFACE, agent_input=_built(_Agent(), _Observation(), calls)
    )

    asyncio.run(_collect(chat_service.run_turn(
        _turn(source_lang="en", target_lang="en"), surface, scheduler=_Scheduler()
    )))

    assert calls[0]["translate_to"] is None


@pytest.mark.parametrize("outcome", ["success", "error"])
def test_an_answer_instead_is_said_like_the_gates(monkeypatch, outcome):
    seen = patch_turn(monkeypatch)
    asked = []

    async def _is_stale(reason):
        asked.append(reason)
        return None

    result = ClassifierResult(canned_text="Goodbye.", label="blocked", history_pair=_PAIR, raw=True, outcome=outcome)
    surface = dataclasses.replace(chat_service.CHAT_SURFACE, agent_input=_answering(result))

    emissions = asyncio.run(_collect(chat_service.run_turn(
        _turn(), surface, scheduler=_Scheduler(), is_stale=_is_stale
    )))

    assert emissions == [TextEmission("Goodbye.", raw=True)]
    assert "before_blocked_response" in asked
    assert seen["deps"] == [], "the agent must not run"
    assert seen["history_writes"] == [("agent-input", [*_PRIOR, *_PAIR])]
    assert _trace_outputs(seen) == ["Goodbye."]
    assert _outcomes(seen) == [outcome]


class _Gate:
    def __init__(self, turn, *, execution):
        pass

    async def gate(self):
        return _FAILED

    async def close(self):
        pass


_FAILED = ClassifierResult(canned_text="Try again later.", label="failed", outcome="error")


async def _classify(turn):
    return _FAILED


async def _pretranslate(turn, **_kw):
    return _FAILED


@pytest.mark.parametrize("slot", [
    dict(classifiers=(_classify,)),
    dict(pretranslation=_pretranslate),
    dict(background=_Gate),
    dict(agent_input=_answering(_FAILED)),
])
def test_every_answer_instead_records_the_outcome_it_carries(monkeypatch, slot):
    seen = patch_turn(monkeypatch)
    surface = dataclasses.replace(chat_service.CHAT_SURFACE, **slot)

    emissions = asyncio.run(_collect(chat_service.run_turn(_turn(), surface, scheduler=_Scheduler())))

    assert emissions == [TextEmission("Try again later.")]
    assert _outcomes(seen) == ["error"]


_FAREWELL = ClassifierResult(canned_text="Bye for now.", label="farewell", raw_tail=" Goodbye.")


async def _classify_farewell(turn):
    return _FAREWELL


async def _pretranslate_farewell(turn, **_kw):
    return _FAREWELL


class _FarewellGate(_Gate):
    async def gate(self):
        return _FAREWELL


@pytest.mark.parametrize("slot", [
    dict(classifiers=(_classify_farewell,)),
    dict(pretranslation=_pretranslate_farewell),
    dict(background=_FarewellGate),
    dict(agent_input=_answering(_FAREWELL)),
])
def test_every_answer_instead_says_its_raw_tail_after_it(monkeypatch, slot):
    seen = patch_turn(monkeypatch)
    surface = dataclasses.replace(chat_service.CHAT_SURFACE, **slot)

    emissions = asyncio.run(_collect(chat_service.run_turn(_turn(), surface, scheduler=_Scheduler())))

    assert emissions == [TextEmission("Bye for now."), TextEmission(" Goodbye.", raw=True)]
    assert _outcomes(seen) == ["success"]


def test_a_stale_turn_says_nothing_instead(monkeypatch):
    seen = patch_turn(monkeypatch)

    async def _is_stale(reason):
        return "stale_request" if reason == "before_blocked_response" else None

    result = ClassifierResult(canned_text="Goodbye.", label="blocked", history_pair=_PAIR)
    surface = dataclasses.replace(chat_service.CHAT_SURFACE, agent_input=_answering(result))

    emissions = asyncio.run(_collect(chat_service.run_turn(
        _turn(), surface, scheduler=_Scheduler(), is_stale=_is_stale
    )))

    assert emissions == []
    assert seen["history_writes"] == []
    assert _outcomes(seen) == ["stale_request"]


def test_liveness_stops_before_the_answer_instead(monkeypatch):
    patch_turn(monkeypatch)
    log = []

    class _Liveness:
        def __init__(self, turn, **_kw):
            log.append("started")

        async def stop(self):
            log.append("stopped")

    class _Sender:
        async def send(self, emission):
            pass

    async def _collect_logged(agen):
        async for emission in agen:
            log.append(emission.text)

    result = ClassifierResult(canned_text="Declined.", label="blocked")
    surface = dataclasses.replace(
        chat_service.CHAT_SURFACE, agent_input=_answering(result), liveness=_Liveness
    )

    asyncio.run(_collect_logged(chat_service.run_turn(
        _turn(), surface, scheduler=_Scheduler(), side_channel=_Sender()
    )))

    assert log == ["started", "stopped", "Declined."]


def test_a_surface_without_an_agent_input_fails_before_the_agent(monkeypatch):
    seen = patch_turn(monkeypatch)
    surface = dataclasses.replace(chat_service.CHAT_SURFACE, agent_input=None)

    with pytest.raises(TypeError, match="without an agent input"):
        asyncio.run(_collect(chat_service.run_turn(_turn(), surface, scheduler=_Scheduler())))

    assert seen["deps"] == []
    assert _outcomes(seen) == ["error"]


# ── chat's agent input ──────────────────────────────────────────────────────


def _chat_agent_input(turn, *, translate_to="gu", scheduler=None):
    async def _go():
        execution = await llm_core.context(turn.session_id)
        return await chat_service._chat_agent_input(
            turn,
            # A language pretranslation left alone, as a disabled one is.
            Pretranslated(query="How much water?", lang="gu"),
            execution=execution,
            scheduler=scheduler or _Scheduler(),
            translate_to=translate_to,
        )

    return asyncio.run(_go())


def _moderation(monkeypatch, *, category="valid_agricultural", action="allow", fails=False):
    async def _moderate(user_message, model=None):
        if fails:
            raise RuntimeError("moderation down")
        return SimpleNamespace(output=SimpleNamespace(category=category, action=action))

    monkeypatch.setattr(chat_service.moderation_agent, "run", _moderate)
    monkeypatch.setattr(chat_service.doctor_moderation_agent, "run", _moderate)


async def _as_is(text, **_kw):
    return text


def test_chat_runs_its_agent_with_the_farmer_context(monkeypatch):
    patch_turn(monkeypatch)

    built = _chat_agent_input(_turn())

    assert isinstance(built, AgentInput)
    assert built.agent is chat_service.agrinet_agent
    assert built.history == _PRIOR
    assert built.deps.query == "How much water?"
    assert built.deps.lang_code == "en"
    assert built.deps.moderation_str
    assert built.prompt == built.deps.get_user_message()


def test_chat_answers_in_the_pretranslated_language_when_nothing_is_translated(monkeypatch):
    patch_turn(monkeypatch)
    _moderation(monkeypatch)

    built = _chat_agent_input(_turn(), translate_to=None)

    assert built.deps.lang_code == "gu"
    doctor = _chat_agent_input(_turn(persona="doctor"), translate_to=None)
    assert doctor.agent is chat_service.doctor_agent


def test_a_rejected_query_is_declined_as_a_success(monkeypatch):
    patch_turn(monkeypatch)
    _moderation(monkeypatch, category="unsafe", action="I cannot help with that.")
    monkeypatch.setattr(chat_service, "_localize_system_text", _as_is)

    declined = _chat_agent_input(_turn())

    assert declined == ClassifierResult(canned_text="I cannot help with that.", label="moderation decline")
    assert declined.outcome == "success" and declined.history_pair is None


def test_a_failed_moderation_gets_the_fail_closed_line_as_an_error(monkeypatch):
    patch_turn(monkeypatch)
    _moderation(monkeypatch, fails=True)
    monkeypatch.setattr(chat_service, "_localize_system_text", _as_is)

    failed = _chat_agent_input(_turn())

    assert failed == ClassifierResult(
        canned_text=chat_service.GENERIC_UNAVAILABLE_MESSAGE_EN, label="fail-closed", outcome="error"
    )


def test_a_decline_that_cannot_be_rendered_is_an_error(monkeypatch):
    """Rendering fails inside run_turn's guard, not inside chat's moderation
    handler, so it is recorded once, as an error, and not answered again."""
    seen = patch_turn(monkeypatch)
    _moderation(monkeypatch, category="unsafe", action="I cannot help with that.")
    monkeypatch.setattr(chat_service, "_localize_system_text", _as_is)
    rendered = []

    def _render(emission, *_a):
        rendered.append(emission)
        raise TypeError("cannot render")

    monkeypatch.setattr(chat_service, "_render_chat_emission", _render)

    async def _go():
        return [chunk async for chunk in chat_service.stream_chat_messages(
            query="q",
            session_id="decline",
            source_lang="en",
            target_lang="en",
            channel="web",
            user_id="anonymous",
            history=[],
            user_info={},
            background_tasks=BackgroundTasks(),
        )]

    with pytest.raises(TypeError, match="cannot render"):
        asyncio.run(_go())

    assert rendered == [TextEmission("I cannot help with that.")]
    assert _outcomes(seen) == ["error"]
