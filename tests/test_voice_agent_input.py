"""Voice's agent input: what the voice agent is run with on a call turn, the
outbound consent gate, and the goodbye after an answer that closes the call.

The farmer fetch, the checks and the outbound state are stood in for; the
agent input itself, the background set and run_turn are the real ones.
"""
import asyncio
import dataclasses
import os
from types import MappingProxyType, SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart

from agents.voice import agent as voice_agents
from agents.voice.models.farmer import FarmerDataEnvelope
from app import model_boundary_capture
from app.channels.chat import WEB
from app.services import chat as chat_service
from app.turn.types import AgentInput, ClassifierResult, Pretranslated, TelephonyCall, TextEmission, Turn
from app.voice import agent_input as ai
from app.voice import background as bg
from app.voice import outbound
from app.voice.moderation import ModerationVerdict
from app.voice.non_meaningful import NonMeaningfulVerdict
from app.voice.outbound_consent import ConsentVerdict
from tests.test_chat_turn_contract import patch_turn
from tests.test_chat_turn_sequence import _Node

_EXECUTION = object()
_MOBILE = "9876543210"
_FOUND = FarmerDataEnvelope.from_records(
    [{
        "farmerName": "Rameshbhai",
        "farmerCode": "5058",
        "societyCode": "00002",
        "societyName": "Chikhodra",
        "unionCode": "159",
        "unionName": "Kaira",
        "district": "Anand",
    }],
    source="api",
    lookup_status="found",
)
_PRIOR = (
    ModelRequest(parts=[UserPromptPart(content="earlier")]),
    ModelResponse(parts=[TextPart(content="answered")]),
)


def _turn(query="My cow has fever", *, user_id=f"+91 {_MOBILE}", consent_turn=False, history=_PRIOR,
          source_lang="gu", target_lang="gu"):
    return Turn(
        query=query,
        session_id="call-1",
        source_lang=source_lang,
        target_lang=target_lang,
        user_id=user_id,
        authenticated_user=MappingProxyType({}),
        history=tuple(history),
        history_session_id="call-1",
        channel=WEB,
        persona="farmer",
        call=TelephonyCall(process_id="proc-1", outbound_consent_turn=consent_turn),
    )


async def _render(text_en, target_lang, *, execution):
    return f"<{target_lang}> {text_en}"


@pytest.fixture
def world(monkeypatch):
    """Stand-ins for everything outside voice's code, recording what they saw."""
    seen = {
        "envelope": _FOUND,
        "cached": None,
        "consent": None,
        "prefetched": None,
        "fetches": [],
        "consent_asked": [],
        "spawned": [],
        "stages": [],
        "refreshes": [],
        "history_writes": [],
        "prefetches": [],
        "scheme_summary": "",
        "trims": [],
    }

    async def _moderation(**_kw):
        return ModerationVerdict(category="in_scope", reason="ok")

    async def _streak(**_kw):
        return NonMeaningfulVerdict(five_consecutive_non_meaningful=False, reason="fine")

    async def _fetch(mobile):
        seen["fetches"].append(mobile)
        result = seen["envelope"]
        if isinstance(result, Exception):
            raise result
        if callable(result):
            return await result()
        return result

    async def _consent(**kwargs):
        seen["consent_asked"].append(kwargs)
        result = seen["consent"]
        if callable(result):
            return await result()
        return result

    async def _cached(mobile):
        return seen["cached"]

    async def _set_stage(session_id, stage):
        seen["stages"].append((session_id, stage))

    async def _prefetched(session_id):
        return seen["prefetched"]

    async def _refresh(mobile):
        seen["refreshes"].append(mobile)

    async def _history(key, messages):
        seen["history_writes"].append((key, list(messages)))

    async def _prefetch(session_id, accounts):
        seen["prefetches"].append((session_id, [a.farmer_code for a in accounts]))

    async def _schemes(unions):
        return seen["scheme_summary"]

    def _spawn(coro, *, label):
        seen["spawned"].append((coro, label))

    def _trim(history, **kwargs):
        # The real one counts tokens with tiktoken, which fetches its encoding.
        seen["trims"].append(kwargs)
        return list(history)

    monkeypatch.setattr(bg, "check_moderation", _moderation)
    monkeypatch.setattr(bg, "check_non_meaningful_streak", _streak)
    monkeypatch.setattr(bg, "get_or_fetch_farmer_data", _fetch)
    monkeypatch.setattr(bg, "classify_consent", _consent)
    monkeypatch.setattr(outbound, "spawn", _spawn)
    monkeypatch.setattr(outbound, "set_stage", _set_stage)
    monkeypatch.setattr(outbound, "get_prefetched_milk_summary", _prefetched)
    monkeypatch.setattr(outbound, "prefetch_milk_summary", _prefetch)
    monkeypatch.setattr(ai, "get_farmer_data_cached_only", _cached)
    monkeypatch.setattr(ai, "enqueue_farmer_refresh", _refresh)
    monkeypatch.setattr(ai, "update_message_history", _history)
    monkeypatch.setattr(ai, "_build_union_scheme_summary", _schemes)
    monkeypatch.setattr(ai, "trim_history", _trim)
    yield seen
    for coro, _label in seen["spawned"]:
        coro.close()


def _agent_input(turn, *, query=None, is_stale=None, keep_open=False):
    """Build the background (which starts its tasks), then voice's agent input."""

    async def _go():
        background = bg.VoiceBackground(turn, execution=_EXECUTION, render=_render)
        try:
            built = await ai._voice_agent_input(
                turn,
                Pretranslated(query=query or turn.query, lang="en"),
                execution=_EXECUTION,
                scheduler=None,
                translate_to=turn.target_lang,
                background=background,
                is_stale=is_stale,
                render=_render,
            )
            return built, background
        finally:
            if not keep_open:
                await background.close()

    return asyncio.run(_go())


def _content(message):
    return "".join(getattr(part, "content", "") for part in message.parts)


# ── the background set ──────────────────────────────────────────────────────


def test_the_farmer_fetch_starts_for_the_callers_mobile(world):
    built, background = _agent_input(_turn())

    assert world["fetches"] == [_MOBILE]
    assert background.mobile == _MOBILE


def test_there_is_no_fetch_without_a_mobile(world):
    built, background = _agent_input(_turn(user_id="anonymous"))

    assert world["fetches"] == []
    assert background.mobile is None
    assert asyncio.run(background.farmer_data()) is None


def test_the_consent_check_runs_only_on_a_consent_turn(world):
    world["consent"] = ConsentVerdict(intent="other", reason="asked a question")

    _agent_input(_turn())
    assert world["consent_asked"] == []

    _agent_input(_turn(query="હા બોલો", consent_turn=True))
    assert world["consent_asked"] == [dict(reply="હા બોલો", source_lang="gu", execution=_EXECUTION)]


def test_a_consent_turn_warms_the_milk_summary(world):
    world["consent"] = ConsentVerdict(intent="other", reason="asked a question")

    _agent_input(_turn(consent_turn=True))
    ((coro, label),) = world["spawned"]
    asyncio.run(coro)

    assert label == "outbound_milk_prefetch"
    assert world["prefetches"] == [("call-1", ["5058"])]


@pytest.mark.parametrize("turn", [
    _turn(consent_turn=False),
    _turn(consent_turn=True, user_id="anonymous"),
])
def test_nothing_is_warmed_off_a_consent_turn_or_without_a_mobile(world, turn):
    world["consent"] = ConsentVerdict(intent="other", reason="asked a question")

    _agent_input(turn)

    assert world["spawned"] == []


def test_closing_cancels_the_consent_check_but_lets_the_farmer_fetch_finish(world):
    release = asyncio.Event

    async def _run():
        never = release()

        async def _slow():
            await never.wait()

        world["envelope"] = _slow
        world["consent"] = _slow
        background = bg.VoiceBackground(_turn(consent_turn=True), execution=_EXECUTION, render=_render)
        await asyncio.sleep(0)
        await background.close()
        state = (background._consent_task.cancelled(), background._farmer_task.done())
        background._farmer_task.cancel()
        return state

    consent_cancelled, fetch_done = asyncio.run(_run())

    assert consent_cancelled
    assert not fetch_done, "voice leaves the farmer fetch to fill the cache"


# ── what the agent is run with ──────────────────────────────────────────────


def test_a_known_farmer_gets_the_signed_in_agent_and_their_details(world):
    built, background = _agent_input(_turn())

    assert isinstance(built, AgentInput)
    assert built.agent is voice_agents.voice_agent_signed_in
    assert built.usage_limits.request_limit == 6
    deps = built.deps
    assert deps.farmer_profile_status == "found"
    assert (deps.mobile, deps.signed_in, deps.process_id) == (_MOBILE, True, "proc-1")
    assert (deps.query, deps.lang_code, deps.target_lang) == ("My cow has fever", "en", "gu")
    assert [a.farmer_code for a in deps.farmer_accounts] == ["5058"]
    assert deps.farmer_unions == ["kaira"]
    assert (deps.farmer_village, deps.farmer_district) == ("Chikhodra", "Anand")
    assert "- Farmer name: Rameshbhai" in deps.farmer_info
    assert deps._moderation_task is background.moderation_task
    assert built.prompt == '**User:** "My cow has fever"'
    assert built.history == list(_PRIOR)
    context = _content(built.message_history[0])
    assert "- Today date:" in context
    assert "- Farmer-data tools may be available for this turn." in context
    assert "- Tool groups in this run: retrieval, booking, signed-in-farmer-data" in context
    assert built.message_history[1:3] == list(_PRIOR)
    assert "action-first symptom" in _content(built.message_history[-1])
    assert world["trims"] == [dict(max_tokens=32_000, include_system_prompts=False, include_tool_calls=True)]


def test_an_anonymous_caller_gets_the_base_agent(world):
    built, _ = _agent_input(_turn(user_id="anonymous"))

    assert built.agent is voice_agents.voice_agent
    assert built.usage_limits.request_limit == 4
    assert built.deps.farmer_profile_status == "anonymous"
    assert built.deps.mobile is None and built.deps.signed_in is False
    assert "no registered mobile number" in built.deps.farmer_info
    assert "- Tool groups in this run: retrieval" in _content(built.message_history[0])


def test_an_unresolved_fetch_is_read_again_from_the_cache(world):
    world["envelope"] = None
    world["cached"] = _FOUND

    built, _ = _agent_input(_turn())

    assert built.deps.farmer_profile_status == "found"


def test_a_fetch_that_never_resolves_leaves_the_identity_unavailable(world):
    world["envelope"] = None

    built, _ = _agent_input(_turn())

    assert built.deps.farmer_profile_status == "unavailable"
    assert "could not be loaded" in built.deps.farmer_info
    assert built.agent is voice_agents.voice_agent_signed_in


def test_a_failed_fetch_leaves_the_turn_unresolved(world):
    world["envelope"] = RuntimeError("farmer api down")

    built, _ = _agent_input(_turn())

    assert built.deps.farmer_profile_status == "unavailable"
    assert built.deps.farmer_accounts == []


def test_a_stale_record_is_queued_for_refresh(world):
    world["envelope"] = _FOUND.model_copy(update={"stale": True})
    _agent_input(_turn())
    assert world["refreshes"] == [_MOBILE]

    world["refreshes"].clear()
    world["envelope"] = _FOUND
    _agent_input(_turn())
    assert world["refreshes"] == []


def test_union_schemes_join_the_farmer_block(world):
    world["scheme_summary"] = "\n## Union schemes available"

    built, _ = _agent_input(_turn())

    assert built.deps.farmer_info.endswith("\n\n## Union schemes available")


def test_per_query_hints_come_right_before_the_prompt(world):
    built, _ = _agent_input(_turn(), query="What is the difference between cow and buffalo milk?")

    hints = _content(built.message_history[-1])
    assert hints.startswith("Hints for the current user query:")
    assert "Voice answer mode: compact comparison" in hints
    assert built.message_history[1:-1] == list(_PRIOR)


_ORPHANED = (
    *_PRIOR,
    ModelResponse(parts=[ToolCallPart(tool_name="search_documents", args={"query": "x"}, tool_call_id="lost")]),
    ModelRequest(parts=[UserPromptPart(content="and then?")]),
)


def test_orphaned_tool_calls_are_cleaned_and_saved(world):
    built, _ = _agent_input(_turn(history=_ORPHANED))

    cleaned = [m for m in _ORPHANED if not any(getattr(p, "tool_call_id", None) == "lost" for p in m.parts)]
    assert [_content(m) for m in built.history] == [_content(m) for m in cleaned]
    assert world["history_writes"] == [("call-1", built.history)]


def test_a_stale_turn_does_not_rewrite_history(world):
    asked = []

    async def _is_stale(reason):
        asked.append(reason)
        return "stale_request"

    built, _ = _agent_input(_turn(history=_ORPHANED), is_stale=_is_stale)

    assert asked == ["before_cleaned_history_write"]
    assert world["history_writes"] == []
    assert len(built.history) == len(_ORPHANED) - 1


def test_tool_calls_with_their_returns_are_kept(world):
    kept = (
        *_PRIOR,
        ModelResponse(parts=[ToolCallPart(tool_name="search_documents", args={}, tool_call_id="t1")]),
        ModelRequest(parts=[ToolReturnPart(tool_name="search_documents", content="doc", tool_call_id="t1")]),
    )

    built, _ = _agent_input(_turn(history=kept))

    assert world["history_writes"] == []
    assert built.message_history[1:1 + len(kept)] == list(kept)


def test_the_agent_runs_inside_voices_boundary_capture(world):
    built, _ = _agent_input(_turn())

    with built.observe() as handle:
        inside = model_boundary_capture._SESSION_ID_VAR.get(), model_boundary_capture._PROCESS_ID_VAR.get()

    assert handle is None
    assert inside == ("call-1", "proc-1")


# ── the outbound consent gate ───────────────────────────────────────────────


def test_a_no_says_the_farewell_and_hangs_up(world):
    world["consent"] = ConsentVerdict(intent="negative", reason="not now")

    built, _ = _agent_input(_turn(query="ના", consent_turn=True))

    assert built == ClassifierResult(
        canned_text=outbound.OUTBOUND_DECLINE_FAREWELL["gu"],
        label="outbound_declined",
        history_pair=built.history_pair,
        outcome="outbound_declined",
        raw_tail=" Goodbye.",
    )
    assert [_content(m) for m in built.history_pair] == [
        "ના", f"{outbound.OUTBOUND_DECLINE_FAREWELL['en']} Goodbye.",
    ]
    assert world["stages"] == [("call-1", outbound.STAGE_RESOLVED)]


def test_a_yes_reads_out_the_prefetched_milk_summary(world):
    world["consent"] = ConsentVerdict(intent="affirmative", reason="yes")
    world["prefetched"] = "10 litres, Rs 500"

    built, _ = _agent_input(_turn(query="હા", consent_turn=True), query="What is the difference?")

    last = _content(built.message_history[-1])
    assert last.startswith("Hints for the current user query:\n- Outbound call:")
    assert last.endswith("Milk deposit summary:\n10 litres, Rs 500")
    assert "compact comparison" in _content(built.message_history[-2]), "milk hint goes after the query hints"
    assert world["stages"] == [("call-1", outbound.STAGE_RESOLVED)]


def test_a_yes_without_a_prefetch_has_the_agent_fetch_the_pinned_window(world):
    world["consent"] = ConsentVerdict(intent="affirmative", reason="yes")
    fromdate, todate = outbound.milk_window(7)

    built, _ = _agent_input(_turn(query="હા", consent_turn=True))

    last = _content(built.message_history[-1])
    assert f"for fromdate {fromdate} to todate {todate}" in last


def test_a_yes_with_nothing_to_read_out_says_so(world):
    world["consent"] = ConsentVerdict(intent="affirmative", reason="yes")
    world["envelope"] = FarmerDataEnvelope.not_found()

    built, _ = _agent_input(_turn(query="હા", consent_turn=True))

    assert isinstance(built, ClassifierResult)
    assert (built.canned_text, built.label, built.outcome, built.raw_tail) == (
        outbound.OUTBOUND_NO_DATA["gu"], "outbound_no_data", "outbound_no_data", None,
    )
    assert [_content(m) for m in built.history_pair] == ["હા", outbound.OUTBOUND_NO_DATA["en"]]


def test_a_reply_that_is_neither_goes_to_the_agent(world):
    world["consent"] = ConsentVerdict(intent="other", reason="asked about fever")

    built, _ = _agent_input(_turn(consent_turn=True))

    assert isinstance(built, AgentInput)
    assert "Outbound call" not in _content(built.message_history[-1])
    assert world["stages"] == [("call-1", outbound.STAGE_RESOLVED)]


def test_an_ordinary_turn_leaves_the_outbound_stage_alone(world):
    _agent_input(_turn())

    assert world["stages"] == []


# ── the goodbye after an answer ─────────────────────────────────────────────


def _closing(state):
    return [ModelResponse(parts=[ToolCallPart(tool_name="signal_conversation_state", args={"state": state})])]


def test_an_answer_that_closes_the_call_is_followed_by_the_goodbye(world):
    built, _ = _agent_input(_turn())

    assert built.closing_line(_closing("conversation_closing")) == " Goodbye."
    assert built.closing_line(_closing("in_progress")) is None
    assert built.closing_line([ModelResponse(parts=[TextPart(content="conversation_closing")])]) is None


# ── through run_turn ────────────────────────────────────────────────────────


class _Run:
    def __init__(self, chunks, new_messages):
        self._chunks = chunks
        self.ctx = object()
        self.result = SimpleNamespace(new_messages=lambda: list(new_messages))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_e):
        return False

    async def __aiter__(self):
        yield _Node(self._chunks)


def _outcomes(seen):
    return [
        call.kwargs["value"]
        for call in seen["langfuse"].score_current_trace.call_args_list
        if call.kwargs.get("name") == "turn_outcome"
    ]


def _voice_surface():
    return dataclasses.replace(
        chat_service.CHAT_SURFACE,
        background=bg.voice_background(_render),
        agent_input=ai.voice_agent_input(_render),
    )


def _run_turn(turn, *, is_stale=None):
    async def _go():
        return [e async for e in chat_service.run_turn(
            turn, _voice_surface(), scheduler=SimpleNamespace(schedule=lambda *_a: None), is_stale=is_stale
        )]

    return asyncio.run(_go())


def _english(**kw):
    return _turn(source_lang="en", target_lang="en", **kw)


def test_run_turn_says_the_farewell_then_the_raw_goodbye(world, monkeypatch):
    seen = patch_turn(monkeypatch)
    world["consent"] = ConsentVerdict(intent="negative", reason="not now")

    emissions = _run_turn(_english(query="No", consent_turn=True))

    assert emissions == [
        TextEmission(outbound.OUTBOUND_DECLINE_FAREWELL["en"]),
        TextEmission(" Goodbye.", raw=True),
    ]
    assert _outcomes(seen) == ["outbound_declined"]
    ((key, messages),) = seen["history_writes"]
    assert [_content(m) for m in messages[-2:]] == ["No", f"{outbound.OUTBOUND_DECLINE_FAREWELL['en']} Goodbye."]


def test_run_turn_ends_the_call_after_an_answer_that_closes_it(world, monkeypatch):
    seen = patch_turn(monkeypatch)
    runs = []

    def _iter(**kw):
        runs.append(kw)
        return _Run(["Take care of her."], _closing("conversation_closing"))

    monkeypatch.setattr(voice_agents.voice_agent_signed_in, "iter", _iter)

    emissions = _run_turn(_english())

    assert emissions[-1] == TextEmission(" Goodbye.", raw=True)
    assert "".join(e.text for e in emissions[:-1]).strip() == "Take care of her."
    assert runs[0]["usage_limits"].request_limit == 6
    assert runs[0]["message_history"][0].parts[0].content.startswith("Runtime context for this turn:")
    ((key, messages),) = seen["history_writes"]
    assert messages[-1].parts[0].tool_name == "signal_conversation_state"
    assert _outcomes(seen) == ["success"]


def test_run_turn_says_no_goodbye_once_the_turn_is_stale(world, monkeypatch):
    seen = patch_turn(monkeypatch)
    monkeypatch.setattr(
        voice_agents.voice_agent_signed_in, "iter",
        lambda **_kw: _Run(["Take care of her."], _closing("conversation_closing")),
    )

    async def _is_stale(reason):
        return "stale_request" if reason == "before_goodbye" else None

    emissions = _run_turn(_english(), is_stale=_is_stale)

    assert TextEmission(" Goodbye.", raw=True) not in emissions
    assert seen["history_writes"] == []
    assert _outcomes(seen) == ["stale_request"]


def test_run_turn_says_nothing_extra_after_an_ordinary_answer(world, monkeypatch):
    patch_turn(monkeypatch)
    monkeypatch.setattr(
        voice_agents.voice_agent_signed_in, "iter",
        lambda **_kw: _Run(["Give her water."], []),
    )

    emissions = _run_turn(_english())

    assert all(not e.raw for e in emissions)


def test_run_turn_hands_the_agent_input_its_staleness_check(world, monkeypatch):
    patch_turn(monkeypatch)
    monkeypatch.setattr(
        voice_agents.voice_agent_signed_in, "iter",
        lambda **_kw: _Run(["Give her water."], []),
    )
    asked = []

    async def _is_stale(reason):
        asked.append(reason)
        return "stale_request" if reason == "before_cleaned_history_write" else None

    _run_turn(_english(history=_ORPHANED), is_stale=_is_stale)

    assert "before_cleaned_history_write" in asked
    assert world["history_writes"] == [], "a stale request must not rewrite history"


def test_a_voice_tool_call_wakes_the_nudge(monkeypatch):
    """Voice's tools and the nudge share one tool-call signal."""
    import time

    from agents.voice.tools import _with_nudge_signal
    from app.config import settings
    from app.voice import liveness as lv

    monkeypatch.setattr(settings, "nudge_timeout_seconds", 60)
    monkeypatch.setattr(settings, "enable_voice_nudges", True)
    sent = []

    class _Sender:
        async def send(self, emission):
            sent.append(emission)

    async def _tool():
        return "done"

    async def _run():
        liveness = lv.VoiceLiveness(_english(), started_at=time.monotonic(), send=_Sender(), is_stale=None)
        # pydantic-ai runs a tool in a child task.
        await asyncio.create_task(_with_nudge_signal(_tool)())
        await asyncio.sleep(0.05)
        await liveness.stop()

    asyncio.run(asyncio.wait_for(_run(), timeout=5))

    (emission,) = sent
    assert emission.text in lv._TOOL_NUDGE_MESSAGES["en"]
