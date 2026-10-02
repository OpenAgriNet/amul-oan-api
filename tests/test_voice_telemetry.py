"""Voice's telemetry on the seam: one agent_journey trace per call turn, stamped
voice.turn.v1, with what voice's trace recorded along the way.

Turns run through the real run_turn with voice's pieces; Langfuse, the farmer
fetch, the checks and the agent are stood in for.
"""
import asyncio
import contextlib
import dataclasses
import json
import os
import time
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import langfuse
import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from agents.voice import agent as voice_agents
from app import observability
from app.channels.chat import WEB
from app.config import settings
from app.services import chat as chat_service
from app.services.telemetry_stamps import chat_telemetry_release
from app.turn.types import TelephonyCall, Turn
from app.voice import agent_input as ai
from app.voice import background as bg
from app.voice import classifiers as vc
from app.voice import outbound
from app.voice import telemetry as voice_telemetry
from app.voice import trace as voice_trace
from app.voice.agent_input import voice_agent_input
from app.voice.background import VoiceBackground, voice_background
from app.voice.classifiers import _GREETING_RESPONSES, voice_classifiers
from app.voice.liveness import VoiceLiveness, nudge_stopped
from app.voice.moderation import ModerationVerdict
from app.voice.non_meaningful import NonMeaningfulVerdict
from app.voice.outbound_consent import ConsentVerdict
from app.voice.pretranslation import voice_pretranslation
from app.voice.sink import VoiceSink
from app.voice.telemetry import VoiceTelemetry
from tests.test_chat_turn_contract import patch_turn
from tests.test_voice_agent_input import _FOUND, _Run, _closing, _render
from tests.test_voice_pretranslation import _execution, _pretranslate
from tests.test_voice_pretranslation import _turn as _gujarati_turn
from tests.test_voice_sink import _English, _speak, _Translation

_MOBILE = "9876543210"


def _turn(query="My cow has fever", *, history=(), consent_turn=False, lang="en", user_id=f"+91 {_MOBILE}"):
    return Turn(
        query=query,
        session_id="call-1",
        source_lang=lang,
        target_lang=lang,
        user_id=user_id,
        authenticated_user=MappingProxyType({}),
        history=tuple(history),
        history_session_id="call-1",
        channel=WEB,
        persona="farmer",
        call=TelephonyCall(process_id="proc-1", outbound_consent_turn=consent_turn, provider="raya"),
    )


_PRIOR = (
    ModelRequest(parts=[UserPromptPart(content="earlier")]),
    ModelResponse(parts=[TextPart(content="answered")]),
)


def _answer(text):
    """What the agent streams, and the new messages its run leaves."""
    return _Run([text], [ModelRequest(parts=[UserPromptPart(content="q")]), ModelResponse(parts=[TextPart(content=text)])])


class _Observation:
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.updates = []
        self.ended = 0

    def update(self, **kwargs):
        self.updates.append(kwargs)

    def end(self):
        self.ended += 1


class _Langfuse:
    """Records what voice's trace hands Langfuse."""

    def __init__(self):
        self.observations = []
        self.propagated = []
        self.scores = []

    @contextlib.contextmanager
    def start_as_current_observation(self, **kwargs):
        observation = _Observation(kwargs)
        self.observations.append(observation)
        yield observation

    @contextlib.contextmanager
    def propagate_attributes(self, **kwargs):
        self.propagated.append(kwargs)
        yield

    def score_current_trace(self, **kwargs):
        self.scores.append(kwargs)

    @property
    def root(self):
        (root,) = [o for o in self.observations if o.kwargs["name"] == "agent_journey"]
        return root

    @property
    def summary(self):
        return self.root.updates[-1]["metadata"]


@pytest.fixture
def lf(monkeypatch):
    client = _Langfuse()
    monkeypatch.setattr(observability, "get_langfuse_client", lambda: client)
    monkeypatch.setattr(langfuse, "propagate_attributes", client.propagate_attributes)
    monkeypatch.setattr(voice_telemetry, "_get_langfuse_client", lambda: client)
    monkeypatch.setattr(settings, "voice_trace_text_mode", "full")
    monkeypatch.setattr(settings, "enable_voice_tracing", True)
    return client


@pytest.fixture
def call(monkeypatch):
    """Stand-ins for everything outside voice's code on a call turn."""
    seen = {
        "moderation": ModerationVerdict(category="in_scope", reason="ok"),
        "consent": None,
        "prefetched": None,
        "history_writes": [],
    }
    patch_turn(monkeypatch)

    async def _moderation(**_kw):
        return seen["moderation"]

    async def _streak(**_kw):
        return NonMeaningfulVerdict(five_consecutive_non_meaningful=False, reason="fine")

    async def _fetch(mobile):
        return _FOUND

    async def _consent(**_kw):
        return seen["consent"]

    async def _nothing(*_a, **_k):
        return None

    async def _prefetched(session_id):
        return seen["prefetched"]

    async def _history(key, messages):
        seen["history_writes"].append((key, list(messages)))

    async def _schemes(unions):
        return ""

    def _spawn(coro, *, label):
        coro.close()

    monkeypatch.setattr(bg, "check_moderation", _moderation)
    monkeypatch.setattr(bg, "check_non_meaningful_streak", _streak)
    monkeypatch.setattr(bg, "get_or_fetch_farmer_data", _fetch)
    monkeypatch.setattr(bg, "classify_consent", _consent)
    monkeypatch.setattr(outbound, "spawn", _spawn)
    monkeypatch.setattr(outbound, "set_stage", _nothing)
    monkeypatch.setattr(outbound, "get_prefetched_milk_summary", _prefetched)
    monkeypatch.setattr(ai, "get_farmer_data_cached_only", _nothing)
    monkeypatch.setattr(ai, "enqueue_farmer_refresh", _nothing)
    monkeypatch.setattr(ai, "update_message_history", _history)
    monkeypatch.setattr(ai, "_build_union_scheme_summary", _schemes)
    monkeypatch.setattr(ai, "trim_history", lambda history, **_kw: list(history))
    monkeypatch.setattr(chat_service, "update_message_history", _history)
    monkeypatch.setattr(voice_agents.voice_agent_signed_in, "iter", lambda **_kw: _answer("Give her water."))
    return seen


_VOICE = dataclasses.replace(
    chat_service.CHAT_SURFACE,
    classifiers=voice_classifiers(_render),
    sink=VoiceSink,
    telemetry=VoiceTelemetry,
    background=voice_background(_render),
    liveness=VoiceLiveness,
    pretranslation=voice_pretranslation(_render),
    agent_input=voice_agent_input(_render),
)


class _Sender:
    async def send(self, emission):
        pass


def _run(turn, *, is_stale=None, take=None):
    async def _go():
        out = []
        turn_stream = chat_service.run_turn(
            turn, _VOICE, scheduler=SimpleNamespace(schedule=lambda *_a: None),
            side_channel=_Sender(), is_stale=is_stale,
        )
        async for emission in turn_stream:
            out.append(emission)
            if take is not None and len(out) == take:
                await turn_stream.aclose()
                break
        return out

    return asyncio.run(asyncio.wait_for(_go(), timeout=10))


def _stage_names(summary):
    return [stage["name"] for stage in summary["stages"]]


# ── one call turn, one trace ────────────────────────────────────────────────


def test_an_answered_turn_is_one_agent_journey_stamped_voice_turn_v1(lf, call):
    _run(_turn(history=_PRIOR))

    root, summary = lf.root, lf.summary
    assert root.kwargs["name"] == "agent_journey" and root.ended == 1
    (propagated,) = lf.propagated
    assert propagated["trace_name"] == "agent_journey"
    assert propagated["session_id"] == "call-1" and propagated["user_id"] == f"+91 {_MOBILE}"
    assert propagated["tags"] == ["voice", "raya", f"pipeline_profile:{summary['pipeline_profile']}"]
    assert summary["amul.schema_version"] == "voice.turn.v1"
    assert summary["service"] == "amul-oan-api"
    assert summary["release"] == chat_telemetry_release()
    assert summary["outcome"] == "success" and "route" not in summary
    assert (summary["provider"], summary["process_id"], summary["call_type"]) == ("raya", "proc-1", "inbound")
    assert summary["request_model"] and summary["request_provider"]
    assert summary["response"]["text"] == "Give her water."
    assert summary["agent"]["signed_in"] is True
    assert summary["agent"]["output_chars"] == len("Give her water.")
    assert (summary["agent"]["new_message_count"], summary["agent"]["tool_call_count"]) == (2, 0)
    assert summary["pretranslation"]["provider"] == "none"
    assert {"ttft_ms", "ttfr_ms", "first_agent_text_ms"} <= set(summary["timings_ms"])
    assert {"moderation", "non_meaningful", "farmer_context", "scheme_summary", "agent"} <= set(_stage_names(summary))
    (agent,) = [stage for stage in summary["stages"] if stage["name"] == "agent"]
    assert (agent["signed_in"], agent["request_limit"]) == (True, 6)
    assert "pipeline_flags" in summary
    assert summary["farmer_context"]["unions"] == ["kaira"]
    assert summary["nudge"] == {"armed": True, "sent": False, "cancel_reason": "first_text_chunk_received"}
    assert [score["name"] for score in lf.scores] == ["pipeline_profile"]
    assert lf.scores[0]["score_id"] == "voice-variant-call-1"


def test_an_answered_turn_sends_only_what_the_contract_lists(lf, call, monkeypatch):
    monkeypatch.setattr(settings, "voice_trace_text_mode", "preview_hash")
    contract = json.loads(
        (Path(__file__).resolve().parents[1] / "telemetry" / "contracts" / "voice.turn.v1.json").read_text()
    )

    _run(_turn(history=_PRIOR))

    summary = lf.summary
    # pipeline_flags and pc_<step> come from llm_core and are left out of the contract.
    sent = {key for key in summary if key != "pipeline_flags" and not key.startswith("pc_")}
    assert sent <= set(contract["metadata_keys"])
    for block, keys in contract["nested_keys"].items():
        if block != "error":
            assert set(keys) <= set(summary[block]), block


@pytest.mark.parametrize("query, path, stage", [
    ("hello", "greeting_fast_path", "greeting_fast_path"),
    ("What is your name?", "identity_fast_path", "identity_fast_path"),
    ("hm", "fragment_fast_path", "fragment_fast_path"),
    ("No audio/User is speaking softly", "stt_signal", "stt_signal_response"),
    ("please remain on the line", "hold_message", None),
])
def test_a_short_circuit_records_its_path_as_route_and_outcome(lf, call, monkeypatch, query, path, stage):
    async def _please_repeat(**_kw):
        return "Please say that again."

    monkeypatch.setattr(vc, "generate_stt_signal_response", _please_repeat)

    emissions = _run(_turn(query))

    summary = lf.summary
    assert (summary["route"], summary["outcome"]) == (path, path)
    if stage is not None:
        assert stage in _stage_names(summary)
    assert summary["response"]["text"] == "".join(e.text for e in emissions)
    assert "ttft_ms" in summary["timings_ms"]


def test_a_greeting_is_recorded_as_the_caller_hears_it(lf, call):
    _run(_turn("hello"))

    assert lf.summary["response"]["text"] == _GREETING_RESPONSES["en"]


def test_a_rejected_query_is_recorded_as_declined(lf, call):
    call["moderation"] = ModerationVerdict(category="irrelevant", reason="off topic")

    _run(_turn("Who won the match?", history=_PRIOR))

    summary = lf.summary
    assert (summary["route"], summary["outcome"]) == ("moderation_rejected", "moderation_rejected")
    assert summary["moderation"]["rejected"] is True
    assert summary["nudge"]["cancel_reason"] == "moderation_rejected"


def test_the_outbound_decline_records_the_farewell_and_the_goodbye(lf, call):
    call["consent"] = ConsentVerdict(intent="negative", reason="not now")

    _run(_turn("No", history=_PRIOR, consent_turn=True))

    summary = lf.summary
    assert (summary["route"], summary["outcome"]) == ("outbound_declined", "outbound_declined")
    assert summary["response"]["text"] == f"{outbound.OUTBOUND_DECLINE_FAREWELL['en']} Goodbye."
    assert summary["outbound_consent_intent"] == "negative"
    assert "outbound_consent" in _stage_names(summary)
    assert summary["nudge"]["cancel_reason"] == "outbound_declined"


@pytest.mark.parametrize("prefetched, route", [
    ("Total 40 litres.", "outbound_milk_readout"),
    (None, "outbound_milk_readout_cold"),
])
def test_a_milk_readout_is_routed_as_one(lf, call, prefetched, route):
    call["consent"] = ConsentVerdict(intent="affirmative", reason="yes")
    call["prefetched"] = prefetched

    _run(_turn("Yes", history=_PRIOR, consent_turn=True))

    assert (lf.summary["route"], lf.summary["outcome"]) == (route, "success")


def test_an_anonymous_caller_is_recorded_as_not_signed_in(lf, call, monkeypatch):
    monkeypatch.setattr(voice_agents.voice_agent, "iter", lambda **_kw: _answer("Give her water."))

    _run(_turn(history=_PRIOR, user_id="anonymous"))

    summary = lf.summary
    assert summary["agent"]["signed_in"] is False
    (agent,) = [stage for stage in summary["stages"] if stage["name"] == "agent"]
    assert agent["request_limit"] == 4
    assert "farmer_context" not in _stage_names(summary)


def test_the_goodbye_after_a_closing_answer_is_part_of_the_answer(lf, call, monkeypatch):
    monkeypatch.setattr(
        voice_agents.voice_agent_signed_in, "iter",
        lambda **_kw: _Run(["Take care of her."], _closing("conversation_closing")),
    )

    _run(_turn(history=_PRIOR))

    summary = lf.summary
    assert summary["response"]["text"] == "Take care of her. Goodbye."
    assert "route" not in summary and summary["outcome"] == "success"
    assert summary["agent"]["tool_call_count"] == 1


def test_a_caller_who_hangs_up_mid_answer_is_a_disconnect(lf, call, monkeypatch):
    monkeypatch.setattr(
        voice_agents.voice_agent_signed_in, "iter",
        lambda **_kw: _answer("Give her water. Keep her in the shade."),
    )

    _run(_turn(history=_PRIOR), take=1)

    assert lf.summary["outcome"] == "client_disconnected"
    assert lf.root.ended == 1


def test_a_stale_turn_is_recorded_as_stale(lf, call):
    async def _is_stale(reason):
        return "stale_request" if reason == "before_history_write" else None

    _run(_turn(history=_PRIOR), is_stale=_is_stale)

    assert lf.summary["outcome"] == "stale_request"


def test_a_failing_turn_records_the_error_and_ends_the_root_once(lf, call, monkeypatch):
    async def _down(key, messages):
        raise RuntimeError("redis down")

    monkeypatch.setattr(chat_service, "update_message_history", _down)

    with pytest.raises(RuntimeError):
        _run(_turn(history=_PRIOR))

    summary = lf.summary
    assert summary["outcome"] == "error"
    assert summary["error"] == {"type": "RuntimeError", "message": "redis down"}
    assert lf.root.updates[-1]["level"] == "ERROR"
    assert lf.root.ended == 1


def test_the_request_is_recorded_as_voice_records_it(lf):
    turn = dataclasses.replace(
        _turn(), source_lang=" GU ", target_lang="Gu", call=TelephonyCall(process_id="proc-1", call_type="Outbound")
    )
    telemetry = VoiceTelemetry(turn, pipeline_profile="oss", pipeline_trace=_execution().begin_trace())
    with telemetry.root():
        pass

    summary = lf.summary
    assert summary["call_type"] == "outbound"
    assert (summary["source_lang"], summary["target_lang"]) == ("gu", "gu")
    assert summary["provider"] is None
    # The agent's primary tier, as voice's request_model and request_provider.
    assert (summary["request_model"], summary["request_provider"]) == ("gemma", "vllm")
    assert summary["pc_agent"].startswith("vllm:gemma@")


def test_the_turns_trace_is_current_only_inside_its_root(lf):
    telemetry = VoiceTelemetry(_turn(), pipeline_profile="oss", pipeline_trace=None)
    outside = voice_trace.current_trace()
    with telemetry.root():
        inside = voice_trace.current_trace()
    after = voice_trace.current_trace()

    assert inside is telemetry._trace
    assert not outside.enabled and not after.enabled
    assert after is not inside


def test_a_voice_turn_sends_no_served_tier_and_a_chat_turn_still_does(lf, call, monkeypatch):
    monkeypatch.setattr(chat_service._pipeline_trace, "served_summary", lambda _pt: "agent=vllm:gemma[0]")

    _run(_turn(history=_PRIOR))
    assert [score["name"] for score in lf.scores] == ["pipeline_profile"]

    seen = patch_turn(monkeypatch)
    chat_turn = dataclasses.replace(_turn(history=_PRIOR), call=None, user_id="anonymous")
    asyncio.run(_collect(chat_service.run_turn(
        chat_turn, chat_service.CHAT_SURFACE, scheduler=SimpleNamespace(schedule=lambda *_a: None)
    )))
    served = [
        c.kwargs["value"] for c in seen["langfuse"].score_current_trace.call_args_list
        if c.kwargs.get("name") == "served_tier"
    ]
    assert served == ["agent=vllm:gemma[0]"]


async def _collect(agen):
    return [item async for item in agen]


# ── what each voice module records ──────────────────────────────────────────


@pytest.fixture
def trace(monkeypatch):
    """A current trace to record into, as a voice turn's root sets one."""
    monkeypatch.setattr(settings, "voice_trace_text_mode", "full")
    current = voice_trace.VoiceTrace(
        session_id="call-1", user_id="u", source_lang="gu", target_lang="gu", query="q", enabled=False
    )
    client = _Langfuse()
    current.enabled, current.langfuse_client = True, client
    current.sent = client
    token = voice_trace._current.set(current)
    yield current
    voice_trace._current.reset(token)


def test_outside_a_turn_the_current_trace_is_never_sent():
    first, second = voice_trace.current_trace(), voice_trace.current_trace()

    assert not first.enabled and first.langfuse_client is None
    assert first is not second, "records outside a turn must not pile up"


def _checks(monkeypatch, moderation, streak):
    async def _fetch(mobile):
        return None

    monkeypatch.setattr(bg, "check_moderation", moderation)
    monkeypatch.setattr(bg, "check_non_meaningful_streak", streak)
    monkeypatch.setattr(bg, "get_or_fetch_farmer_data", _fetch)


async def _streak_fine(**_kw):
    return NonMeaningfulVerdict(five_consecutive_non_meaningful=False, reason="fine")


def test_moderation_latency_is_the_checks_own_not_the_wait_for_it(trace, monkeypatch):
    async def _moderation(**_kw):
        return ModerationVerdict(category="in_scope", reason="ok")

    _checks(monkeypatch, _moderation, _streak_fine)

    async def _go():
        background = VoiceBackground(_turn(history=_PRIOR), execution=object(), render=_render)
        await asyncio.sleep(0.3)
        await background.gate()
        await background.close()

    asyncio.run(_go())

    (moderation,) = [s for s in trace.stages if s["name"] == "moderation"]
    assert moderation["duration_ms"] < 200
    assert trace.metadata["moderation"]["available"] is True
    assert trace.metadata["moderation"]["category"] == "in_scope"
    (observation,) = [o for o in trace.sent.observations if o.kwargs["name"] == "moderation"]
    assert observation.updates[-1]["output"]["category"] == "in_scope"
    (streak,) = [s for s in trace.stages if s["name"] == "non_meaningful"]
    assert streak["turn_count"] == 2
    # Fewer than five turns: voice does not gate on the classifier.
    assert trace.metadata["non_meaningful"]["gate_skipped"] is True


def test_a_streak_check_still_running_when_skipped_is_timed_as_skipped(trace, monkeypatch):
    async def _moderation(**_kw):
        return ModerationVerdict(category="in_scope", reason="ok")

    async def _slow_streak(**_kw):
        await asyncio.sleep(5)

    _checks(monkeypatch, _moderation, _slow_streak)

    async def _go():
        background = VoiceBackground(_turn(history=_PRIOR), execution=object(), render=_render)
        await background.gate()
        await background.close()

    asyncio.run(_go())

    # Voice records the skip, then the check as it records every resolution.
    streaks = [s for s in trace.stages if s["name"] == "non_meaningful"]
    assert [s.get("gate_skipped") for s in streaks] == [True, None]


def test_a_moderation_check_that_raises_is_recorded_as_an_error(trace, monkeypatch):
    async def _moderation(**_kw):
        raise RuntimeError("moderation blew up")

    _checks(monkeypatch, _moderation, _streak_fine)

    async def _go():
        background = VoiceBackground(_turn(history=_PRIOR), execution=object(), render=_render)
        await background.gate()
        await background.close()

    asyncio.run(_go())

    (moderation,) = [s for s in trace.stages if s["name"] == "moderation"]
    assert moderation["status"] == "error"
    assert trace.metadata["moderation"] == {**trace.metadata["moderation"], "available": False}
    (observation,) = [o for o in trace.sent.observations if o.kwargs["name"] == "moderation"]
    assert observation.updates[-1]["level"] == "ERROR"


def test_pretranslation_records_each_attempt_and_the_fallback(trace, monkeypatch):
    _pretranslate(monkeypatch, _gujarati_turn("ભેંસ"), replies=[RuntimeError("oss down"), "For a buffalo"])

    recorded = trace.metadata["pretranslation"]
    assert (recorded["provider"], recorded["fallback_used"]) == ("vllm", True)
    assert [a["status"] for a in recorded["attempts"]] == ["error", "ok"]
    assert recorded["attempts"][0]["error_class"] == "RuntimeError"
    assert (recorded["actual_tier"], recorded["actual_provider"]) == ("managed", "openai")
    assert recorded["text"]["text"] == "For a buffalo"
    assert [s["name"] for s in trace.stages] == ["pretranslation"]


def test_pretranslation_that_fails_everywhere_is_recorded_as_failed(trace, monkeypatch):
    _pretranslate(monkeypatch, _gujarati_turn("ભેંસ"), replies=[RuntimeError("down"), RuntimeError("down")])

    recorded = trace.metadata["pretranslation"]
    assert (recorded["provider"], recorded["actual_tier"]) == ("failed", "failed")
    assert [s["status"] for s in trace.stages] == ["error"]


def test_the_sink_records_translation_marks_and_why_the_nudge_stopped(trace, monkeypatch):
    trace.set_nudge(armed=True, sent=False)
    _Translation(monkeypatch, replies=[["ગાયને પાણી આપો."]])

    out, _sink = _speak(_English(["Give the cow water."]))

    assert out
    assert trace.response_parts == out
    assert {"first_agent_text_ms", "first_translation_chunk_ms", "ttft_ms", "ttfr_ms"} <= set(trace.timings_ms)
    assert [s["name"] for s in trace.stages] == ["output_translation"]
    assert trace.metadata["nudge"]["cancel_reason"] == "tail_translated_fragment"


def test_a_failed_output_translation_is_counted(trace, monkeypatch):
    _Translation(monkeypatch, fail_on={1})

    _speak(_English(["Give the cow water."]))

    assert trace.counters["output_translation_errors"] == 1


def test_a_stream_nothing_was_heard_from_stops_the_nudge_as_stream_ended(trace):
    trace.set_nudge(armed=True, sent=False)

    _speak(_English([]), target_lang="en")

    assert trace.metadata["nudge"]["cancel_reason"] == "stream_ended"


def test_nudge_stopped_is_only_for_a_nudge_still_waiting(trace):
    nudge_stopped("moderation_rejected")
    assert "nudge" not in trace.metadata

    trace.set_nudge(armed=True, sent=True)
    nudge_stopped("moderation_rejected")
    assert "cancel_reason" not in trace.metadata["nudge"]

    trace.set_nudge(sent=False)
    nudge_stopped("moderation_rejected")
    nudge_stopped("stream_ended")
    assert trace.metadata["nudge"]["cancel_reason"] == "moderation_rejected"


def test_the_nudge_records_when_and_why_it_was_sent(trace, monkeypatch):
    monkeypatch.setattr(settings, "enable_voice_nudges", True)
    monkeypatch.setattr(settings, "nudge_timeout_seconds", 0)

    async def _go():
        liveness = VoiceLiveness(_turn(), started_at=time.monotonic(), send=_Sender(), is_stale=None)
        await asyncio.sleep(0.05)
        await liveness.stop()

    asyncio.run(_go())

    nudge = trace.metadata["nudge"]
    assert (nudge["armed"], nudge["sent"], nudge["trigger"]) == (True, True, "timeout")
    assert nudge["sent_ms"] >= 0


def test_nudges_switched_off_are_recorded_as_not_armed(trace, monkeypatch):
    monkeypatch.setattr(settings, "enable_voice_nudges", False)

    async def _go():
        await VoiceLiveness(_turn(), started_at=time.monotonic(), send=_Sender(), is_stale=None).stop()

    asyncio.run(_go())

    assert trace.metadata["nudge"] == {"armed": False, "sent": False}
