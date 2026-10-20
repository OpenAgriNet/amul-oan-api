"""Voice's telemetry on the seam: one turn.v1 trace per call turn (a root named
turn, flat amul.* keys, the steps under it), and what voice's trace recorded
along the way for its summary log.

Turns run through the real run_turn with voice's pieces; Langfuse, the farmer
fetch, the checks and the agent are stood in for.
"""
import asyncio
import contextlib
import dataclasses
import json
import os
import time
from types import MappingProxyType, SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import langfuse
import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from agents.voice import agent as voice_agents
from app import observability
from app.channels.chat import WEB
from app.config import settings
from app.llm_core import Step
from app.llm_core import runtime as llm_runtime
from app.llm_core.config_model import StepClientKind
from app.llm_core.execution import ExecutionContext, ExecutionTarget
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
from tests.test_chat_turn_sequence import _Node
from tests.test_voice_agent_input import _FOUND, _Run, _closing, _render
from tests.test_voice_pretranslation import _MANAGED, _OSS, _execution, _pretranslate
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
    def __init__(self, client, kwargs, parent=None):
        self.client = client
        self.kwargs = kwargs
        self.parent = parent
        self.updates = []
        self.ended = 0

    def update(self, **kwargs):
        self.updates.append(kwargs)

    def end(self):
        self.ended += 1

    def start_observation(self, **kwargs):
        """A child of this observation, as the SDK opens one off a span."""
        child = _Observation(self.client, kwargs, parent=self.kwargs["name"])
        self.client.observations.append(child)
        return child

    @property
    def sent(self):
        """What the observation was opened with and told since, last word winning."""
        merged = {key: value for key, value in self.kwargs.items() if value is not None}
        for update in self.updates:
            merged.update({key: value for key, value in update.items() if value is not None})
        return merged


class _Langfuse:
    """Records what voice's trace hands Langfuse, and the traces voice built."""

    def __init__(self):
        self.observations = []
        self.propagated = []
        self.scores = []
        self.io = {}
        self.traces = []

    @contextlib.contextmanager
    def start_as_current_observation(self, **kwargs):
        observation = _Observation(self, kwargs)
        self.observations.append(observation)
        yield observation

    @contextlib.contextmanager
    def propagate_attributes(self, **kwargs):
        self.propagated.append(kwargs)
        yield

    def score_current_trace(self, **kwargs):
        self.scores.append(kwargs)

    def set_current_trace_io(self, **kwargs):
        self.io.update({key: value for key, value in kwargs.items() if value is not None})

    @property
    def root(self):
        (root,) = [o for o in self.observations if o.kwargs["name"] == "turn"]
        return root

    @property
    def names(self):
        return [o.kwargs["name"] for o in self.observations]

    def steps(self, name):
        return [o for o in self.observations if o.kwargs["name"] == name]

    def step(self, name):
        (step,) = self.steps(name)
        return step

    @property
    def opened(self):
        """The trace-level fields set on the root before any step."""
        return self.propagated[1]

    @property
    def closed(self):
        """The amul.* keys set on the root when the turn ended."""
        return self.propagated[2]["metadata"]

    @property
    def trace(self):
        """The VoiceTrace of the one turn run: voice's record, for its summary log."""
        (trace,) = self.traces
        return trace


@pytest.fixture
def lf(monkeypatch):
    client = _Langfuse()
    create_voice_trace = voice_telemetry.create_voice_trace

    def _create(**kwargs):
        trace = create_voice_trace(**kwargs)
        client.traces.append(trace)
        return trace

    monkeypatch.setattr(voice_telemetry, "create_voice_trace", _create)
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
    # Voice has no prefetch: its background set fetches the farmer data.
    prefetch=None,
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


def _stage_names(trace):
    return [stage["name"] for stage in trace.stages]


# ── one call turn, one trace ────────────────────────────────────────────────


def test_an_answered_turn_is_one_turn_v1_root_with_its_steps_under_it(lf, call):
    _run(_turn(history=_PRIOR))

    root, record = lf.root, lf.trace
    assert (root.kwargs["name"], root.kwargs["as_type"], root.ended) == ("turn", "span", 1)
    # Three propagations: voice's environment for every span; the trace-level fields
    # on the root before any step; the end of the turn on the root as it closes.
    environment, opened, closed = lf.propagated
    assert set(environment) == {"environment"}
    assert opened["environment"] == environment["environment"]
    assert opened["session_id"] == "call-1" and opened["user_id"] == record.metadata["user_id_hash"]
    assert "tags" not in opened and "trace_name" not in opened
    assert opened["metadata"] == root.kwargs["metadata"] == {
        "amul.schema_version": "turn.v1",
        "amul.kind": "turn",
        "amul.service": "amul-oan-api",
        "amul.release": chat_telemetry_release(),
        "amul.pipeline": "telephony",
        "amul.direction": "inbound",
        "amul.provider": "raya",
        "amul.source_lang": "en",
        "amul.target_lang": "en",
        "amul.translation_mode": "direct",
        "amul.pipeline_profile": record.metadata["pipeline_profile"],
        "amul.pipeline_flags": record.metadata["pipeline_flags"],
        "amul.persona": "farmer",
        "amul.process_id": "proc-1",
    }
    assert closed["metadata"]["amul.outcome"] == "success"
    assert closed["metadata"]["amul.served_tier"] == lf.step("agent").sent["metadata"]["amul.served_tier"]
    assert closed["metadata"]["amul.ttft_ms"].isdigit() and closed["metadata"]["amul.ttfr_ms"].isdigit()
    assert set(closed["metadata"]) == {"amul.outcome", "amul.served_tier", "amul.ttft_ms", "amul.ttfr_ms"}
    # The words, in this fixture's full text mode: the question and the answer as heard.
    assert lf.io == {"input": "My cow has fever", "output": "Give her water."}
    assert "input" not in root.kwargs and root.updates == []
    assert set(lf.names) == {"turn", "moderation", "non_meaningful", "farmer_context", "scheme_summary", "agent", "nudge"}
    # The checks resolve during the agent step; their steps still sit under the root.
    assert {lf.step(name).parent for name in ("moderation", "non_meaningful", "nudge")} == {"turn"}
    assert lf.step("farmer_context").sent["output"] == {
        "source": "api", "stale": False, "unions": ["kaira"],
        "farmer_info_chars": record.metadata["farmer_context"]["farmer_info_chars"],
        "technician_info_chars": record.metadata["farmer_context"]["technician_info_chars"],
    }
    agent = lf.step("agent").sent
    assert (agent["as_type"], agent["input"], agent["output"]) == ("agent", "My cow has fever", "Give her water.")
    assert set(agent["metadata"]) == {"amul.configured_tier", "amul.served_tier", "amul.attempts"}
    assert agent["metadata"]["amul.attempts"] == "1" and agent["metadata"]["amul.served_tier"].endswith("[0]")
    assert lf.step("nudge").sent["output"] == {"armed": True, "sent": False, "cancel_reason": "first_text_chunk_received"}
    (score,) = lf.scores
    assert (score["name"], score["value"], score["data_type"]) == ("turn_outcome", "success", "CATEGORICAL")
    # Voice's record, for its summary log, as voice kept it.
    assert record.metadata["amul.schema_version"] == "turn.v1" and record.metadata["amul.service"] == "amul-oan-api"
    assert record.metadata["outcome"] == "success" and "route" not in record.metadata
    assert (record.metadata["provider"], record.metadata["process_id"], record.metadata["call_type"]) == ("raya", "proc-1", "inbound")
    assert record.metadata["request_model"] and record.metadata["request_provider"]
    assert record.metadata["agent"]["signed_in"] is True
    assert record.metadata["agent"]["output_chars"] == len("Give her water.")
    assert (record.metadata["agent"]["new_message_count"], record.metadata["agent"]["tool_call_count"]) == (2, 0)
    assert record.metadata["pretranslation"]["provider"] == "none"
    assert {"ttft_ms", "ttfr_ms", "first_agent_text_ms"} <= set(record.timings_ms)
    assert {"moderation", "non_meaningful", "farmer_context", "scheme_summary", "agent"} <= set(_stage_names(record))
    (stage,) = [stage for stage in record.stages if stage["name"] == "agent"]
    assert (stage["signed_in"], stage["request_limit"]) == (True, 6)
    assert "pipeline_flags" in record.metadata
    assert record.metadata["nudge"] == {"armed": True, "sent": False, "cancel_reason": "first_text_chunk_received"}


def test_a_turn_with_a_request_epoch_sends_it(lf, call):
    turn = _turn(history=_PRIOR)
    _run(dataclasses.replace(turn, call=dataclasses.replace(turn.call, request_epoch=7)))

    assert lf.root.kwargs["metadata"]["amul.request_epoch"] == "7"


def _everything_sent(client):
    return json.dumps(
        [[o.kwargs, o.updates] for o in client.observations] + client.propagated + [client.io],
        ensure_ascii=False, default=str,
    )


def test_by_default_a_turn_sends_langfuse_no_phone_and_no_words(lf, call, monkeypatch):
    default = type(settings).model_fields["voice_trace_text_mode"].default
    monkeypatch.setattr(settings, "voice_trace_text_mode", default)

    _run(_turn(history=_PRIOR))

    sent = _everything_sent(lf)
    for private in (_MOBILE, "fever", "water"):
        assert private not in sent
    assert lf.io == {}
    assert "input" not in lf.step("agent").sent and "output" not in lf.step("agent").sent
    assert "amul.farmer_id" not in lf.root.kwargs["metadata"]
    # The record keeps the hash, for the summary log.
    assert lf.trace.metadata["query"].keys() == {"chars", "sha256"}


def test_the_preview_hash_mode_sends_langfuse_no_words_either(lf, call, monkeypatch):
    # turn.v1 has no place for a preview: the trace carries the words or nothing.
    monkeypatch.setattr(settings, "voice_trace_text_mode", "preview_hash")

    _run(_turn(history=_PRIOR))

    for private in ("fever", "water"):
        assert private not in _everything_sent(lf)
    assert lf.io == {}
    assert lf.trace.metadata["query"]["preview"] == "My cow has fever"


def test_the_phone_user_id_mode_sends_langfuse_the_callers_phone(lf, call, monkeypatch):
    monkeypatch.setattr(settings, "voice_trace_user_id_mode", "phone")

    _run(_turn(history=_PRIOR))

    assert lf.opened["user_id"] == f"+91 {_MOBILE}"
    # A signed-in caller's number is the farmer id; only the record keeps the hash.
    assert lf.root.kwargs["metadata"]["amul.farmer_id"] == _MOBILE
    assert _MOBILE not in json.dumps(lf.trace.metadata, ensure_ascii=False, default=str)


@pytest.mark.parametrize("mode", sorted(voice_trace._VALID_TEXT_MODES))
def test_the_summary_log_holds_no_words_whatever_the_text_mode(lf, call, caplog, monkeypatch, mode):
    monkeypatch.setattr(settings, "voice_trace_text_mode", mode)
    caplog.set_level("INFO", logger=voice_trace.logger.name)

    _run(_turn(history=_PRIOR))

    (logged,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("VOICE_TRACE_SUMMARY")]
    for private in (_MOBILE, "fever", "water"):
        assert private not in logged
    assert json.loads(logged.split(" ", 1)[1])["query"].keys() == {"chars", "sha256"}


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

    record = lf.trace
    # The path is the outcome; voice's record keeps it as the route too.
    assert (record.metadata["route"], record.metadata["outcome"]) == (path, path)
    assert lf.closed["amul.outcome"] == path
    (score,) = lf.scores
    assert score["value"] == path
    if stage is not None:
        assert stage in _stage_names(record) and stage in lf.names
    assert lf.io["output"] == "".join(e.text for e in emissions)
    assert "ttft_ms" in record.timings_ms and lf.closed["amul.ttft_ms"].isdigit()


def test_a_greeting_is_recorded_as_the_caller_hears_it(lf, call):
    _run(_turn("hello"))

    assert lf.io["output"] == _GREETING_RESPONSES["en"]
    assert lf.step("greeting_fast_path").parent is None


def test_a_rejected_query_is_recorded_as_declined(lf, call):
    call["moderation"] = ModerationVerdict(category="irrelevant", reason="off topic")

    _run(_turn("Who won the match?", history=_PRIOR))

    record = lf.trace
    assert (record.metadata["route"], record.metadata["outcome"]) == ("moderation_rejected", "moderation_rejected")
    assert lf.closed["amul.outcome"] == "moderation_rejected"
    moderation = lf.step("moderation").sent
    assert moderation["output"] == {"category": "irrelevant", "reason": "off topic", "rejected": True}
    assert moderation["input"] == "Who won the match?"
    assert moderation["metadata"]["amul.check_mode"] == "model"
    assert set(moderation["metadata"]) <= {"amul.check_mode", "amul.configured_tier", "amul.served_tier", "amul.attempts"}
    assert record.metadata["moderation"]["rejected"] is True
    assert record.metadata["nudge"]["cancel_reason"] == "moderation_rejected"
    assert lf.step("nudge").sent["output"] == {"armed": True, "sent": False, "cancel_reason": "moderation_rejected"}


def test_the_outbound_decline_records_the_farewell_and_the_goodbye(lf, call):
    call["consent"] = ConsentVerdict(intent="negative", reason="not now")

    _run(_turn("No", history=_PRIOR, consent_turn=True))

    record = lf.trace
    assert (record.metadata["route"], record.metadata["outcome"]) == ("outbound_declined", "outbound_declined")
    assert lf.io["output"] == f"{outbound.OUTBOUND_DECLINE_FAREWELL['en']} Goodbye."
    consent = lf.step("outbound_consent").sent
    assert (consent["as_type"], consent["output"]) == ("generation", {"intent": "negative"})
    assert consent["metadata"]["amul.check_mode"] == "model"
    assert record.metadata["outbound_consent_intent"] == "negative"
    assert "outbound_consent" in _stage_names(record)
    assert record.metadata["nudge"]["cancel_reason"] == "outbound_declined"


@pytest.mark.parametrize("prefetched, route, source", [
    ("Total 40 litres.", "outbound_milk_readout", "cache"),
    (None, "outbound_milk_readout_cold", "cold"),
])
def test_a_milk_readout_is_routed_as_one(lf, call, prefetched, route, source):
    call["consent"] = ConsentVerdict(intent="affirmative", reason="yes")
    call["prefetched"] = prefetched

    _run(_turn("Yes", history=_PRIOR, consent_turn=True))

    assert (lf.trace.metadata["route"], lf.trace.metadata["outcome"]) == (route, "success")
    # On the trace the readout is a step, and the outcome stays success.
    assert lf.step("milk_readout").sent["output"] == {"source": source}
    assert lf.closed["amul.outcome"] == "success"


def test_an_anonymous_caller_is_recorded_as_not_signed_in(lf, call, monkeypatch):
    monkeypatch.setattr(voice_agents.voice_agent, "iter", lambda **_kw: _answer("Give her water."))
    monkeypatch.setattr(settings, "voice_trace_user_id_mode", "phone")

    _run(_turn(history=_PRIOR, user_id="anonymous"))

    record = lf.trace
    assert record.metadata["agent"]["signed_in"] is False
    (agent,) = [stage for stage in record.stages if stage["name"] == "agent"]
    assert agent["request_limit"] == 4
    assert "farmer_context" not in _stage_names(record) and "farmer_context" not in lf.names
    assert "amul.farmer_id" not in lf.root.kwargs["metadata"]


def test_the_goodbye_after_a_closing_answer_is_part_of_the_answer(lf, call, monkeypatch):
    monkeypatch.setattr(
        voice_agents.voice_agent_signed_in, "iter",
        lambda **_kw: _Run(["Take care of her."], _closing("conversation_closing")),
    )

    _run(_turn(history=_PRIOR))

    record = lf.trace
    assert lf.io["output"] == "Take care of her. Goodbye."
    assert "route" not in record.metadata and lf.closed["amul.outcome"] == "success"
    assert record.metadata["agent"]["tool_call_count"] == 1


def test_a_caller_who_hangs_up_mid_answer_is_cancelled(lf, call, monkeypatch):
    monkeypatch.setattr(
        voice_agents.voice_agent_signed_in, "iter",
        lambda **_kw: _answer("Give her water. Keep her in the shade."),
    )

    _run(_turn(history=_PRIOR), take=1)

    # turn.v1's value for a caller who went away, for every pipeline.
    assert lf.closed["amul.outcome"] == "cancelled" and lf.trace.metadata["outcome"] == "cancelled"
    assert [score["value"] for score in lf.scores] == ["cancelled"]
    assert lf.root.ended == 1


def test_a_stale_turn_is_recorded_as_stale(lf, call):
    async def _is_stale(reason):
        return "stale_request" if reason == "before_history_write" else None

    _run(_turn(history=_PRIOR), is_stale=_is_stale)

    assert lf.closed["amul.outcome"] == "stale_request"


def test_a_failing_turn_records_the_error_and_ends_the_root_once(lf, call, monkeypatch):
    async def _down(key, messages):
        raise RuntimeError("redis down")

    monkeypatch.setattr(chat_service, "update_message_history", _down)

    with pytest.raises(RuntimeError):
        _run(_turn(history=_PRIOR))

    assert lf.closed["amul.outcome"] == "error"
    assert lf.trace.metadata["error"] == {"type": "RuntimeError", "message": "redis down"}
    assert lf.root.updates == [{"level": "ERROR", "status_message": "redis down"}]
    assert lf.root.ended == 1


def test_the_request_is_recorded_as_voice_records_it(lf):
    turn = dataclasses.replace(
        _turn(), source_lang=" GU ", target_lang="Gu", call=TelephonyCall(process_id="proc-1", call_type="Outbound")
    )
    telemetry = VoiceTelemetry(turn, pipeline_profile="oss", pipeline_trace=_execution().begin_trace())
    with telemetry.root():
        pass

    record, sent = lf.trace, lf.root.kwargs["metadata"]
    assert record.metadata["call_type"] == "outbound" and sent["amul.direction"] == "outbound"
    assert (record.metadata["source_lang"], record.metadata["target_lang"]) == ("gu", "gu")
    assert (sent["amul.source_lang"], sent["amul.target_lang"], sent["amul.translation_mode"]) == ("gu", "gu", "translated")
    # No provider on the request: no provider key, as the record says None.
    assert record.metadata["provider"] is None and "amul.provider" not in sent
    # The agent's primary tier, as voice's request_model and request_provider.
    assert (record.metadata["request_model"], record.metadata["request_provider"]) == ("gemma", "vllm")
    assert record.metadata["pc_agent"].startswith("vllm:gemma@")
    # Nothing a tier served and nothing timed: the root closes with the outcome alone.
    assert lf.closed == {"amul.outcome": "success"}


def test_the_turns_trace_is_current_only_inside_its_root(lf):
    telemetry = VoiceTelemetry(_turn(), pipeline_profile="oss", pipeline_trace=None)
    outside = voice_trace.current_trace()
    with telemetry.root():
        inside = voice_trace.current_trace()
    after = voice_trace.current_trace()

    assert inside is telemetry._trace
    assert not outside.enabled and not after.enabled
    assert after is not inside


def test_both_surfaces_record_the_served_tier_as_metadata_and_score_only_the_outcome(lf, call, monkeypatch):
    _run(_turn(history=_PRIOR))
    assert [score["name"] for score in lf.scores] == ["turn_outcome"]
    assert lf.closed["amul.served_tier"].endswith("[0]")

    # Chat (turn.v1): the agent's tier is amul.served_tier on the root, written when the
    # root closes, and no longer a score.
    seen = patch_turn(monkeypatch)
    monkeypatch.setattr(chat_service, "_agent_served_tier", lambda _pt: "vllm:gemma[0]")
    propagated = []

    @contextlib.contextmanager
    def _propagate(**kwargs):
        propagated.append(kwargs)
        yield

    monkeypatch.setattr(chat_service, "propagate_attributes", _propagate)
    chat_turn = dataclasses.replace(_turn(history=_PRIOR), call=None, user_id="anonymous")
    asyncio.run(_collect(chat_service.run_turn(
        chat_turn, chat_service.CHAT_SURFACE, scheduler=SimpleNamespace(schedule=lambda *_a: None)
    )))
    assert propagated[-1]["metadata"]["amul.served_tier"] == "vllm:gemma[0]"
    assert "served_tier" not in {c.kwargs.get("name") for c in seen["langfuse"].score_current_trace.call_args_list}


async def _collect(agen):
    return [item async for item in agen]


class _Down:
    """An agent run whose tier fails before it sends anything."""

    async def __aenter__(self):
        raise RuntimeError("tier down")

    async def __aexit__(self, *_e):
        return False


class _BreaksOff(_Down):
    """An agent run that sends its first words, then fails."""

    ctx = None

    async def __aenter__(self):
        return self

    async def __aiter__(self):
        yield _Node(["Give her "])
        raise RuntimeError("stream reset")


_RUNS = {"up": lambda: _answer("Give her water."), "down": _Down, "breaks off": _BreaksOff}


def _agent_on(monkeypatch, *tiers):
    """The agent's chain for the turn: each tier, and how its run goes."""
    chain = []
    for tier, run in tiers:
        target = ExecutionTarget(tier, StepClientKind.AGENT)
        object.__setattr__(target, "handle", run)
        chain.append(target)
    real_chain = ExecutionContext._chain

    async def _chain(self, step, client_kind=None):
        return chain if step is Step.AGENT else await real_chain(self, step, client_kind)

    monkeypatch.setattr(ExecutionContext, "_chain", _chain)
    monkeypatch.setattr(voice_agents.voice_agent_signed_in, "iter", lambda **kw: _RUNS[kw["model"]]())


_GPT_MINI = ("managed", "openai", "gpt-mini")


@pytest.mark.parametrize("tiers, answered, fell_back, served", [
    pytest.param([(_MANAGED, "up")], _GPT_MINI, False, "openai:gpt-mini[0]", id="answers"),
    pytest.param([(_OSS, "down"), (_MANAGED, "up")], _GPT_MINI, True, "openai:gpt-mini[1]", id="falls back"),
    pytest.param([(_MANAGED, "breaks off")], _GPT_MINI, False, None, id="breaks off after its first words"),
    pytest.param([(_OSS, "down"), (_MANAGED, "down")], ("failed",) * 3, True, None, id="every tier is down"),
])
def test_the_agent_block_names_the_model_that_answered_as_voice_did(lf, call, monkeypatch, tiers, answered, fell_back, served):
    _agent_on(monkeypatch, *tiers)

    _run(_turn(history=_PRIOR))

    record = lf.trace
    (stage,) = [stage for stage in record.stages if stage["name"] == "agent"]
    committed = (None,) * 3 if answered[0] == "failed" else answered
    for block in (record.metadata["agent"], stage):
        assert (block["actual_tier"], block["actual_provider"], block["actual_model"]) == answered
        assert tuple(block.get(f"first_token_committed_{key}") for key in ("tier", "provider", "model")) == committed
        assert block["fallback_used"] is fell_back
    # On the trace: the tier that served the answer, as route[index], and the attempts
    # it took; none when no tier finished an answer (the step and the turn are errors).
    sent = lf.step("agent").sent
    assert (sent["metadata"].get("amul.served_tier"), lf.closed.get("amul.served_tier")) == (served, served)
    if served is None:
        assert lf.closed["amul.outcome"] == "error" and lf.root.updates == [{"level": "ERROR", "status_message": None}]
    else:
        assert sent["metadata"]["amul.attempts"] == str(int(served[-2]) + 1)


def test_with_fallback_off_the_agent_block_names_the_primary_that_answered(lf, call, monkeypatch):
    pipeline = llm_runtime.get_pipeline()
    monkeypatch.setattr(llm_runtime, "get_pipeline", lambda surface="chat": pipeline.model_copy(update={"fallback_enabled": False}))
    monkeypatch.setattr(ExecutionTarget, "handle", property(lambda _target: "up"))
    monkeypatch.setattr(voice_agents.voice_agent_signed_in, "iter", lambda **kw: _RUNS[kw["model"]]())

    _run(_turn(history=_PRIOR))

    agent = lf.trace.metadata["agent"]
    assert (agent["actual_tier"], agent["actual_provider"], agent["actual_model"]) == (
        agent["requested_tier"], agent["requested_provider"], agent["requested_model"]
    )
    assert agent["fallback_used"] is False


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
    observation = trace.sent.step("moderation").sent
    assert observation["output"] == {"category": "in_scope", "reason": "ok", "rejected": False}
    assert observation["metadata"]["amul.check_mode"] == "model"
    (streak,) = [s for s in trace.stages if s["name"] == "non_meaningful"]
    assert streak["turn_count"] == 2
    # Fewer than five turns: voice does not gate on the classifier.
    assert trace.metadata["non_meaningful"]["gate_skipped"] is True
    streak_step = trace.sent.step("non_meaningful").sent
    assert streak_step["output"] == {"non_meaningful": False, "reason": "fine", "streak": 2}
    assert streak_step["metadata"]["amul.check_mode"] == "skipped"


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
    observation = trace.sent.step("moderation").sent
    # No verdict to send; the check that raised let the turn through.
    assert (observation["level"], observation["status_message"]) == ("ERROR", "moderation blew up")
    assert "output" not in observation and observation["metadata"]["amul.check_mode"] == "failed_open"


def _turn_execution():
    """The turn's execution context as run_turn hands it to voice's steps: its
    pipeline trace begun, so the walker records the tier that served."""
    execution = _execution()
    execution.begin_trace()
    return execution


def test_pretranslation_records_each_attempt_and_the_fallback(trace, monkeypatch):
    _pretranslate(
        monkeypatch, _gujarati_turn("ભેંસ"), replies=[RuntimeError("oss down"), "For a buffalo"], execution=_turn_execution()
    )

    recorded = trace.metadata["pretranslation"]
    assert (recorded["provider"], recorded["fallback_used"]) == ("vllm", True)
    assert [a["status"] for a in recorded["attempts"]] == ["error", "ok"]
    assert recorded["attempts"][0]["error_class"] == "RuntimeError"
    assert (recorded["actual_tier"], recorded["actual_provider"]) == ("managed", "openai")
    assert recorded["text"]["text"] == "For a buffalo"
    assert [s["name"] for s in trace.stages] == ["pre_translation"]
    step = trace.sent.step("pre_translation").sent
    assert (step["as_type"], step["input"], step["output"]) == ("generation", "ભેંસ", "For a buffalo")
    assert step["metadata"]["amul.served_tier"] == "openai:gpt-mini[1]" and step["metadata"]["amul.attempts"] == "2"
    assert step["metadata"]["amul.configured_tier"].startswith("vllm:gemma@")


def test_pretranslation_that_fails_everywhere_is_recorded_as_failed(trace, monkeypatch):
    _pretranslate(
        monkeypatch, _gujarati_turn("ભેંસ"), replies=[RuntimeError("down"), RuntimeError("down")], execution=_turn_execution()
    )

    recorded = trace.metadata["pretranslation"]
    assert (recorded["provider"], recorded["actual_tier"]) == ("failed", "failed")
    assert [s["status"] for s in trace.stages] == ["error"]
    step = trace.sent.step("pre_translation").sent
    assert step["level"] == "ERROR" and "output" not in step and "amul.served_tier" not in step["metadata"]


def test_the_sink_records_translation_marks_and_why_the_nudge_stopped(trace, monkeypatch):
    trace.set_nudge(armed=True, sent=False)
    _Translation(monkeypatch, replies=[["ગાયને પાણી આપો."]])

    out, _sink = _speak(_English(["Give the cow water."]))

    assert out
    assert trace.response_parts == out
    assert {"first_agent_text_ms", "first_translation_chunk_ms", "ttft_ms", "ttfr_ms"} <= set(trace.timings_ms)
    assert [s["name"] for s in trace.stages] == ["post_translation"]
    step = trace.sent.step("post_translation").sent
    assert (step["as_type"], step["input"], step["output"]) == ("generation", "Give the cow water.", "".join(out))
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
    # The nudge step is written when the nudge goes out, once.
    assert trace.sent.step("nudge").sent["output"] == {"armed": True, "sent": True}

    trace.set_nudge(sent=False)
    nudge_stopped("moderation_rejected")
    nudge_stopped("stream_ended")
    assert trace.metadata["nudge"]["cancel_reason"] == "moderation_rejected"
    assert len(trace.sent.steps("nudge")) == 1


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
    assert trace.sent.step("nudge").sent["output"] == {"armed": True, "sent": True, "trigger": "timeout"}


def test_nudges_switched_off_are_recorded_as_not_armed(trace, monkeypatch):
    monkeypatch.setattr(settings, "enable_voice_nudges", False)

    async def _go():
        await VoiceLiveness(_turn(), started_at=time.monotonic(), send=_Sender(), is_stale=None).stop()

    asyncio.run(_go())

    assert trace.metadata["nudge"] == {"armed": False, "sent": False}
