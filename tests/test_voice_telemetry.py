"""Voice's telemetry on the seam: one turn.v1 trace per call turn (a root named
turn, flat amul.* keys, the steps under it), and what voice's trace recorded
along the way for its summary log.

Turns run through the real run_turn with voice's pieces; Langfuse, the farmer
fetch, the checks and the agent are stood in for.
"""
import asyncio
import contextlib
import dataclasses
import importlib
import json
import os
import re
import time
from types import MappingProxyType, SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import langfuse
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
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
from app.utils import SessionRequestOwner
from app.voice import agent_input as ai
from app.voice import background as bg
from app.voice import classifiers as vc
from app.voice import liveness as lv_module
from app.voice import outbound
from app.voice import pretranslation as vp
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
_SHA256 = re.compile(r"\b[0-9a-f]{64}\b")


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
        # The trace id OpenTelemetry would give a root opened now (observability.root_trace_id).
        self.pinned = observability._root_trace_id.get()

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
    assert opened["session_id"] == "call-1" and opened["user_id"] == f"+91 {_MOBILE}"
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
        "amul.call_id": "call-1",
        "amul.utterance_id": record.trace_id,
        "amul.signed_in": "true",
        "amul.farmer_id": _MOBILE,
    }
    # The root opens with the turn's trace id, the one its summary log names; no step does.
    assert root.pinned == int(record.trace_id, 16) and record.metadata["trace_id"] == record.trace_id
    assert all(o.pinned is None for o in lf.observations if o is not root)
    assert closed["metadata"]["amul.outcome"] == "success"
    assert closed["metadata"]["amul.served_tier"] == lf.step("agent").sent["metadata"]["amul.served_tier"]
    assert closed["metadata"]["amul.ttft_ms"].isdigit() and closed["metadata"]["amul.ttfr_ms"].isdigit()
    # Every mark and every stage the turn timed goes out, in whole milliseconds.
    timed = {f"amul.{mark}" for mark in record.timings_ms} | {f"amul.{stage}_ms" for stage in record.stage_totals_ms}
    assert {"amul.agent_ms", "amul.first_agent_text_ms"} <= timed
    assert set(closed["metadata"]) == {"amul.outcome", "amul.served_tier", "amul.total_ms"} | timed
    assert all(closed["metadata"][key].isdigit() for key in timed | {"amul.total_ms"})
    # The words: the question and the answer as heard.
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


_CLAIMED, _UTTERANCE, _TOOK_OVER = "ab" * 16, "cd" * 16, "ef" * 16


def test_a_turn_opens_with_the_trace_id_its_request_claimed_and_sends_its_utterance(lf, call):
    turn = _turn(history=_PRIOR)
    _run(dataclasses.replace(turn, call=dataclasses.replace(turn.call, trace_id=_CLAIMED, utterance_id=_UTTERANCE)))

    assert lf.root.pinned == int(_CLAIMED, 16) and lf.trace.trace_id == _CLAIMED
    assert lf.root.kwargs["metadata"]["amul.utterance_id"] == _UTTERANCE
    assert lf.root.kwargs["metadata"]["amul.call_id"] == "call-1"
    assert "amul.superseded_by" not in lf.closed


def test_only_a_root_opened_inside_root_trace_id_takes_its_trace_id():
    exported = InMemorySpanExporter()
    provider = TracerProvider(id_generator=observability._trace_ids())
    provider.add_span_processor(SimpleSpanProcessor(exported))
    tracer = provider.get_tracer("test")

    with observability.root_trace_id(_CLAIMED):
        with tracer.start_as_current_span("turn"):
            pass
    with observability.root_trace_id(_CLAIMED):
        with tracer.start_as_current_span("turn") as root:
            # Opened under the root, as a step is: the root's trace, not a new one.
            with tracer.start_as_current_span("agent") as step:
                pass
    with tracer.start_as_current_span("unpinned"):
        pass
    with observability.root_trace_id(_CLAIMED.upper()), tracer.start_as_current_span("not hex"):
        pass
    with tracer.start_as_current_span("parent"), observability.root_trace_id(_UTTERANCE):
        # A span with a parent takes its parent's trace: the pin goes unused.
        with tracer.start_as_current_span("child"):
            pass

    spans = {span.name: span for span in exported.get_finished_spans()}
    trace_id = {name: format(span.context.trace_id, "032x") for name, span in spans.items()}
    assert trace_id["turn"] == trace_id["agent"] == _CLAIMED and spans["turn"].parent is None
    assert step.parent.span_id == root.get_span_context().span_id
    assert trace_id["unpinned"] != _CLAIMED and trace_id["not hex"] != _CLAIMED
    assert trace_id["child"] == trace_id["parent"] != _UTTERANCE
    assert observability._root_trace_id.get() is None


def test_langfuse_is_set_up_with_the_id_generator_a_call_turns_root_needs(monkeypatch):
    made = {}

    class _Client:
        def __init__(self, **kwargs):
            made.update(kwargs)

        def auth_check(self):
            return False

    had_client = observability.langfuse_client is not None
    monkeypatch.setattr(langfuse, "Langfuse", _Client)
    monkeypatch.setattr(settings, "langfuse_public_key", "pk-lf-test")
    monkeypatch.setattr(settings, "langfuse_secret_key", "sk-lf-test")
    try:
        importlib.reload(observability)
        with observability.root_trace_id(_CLAIMED):
            assert made["id_generator"].generate_trace_id() == int(_CLAIMED, 16)
        assert made["id_generator"].generate_trace_id() != int(_CLAIMED, 16)
    finally:
        monkeypatch.undo()
        importlib.reload(observability)
    assert (observability.langfuse_client is not None) == had_client


def test_a_root_that_opens_with_another_trace_id_is_logged(lf, call, monkeypatch, caplog):
    # As if a tracer provider registered before Langfuse's had kept its own ids.
    monkeypatch.setattr(_Observation, "trace_id", "00" * 16, raising=False)
    turn = _turn(history=_PRIOR)

    with caplog.at_level("WARNING", logger="app.voice.trace"):
        _run(dataclasses.replace(turn, call=dataclasses.replace(turn.call, trace_id=_CLAIMED)))

    assert "did not open with its claimed trace id" in caplog.text
    assert lf.closed["amul.outcome"] == "success"


def _taken_over(monkeypatch, *, by, owner=True):
    """The real staleness check, losing the session just before the history write."""
    async def _yes(_owner):
        return True

    async def _owns(_owner):
        return owner

    async def _by(_owner):
        if isinstance(by, Exception):
            raise by
        return by

    monkeypatch.setattr(lv_module, "refresh_session_request_ownership", _yes)
    monkeypatch.setattr(lv_module, "is_session_request_owner", _owns)
    monkeypatch.setattr(lv_module, "session_request_taken_over_by", _by)
    check = lv_module.CallStaleness(
        http_request=None, owner=SessionRequestOwner(session_id="call-1", request_token="t", epoch=4),
        session_id="call-1", process_id="proc-1",
    )

    async def _is_stale(reason):
        return await check(reason) if reason == "before_history_write" else None

    return _is_stale


def test_a_stale_turn_names_the_request_that_took_over(lf, call, monkeypatch):
    _run(_turn(history=_PRIOR), is_stale=_taken_over(monkeypatch, by=_TOOK_OVER, owner=False))

    assert lf.closed["amul.outcome"] == "stale_request"
    assert lf.closed["amul.superseded_by"] == _TOOK_OVER


@pytest.mark.parametrize("by", [None, ConnectionError("redis down")])
def test_a_stale_turn_no_request_took_over_names_none(lf, call, monkeypatch, by):
    # The session's ownership ran out with no one taking over, or Redis could not say.
    _run(_turn(history=_PRIOR), is_stale=_taken_over(monkeypatch, by=by, owner=False))

    assert lf.closed["amul.outcome"] == "stale_request"
    assert "amul.superseded_by" not in lf.closed


def test_only_a_stale_turn_names_a_request_that_took_over(lf, call):
    async def _is_stale(reason):
        # Known, but the turn ended otherwise: it was not superseded.
        voice_trace.current_trace().superseded_by = _TOOK_OVER
        return None

    _run(_turn(history=_PRIOR), is_stale=_is_stale)

    assert lf.closed["amul.outcome"] == "success"
    assert "amul.superseded_by" not in lf.closed


def _everything_sent(client):
    return json.dumps(
        [[o.kwargs, o.updates] for o in client.observations] + client.propagated + [client.io],
        ensure_ascii=False, default=str,
    )


def test_a_turn_sends_langfuse_the_words_and_the_callers_number_as_they_are(lf, call):
    """turn.v1 keeps the caller's words and number raw (decided 2026-10-13 and 2026-10-20):
    no mode hashes them or leaves them out, and nothing sent is a hash."""
    _run(_turn(history=_PRIOR))

    assert lf.io == {"input": "My cow has fever", "output": "Give her water."}
    assert lf.opened["user_id"] == f"+91 {_MOBILE}"
    assert lf.root.kwargs["metadata"]["amul.farmer_id"] == _MOBILE
    assert not _SHA256.search(_everything_sent(lf))
    # The record keeps the length of the words, not the words, the number or a hash.
    assert lf.trace.metadata["query"] == {"chars": len("My cow has fever")}
    assert _MOBILE not in json.dumps(lf.trace.metadata, ensure_ascii=False, default=str)


def test_the_summary_log_holds_no_words_no_number_and_no_hash(lf, call, caplog):
    caplog.set_level("INFO", logger=voice_trace.logger.name)

    _run(_turn(history=_PRIOR))

    (logged,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("VOICE_TRACE_SUMMARY")]
    for private in (_MOBILE, "fever", "water"):
        assert private not in logged
    assert not _SHA256.search(logged)
    assert json.loads(logged.split(" ", 1)[1])["query"] == {"chars": len("My cow has fever")}


def test_the_root_carries_the_turns_whole_time_from_the_request_coming_in(lf, caplog):
    """The root opens after claiming the session and loading its history, so its span is not
    the whole turn; amul.total_ms is, from the request coming in, as the summary log has it."""
    caplog.set_level("INFO", logger=voice_trace.logger.name)
    route = voice_telemetry.RouteTimings(
        started_at=time.perf_counter() - 1.0, ownership_claim_ms=3.0, history_load_ms=4.0, history_messages=0
    )
    telemetry = voice_telemetry.VoiceTelemetry(
        _turn(), pipeline_profile="oss", pipeline_trace=_execution().begin_trace(), route=route
    )
    with telemetry.root():
        pass

    (logged,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("VOICE_TRACE_SUMMARY")]
    total_ms = json.loads(logged.split(" ", 1)[1])["total_ms"]
    assert total_ms >= 1000.0
    assert lf.closed["amul.total_ms"] == str(int(round(total_ms)))


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
    # Nothing a tier served and nothing timed: the root closes with the outcome and the
    # turn's whole time alone.
    assert set(lf.closed) == {"amul.outcome", "amul.total_ms"}
    assert lf.closed["amul.outcome"] == "success" and lf.closed["amul.total_ms"].isdigit()


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


@pytest.mark.parametrize("tiers, answered, fell_back, served, reason", [
    pytest.param([(_MANAGED, "up")], _GPT_MINI, False, "openai:gpt-mini[0]", None, id="answers"),
    pytest.param([(_OSS, "down"), (_MANAGED, "up")], _GPT_MINI, True, "openai:gpt-mini[1]", None, id="falls back"),
    pytest.param([(_MANAGED, "breaks off")], _GPT_MINI, False, None, "stream reset", id="breaks off after its first words"),
    pytest.param([(_OSS, "down"), (_MANAGED, "down")], ("failed",) * 3, True, None, "tier down", id="every tier is down"),
])
def test_the_agent_block_names_the_model_that_answered_as_voice_did(
    lf, call, monkeypatch, tiers, answered, fell_back, served, reason
):
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
    # it took; none when no tier finished an answer (the step and the turn are errors, and
    # the turn's reason is the failure the sink caught and spoke the trouble line for).
    sent = lf.step("agent").sent
    assert (sent["metadata"].get("amul.served_tier"), lf.closed.get("amul.served_tier")) == (served, served)
    if served is None:
        assert lf.closed["amul.outcome"] == "error" and lf.root.updates == [{"level": "ERROR", "status_message": reason}]
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
    assert recorded["text"] == {"chars": len("For a buffalo")}
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
    # One span for the turn's translation, the batch timed into it without a step of its own.
    step = trace.sent.step("post_translation")
    assert (step.sent["as_type"], step.sent["input"], step.sent["output"]) == ("span", "Give the cow water.", "".join(out))
    assert step.sent["level"] == "DEFAULT"
    assert trace.metadata["nudge"]["cancel_reason"] == "tail_translated_fragment"


def _english_pretranslation(monkeypatch):
    async def _english(text, source_lang, **_kw):
        return "My cow"

    monkeypatch.setattr(vp, "_translate_to_english_pretranslation", _english)


def test_the_sinks_translation_is_one_span_under_the_root_across_its_batches(lf, call, monkeypatch):
    """Through a whole turn: one post_translation span, a direct child of the root, the
    English of every batch in and what the caller heard out; the step's time is the
    batches' time."""
    _english_pretranslation(monkeypatch)
    monkeypatch.setattr(voice_agents.voice_agent_signed_in, "iter", lambda **_kw: _answer("Give her water. Twice a day."))
    _Translation(monkeypatch, replies=[["ગાયને પાણી આપો."], ["દિવસમાં બે વાર."]])

    _run(_turn("મારી ગાય", lang="gu", history=_PRIOR))

    (step,) = lf.steps("post_translation")
    assert (step.parent, step.sent["as_type"]) == ("turn", "span")
    assert step.sent["input"] == "Give her water. Twice a day."
    # Each batch's translation as voice cleaned it for the phone, in order.
    assert step.sent["output"] == "ગાયને પાણી આપો.દિવસમાં બે વાર."
    assert step.ended == 1 and step.sent["level"] == "DEFAULT"
    assert [s["name"] for s in lf.trace.stages if s["name"] == "post_translation"] == ["post_translation"] * 2
    assert lf.closed["amul.post_translation_ms"].isdigit()


def test_a_failed_batch_marks_the_turns_translation_step_an_error(lf, call, monkeypatch):
    _english_pretranslation(monkeypatch)
    monkeypatch.setattr(voice_agents.voice_agent_signed_in, "iter", lambda **_kw: _answer("Give her water. Twice a day."))
    _Translation(monkeypatch, replies=[["ગાયને પાણી આપો."]], fail_on={2})

    _run(_turn("મારી ગાય", lang="gu", history=_PRIOR))

    step = lf.step("post_translation").sent
    assert (step["level"], step["status_message"]) == ("ERROR", "translategemma down")
    assert step["input"] == "Give her water. Twice a day." and step["output"] == "ગાયને પાણી આપો."
    assert lf.closed["amul.outcome"] == "error"


def test_a_failed_output_translation_is_counted(trace, monkeypatch):
    _Translation(monkeypatch, fail_on={1})

    _speak(_English(["Give the cow water."]))

    assert trace.counters["output_translation_errors"] == 1
    # The caller heard the trouble line; the failure is the turn's reason if it ends as an error.
    assert trace.failure == "translategemma down"


def test_an_error_turns_reason_is_what_it_raised_else_what_it_caught_else_its_path():
    trace = voice_trace.VoiceTrace(session_id="s", user_id="u", source_lang="gu", target_lang="gu", query="q", enabled=False)

    assert trace._error_reason(None) == "the turn ended without an answer"
    trace.set_route("pretranslation_empty")
    assert trace._error_reason(None) == "pretranslation_empty"
    trace.record_failure(RuntimeError("tier down"))
    trace.record_failure(RuntimeError("translategemma down"))
    assert trace._error_reason(None) == "tier down"
    assert trace._error_reason(RuntimeError("redis down")) == "redis down"
    assert trace._error_reason(TimeoutError()) == "TimeoutError"
    assert trace._error_reason(RuntimeError("x" * 400)) == "x" * 300


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
