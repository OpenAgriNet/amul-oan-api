"""The traces sent beside the turns, in turn.v1 (amul-telemetry telemetry/turn.v1.yaml).

A background job or a frontend event is its own trace: a root named after its kind,
the amul.* keys, session and user written on the root only, and no input or output,
which turn.v1 keeps for turns. Suggestions are not one: they are a step of the turn
that asked for them, opened under its root after the response.
"""

import asyncio
import contextlib
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import langfuse
import pytest

import agents.tools.farmer_cache as farmer_cache
import app.services.langfuse_telemetry_writer as writer
import app.tasks.suggestions as suggestions
from opentelemetry import trace as otel_trace
from app import observability
from app.models.telemetry import CanonicalTelemetryEvent
from app.routers import health
from app.services.telemetry_stamps import forward_chat_telemetry_metadata


class _Observation:
    def __init__(self, client, name):
        self._client = client
        self._name = name

    def update(self, **kwargs):
        self._client.log.append(("update", self._name, kwargs))

    def score(self, **kwargs):
        self._client.log.append(("score", self._name, kwargs))


class _Langfuse:
    """Records, in order, the observations opened and closed and what was propagated
    with which observations open."""

    def __init__(self):
        self.log = []
        self.open = []
        self.parents = {}

    @contextlib.contextmanager
    def start_as_current_observation(self, **kwargs):
        self.log.append(("open", kwargs["name"], kwargs))
        current = otel_trace.get_current_span().get_span_context()
        self.parents[kwargs["name"]] = (format(current.trace_id, "032x"), format(current.span_id, "016x"))
        self.open.append(kwargs["name"])
        try:
            yield _Observation(self, kwargs["name"])
        finally:
            self.open.pop()
            self.log.append(("close", kwargs["name"]))

    @contextlib.contextmanager
    def propagate_attributes(self, **kwargs):
        self.log.append(("propagate", tuple(self.open), kwargs))
        yield

    def set_current_trace_io(self, **kwargs):
        self.log.append(("trace_io", kwargs))

    def create_score(self, **kwargs):
        self.log.append(("score_trace", kwargs))

    def flush(self):
        pass

    def opened(self, name):
        (kwargs,) = [entry[2] for entry in self.log if entry[0] == "open" and entry[1] == name]
        return kwargs

    def propagated(self):
        return [(open_, kwargs) for kind, open_, kwargs in (e for e in self.log if e[0] == "propagate")]


@pytest.fixture
def lf(monkeypatch):
    client = _Langfuse()
    monkeypatch.setattr(observability, "langfuse_client", client)
    monkeypatch.setattr(langfuse, "propagate_attributes", client.propagate_attributes)
    return client


def _root_metadata(kind, **keys):
    return {**forward_chat_telemetry_metadata(), "amul.kind": kind, **keys}


def test_a_farmer_refresh_is_a_background_job_no_turn_caused(lf):
    open_during_refresh = []

    async def _refresh(phone, *, background):
        open_during_refresh.append(tuple(lf.open))

    fake_redis = AsyncMock()
    fake_redis.spop.return_value = ["9876543210"]
    with patch.object(farmer_cache, "redis_client", fake_redis), patch.object(farmer_cache, "refresh_farmer_data", new=_refresh):
        assert asyncio.run(farmer_cache.drain_farmer_refresh_queue_once(batch=1)) == 1

    metadata = _root_metadata("background_job", **{"amul.job": "farmer_background_refresh", "amul.farmer_id": "9876543210"})
    root = lf.opened("background_job")
    assert (root["as_type"], root["metadata"], root.get("input")) == ("span", metadata, None)
    # The API calls of the refresh nest under it. The queue is chat's and voice's: no
    # session, no pipeline.
    assert open_during_refresh == [("background_job",)]
    assert lf.propagated() == [(("background_job",), {"session_id": None, "user_id": None, "metadata": metadata})]


def test_the_health_canary_translates_inside_a_health_check_job(lf, monkeypatch):
    open_during_translation = []

    async def _translate(**_kwargs):
        open_during_translation.append(tuple(lf.open))
        return "તમારી ગાયે આજે 12 લિટર દૂધ આપ્યું."

    monkeypatch.setattr("app.services.translation.translate_text", _translate)

    assert asyncio.run(health.check_translation_service())["status"] == "healthy"
    assert open_during_translation == [("background_job",)]
    assert lf.opened("background_job")["metadata"] == _root_metadata("background_job", **{"amul.job": "health_check"})


def _feedback(trace_id=None):
    return CanonicalTelemetryEvent(
        event_name="feedback",
        session_id="web-session",
        question_id="q-1",
        trace_id=trace_id,
        pipeline="default",
        payload={"target": {"type": "Feedback", "feedbackDetails": {
            "feedbackText": "useful", "questionText": "q", "answerText": "a", "feedbackType": "like",
        }}},
    )


def test_a_frontend_event_is_its_own_frontend_event_trace(lf, monkeypatch):
    monkeypatch.setattr(writer, "get_langfuse_client", lambda: lf)

    assert asyncio.run(writer.write_canonical_event_to_langfuse(_feedback())) == ("queued", None)

    metadata = _root_metadata("frontend_event", **{"amul.pipeline": "webapp", "amul.event": "feedback"})
    assert [entry[:2] for entry in lf.log if entry[0] in ("open", "close")] == [
        ("open", "frontend_event"), ("open", "frontend.feedback"), ("close", "frontend.feedback"), ("close", "frontend_event"),
    ]
    root = lf.opened("frontend_event")
    assert (root["metadata"], root.get("input"), root.get("output")) == (metadata, None, None)
    # Trace-level fields on the root only, before the event's observation opens.
    assert lf.propagated() == [(("frontend_event",), {"session_id": "web-session", "user_id": None, "metadata": metadata})]
    event = lf.opened("frontend.feedback")
    assert event["input"] == {"question": "q", "answer": "a"}
    assert event["output"] == {"feedback_text": "useful", "feedback_type": "like", "rating": None}
    assert event["metadata"]["pipeline"] == "default"
    assert ("score", "frontend.feedback", {"name": "user_feedback", "value": 1.0, "comment": "useful"}) in lf.log
    assert not [entry for entry in lf.log if entry[0] == "trace_io"]


def test_a_feedback_that_names_its_turn_scores_the_turn(lf, monkeypatch):
    """turn.v1: user_feedback goes on the turn when the frontend sends back the trace id
    the chat response carried (X-Trace-Id); the frontend_event trace is still written."""
    monkeypatch.setattr(writer, "get_langfuse_client", lambda: lf)
    turn = "b" * 32

    assert asyncio.run(writer.write_canonical_event_to_langfuse(_feedback(trace_id=turn))) == ("queued", None)

    assert [entry[:2] for entry in lf.log if entry[0] in ("open", "close")] == [
        ("open", "frontend_event"), ("open", "frontend.feedback"), ("close", "frontend.feedback"), ("close", "frontend_event"),
    ]
    scores = [entry for entry in lf.log if entry[0] in ("score", "score_trace")]
    assert scores == [("score_trace", {
        "trace_id": turn, "name": "user_feedback", "value": 1.0, "comment": "useful",
        "environment": observability.surface_environment("chat"),
    })]


@pytest.mark.parametrize("trace_id", ["", "not-a-trace", "B" * 32, "c" * 31])
def test_a_feedback_whose_trace_id_is_not_one_stays_on_its_own_trace(lf, monkeypatch, trace_id):
    monkeypatch.setattr(writer, "get_langfuse_client", lambda: lf)

    assert asyncio.run(writer.write_canonical_event_to_langfuse(_feedback(trace_id=trace_id))) == ("queued", None)

    scores = [entry for entry in lf.log if entry[0] in ("score", "score_trace")]
    assert scores == [("score", "frontend.feedback", {"name": "user_feedback", "value": 1.0, "comment": "useful"})]


def test_the_anonymous_token_carries_its_user(lf, monkeypatch):
    monkeypatch.setattr(writer, "get_langfuse_client", lambda: lf)
    event = CanonicalTelemetryEvent(event_name="anonymous_token_issued", session_id="web-session", user_id="anon-uid", payload={})

    asyncio.run(writer.write_canonical_event_to_langfuse(event))

    ((_, propagated),) = lf.propagated()
    assert (propagated["session_id"], propagated["user_id"]) == ("web-session", "anon-uid")
    assert propagated["metadata"]["amul.event"] == "anonymous_token_issued"


def test_suggestions_are_a_step_under_the_root_of_their_turn(lf, monkeypatch):
    async def _no_history(_session_id):
        return []

    async def _done(*_args, **_kwargs):
        return True

    monkeypatch.setattr(suggestions, "_get_message_history", _no_history)
    monkeypatch.setattr(suggestions, "set_cache", _done)
    monkeypatch.setattr(suggestions.cache, "delete", _done)
    monkeypatch.setattr(suggestions, "get_langfuse_client", lambda: lf)
    monkeypatch.setattr(suggestions, "propagate_attributes", lf.propagate_attributes)
    served = {"amul.served_tier": "vllm:gemma[0]", "amul.attempts": "1"}
    monkeypatch.setattr(suggestions._pipeline_trace, "step_metadata", lambda pt, step: served if pt == "turn-trace" else {})

    class _Execution:
        def info(self, _step):
            return SimpleNamespace(model_name="test-model")

        async def run(self, *_args):
            return SimpleNamespace(output=["q1", "q2", "q3"])

        def begin_trace(self):
            return "turn-trace"

    turn_root = {"trace_id": "a" * 32, "parent_span_id": "b" * 16}
    assert asyncio.run(suggestions.create_suggestions("s1", "gu", _Execution(), turn_root)) == ["q1", "q2", "q3"]

    step = lf.opened("suggestions")
    # A child of the turn's root span, not opened with trace_context=: Langfuse marks
    # such a span as the trace's root and writes its name, input and output over the turn's.
    assert "trace_context" not in step and step["as_type"] == "generation"
    assert lf.parents["suggestions"] == (turn_root["trace_id"], turn_root["parent_span_id"])
    # A late step writes no trace-level field: only the environment is propagated.
    assert [set(kwargs) for _, kwargs in lf.propagated()] == [{"environment"}]
    assert ("update", "suggestions", {"output": {"suggestions": ["q1", "q2", "q3"]}, "metadata": served}) in lf.log
