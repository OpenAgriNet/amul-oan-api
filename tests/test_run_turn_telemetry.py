"""Telemetry on the seam: run_turn owns the lifecycle, the surface owns the contract.

``run_turn`` opens exactly one root span per turn, records what the caller got
and how the turn ended on every exit, and takes all three from the surface.
Chat's telemetry is ``_ChatTelemetry`` and writes ``turn.v1``; voice will bring
its own root, stamps and outcome vocabulary without touching the skeleton.
"""
import asyncio
import contextlib
import dataclasses
import json
import os
from pathlib import Path
from types import MappingProxyType

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

from app.channels.chat import WEB
from app.llm_core.trace import snapshot_flags
from app.services import chat as chat_service
from app.turn.types import TextEmission, Turn
from tests.test_chat_turn_contract import _Run, patch_turn

CONTRACT = Path(__file__).resolve().parents[1] / "telemetry" / "contracts" / "turn.v1" / "chat.json"


def _turn(**overrides):
    values = dict(
        query="How much water?",
        session_id="telemetry",
        source_lang="gu",
        target_lang="gu",
        user_id="anonymous",
        authenticated_user=MappingProxyType({}),
        history=(),
        history_session_id="telemetry",
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


class _RecordingTelemetry:
    """Records the lifecycle run_turn drives, in order."""

    events = []

    def __init__(self, turn, *, pipeline_profile, pipeline_trace):
        _RecordingTelemetry.events.append(("built", pipeline_profile))

    @contextlib.contextmanager
    def root(self):
        _RecordingTelemetry.events.append(("enter",))
        try:
            yield
        finally:
            _RecordingTelemetry.events.append(("exit",))

    def record_output(self, text, label):
        _RecordingTelemetry.events.append(("output", label, text))

    def record_outcome(self, outcome):
        _RecordingTelemetry.events.append(("outcome", outcome))


def _recording_surface():
    _RecordingTelemetry.events = []
    return dataclasses.replace(chat_service.CHAT_SURFACE, telemetry=_RecordingTelemetry)


def _run(turn, surface):
    return asyncio.run(_collect(chat_service.run_turn(turn, surface, scheduler=_Scheduler())))


def test_chat_surface_populates_its_real_telemetry():
    assert chat_service.CHAT_SURFACE.telemetry is chat_service._ChatTelemetry


def test_run_turn_opens_the_surface_root_once_and_records_through_it(monkeypatch):
    patch_turn(monkeypatch)

    emissions = _run(_turn(), _recording_surface())

    events = _RecordingTelemetry.events
    kinds = [e[0] for e in events]
    assert kinds.count("enter") == 1 and kinds.count("exit") == 1
    assert kinds[0] == "built" and kinds[1] == "enter" and kinds[-1] == "exit"
    assert events[-2] == ("outcome", "success"), "the outcome must be recorded inside the root"
    outputs = [e for e in events if e[0] == "output"]
    caller_text = "".join(e.text for e in emissions if isinstance(e, TextEmission))
    assert outputs == [("output", "final", caller_text)]


def test_a_short_circuit_records_its_output_and_outcome(monkeypatch):
    patch_turn(monkeypatch)

    emissions = _run(
        _turn(query="who are you", source_lang="en", target_lang="en"), _recording_surface()
    )

    events = _RecordingTelemetry.events
    assert [e for e in events if e[0] == "output"] == [("output", "identity", emissions[0].text)]
    assert ("outcome", "success") in events
    assert events[-1] == ("exit",)


def test_a_failing_turn_records_error_and_still_closes_the_root(monkeypatch):
    patch_turn(monkeypatch)

    def _explode(**_kw):
        raise RuntimeError("agent exploded")

    monkeypatch.setattr(chat_service.agrinet_agent, "iter", _explode)

    with pytest.raises(RuntimeError):
        _run(_turn(), _recording_surface())

    events = _RecordingTelemetry.events
    assert events[-2:] == [("outcome", "error"), ("exit",)]
    assert not [e for e in events if e[0] == "output"], "nothing reached the caller, so there is no output"


def _answer_in_two_parts(monkeypatch, second="Second part."):
    """An English turn (passed through as the agent streams it) whose answer comes in two
    chunks; ``second`` may raise instead."""
    def _chunks():
        yield "First part. "
        if isinstance(second, BaseException):
            raise second
        yield second

    monkeypatch.setattr(chat_service.agrinet_agent, "iter", lambda **kw: _Run(_chunks(), deps=kw["deps"]))
    return _turn(source_lang="en", target_lang="en")


def _leave_after_first(turn, surface):
    async def _go():
        stream = chat_service.run_turn(turn, surface, scheduler=_Scheduler())
        first = await stream.__anext__()
        await stream.aclose()
        return first

    return asyncio.run(_go())


def test_a_turn_the_caller_left_mid_answer_records_what_they_got(monkeypatch):
    """turn.v1's output is the answer as delivered: a hang-up after the first chunk keeps that chunk."""
    patch_turn(monkeypatch)
    turn = _answer_in_two_parts(monkeypatch)

    first = _leave_after_first(turn, _recording_surface())

    events = _RecordingTelemetry.events
    assert first.text == "First part. "
    assert [e for e in events if e[0] == "output"] == [("output", "final", "First part. ")]
    assert events[-2:] == [("outcome", "cancelled"), ("exit",)]


def test_a_turn_that_failed_mid_answer_records_what_was_sent(monkeypatch):
    patch_turn(monkeypatch)
    turn = _answer_in_two_parts(monkeypatch, second=RuntimeError("model dropped"))

    with pytest.raises(RuntimeError, match="model dropped"):
        _run(turn, _recording_surface())

    events = _RecordingTelemetry.events
    assert [e for e in events if e[0] == "output"] == [("output", "final", "First part. ")]
    assert events[-2:] == [("outcome", "error"), ("exit",)]


def test_chat_writes_the_delivered_part_as_the_roots_output(monkeypatch):
    """Through chat's own telemetry: the root's output after a hang-up, and after a failure,
    which also marks it an error."""
    seen = patch_turn(monkeypatch)
    _leave_after_first(_answer_in_two_parts(monkeypatch), chat_service.CHAT_SURFACE)
    assert [kw["output"] for kw in seen["trace_io"] if "output" in kw] == ["First part. "]

    seen = patch_turn(monkeypatch)
    updates = _root_status(seen)
    with pytest.raises(RuntimeError):
        _run(_answer_in_two_parts(monkeypatch, second=RuntimeError("model dropped")), chat_service.CHAT_SURFACE)
    assert [kw["output"] for kw in seen["trace_io"] if "output" in kw] == ["First part. "]
    assert updates == [({"level": "ERROR", "status_message": "model dropped"}, "turn")]


def _open_span(seen):
    open_spans = []
    for event, name in seen["spans"]:
        open_spans.append(name) if event == "enter" else open_spans.remove(name)
    return open_spans[-1] if open_spans else None


def _root_status(seen):
    """What the turn writes on the current span's level and status, with the span open then."""
    updates = []
    seen["langfuse"].update_current_span.side_effect = lambda **kw: updates.append((kw, _open_span(seen)))
    return updates


def test_a_failing_turn_marks_its_root_an_error_with_what_it_raised(monkeypatch):
    """turn.v1: level ERROR and the reason on the root whenever the outcome is error."""
    seen = patch_turn(monkeypatch)
    updates = _root_status(seen)

    def _explode(**_kw):
        raise RuntimeError("agent exploded")

    monkeypatch.setattr(chat_service.agrinet_agent, "iter", _explode)

    with pytest.raises(RuntimeError):
        _run(_turn(), chat_service.CHAT_SURFACE)

    assert updates == [({"level": "ERROR", "status_message": "agent exploded"}, "turn")]
    assert {"level", "status_message"} <= set(json.loads(CONTRACT.read_text(encoding="utf-8"))["trace_fields"])


def test_a_fail_closed_answer_marks_its_root_an_error_with_its_path(monkeypatch):
    """Moderation failed and the farmer got the placeholder: an error the turn did not raise,
    so the reason is the path that answered it."""
    seen = patch_turn(monkeypatch)
    updates = _root_status(seen)

    async def _moderation_down(user_message, model=None):
        raise RuntimeError("moderation down")

    monkeypatch.setattr(chat_service.moderation_agent, "run", _moderation_down)

    emissions = _run(_turn(source_lang="en", target_lang="en"), chat_service.CHAT_SURFACE)

    assert emissions, "the fail-closed line still reaches the farmer"
    assert updates == [({"level": "ERROR", "status_message": "fail-closed"}, "turn")]


def test_an_error_turns_reason_is_what_it_raised_else_the_path_that_answered():
    telemetry = chat_service._ChatTelemetry(_turn(), pipeline_profile="oss", pipeline_trace=None)

    assert telemetry._error_reason() == "the turn ended without an answer"
    telemetry.record_output("Please try again later.", "fail-closed")
    assert telemetry._error_reason() == "fail-closed"
    telemetry._raised = TimeoutError()
    assert telemetry._error_reason() == "TimeoutError"
    telemetry._raised = RuntimeError("x" * 400)
    assert telemetry._error_reason() == "x" * 300


def test_a_turn_that_did_not_fail_leaves_its_root_level_alone(monkeypatch):
    """An answered turn, and one the caller hung up on (cancelled), are not errors."""
    seen = patch_turn(monkeypatch)
    updates = _root_status(seen)

    _run(_turn(), chat_service.CHAT_SURFACE)

    async def _hang_up():
        turn = chat_service.run_turn(_turn(), chat_service.CHAT_SURFACE, scheduler=_Scheduler())
        async for _ in turn:
            break
        await turn.aclose()

    asyncio.run(_hang_up())

    outcomes = [
        call.kwargs["value"] for call in seen["langfuse"].score_current_trace.call_args_list
        if call.kwargs.get("name") == "turn_outcome"
    ]
    assert outcomes == ["success", "cancelled"]
    assert updates == []


def test_a_bare_surface_runs_without_writing_a_trace(monkeypatch):
    seen = patch_turn(monkeypatch)
    bare = dataclasses.replace(chat_service.CHAT_SURFACE, telemetry=None)

    emissions = _run(_turn(), bare)

    assert emissions, "the turn produced nothing"
    assert ("enter", chat_service.TURN_ROOT) not in seen["spans"]
    outcomes = [
        call for call in seen["langfuse"].score_current_trace.call_args_list
        if call.kwargs.get("name") == "turn_outcome"
    ]
    assert outcomes == []


def test_chat_telemetry_writes_the_turn_v1_contract(monkeypatch):
    """The contract test reads the source; this one checks what is actually sent."""
    seen = patch_turn(monkeypatch)
    propagated = []

    @contextlib.contextmanager
    def _propagate(**kwargs):
        propagated.append((kwargs, len(seen["spans"])))
        yield

    monkeypatch.setattr(chat_service, "propagate_attributes", _propagate)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    _no_farmer_lookup(monkeypatch)

    _run(_turn(user_id="9000000000", authenticated_user=MappingProxyType({"phone": "9000000000"})), chat_service.CHAT_SURFACE)

    assert seen["spans"][0] == ("enter", contract["root"]) == ("enter", "turn")
    environment, root, close = propagated
    # The surface's environment goes to every span; it is not a trace-level field.
    assert set(environment[0]) == {"environment"} and environment[1] == 0
    # The trace-level fields are set on the root, before any step opened, and never propagated to one.
    assert root[0]["session_id"] == "telemetry" and root[0]["user_id"] == "9000000000"
    assert root[1] == 1 and seen["spans"][1][0] != "enter" or seen["spans"][1][1] != "turn"
    metadata = root[0]["metadata"]
    assert set(metadata) <= set(contract["metadata_keys"])
    assert metadata["amul.schema_version"] == "turn.v1" and metadata["amul.kind"] == "turn"
    assert metadata["amul.pipeline"] == "webapp" and metadata["amul.direction"] == "inbound"
    assert metadata["amul.translation_mode"] == "translated" and metadata["amul.farmer_id"] == "9000000000"
    assert (metadata["amul.source_lang"], metadata["amul.target_lang"], metadata["amul.persona"]) == ("gu", "gu", "farmer")
    # The guard flags that were on, as llm_core joins them; they can change while a release runs.
    flags_on = ",".join(name.replace("_enabled", "") for name, on in snapshot_flags().items() if on)
    assert metadata.get("amul.pipeline_flags") == (flags_on or None)
    assert all(isinstance(value, str) for value in metadata.values())
    # What the turn knows at its end lands on the root when it closes, with the root current again.
    assert close[1] == len(seen["spans"]) - 1 and seen["spans"][-1] == ("exit", "turn")
    assert set(close[0]["metadata"]) <= set(contract["metadata_keys"]) and close[0]["metadata"]["amul.outcome"] == "success"
    inputs = [kw["input"] for kw in seen["trace_io"] if "input" in kw]
    assert inputs == ["How much water?"]  # the raw question, not an object
    scores = {call.kwargs["name"] for call in seen["langfuse"].score_current_trace.call_args_list}
    assert scores == set(contract["scores"]) == {"turn_outcome"}


def _no_farmer_lookup(monkeypatch):
    async def _lookup(_phone):
        return chat_service.FarmerContextBundle(markdown="")

    monkeypatch.setattr(chat_service, "get_farmer_context_bundle_by_mobile", _lookup)


def _root_fields(monkeypatch, turn):
    patch_turn(monkeypatch)
    _no_farmer_lookup(monkeypatch)
    propagated = []

    @contextlib.contextmanager
    def _propagate(**kwargs):
        propagated.append(kwargs)
        yield

    monkeypatch.setattr(chat_service, "propagate_attributes", _propagate)
    _run(turn, chat_service.CHAT_SURFACE)
    return next(kwargs for kwargs in propagated if "session_id" in kwargs)


@pytest.mark.parametrize(
    ("user_id", "authenticated_user", "farmer_id"),
    [
        ("anonymous", {"phone": "+91 98765 43210"}, "9876543210"),  # the JWT's phone, as the farmer context reads it
        ("9000000000", {}, None),  # a phone-shaped query-param user id is not verified
        ("9000000000", {"sub": "9000000000"}, None),  # nor is a JWT subject that is not its phone
    ],
    ids=["jwt-phone", "query-param", "jwt-sub"],
)
def test_a_farmer_id_is_only_the_phone_the_verified_jwt_carries(monkeypatch, user_id, authenticated_user, farmer_id):
    root = _root_fields(monkeypatch, _turn(user_id=user_id, authenticated_user=MappingProxyType(authenticated_user)))
    assert root["metadata"].get("amul.farmer_id") == farmer_id


@pytest.mark.parametrize(
    ("source_lang", "target_lang", "codes", "mode"),
    [("en", "english", ("en", "en"), "direct"), ("gujarati", "Gujarati", ("gu", "gu"), "translated"), ("gu", "hi", ("gu", "hi"), "translated")],
)
def test_the_root_names_the_languages_by_code(monkeypatch, source_lang, target_lang, codes, mode):
    """A client may name the language (dev's sent "english", 2026-08 to 10); turn.v1 keeps its code.
    Whether the answer is translated is decided as before."""
    root = _root_fields(monkeypatch, _turn(source_lang=source_lang, target_lang=target_lang))
    metadata = root["metadata"]
    assert (metadata["amul.source_lang"], metadata["amul.target_lang"]) == codes
    assert metadata["amul.translation_mode"] == mode


def test_an_anonymous_user_has_no_user_id_no_farmer_id_and_english_is_direct(monkeypatch):
    """turn.v1's userId is written when known: the request's default "anonymous" is not a user."""
    patch_turn(monkeypatch)
    propagated = []

    @contextlib.contextmanager
    def _propagate(**kwargs):
        propagated.append(kwargs)
        yield

    monkeypatch.setattr(chat_service, "propagate_attributes", _propagate)

    _run(_turn(source_lang="en", target_lang="en"), chat_service.CHAT_SURFACE)

    root = next(kwargs for kwargs in propagated if "session_id" in kwargs)
    assert root["user_id"] is None and "amul.farmer_id" not in root["metadata"]
    assert root["metadata"]["amul.translation_mode"] == "direct"


@pytest.mark.parametrize(
    ("user_id", "authenticated_user", "recorded"),
    [
        ("anonymous", {}, None),
        ("", {}, None),
        ("9000000000", {}, "9000000000"),  # the request's user id, unverified but known
        ("anonymous", {"sub": "farmer-7"}, "farmer-7"),  # the JWT's subject over the request's default
        ("9000000000", {"phone": "+91 98765 43210", "sub": "x"}, "+91 98765 43210"),  # the JWT's phone first
    ],
    ids=["anonymous", "empty", "query-param", "jwt-sub", "jwt-phone"],
)
def test_the_root_names_the_user_only_when_one_is_known(monkeypatch, user_id, authenticated_user, recorded):
    root = _root_fields(monkeypatch, _turn(user_id=user_id, authenticated_user=MappingProxyType(authenticated_user)))
    assert root["user_id"] == recorded


def test_the_root_opens_with_the_trace_id_the_transport_made(monkeypatch):
    """The chat router makes the turn's trace id before the turn, so the response can
    name it; the root span, and only the root, is opened inside root_trace_id with it."""
    seen = patch_turn(monkeypatch)
    pinned = []

    @contextlib.contextmanager
    def _root_trace_id(trace_id):
        pinned.append((trace_id, list(seen["spans"])))
        yield

    monkeypatch.setattr(chat_service, "root_trace_id", _root_trace_id)

    _run(_turn(trace_id="a" * 32), chat_service.CHAT_SURFACE)

    assert pinned == [("a" * 32, [])]
    assert seen["spans"][0] == ("enter", "turn")
