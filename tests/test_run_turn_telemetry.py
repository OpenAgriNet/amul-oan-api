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
from app.services import chat as chat_service
from app.turn.types import TextEmission, Turn
from tests.test_chat_turn_contract import patch_turn

CONTRACT = Path(__file__).resolve().parents[1] / "telemetry" / "contracts" / "turn.v1.json"


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

    _run(_turn(user_id="9000000000"), chat_service.CHAT_SURFACE)

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
    assert all(isinstance(value, str) for value in metadata.values())
    # What the turn knows at its end lands on the root when it closes, with the root current again.
    assert close[1] == len(seen["spans"]) - 1 and seen["spans"][-1] == ("exit", "turn")
    assert set(close[0]["metadata"]) <= set(contract["metadata_keys"]) and close[0]["metadata"]["amul.outcome"] == "success"
    inputs = [kw["input"] for kw in seen["trace_io"] if "input" in kw]
    assert inputs == ["How much water?"]  # the raw question, not an object
    scores = {call.kwargs["name"] for call in seen["langfuse"].score_current_trace.call_args_list}
    assert scores == set(contract["scores"]) == {"turn_outcome"}


def test_an_anonymous_user_has_no_farmer_id_and_english_is_direct(monkeypatch):
    patch_turn(monkeypatch)
    propagated = []

    @contextlib.contextmanager
    def _propagate(**kwargs):
        propagated.append(kwargs)
        yield

    monkeypatch.setattr(chat_service, "propagate_attributes", _propagate)

    _run(_turn(source_lang="en", target_lang="en"), chat_service.CHAT_SURFACE)

    root = next(kwargs for kwargs in propagated if "session_id" in kwargs)
    assert root["user_id"] == "anonymous" and "amul.farmer_id" not in root["metadata"]
    assert root["metadata"]["amul.translation_mode"] == "direct"
