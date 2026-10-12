"""The voice route and its adapter over run_turn, behind VOICE_ROUTE_ENABLED.

The route and the adapter are the real ones; Redis, the checks, the agent and
the telephony provider are stood in for.
"""
import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from agents.voice import agent as voice_agents
from app.auth.jwt_auth import get_current_user
from app.config import settings
from app.routers import voice as voice_router
from app.services import chat as chat_service
from app.services import voice as voice_service
from app.turn.types import (
    AgentActivityEmission,
    ArtifactEmission,
    SideChannelEmission,
    Surface,
    TextEmission,
)
from app.utils import SessionRequestOwner
from app.voice import background as bg
from app.voice import liveness as lv
from app.voice import outbound
from app.voice.classifiers import _GREETING_RESPONSES
from app.voice.liveness import CallStaleness, RayaNudgeSender
from app.voice.moderation import ModerationVerdict
from app.voice.non_meaningful import NonMeaningfulVerdict
from app.voice.sink import _prepare_voice_output
from app.voice.telemetry import RouteTimings, VoiceTelemetry
from tests.test_chat_turn_contract import patch_turn
from tests.test_voice_agent_input import _Run

REPO = Path(__file__).resolve().parents[1]
_OWNER = SessionRequestOwner(session_id="call-1", request_token="token-1", epoch=1)


@pytest.fixture
def world(monkeypatch):
    """Stand-ins for Redis, the checks, the agent and the provider."""
    seen = {"released": [], "history_writes": [], "stage": None, "stage_reads": 0}
    patch_turn(monkeypatch)

    async def _moderation(**_kw):
        return ModerationVerdict(category="in_scope", reason="ok")

    async def _streak(**_kw):
        return NonMeaningfulVerdict(five_consecutive_non_meaningful=False, reason="fine")

    async def _claim(session_id):
        return _OWNER

    async def _release(owner):
        seen["released"].append(owner)
        return True

    async def _owner(_owner):
        return True

    async def _no_history(_key):
        return []

    async def _history(key, messages):
        seen["history_writes"].append((key, list(messages)))

    async def _stage(session_id):
        seen["stage_reads"] += 1
        return seen["stage"]

    answer = [ModelRequest(parts=[UserPromptPart(content="q")]), ModelResponse(parts=[TextPart(content="Give her water.")])]
    monkeypatch.setattr(bg, "check_moderation", _moderation)
    monkeypatch.setattr(bg, "check_non_meaningful_streak", _streak)
    monkeypatch.setattr(voice_router, "claim_session_request_ownership", _claim)
    monkeypatch.setattr(voice_router, "_get_message_history", _no_history)
    monkeypatch.setattr(voice_service, "release_session_request_ownership", _release)
    monkeypatch.setattr(lv, "refresh_session_request_ownership", _owner)
    monkeypatch.setattr(lv, "is_session_request_owner", _owner)
    monkeypatch.setattr(outbound, "get_stage", _stage)
    monkeypatch.setattr(chat_service, "update_message_history", _history)
    monkeypatch.setattr(voice_agents.voice_agent, "iter", lambda **_kw: _Run(["Give her water."], answer))
    monkeypatch.setattr(settings, "enable_voice_nudges", False)
    return seen


def _client():
    app = FastAPI()
    app.include_router(voice_router.router)
    app.dependency_overrides[get_current_user] = lambda: {}
    return TestClient(app)


def _get(**params):
    return _client().get("/voice/", params={"session_id": "call-1", "process_id": "proc-1", **params})


# ── through the route ───────────────────────────────────────────────────────


def test_a_question_is_answered_over_the_voice_route(world):
    response = _get(query="My cow has fever", source_lang="en", target_lang="en")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text == "Give her water."
    ((key, messages),) = world["history_writes"]
    assert key == "call-1" and messages[-1].parts[0].content == "Give her water."
    assert world["released"] == [_OWNER]


def _logged_by(caplog, *loggers):
    return "\n".join(record.getMessage() for record in caplog.records if record.name in loggers)


def test_the_route_logs_no_caller_phone_words_or_ownership_token(world, caplog):
    caplog.set_level("INFO")

    _get(query="My cow has fever", user_id="9990001112", source_lang="en", target_lang="en")

    logged = _logged_by(caplog, voice_router.logger.name, voice_service.logger.name)
    assert "query_chars: 16" in logged
    for private in ("9990001112", "My cow has fever", "token-1"):
        assert private not in logged


def test_a_consent_turn_logs_no_caller_phone_or_words(monkeypatch, caplog):
    caplog.set_level("INFO")

    async def _stage(session_id):
        return outbound.STAGE_INTRO_SENT

    monkeypatch.setattr(settings, "outbound_intro_enabled", True)
    monkeypatch.setattr(outbound, "get_stage", _stage)
    _stream(monkeypatch, [], call_type="inbound", history=("earlier",))

    logged = _logged_by(caplog, voice_service.logger.name)
    assert "Outbound consent turn" in logged
    for private in ("9876543210", "હા"):
        assert private not in logged


def test_a_canned_answer_goes_through_voices_normalizer(world):
    response = _get(query="hello", source_lang="gu", target_lang="gu")

    assert response.text == _prepare_voice_output(_GREETING_RESPONSES["gu"], "gu")


def test_the_hang_up_token_reaches_the_provider_exactly(world):
    response = _get(query="please remain on the line", source_lang="gu", target_lang="gu")

    assert response.text == "Goodbye."
    assert world["released"] == [_OWNER]


def test_a_request_without_a_session_gets_one(world):
    _client().get("/voice/", params={"query": "My cow has fever", "source_lang": "en", "target_lang": "en"})

    ((key, _messages),) = world["history_writes"]
    assert key and key != "None"


def test_the_route_times_what_it_does_before_the_turn(world, monkeypatch):
    seen = {}

    async def _slow_claim(session_id):
        await asyncio.sleep(0.05)
        return _OWNER

    async def _history(_key):
        return [ModelRequest(parts=[UserPromptPart(content="earlier")])]

    async def _nothing():
        if False:
            yield ""

    def _stream(**kwargs):
        seen.update(kwargs, called_at=time.perf_counter())
        return _nothing()

    monkeypatch.setattr(voice_router, "claim_session_request_ownership", _slow_claim)
    monkeypatch.setattr(voice_router, "_get_message_history", _history)
    monkeypatch.setattr(voice_router, "stream_voice_message", _stream)

    _get(query="hello", call_type="outbound", provider="RAYA")

    route = seen["route"]
    assert route.ownership_claim_ms >= 50
    assert route.history_messages == 1
    assert seen["called_at"] - route.started_at >= 0.05, "the clock starts before the claim"
    assert (seen["call_type"], seen["provider"], seen["process_id"]) == ("outbound", "RAYA", "proc-1")


def test_the_route_is_registered_only_when_switched_on():
    def _voice_paths(enabled):
        script = (
            "import sys, main; "
            "print(sorted(r.path for r in main.app.routes if r.path.startswith('/api/voice'))); "
            "print('app.voice.agent_input' in sys.modules)"
        )
        env = {**os.environ, "OPENAI_API_KEY": "test-key", "VOICE_ROUTE_ENABLED": enabled}
        out = subprocess.run([sys.executable, "-c", script], cwd=REPO, env=env, capture_output=True, text=True)
        assert out.returncode == 0, out.stderr[-2000:]
        return out.stdout.strip().splitlines()[-2:]

    assert _voice_paths("false") == ["[]", "False"]
    assert _voice_paths("true") == ["['/api/voice/']", "True"]


# ── the adapter ─────────────────────────────────────────────────────────────


class _FakeTurn:
    """Stands in for run_turn: records how the turn was composed, then yields
    ``emissions``, recording anything thrown into it."""

    def __init__(self, emissions):
        self.emissions = emissions
        self.calls = []
        self.thrown = []
        self.closed = False

    def __call__(self, turn, surface, **kwargs):
        self.calls.append((turn, surface, kwargs))
        return self._run()

    async def _run(self):
        try:
            for emission in self.emissions:
                yield emission
        except Exception as exc:
            self.thrown.append(exc)
            raise
        finally:
            self.closed = True


_ROUTE = RouteTimings(started_at=0.0, ownership_claim_ms=1.0, history_load_ms=1.0, history_messages=0)


def _stream(monkeypatch, emissions, *, call_type="inbound", history=(), take=None, target_lang="gu"):
    fake = _FakeTurn(emissions)
    monkeypatch.setattr(voice_service, "run_turn", fake)
    released = []

    async def _release(owner):
        released.append(owner)
        return True

    monkeypatch.setattr(voice_service, "release_session_request_ownership", _release)

    async def _go():
        out = []
        stream = voice_service.stream_voice_message(
            query="હા", session_id="call-1", source_lang="gu", target_lang=target_lang,
            user_id="+91 9876543210", history=list(history), user_info={"sub": "x"},
            provider="RAYA", process_id="proc-1", call_type=call_type, owner=_OWNER,
            http_request=None, background_tasks=SimpleNamespace(add_task=lambda *_a: None), route=_ROUTE,
        )
        async for chunk in stream:
            out.append(chunk)
            if take is not None and len(out) == take:
                await stream.aclose()
                break
        return out

    return asyncio.run(_go()), fake, released


def test_the_adapter_composes_a_voice_turn(monkeypatch):
    _out, fake, _released = _stream(monkeypatch, [])

    ((turn, surface, kwargs),) = fake.calls
    assert surface.surface is Surface.VOICE
    assert surface.telemetry.func is VoiceTelemetry and surface.telemetry.keywords == {"route": _ROUTE}
    assert (turn.session_id, turn.history_session_id, turn.persona, turn.channel) == ("call-1", "call-1", "farmer", None)
    assert dict(turn.authenticated_user) == {"sub": "x"}
    assert (turn.call.process_id, turn.call.provider, turn.call.call_type) == ("proc-1", "RAYA", "inbound")
    assert isinstance(kwargs["is_stale"], CallStaleness)
    assert isinstance(kwargs["side_channel"], RayaNudgeSender)


@pytest.mark.parametrize("enabled, call_type, history, stage, consent_turn", [
    (True, "outbound", (), None, True),
    (True, "OUTBOUND", (), None, True),
    (True, "inbound", (), None, False),
    (True, "outbound", ("earlier",), None, False),
    (True, "outbound", (), outbound.STAGE_RESOLVED, False),
    (True, "inbound", ("earlier",), outbound.STAGE_INTRO_SENT, True),
    (False, "outbound", (), None, False),
])
def test_the_consent_turn_is_worked_out_as_voice_did(monkeypatch, enabled, call_type, history, stage, consent_turn):
    reads = []

    async def _stage(session_id):
        reads.append(session_id)
        return stage

    monkeypatch.setattr(settings, "outbound_intro_enabled", enabled)
    monkeypatch.setattr(outbound, "get_stage", _stage)

    _out, fake, _released = _stream(monkeypatch, [], call_type=call_type, history=history)

    ((turn, _surface, _kwargs),) = fake.calls
    assert turn.call.outbound_consent_turn is consent_turn
    assert turn.call.call_type == call_type.lower()
    assert reads == (["call-1"] if enabled else [])


def test_only_a_raw_emission_skips_the_normalizer(monkeypatch):
    out, _fake, _released = _stream(monkeypatch, [
        TextEmission("**ગાયને** 3 લિટર"),
        TextEmission("Goodbye.", raw=True),
        SideChannelEmission("please wait"),
        AgentActivityEmission(),
        ArtifactEmission(artifacts=({"kind": "shc"},)),
    ])

    assert out == [_prepare_voice_output("**ગાયને** 3 લિટર", "gu"), "Goodbye."]


def test_the_session_is_released_when_the_caller_hangs_up(monkeypatch):
    out, fake, released = _stream(monkeypatch, [TextEmission("one"), TextEmission("two")], take=1, target_lang="en")

    assert out == ["one"]
    assert fake.closed and released == [_OWNER]


def test_a_rendering_failure_is_thrown_into_the_turn_and_the_session_released(monkeypatch):
    fake = _FakeTurn([object()])
    monkeypatch.setattr(voice_service, "run_turn", fake)
    released = []

    async def _release(owner):
        released.append(owner)
        return True

    monkeypatch.setattr(voice_service, "release_session_request_ownership", _release)

    async def _go():
        async for _chunk in voice_service.stream_voice_message(
            query="q", session_id="call-1", source_lang="en", target_lang="en", user_id="anonymous",
            history=[], user_info={}, provider=None, process_id=None, call_type="inbound", owner=_OWNER,
            http_request=None, background_tasks=SimpleNamespace(add_task=lambda *_a: None),
        ):
            pass

    with pytest.raises(TypeError):
        asyncio.run(_go())
    # Thrown into run_turn, so it records an error rather than a hang-up.
    assert [type(exc) for exc in fake.thrown] == [TypeError]
    assert released == [_OWNER]


# ── what the route timed, on the trace ──────────────────────────────────────


def test_the_routes_timings_open_the_trace(monkeypatch):
    monkeypatch.setattr(settings, "enable_voice_tracing", False)
    turn = SimpleNamespace(
        session_id="call-1", user_id="u", query="q", source_lang="gu", target_lang="gu",
        call=SimpleNamespace(provider=None, process_id=None, call_type="inbound"),
    )
    route = RouteTimings(started_at=time.perf_counter() - 1.0, ownership_claim_ms=3.0, history_load_ms=4.0, history_messages=6)

    trace = VoiceTelemetry(turn, pipeline_profile="oss", pipeline_trace=None, route=route)._trace
    trace.finish()

    assert [(s["name"], s["duration_ms"]) for s in trace.stages[:2]] == [("ownership_claim", 3.0), ("history_load", 4.0)]
    assert trace.stages[1]["history_messages"] == 6
    assert trace.stage_totals_ms["history_load"] == 4.0
    assert trace.started_at == route.started_at


# ── the normalizer runs again on what the sink already cleaned ──────────────


@pytest.mark.parametrize("lang, text", [
    ("gu", "ગાયને 3-4 કિ.ગ્રા. ખોરાક આપો."),
    ("gu", " ગાયને પાણી આપો!"),
    ("gu", "**૧. ગાય** (ભેંસ) / બકરી"),
    ("en", "**Give** 5 kg of feed, twice a day."),
    ("en", " Keep her in the shade; check again..."),
    ("en", "- First item\n- Second item"),
])
def test_normalizing_twice_changes_nothing(lang, text):
    once = _prepare_voice_output(text, lang)

    assert _prepare_voice_output(once, lang) == once
