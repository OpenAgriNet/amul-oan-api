"""The #307 seam contract: ``Turn`` in, ``Emission`` out, services wired outside.

tests/test_chat_turn_contract.py pins that the chat turn behaves exactly as it
did; this file pins the seam itself — what the adapter hands ``run_turn``, how
the adapter renders each emission, that the turn stays transport-free, and that
deferred work really runs after the response through the real router.
"""
import ast
import asyncio
import dataclasses
import os
import pathlib
from types import MappingProxyType, SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest
from fastapi import BackgroundTasks, FastAPI
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect

from app.auth.jwt_auth import get_chat_user
from app.channels.base import Channel
from app.channels.chat import WEB, WHATSAPP, WHATSAPP_RESPONSE_MAX_CHARS
from app.chat_artifacts import CHAT_ARTIFACTS_START, encode_chat_artifacts
from app.routers import chat as chat_router
from app.services import chat as chat_service
from app.turn.types import (
    AgentActivityEmission,
    ArtifactEmission,
    ClassifierResult,
    SideChannelEmission,
    Surface,
    SurfaceProfile,
    TextEmission,
    Turn,
)
from tests.test_chat_turn_contract import _ARTIFACT, patch_turn

_TURN_DIR = pathlib.Path(__file__).resolve().parents[1] / "app" / "turn"


def _turn(**overrides):
    values = dict(
        query="How much water?",
        session_id="seam",
        source_lang="en",
        target_lang="en",
        user_id="anonymous",
        authenticated_user=MappingProxyType({}),
        history=(),
        history_session_id="seam",
        channel=WEB,
        persona="farmer",
    )
    values.update(overrides)
    return Turn(**values)


def _stream(**overrides):
    kwargs = dict(
        query="How much water?",
        session_id="seam",
        source_lang="en",
        target_lang="en",
        channel="web",
        user_id="anonymous",
        history=[],
        user_info={},
        background_tasks=BackgroundTasks(),
    )
    kwargs.update(overrides)
    return chat_service.stream_chat_messages(**kwargs)


async def _collect(agen):
    return [item async for item in agen]


def _fake_run_turn(monkeypatch, emissions, seen=None):
    """Replace run_turn so the adapter can be tested on its own."""

    async def _run_turn(turn, surface, *, scheduler):
        if seen is not None:
            seen.update(turn=turn, surface=surface, scheduler=scheduler)
        try:
            for emission in emissions:
                yield emission
        except BaseException as exc:
            # How the turn was ended: GeneratorExit is a hang-up, anything else
            # is a failure the turn must record as an error.
            if seen is not None:
                seen["ended_by"] = type(exc).__name__
            raise
        finally:
            if seen is not None:
                seen["closed"] = True

    monkeypatch.setattr(chat_service, "run_turn", _run_turn)


class _RecordingScheduler:
    def __init__(self):
        self.scheduled = []

    def schedule(self, fn, /, *args):
        self.scheduled.append((fn, args))


# ── the Turn the adapter composes ───────────────────────────────────────────


def test_turn_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        _turn().query = "changed"


def test_adapter_composes_the_turn_from_request_values(monkeypatch):
    seen = {}
    _fake_run_turn(monkeypatch, [], seen)
    claims = {"phone": "9999999999", "user_type": "farmer"}
    history = [SimpleNamespace(kind="prior")]

    asyncio.run(_collect(_stream(
        channel="whatsapp",
        user_info=claims,
        history=history,
        persona="doctor",
        history_session_id="seam:persona:doctor",
    )))

    turn = seen["turn"]
    assert turn.channel is WHATSAPP
    assert turn.persona == "doctor"
    assert turn.history_session_id == "seam:persona:doctor"
    assert turn.history == tuple(history)
    assert seen["surface"] is chat_service.CHAT_SURFACE


def test_history_key_defaults_to_session_id(monkeypatch):
    seen = {}
    _fake_run_turn(monkeypatch, [], seen)
    asyncio.run(_collect(_stream(history_session_id=None)))
    assert seen["turn"].history_session_id == "seam"


def test_turn_holds_a_read_only_copy_of_the_claims(monkeypatch):
    seen = {}
    _fake_run_turn(monkeypatch, [], seen)
    claims = {"phone": "9999999999"}

    asyncio.run(_collect(_stream(user_info=claims)))
    claims["phone"] = "0000000000"

    user = seen["turn"].authenticated_user
    assert user["phone"] == "9999999999", "the turn must not see later mutation"
    with pytest.raises(TypeError):
        user["phone"] = "1111111111"


def test_missing_claims_become_an_empty_mapping(monkeypatch):
    seen = {}
    _fake_run_turn(monkeypatch, [], seen)
    asyncio.run(_collect(_stream(user_info=None)))
    assert dict(seen["turn"].authenticated_user) == {}


def test_adapter_wires_background_tasks_behind_the_scheduler(monkeypatch):
    seen = {}
    _fake_run_turn(monkeypatch, [], seen)
    tasks = BackgroundTasks()

    asyncio.run(_collect(_stream(background_tasks=tasks)))
    seen["scheduler"].schedule(print, "later")

    assert [(t.func, t.args) for t in tasks.tasks] == [(print, ("later",))]


# ── how the chat adapter renders each emission ──────────────────────────────


def test_adapter_renders_text_and_artifacts_and_drops_what_chat_cannot_carry(monkeypatch):
    _fake_run_turn(monkeypatch, [
        TextEmission("Give clean water."),
        AgentActivityEmission(),
        SideChannelEmission("Please hold on."),
        ArtifactEmission(artifacts=(_ARTIFACT,)),
    ])
    sink = []

    chunks = asyncio.run(_collect(_stream(artifact_sink=sink)))

    assert chunks == ["Give clean water.", encode_chat_artifacts([_ARTIFACT])]
    assert sink == [_ARTIFACT]
    assert "Please hold on." not in "".join(chunks), "a side-channel emission went in-band"


def test_adapter_without_frames_fills_only_the_sink(monkeypatch):
    _fake_run_turn(monkeypatch, [TextEmission("ok"), ArtifactEmission(artifacts=(_ARTIFACT,))])
    sink = []

    chunks = asyncio.run(_collect(_stream(artifact_sink=sink, emit_artifact_frames=False)))

    assert chunks == ["ok"]
    assert sink == [_ARTIFACT]


def test_adapter_refuses_an_emission_it_cannot_render(monkeypatch):
    seen = {}
    _fake_run_turn(monkeypatch, [TextEmission("ok"), object()], seen)

    with pytest.raises(TypeError, match="cannot render"):
        asyncio.run(_collect(_stream()))
    assert seen["ended_by"] == "TypeError", "a rendering failure must reach the turn as an error"


def test_disconnect_closes_the_turn_immediately(monkeypatch):
    """A client hang-up must unwind run_turn now, not at garbage collection."""
    seen = {}
    _fake_run_turn(monkeypatch, [TextEmission("a"), TextEmission("b")], seen)

    async def _go():
        gen = _stream()
        assert await gen.__anext__() == "a"
        await gen.aclose()
        return seen.get("closed")

    assert asyncio.run(_go()) is True
    assert seen["ended_by"] == "GeneratorExit"


def _outcomes(seen):
    return [
        call.kwargs["value"]
        for call in seen["langfuse"].score_current_trace.call_args_list
        if call.kwargs.get("name") == "turn_outcome"
    ]


def test_artifact_encoding_failure_is_recorded_as_error_not_cancelled(monkeypatch):
    """The adapter renders artifacts outside run_turn; its failure is still the turn's.

    Closing the turn on the way out would inject GeneratorExit and record the
    failure as a client hang-up.
    """
    unserializable = {"id": "bad", "kind": "soil_health_card", "content": object()}
    seen = patch_turn(monkeypatch, artifact=unserializable)

    with pytest.raises(TypeError):
        asyncio.run(_collect(_stream(source_lang="gu", target_lang="gu")))

    assert _outcomes(seen) == ["error"]
    assert ("exit", "chat.translation") in seen["spans"], "the root span was left open"


# ── run_turn itself ─────────────────────────────────────────────────────────


def _no_telemetry(monkeypatch):
    monkeypatch.setattr(chat_service, "get_langfuse_client", None)
    monkeypatch.setattr(chat_service, "propagate_attributes", None)


def test_classifier_chain_first_match_wins_and_raw_rides_on_the_text(monkeypatch):
    _no_telemetry(monkeypatch)
    writes, calls = [], []

    async def _history(key, messages):
        writes.append(key)

    monkeypatch.setattr(chat_service, "update_message_history", _history)

    async def _skip(turn):
        calls.append("skip")
        return None

    async def _hangup(turn):
        calls.append("hangup")
        return ClassifierResult(canned_text="Goodbye.", label="hangup", raw=True)

    async def _never(turn):
        calls.append("never")
        return ClassifierResult(canned_text="unreachable", label="never")

    surface = SurfaceProfile(surface=Surface.CHAT, classifiers=(_skip, _hangup, _never))
    emissions = asyncio.run(_collect(
        chat_service.run_turn(_turn(), surface, scheduler=_RecordingScheduler())
    ))

    assert emissions == [TextEmission("Goodbye.", raw=True)]
    assert calls == ["skip", "hangup"]
    assert writes == [], "a classifier with no history_pair must persist nothing"


def test_run_turn_yields_the_answer_then_one_artifact_emission(monkeypatch):
    seen = patch_turn(monkeypatch, artifact=_ARTIFACT)
    scheduler = _RecordingScheduler()

    emissions = asyncio.run(_collect(
        chat_service.run_turn(_turn(source_lang="gu", target_lang="gu"),
                              chat_service.CHAT_SURFACE, scheduler=scheduler)
    ))

    assert emissions[-1] == ArtifactEmission(artifacts=(_ARTIFACT,))
    texts = emissions[:-1]
    assert texts and all(isinstance(e, TextEmission) for e in texts)
    assert _ARTIFACT["content"] not in "".join(e.text for e in texts)
    assert seen["history_writes"], "the turn never persisted its history"


def test_run_turn_defers_suggestions_through_the_given_scheduler(monkeypatch):
    seen = patch_turn(monkeypatch)
    scheduler = _RecordingScheduler()

    asyncio.run(_collect(
        chat_service.run_turn(_turn(source_lang="gu", target_lang="gu"),
                              chat_service.CHAT_SURFACE, scheduler=scheduler)
    ))

    assert seen["suggestions_ran_inline"] == 0
    assert [(fn, args[:2]) for fn, args in scheduler.scheduled] == [
        (chat_service.create_suggestions, ("seam", "gu"))
    ]


@pytest.mark.parametrize(
    "profile, rich, cap",
    [(WEB, True, None), (WHATSAPP, False, WHATSAPP_RESPONSE_MAX_CHARS)],
)
def test_deps_take_channel_capabilities_from_the_profile(monkeypatch, profile, rich, cap):
    seen = patch_turn(monkeypatch)

    asyncio.run(_collect(
        chat_service.run_turn(_turn(source_lang="gu", target_lang="gu", channel=profile),
                              chat_service.CHAT_SURFACE, scheduler=_RecordingScheduler())
    ))

    (deps,) = seen["deps"]
    assert deps.supports_rich_artifacts is rich
    assert deps.response_max_chars == cap


def test_every_channel_profile_declares_rich_artifact_support():
    assert WEB.supports_rich_artifacts is True
    assert WHATSAPP.supports_rich_artifacts is False
    assert {WEB.channel, WHATSAPP.channel} == set(Channel)


# ── the contract stays transport-free ───────────────────────────────────────


_TRANSPORT_MODULES = ("fastapi", "starlette", "redis", "langfuse", "app.core.cache")


@pytest.mark.parametrize("path", sorted(_TURN_DIR.glob("*.py")), ids=lambda p: p.name)
def test_seam_contract_imports_no_transport_or_runtime_service(path):
    tree = ast.parse(path.read_text())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    leaked = [m for m in imported if m.split(".")[0] in _TRANSPORT_MODULES or m in _TRANSPORT_MODULES]
    assert not leaked, f"{path.name} imports {leaked}"


# ── through the real router ─────────────────────────────────────────────────


def _client(monkeypatch):
    async def _no_history(_key):
        return []

    monkeypatch.setattr(chat_router, "_get_message_history", _no_history)
    app = FastAPI()
    app.include_router(chat_router.router)
    app.dependency_overrides[get_chat_user] = lambda: {}
    return TestClient(app)


def test_streaming_response_ends_with_the_artifact_frame(monkeypatch):
    patch_turn(monkeypatch, artifact=_ARTIFACT)

    response = _client(monkeypatch).get(
        "/chat/", params={"query": "How much water?", "source_lang": "gu", "target_lang": "gu"}
    )

    assert response.status_code == 200
    assert response.text.endswith(encode_chat_artifacts([_ARTIFACT]))
    assert response.text.index(CHAT_ARTIFACTS_START) > 0, "the answer must precede the frame"


def test_json_response_carries_artifacts_without_a_frame(monkeypatch):
    patch_turn(monkeypatch, artifact=_ARTIFACT)

    response = _client(monkeypatch).get(
        "/chat/",
        params={"query": "How much water?", "source_lang": "gu", "target_lang": "gu", "stream": False},
    )

    body = response.json()
    assert body["artifacts"] == [_ARTIFACT]
    assert CHAT_ARTIFACTS_START not in body["response"]
    assert body["response"].strip()


def _asgi_app(monkeypatch):
    async def _no_history(_key):
        return []

    monkeypatch.setattr(chat_router, "_get_message_history", _no_history)
    app = FastAPI()
    app.include_router(chat_router.router)
    app.dependency_overrides[get_chat_user] = lambda: {}
    return app


def _asgi_scope(spec_version):
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": spec_version},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/chat/",
        "raw_path": b"/chat/",
        "root_path": "",
        "query_string": b"query=How+much+water%3F&source_lang=gu&target_lang=gu",
        "headers": [(b"host", b"test")],
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
    }


def test_disconnect_raised_by_send_closes_the_turn_before_the_response_returns(monkeypatch):
    """ASGI 2.4: the server's send raises when the client is gone.

    Starlette's StreamingResponse does not close its body iterator on that path,
    so without an explicit close the turn stays suspended — root span open, no
    outcome — until the loop gets round to finalising the generator.
    """
    seen = patch_turn(monkeypatch)
    app = _asgi_app(monkeypatch)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            raise OSError("client went away")

    async def _go():
        with pytest.raises(ClientDisconnect):
            await app(_asgi_scope("2.4"), receive, send)
        # Read immediately: no await has run since, so no async-generator
        # finalizer can have closed the turn behind the response's back.
        return _outcomes(seen), list(seen["spans"])

    outcomes, spans = asyncio.run(_go())
    assert outcomes == ["cancelled"]
    assert spans[-1] == ("exit", "chat.translation"), spans


def test_disconnect_that_cancels_the_stream_closes_the_turn_before_background_work(monkeypatch):
    """ASGI < 2.4: http.disconnect cancels the stream task mid-send.

    The turn must be closed as part of the response, before background tasks
    run, so suggestions never execute inside a still-open turn span.
    """
    seen = patch_turn(monkeypatch)
    spans_when_suggestions_ran = []
    monkeypatch.setattr(
        chat_service,
        "create_suggestions",
        lambda *_a: spans_when_suggestions_ran.append(list(seen["spans"])),
    )
    app = _asgi_app(monkeypatch)
    first_chunk_sent = asyncio.Event()
    requested = []

    async def receive():
        if not requested:
            requested.append(True)
            return {"type": "http.request", "body": b"", "more_body": False}
        await first_chunk_sent.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            first_chunk_sent.set()
            await asyncio.Event().wait()  # a slow client: blocks until cancelled

    async def _go():
        await app(_asgi_scope("2.3"), receive, send)
        return _outcomes(seen), list(seen["spans"])

    outcomes, spans = asyncio.run(_go())
    assert outcomes == ["cancelled"]
    assert spans[-1] == ("exit", "chat.translation"), spans
    assert len(spans_when_suggestions_ran) == 1, "background work did not run"
    assert spans_when_suggestions_ran[0][-1] == ("exit", "chat.translation"), (
        "suggestions ran while the turn's root span was still open"
    )


def test_disconnect_before_the_first_chunk_is_recorded_as_cancelled(monkeypatch):
    """A hang-up while the turn is still working cancels the stream task mid-await.

    run_turn then sees CancelledError rather than GeneratorExit, but the caller
    still hung up; the turn did not fail.
    """
    seen = patch_turn(monkeypatch)
    pretranslating = asyncio.Event()

    async def _slow_pretranslate(_tier, *, text, **_k):
        pretranslating.set()
        await asyncio.Event().wait()  # until the disconnect cancels it

    monkeypatch.setattr(chat_service, "pretranslate_with_tier", _slow_pretranslate)
    app = _asgi_app(monkeypatch)
    requested = []

    async def receive():
        if not requested:
            requested.append(True)
            return {"type": "http.request", "body": b"", "more_body": False}
        await pretranslating.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        assert not (message["type"] == "http.response.body" and message.get("body")), "nothing was produced"

    async def _go():
        await app(_asgi_scope("2.3"), receive, send)
        return _outcomes(seen), list(seen["spans"])

    outcomes, spans = asyncio.run(_go())
    assert outcomes == ["cancelled"]
    assert spans[-1] == ("exit", "chat.translation"), spans


def test_turn_close_completes_even_if_the_turn_awaits_while_unwinding(monkeypatch):
    """The close runs inside the cancelled stream task, so it must be shielded.

    Today's run_turn unwinds without suspending, but a turn that awaits on the
    way out (voice cancelling its background set, say) would otherwise be cut
    off half-way by the pending cancellation.
    """
    state = {}

    async def _run_turn(turn, surface, *, scheduler):
        try:
            yield TextEmission("a")
            yield TextEmission("b")
        finally:
            await asyncio.sleep(0)
            state["closed"] = True

    monkeypatch.setattr(chat_service, "run_turn", _run_turn)
    app = _asgi_app(monkeypatch)
    first_chunk_sent = asyncio.Event()
    requested = []

    async def receive():
        if not requested:
            requested.append(True)
            return {"type": "http.request", "body": b"", "more_body": False}
        await first_chunk_sent.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            first_chunk_sent.set()
            await asyncio.Event().wait()

    async def _go():
        await app(_asgi_scope("2.3"), receive, send)
        return state.get("closed")

    assert asyncio.run(_go()) is True


def test_suggestions_run_after_the_turn_has_written_its_history(monkeypatch):
    """The scheduler's "after" is load-bearing: suggestions read this history."""
    seen = patch_turn(monkeypatch)

    _client(monkeypatch).get(
        "/chat/", params={"query": "How much water?", "source_lang": "gu", "target_lang": "gu"}
    )

    assert seen["events"].count("suggestions") == 1
    assert seen["events"].index("history_write") < seen["events"].index("suggestions")
