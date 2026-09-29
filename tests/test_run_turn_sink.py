"""The sink on the seam: run_turn streams through the surface's sink, not its own.

Chat's sink is ``_ChatSink`` (``_stream_to_client`` plus the Doctor sanitiser on
both sides of translation). These tests pin that ``run_turn`` builds the sink
from the surface, streams the agent through it, and records what the sink says
the caller received, so a voice sink can replace it without touching the
skeleton.
"""
import asyncio
import dataclasses
import os
from types import MappingProxyType, SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

from app.channels.chat import WEB
from app.services import chat as chat_service
from app.turn.types import TextEmission, Turn
from tests.test_chat_turn_contract import patch_turn


def _turn(**overrides):
    values = dict(
        query="How much water?",
        session_id="sink",
        source_lang="gu",
        target_lang="gu",
        user_id="anonymous",
        authenticated_user=MappingProxyType({}),
        history=(),
        history_session_id="sink",
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


def _trace_outputs(seen):
    return [kw["output"] for kw in seen["trace_io"] if "output" in kw]


def _outcomes(seen):
    return [
        call.kwargs["value"]
        for call in seen["langfuse"].score_current_trace.call_args_list
        if call.kwargs.get("name") == "turn_outcome"
    ]


class _RecordingSink:
    """A sink that shouts, so its output is unmistakably its own."""

    built = []

    def __init__(self, turn, *, execution, deps, translate_to):
        self.kwargs = dict(turn=turn, execution=execution, deps=deps, translate_to=translate_to)
        _RecordingSink.built.append(self)

    async def stream(self, english):
        async for chunk in english:
            yield chunk.upper()

    def final_text(self):
        return "FINAL FROM SINK"


def _recording_surface():
    # Everything else stays chat's own, so only the sink differs from production.
    _RecordingSink.built = []
    return dataclasses.replace(chat_service.CHAT_SURFACE, sink=_RecordingSink)


def test_chat_surface_populates_its_real_sink():
    assert chat_service.CHAT_SURFACE.sink is chat_service._ChatSink


def test_run_turn_streams_through_the_surface_sink(monkeypatch):
    seen = patch_turn(monkeypatch, agent_text="give clean water.")

    emissions = asyncio.run(_collect(
        chat_service.run_turn(_turn(), _recording_surface(), scheduler=_Scheduler())
    ))

    assert emissions == [TextEmission("GIVE CLEAN WATER.")]
    (sink,) = _RecordingSink.built
    assert sink.kwargs["translate_to"] == "gu"
    assert sink.kwargs["deps"] is seen["deps"][0]
    assert sink.kwargs["execution"] is not None


def test_trace_output_is_what_the_sink_reports(monkeypatch):
    seen = patch_turn(monkeypatch)

    asyncio.run(_collect(
        chat_service.run_turn(_turn(), _recording_surface(), scheduler=_Scheduler())
    ))

    assert _trace_outputs(seen)[-1] == "FINAL FROM SINK"


def test_english_turn_asks_the_sink_not_to_translate(monkeypatch):
    patch_turn(monkeypatch)

    asyncio.run(_collect(chat_service.run_turn(
        _turn(source_lang="en", target_lang="en"), _recording_surface(), scheduler=_Scheduler()
    )))

    assert _RecordingSink.built[0].kwargs["translate_to"] is None


def test_a_surface_without_a_sink_fails_before_the_agent_streams(monkeypatch):
    seen = patch_turn(monkeypatch)
    sinkless = dataclasses.replace(chat_service.CHAT_SURFACE, sink=None)

    with pytest.raises(TypeError, match="without a sink"):
        asyncio.run(_collect(chat_service.run_turn(_turn(), sinkless, scheduler=_Scheduler())))

    assert _outcomes(seen) == ["error"]
    assert seen["translated_inputs"] == [], "the agent's output reached translation"


# ── chat's sink on its own ──────────────────────────────────────────────────


def _chat_sink(monkeypatch, *, persona, translate_to, translated_suffix="", sent_to_translation=None):
    async def _translate(text, *_a, **_k):
        if sent_to_translation is not None:
            sent_to_translation.append(text)
        yield f"[{text}]{translated_suffix}"

    monkeypatch.setattr(chat_service, "translate_text_stream_fast", _translate)
    return chat_service._ChatSink(
        _turn(persona=persona),
        execution=SimpleNamespace(),
        deps=SimpleNamespace(response_max_chars=None),
        translate_to=translate_to,
    )


async def _english(*chunks):
    for chunk in chunks:
        yield chunk


def test_chat_sink_passes_english_through_and_keeps_it_for_the_trace(monkeypatch):
    sink = _chat_sink(monkeypatch, persona="farmer", translate_to=None)

    out = asyncio.run(_collect(sink.stream(_english("Give ", "water."))))

    assert "".join(out) == "Give water."
    assert sink.final_text() == "Give water."


def test_chat_sink_with_nothing_produced_reports_none(monkeypatch):
    sink = _chat_sink(monkeypatch, persona="farmer", translate_to="gu")

    assert asyncio.run(_collect(sink.stream(_english()))) == []
    assert sink.final_text() is None


def test_doctor_provenance_is_removed_before_it_reaches_translation(monkeypatch):
    """The first sanitiser pass: source labels never go to the translation model."""
    sent = []
    sink = _chat_sink(monkeypatch, persona="doctor", translate_to="gu", sent_to_translation=sent)

    asyncio.run(_collect(sink.stream(_english("Give fluids.\n", "Sources: NDDB manual\n"))))

    assert sent, "nothing was sent to translation"
    assert "Give fluids." in "".join(sent)
    assert "Sources" not in "".join(sent)


def test_doctor_provenance_invented_by_translation_never_reaches_the_caller(monkeypatch):
    """The second sanitiser pass, after translation, is the one this pins."""
    sink = _chat_sink(
        monkeypatch, persona="doctor", translate_to="gu",
        translated_suffix="\nSources: NDDB manual",
    )

    out = "".join(asyncio.run(_collect(sink.stream(_english("Give fluids.")))))

    assert "Give fluids." in out
    assert "Sources" not in out


def test_doctor_trace_output_is_sanitised_after_translation(monkeypatch):
    sink = _chat_sink(
        monkeypatch, persona="doctor", translate_to="gu",
        translated_suffix="\nSources: NDDB manual",
    )

    asyncio.run(_collect(sink.stream(_english("Give fluids."))))

    assert sink.final_text()
    assert "Sources" not in sink.final_text()
