"""Voice's sink: the agent's English in, what the caller hears out.

Voice's own cases for the batching, the cleanup and the pinned union-ban line
come first, then the sink on its own with the translation stubbed, then the
sink on the seam.
"""
import asyncio
import contextlib
import dataclasses
import logging
import os
from types import MappingProxyType

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

from agents.tools.models.union import UNION_BANNED_MESSAGE, union_banned_message
from app.channels.chat import WEB
from app.llm_core.execution import ExecutionContext
from app.services import chat as chat_service
from app.services import translation
from app.turn.types import TelephonyCall, TextEmission, Turn
from app.voice import sink as voice_sink
from app.voice.classifiers import _IDENTITY_RESPONSE_EN
from tests.test_chat_turn_contract import patch_turn

_TROUBLE_GU = voice_sink.TRANSLATION_TROUBLE_MESSAGE["gu"]


def _turn(target_lang="gu", **overrides):
    values = dict(
        query="મારી ગાયને કેટલું પાણી આપવું?",
        session_id="call-1",
        source_lang="gu",
        target_lang=target_lang,
        user_id="anonymous",
        authenticated_user=MappingProxyType({}),
        history=(),
        history_session_id="call-1",
        channel=WEB,
        persona="farmer",
        call=TelephonyCall(process_id="proc-1"),
    )
    values.update(overrides)
    return Turn(**values)


# ── voice's own cases ───────────────────────────────────────────────────────


def test_gujarati_digits_are_normalized_before_filtering():
    result = voice_sink.clean_output_by_language("3-4 કિ.ગ્રા.", "gu")
    assert "3" not in result
    assert "4" not in result
    assert "ત્રણ" in result
    assert "ચાર" in result
    assert "કિલોગ્રામ" in result


def test_placeholder_dashes_before_units_are_removed():
    result = voice_sink.clean_output_by_language("લીલો ચારો: -- કિ.ગ્રા.", "gu")
    assert "--" not in result
    assert "કિ.ગ્રા." not in result
    assert "લીલો ચારો:" in result


@pytest.mark.parametrize("lang, expected", [
    ("gu", union_banned_message("gu")),
    ("gujarati", union_banned_message("gu")),
    ("hi", union_banned_message("hi")),
    ("hindi", union_banned_message("hi")),
    ("en", UNION_BANNED_MESSAGE),
])
def test_canned_union_ban_translation_pins_agreed_copy(lang, expected):
    assert voice_sink._canned_union_ban_translation(UNION_BANNED_MESSAGE, lang) == expected
    assert voice_sink._canned_union_ban_translation(f"  {UNION_BANNED_MESSAGE}  ", lang) == expected
    assert voice_sink._canned_union_ban_translation("Please try again later.", lang) is None


def test_canned_union_ban_gujarati_survives_voice_cleanup():
    spoken = voice_sink._prepare_voice_output(union_banned_message("gu"), "gu")
    assert "દૂધ મંડળી" in spoken
    assert spoken.strip() == union_banned_message("gu").strip()


def test_extract_translation_units_force_splits_oversized_buffer():
    text = (
        "This is a very long answer without a sentence break that keeps going and going "
        "so the caller should not have to wait forever before translation starts because "
        "we need an earlier forced split in the buffered text for voice delivery "
        * 6
    )
    ready, remaining = voice_sink.extract_translation_units(text)
    assert ready
    assert all(len(unit) <= 600 for unit in ready)
    assert remaining != text


def test_extract_translation_units_breaks_at_structural_headers():
    text = "Call a veterinarian quickly:\n### 1. Base feed\nGive roughage and water"
    ready, remaining = voice_sink.extract_translation_units(text)
    assert ready == ["Call a veterinarian quickly:"]
    assert remaining.startswith("### 1. Base feed")


def test_should_translate_batch_forces_flush_on_large_char_batch():
    batch_text = "word " * 140
    assert voice_sink.should_translate_batch(batch_text, word_count=20, is_first_batch=False) is True


def test_voice_instruction_asks_for_spoken_language_and_has_no_length_rule():
    with translation.translation_channel("voice"):
        instruction, tg_prompt = translation._prepare_translation_inputs(
            "Keep the animal hydrated.", "english", "gujarati", None
        )
    assert instruction in tg_prompt
    assert "Prefer clear spoken language over literal formatting" in instruction
    assert "Preserve newlines" not in instruction
    assert "Length Rule" not in instruction


def test_chat_instruction_still_keeps_the_layout():
    instruction, _ = translation._prepare_translation_inputs(
        "Keep the animal hydrated.", "english", "gujarati", None
    )
    assert "Preserve newlines, paragraph breaks, and list structure" in instruction
    assert "Prefer clear spoken language" not in instruction


# ── the sink on its own ─────────────────────────────────────────────────────


class _Translation:
    """Stands in for ``translate_text_stream_fast``: records what it was asked
    and in which channel, and answers from ``replies`` in order."""

    def __init__(self, monkeypatch, replies=(), fail_on=()):
        self.asked = []
        self.channels = []
        self.closed = 0
        self._replies = list(replies)
        self._fail_on = set(fail_on)
        monkeypatch.setattr(voice_sink, "translate_text_stream_fast", self._stream)

    async def _stream(self, text, source_lang, target_lang, execution=None, **_kw):
        self.asked.append((text, target_lang, execution))
        call = len(self.asked)
        try:
            self.channels.append(translation._is_voice_channel())
            if call in self._fail_on:
                raise RuntimeError("translategemma down")
            for chunk in self._replies[call - 1]:
                yield chunk
                self.channels.append(translation._is_voice_channel())
        finally:
            self.closed += 1


class _Staleness:
    """Stale from the first time ``stale_at`` is asked, and from then on, like a
    hung-up call."""

    def __init__(self, stale_at=None):
        self.asked = []
        self._stale_at = stale_at
        self._stale = False

    async def __call__(self, reason):
        self.asked.append(reason)
        if reason == self._stale_at:
            self._stale = True
        return "stale_request" if self._stale else None


class _English:
    """The agent's English stream; records whether it was closed."""

    def __init__(self, chunks, fail_after=None):
        self.closed = False
        self._chunks = chunks
        self._fail_after = fail_after

    async def __call__(self):
        try:
            for i, chunk in enumerate(self._chunks):
                if i == self._fail_after:
                    raise RuntimeError("agent stream died")
                yield chunk
            if self._fail_after == len(self._chunks):
                raise RuntimeError("agent stream died")
        finally:
            self.closed = True


_EXECUTION = object()


def _speak(english, *, target_lang="gu", is_stale=None, watch=()):
    sink = voice_sink.VoiceSink(
        _turn(target_lang),
        execution=_EXECUTION,
        deps=None,
        translate_to=None if target_lang == "en" else target_lang,
        is_stale=is_stale,
    )

    async def _drain():
        out = []
        channel_between_chunks = []
        async for text in sink.stream(english()):
            out.append(text)
            channel_between_chunks.append(translation._is_voice_channel())
        # What the sink had closed when it finished, before the event loop's own
        # cleanup closes whatever was left open.
        for stream in (english, *watch):
            stream.closed_by_sink = stream.closed
        return out, channel_between_chunks

    out, channels = asyncio.run(asyncio.wait_for(_drain(), timeout=5))
    assert not any(channels), "the voice channel leaked out of the sink"
    return out, sink


def test_english_is_spoken_as_it_streams_cleaned_for_the_phone(monkeypatch):
    tr = _Translation(monkeypatch)

    out, sink = _speak(_English(["**Give** 5 kg", " of green fodder."]), target_lang="en")

    assert out == ["Give 5 kilograms", " of green fodder."]
    assert tr.asked == []
    assert sink.final_text() == "Give 5 kilograms of green fodder."


def test_a_gujarati_answer_is_batched_translated_and_spaced(monkeypatch):
    tr = _Translation(monkeypatch, replies=[
        ["ગાયને રોજ ", "ચોખ્ખું પાણી આપો."],
        ["ગાયને છાંયડામાં રાખો."],
        ["પશુચિકિત્સકને બોલાવો"],
    ])
    english = _English(["Give clean water daily. Keep", " the cow in shade.", " Call a vet"])

    out, sink = _speak(english)

    # The first batch goes as soon as one sentence of three words is done.
    assert [text for text, _, _ in tr.asked] == [
        "Give clean water daily. ", "Keep the cow in shade. ", "Call a vet",
    ]
    assert all(lang == "gu" and execution is _EXECUTION for _, lang, execution in tr.asked)
    assert all(tr.channels), "translation ran outside the voice channel"
    # A chunk after a finished sentence starts with a space.
    assert out == [
        "ગાયને રોજ ", "ચોખ્ખું પાણી આપો.", " ગાયને છાંયડામાં રાખો.", " પશુચિકિત્સકને બોલાવો",
    ]
    assert sink.final_text() == "".join(out)
    assert english.closed_by_sink


def test_translated_text_is_cleaned_for_the_phone(monkeypatch):
    _Translation(monkeypatch, replies=[["**3** કિ.ગ્રા. દાણ."]])

    out, _ = _speak(_English(["Give 3 kg feed."]))

    assert out == [voice_sink.clean_output_by_language("**3** કિ.ગ્રા. દાણ.", "gu")]
    assert "ત્રણ કિલોગ્રામ" in out[0]


def test_a_leaked_model_identity_is_replaced_before_translation(monkeypatch):
    tr = _Translation(monkeypatch, replies=[["હું સરલાબેન છું."]])

    _speak(_English(["I am ChatGPT, made by OpenAI."]))

    assert tr.asked[0][0] == _IDENTITY_RESPONSE_EN


def test_the_union_ban_line_is_the_pinned_copy_not_a_translation(monkeypatch):
    tr = _Translation(monkeypatch)

    out, _ = _speak(_English([UNION_BANNED_MESSAGE]))

    assert out == [voice_sink._prepare_voice_output(union_banned_message("gu"), "gu")]
    assert tr.asked == []


def test_a_failed_translation_is_the_trouble_line_and_the_answer_goes_on(monkeypatch, caplog):
    _Translation(monkeypatch, replies=[None, ["ગાયને છાંયડામાં રાખો."]], fail_on={1})

    out, _ = _speak(_English(["Give clean water daily. Keep the cow in shade."]))

    assert out == [_TROUBLE_GU, " ગાયને છાંયડામાં રાખો."]
    assert "output translation failed" in caplog.text


@pytest.mark.parametrize("fail_after, heard, when", [(0, [], "before"), (1, ["ગાયને પાણી આપો."], "after")])
def test_an_agent_that_fails_is_answered_with_the_trouble_line(monkeypatch, caplog, fail_after, heard, when):
    _Translation(monkeypatch, replies=[["ગાયને પાણી આપો."]])
    english = _English(["Give clean water daily. Keep", " the cow in shade."], fail_after=fail_after)

    with caplog.at_level(logging.ERROR):
        out, sink = _speak(english)

    # What was still buffered is dropped, as on voice.
    assert out == [*heard, _TROUBLE_GU]
    assert f"Voice agent stream failed {when} first token" in caplog.text
    assert sink.final_text() == "".join(out)


def test_a_stale_request_hears_no_trouble_line(monkeypatch):
    _Translation(monkeypatch)
    staleness = _Staleness(stale_at="after_stream_error")

    out, _ = _speak(_English([], fail_after=0), is_stale=staleness)

    assert out == []


def test_going_stale_during_the_agent_stream_stops_everything(monkeypatch):
    tr = _Translation(monkeypatch, replies=[["ગાયને રોજ પાણી આપો."]])
    english = _English(["Give clean water daily. Keep", " the cow in shade.", " Call a vet."])
    staleness = _Staleness(stale_at="during_agent_stream")

    out, _ = _speak(english, is_stale=staleness)

    assert out == []
    assert tr.asked == []
    assert english.closed_by_sink
    assert staleness.asked == ["during_agent_stream", "before_translation_flush"]


def test_going_stale_during_translation_stops_it(monkeypatch):
    tr = _Translation(monkeypatch, replies=[["ગાયને ", "રોજ ", "પાણી આપો."]])
    staleness = _Staleness(stale_at="during_output_translation")

    out, _ = _speak(_English(["Give clean water daily."]), is_stale=staleness, watch=(tr,))

    assert out == []
    assert tr.closed_by_sink == 1


@pytest.mark.parametrize("stale_at, heard", [
    ("before_translated_yield", 0),
    ("before_final_translated_yield", 2),
    ("before_tail_translated_yield", 4),
])
def test_nothing_is_said_once_the_request_is_stale(monkeypatch, stale_at, heard):
    # Two chunks for each batch: the first, the one still buffered at the end,
    # and the tail.
    tr = _Translation(monkeypatch, replies=[["ગાયને ", "પાણી આપો."]] * 3)
    staleness = _Staleness(stale_at=stale_at)
    english = _English(["Give clean water daily. Keep", " it cool. Rest"])

    out, _ = _speak(english, is_stale=staleness, watch=(tr,))

    assert len(out) == heard
    assert tr.closed_by_sink == len(tr.asked)


def test_the_buffered_answer_is_not_translated_once_stale(monkeypatch):
    tr = _Translation(monkeypatch)
    staleness = _Staleness(stale_at="before_translation_flush")

    out, _ = _speak(_English(["Give clean water"]), is_stale=staleness)

    assert out == []
    assert tr.asked == []


def test_english_is_not_spoken_once_stale(monkeypatch):
    _Translation(monkeypatch)
    staleness = _Staleness(stale_at="before_direct_yield")

    out, _ = _speak(_English(["Give clean water daily."]), target_lang="en", is_stale=staleness)

    assert out == []


def test_the_trace_keeps_the_chunks_the_caller_hears(monkeypatch):
    _Translation(monkeypatch)

    out, sink = _speak(_English(["Give", " ", "water."]), target_lang="en")

    assert out == ["Give", " ", "water."]
    # Voice's trace kept only chunks with something to hear.
    assert sink.final_text() == "Givewater."


# ── rendering a fixed line for the caller ───────────────────────────────────


def test_render_speaks_english_as_is(monkeypatch):
    async def _no_translation(**_kw):
        raise AssertionError("English needs no translation")

    monkeypatch.setattr(voice_sink, "translate_text", _no_translation)

    assert asyncio.run(voice_sink.render_for_caller("Give **5 kg** feed.", "en")) == "Give 5 kilograms feed."


def test_render_translates_in_the_voice_channel(monkeypatch):
    seen = []

    async def _translate(text, source_lang, target_lang, **_kw):
        seen.append((text, target_lang, translation._is_voice_channel()))
        return "**૫** કિ.ગ્રા. દાણ આપો."

    monkeypatch.setattr(voice_sink, "translate_text", _translate)

    spoken = asyncio.run(voice_sink.render_for_caller("Give 5 kg feed.", "gu"))

    assert seen == [("Give 5 kg feed.", "gu", True)]
    assert spoken == voice_sink.clean_output_by_language("**૫** કિ.ગ્રા. દાણ આપો.", "gu")
    assert not translation._is_voice_channel()


def test_render_pins_the_union_ban_line(monkeypatch):
    async def _no_translation(**_kw):
        raise AssertionError("the ban line is never translated")

    monkeypatch.setattr(voice_sink, "translate_text", _no_translation)

    assert asyncio.run(voice_sink.render_for_caller(UNION_BANNED_MESSAGE, "gu")) == union_banned_message("gu")


def test_render_falls_back_to_the_trouble_line(monkeypatch):
    async def _down(**_kw):
        raise RuntimeError("translategemma down")

    monkeypatch.setattr(voice_sink, "translate_text", _down)

    assert asyncio.run(voice_sink.render_for_caller("Give feed.", "gu")) == _TROUBLE_GU


# ── the sink on the seam ────────────────────────────────────────────────────


class _Scheduler:
    def schedule(self, fn, /, *args):
        pass


class _Outcomes:
    seen = []

    def __init__(self, turn, *, pipeline_profile, pipeline_trace):
        pass

    @contextlib.contextmanager
    def root(self):
        yield

    def record_output(self, text, label):
        _Outcomes.seen.append(("output", label, text))

    def record_outcome(self, outcome):
        _Outcomes.seen.append(("outcome", outcome))


def _surface(**fields):
    _Outcomes.seen = []
    return dataclasses.replace(
        chat_service.CHAT_SURFACE, sink=voice_sink.VoiceSink, telemetry=_Outcomes, **fields
    )


def _run(turn, surface, *, is_stale=None, side_channel=None):
    async def _drain():
        out = []
        async for emission in chat_service.run_turn(
            turn, surface, scheduler=_Scheduler(), is_stale=is_stale, side_channel=side_channel,
        ):
            out.append(emission)
        return out

    return asyncio.run(asyncio.wait_for(_drain(), timeout=10))


def test_a_call_turn_is_spoken_through_the_voice_sink(monkeypatch):
    seen = patch_turn(monkeypatch, agent_text="Give clean water daily. Keep the cow in shade.")
    _Translation(monkeypatch, replies=[["ગાયને રોજ પાણી આપો."], ["ગાયને છાંયડામાં રાખો."]])

    emissions = _run(_turn(), _surface())

    assert emissions == [TextEmission("ગાયને રોજ પાણી આપો."), TextEmission(" ગાયને છાંયડામાં રાખો.")]
    assert seen["translated_inputs"] == [], "chat's translation ran"
    assert len(seen["history_writes"]) == 1
    assert ("output", "final", "ગાયને રોજ પાણી આપો. ગાયને છાંયડામાં રાખો.") in _Outcomes.seen
    assert _Outcomes.seen[-1] == ("outcome", "success")


def test_a_call_turn_that_goes_stale_mid_answer_stops_and_saves_nothing(monkeypatch):
    seen = patch_turn(monkeypatch, agent_text="Give clean water daily. Keep the cow in shade.")
    _Translation(monkeypatch, replies=[["ગાયને રોજ પાણી આપો."], ["ગાયને છાંયડામાં રાખો."]])
    staleness = _Staleness(stale_at="before_translated_yield")

    emissions = _run(_turn(), _surface(), is_stale=staleness)

    assert emissions == []
    assert seen["history_writes"] == []
    assert _Outcomes.seen[-1] == ("outcome", "stale_request")


def test_an_agent_that_fails_before_its_first_chunk_is_answered_after_the_gate(monkeypatch):
    patch_turn(monkeypatch)
    _Translation(monkeypatch)
    log = []

    class _Pass:
        def __init__(self, turn, *, execution):
            pass

        async def gate(self):
            log.append("gate")
            return None

        async def close(self):
            pass

    async def _dies(self, agent, prompt, *, message_history, deps, new_messages):
        log.append("agent")
        raise RuntimeError("agent stream died")
        yield  # pragma: no cover

    monkeypatch.setattr(ExecutionContext, "stream", _dies)

    emissions = _run(_turn(), _surface(background=_Pass))

    assert log == ["agent", "gate"]
    assert emissions == [TextEmission(_TROUBLE_GU)]
    assert _Outcomes.seen[-1] == ("outcome", "success")


def test_a_blank_first_chunk_does_not_stop_the_nudge(monkeypatch):
    patch_turn(monkeypatch)
    log = []

    class _Liveness:
        def __init__(self, turn, *, started_at, send, is_stale):
            pass

        async def stop(self):
            log.append("stopped")

    class _Sender:
        async def send(self, emission):
            pass

    async def _stream(self, agent, prompt, *, message_history, deps, new_messages):
        yield " "
        yield "Give clean water daily."

    monkeypatch.setattr(ExecutionContext, "stream", _stream)

    async def _drain():
        async for emission in chat_service.run_turn(
            _turn("en"), _surface(liveness=_Liveness), scheduler=_Scheduler(), side_channel=_Sender(),
        ):
            log.append(emission.text)

    asyncio.run(asyncio.wait_for(_drain(), timeout=10))

    assert log == [" ", "stopped", "Give clean water daily."]
