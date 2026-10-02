"""Voice's five pre-turn short-circuits, running as classifiers on the seam.

The detection cases are voice-oan-api's own (tests/test_voice_fixes.py and
tests/test_voice_regressions_apr11_12.py), carried over unchanged. The rest pin
what each short-circuit answers, what it writes to history, when it stands
down, and that ``run_turn`` runs them in voice's order.
"""
import asyncio
import contextlib
import dataclasses
import os
from types import MappingProxyType, SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest
from pydantic_ai.messages import ModelRequest, TextPart, UserPromptPart

from app.channels.chat import WEB
from app.config import settings
from app.services import chat as chat_service
from app.turn.types import TelephonyCall, TextEmission, Turn
from app.voice import classifiers as vc
from app.voice import stt_signals
from app.voice.classifiers import voice_classifiers
from app.voice.stt_signals import detect_stt_signal
from tests.test_chat_turn_contract import patch_turn

_CALL = TelephonyCall(process_id="proc-1")
_CONSENT_CALL = TelephonyCall(process_id="proc-1", outbound_consent_turn=True)

_GREETING_GU = "નમસ્તે, હું સરલાબેન છું. તમારા પશુ વિશે કોઈ સમસ્યા હોય તો મને જણાવો."
_GREETING_EN = "Hello, I am Sarlaben. Please tell me what issue you are facing with your animal."
_FRAGMENT_GU = "મને તમારો પ્રશ્ન સમજાયો નથી. કૃપા કરીને તમારો પ્રશ્ન ફરીથી પૂછો."
_FRAGMENT_EN = "I could not understand your question. Please ask your question again."
_STT_FINAL_HISTORY = "Sorry, I still could not hear you clearly. Please try again later."


class _Render:
    """Stands in for voice's translation to the caller's language."""

    def __init__(self):
        self.calls = []

    async def __call__(self, text_en, target_lang):
        self.calls.append((text_en, target_lang))
        return f"<{target_lang}> {text_en}"


def _turn(query, *, history=(), target_lang="gu", call=_CALL):
    return Turn(
        query=query,
        session_id="call-1",
        source_lang="gu",
        target_lang=target_lang,
        user_id="anonymous",
        authenticated_user=MappingProxyType({}),
        history=tuple(history),
        history_session_id="call-1",
        channel=WEB,
        persona="farmer",
        call=call,
    )


def _answer(turn, render=None):
    """Voice's chain the way run_turn walks it: the first match answers."""

    async def _walk():
        for classify in voice_classifiers(render or _Render()):
            result = await classify(turn)
            if result is not None:
                return result
        return None

    return asyncio.run(_walk())


def _pair_texts(result):
    return [part.content for message in result.history_pair for part in message.parts]


def _exchange(user_text, assistant_text):
    return [
        ModelRequest(parts=[UserPromptPart(content=user_text)]),
        SimpleNamespace(parts=[TextPart(content=assistant_text)]),
    ]


# ── detection, carried over from voice-oan-api's tests ─────────────────────


class TestGreetingDetection:
    @pytest.mark.parametrize("query", [
        "hello", "Hello", "HELLO", "hi", "hey", "hlo", "હલો", "હેલો", "નમસ્તે",
        "नमस्ते", "हेलो", "namaste", "halo",
    ])
    def test_basic_greetings(self, query):
        assert vc._is_bare_greeting(query) is True

    @pytest.mark.parametrize("query", ["hello hello", "hello hello hello", "હલો હલો", "hlo hlo", "hi hi"])
    def test_repeated_greetings(self, query):
        assert vc._is_bare_greeting(query) is True

    @pytest.mark.parametrize("query", ["ha hello", "હા હલો", "ji", "જી", "bolo", "બોલો", "ha bolo", "હા બોલો"])
    def test_greeting_combos(self, query):
        assert vc._is_bare_greeting(query) is True

    @pytest.mark.parametrize("query", ["hello!", "hello.", "  hello  ", "* hello *"])
    def test_greetings_with_punctuation(self, query):
        assert vc._is_bare_greeting(query) is True

    @pytest.mark.parametrize("query", [
        "hello my cow is sick",
        "મારી ગાયને તાવ છે",
        "hello can you help me with my buffalo",
        "હલો મારી ભેંસ",
        "namaste, meri bhains ko bukhar hai",
    ])
    def test_not_bare_greetings(self, query):
        assert vc._is_bare_greeting(query) is False

    def test_affirmative_is_not_treated_as_bare_greeting(self):
        assert vc._is_bare_greeting("હા") is False


class TestFragmentDetection:
    @pytest.mark.parametrize("query", ["", " ", "*", "* *", "ઓ", "બ", "b", "O", "ok", "હા"])
    def test_fragments(self, query):
        assert vc._is_fragment_query(query) is True

    @pytest.mark.parametrize("query", ["મારી ગાય", "hello", "cow is sick", "દૂધ ઓછું", "ભેંસ બીમાર છે"])
    def test_not_fragments(self, query):
        assert vc._is_fragment_query(query) is False


class TestHoldMessageDetection:
    @pytest.mark.parametrize("query", [
        "તેને તમારો કોલ હોલ્ડ પર રાખ્યા છે કૃપા કરી લાઇન પર રહો",
        "તમારો કોલ હોલ્ડ પર રાખ્યા છે કૃપા કરી લાઈન પર રહો",
        "Your call has been put on hold, please stay on the line.",
        "The person you are speaking with has put your call on hold. Please stay on the line.",
        "The person you called has put your call on hold. Please remain on the line.",
        "They have put your call on hold.",
        "હોલ્ડ પર રાખ્યા છે",
        "કોલ હોલ્ડ પર",
        "put your call on hold please stay on the line",
        "આપ કે કોલ કો હોલ્ડ પર રખા હૈ કૃપયા લાઇન પર બને રહી",
        "તમારો કોલ હોલ્ડ પર રાખ્યો છે કૃપા કરીને લાઇન પર રહો",
        "Please stay on the line, your call has been put on hold.",
    ])
    def test_hold_messages_detected(self, query):
        assert vc._is_hold_message(query)

    @pytest.mark.parametrize("query", [
        "મારી ગાયને તાવ આવે છે",
        "My cow is not eating properly",
        "hello",
        "દૂધમાં ફેટ ઓછી છે",
        "I want to book AI for my cow",
        "How long can I hold the calf's milk?",
    ])
    def test_normal_queries_not_detected(self, query):
        assert not vc._is_hold_message(query)

    def test_telephony_terminate_token_stays_goodbye(self):
        assert vc.TELEPHONY_TERMINATE_CALL_TOKEN["gu"] == "Goodbye."
        assert vc.TELEPHONY_TERMINATE_CALL_TOKEN["en"] == "Goodbye."


class TestSttSignalDetection:
    @pytest.mark.parametrize("query", ["*No audio/User is speaking softly*", "No audio/User is speaking softly"])
    def test_no_audio_signal_detected(self, query):
        assert detect_stt_signal(query) == "No audio/User is speaking softly"

    @pytest.mark.parametrize("query", ["*Unclear Speech*", "Unclear Speech"])
    def test_unclear_speech_signal_detected(self, query):
        assert detect_stt_signal(query) == "Unclear Speech"

    @pytest.mark.parametrize("query", ["hello", "મારી ગાયને તાવ છે", "call has been put on hold"])
    def test_normal_queries_are_not_signals(self, query):
        assert detect_stt_signal(query) is None


def test_meaningful_history_detected():
    history = [ModelRequest(parts=[UserPromptPart(content="મારી ગાયને તાવ છે")])]
    assert vc._has_meaningful_history(history) is True


def test_stt_only_history_not_treated_as_meaningful():
    history = [ModelRequest(parts=[UserPromptPart(content="*No audio/User is speaking softly*")])]
    assert vc._has_meaningful_history(history) is False


# ── the chain ──────────────────────────────────────────────────────────────


def test_the_chain_is_voices_five_in_voices_order():
    names = [getattr(c, "func", c).__name__ for c in voice_classifiers(_Render())]
    assert names == [
        "_stt_signal_classifier",
        "_hold_message_classifier",
        "_greeting_classifier",
        "_identity_classifier",
        "_fragment_classifier",
    ]


def test_a_short_greeting_is_a_greeting_not_a_fragment():
    assert _answer(_turn("hi")).label == "greeting_fast_path"


def test_a_real_question_reaches_the_agent():
    render = _Render()
    assert _answer(_turn("મારી ગાયને તાવ છે"), render) is None
    assert render.calls == []


# ── STT signal ─────────────────────────────────────────────────────────────


def test_no_audio_asks_to_repeat_and_keeps_a_marker_in_history():
    result = _answer(_turn("*No audio/User is speaking softly*"))

    assert result.label == "stt_signal"
    assert result.canned_text in stt_signals._FALLBACK_NO_AUDIO["gu"]
    assert _pair_texts(result) == ["[stt:no-audio]", _FRAGMENT_EN]
    assert result.raw is False


def test_unclear_speech_has_its_own_marker():
    result = _answer(_turn("Unclear Speech", target_lang="en"))

    assert result.canned_text in stt_signals._FALLBACK_UNCLEAR["en"]
    assert _pair_texts(result) == ["[stt:unclear-speech]", _FRAGMENT_EN]


def test_stt_reads_the_callers_language_the_way_voice_does():
    result = _answer(_turn("Unclear Speech", target_lang=" GU "))

    assert result.canned_text in stt_signals._FALLBACK_UNCLEAR["gu"]


def test_the_third_stt_signal_in_a_row_says_call_back_later():
    history = [
        *_exchange("[stt:no-audio]", _FRAGMENT_EN),
        *_exchange("[stt:no-audio]", _FRAGMENT_EN),
    ]

    result = _answer(_turn("No audio/User is speaking softly", history=history))

    assert result.canned_text in stt_signals._FINAL_NO_AUDIO["gu"]
    assert _pair_texts(result) == ["[stt:no-audio]", _STT_FINAL_HISTORY]


def test_a_real_turn_between_stt_signals_resets_the_count():
    history = [
        *_exchange("[stt:no-audio]", _FRAGMENT_EN),
        *_exchange("મારી ગાયને તાવ છે", "Please call a vet."),
        *_exchange("[stt:no-audio]", _FRAGMENT_EN),
    ]

    result = _answer(_turn("No audio/User is speaking softly", history=history))

    assert result.canned_text in stt_signals._FALLBACK_NO_AUDIO["gu"]


def test_the_retry_ceiling_is_configurable(monkeypatch):
    monkeypatch.setattr(settings, "stt_signal_retry_ceiling", 1)

    result = _answer(_turn("Unclear Speech"))

    assert result.canned_text in stt_signals._FINAL_UNCLEAR["gu"]


def test_stt_signal_answers_even_mid_call_and_on_the_consent_turn():
    history = _exchange("મારી ગાયને તાવ છે", "Please call a vet.")

    result = _answer(_turn("Unclear Speech", history=history, call=_CONSENT_CALL))

    assert result.label == "stt_signal"


# ── hold message ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("target_lang", ["gu", "en", "hi"])
def test_hold_message_hangs_up_with_the_exact_token_and_persists_nothing(target_lang):
    result = _answer(_turn("તમારો કોલ હોલ્ડ પર રાખ્યો છે", target_lang=target_lang))

    assert result.label == "hold_message"
    assert result.canned_text == "Goodbye."
    assert result.raw is True, "the Gujarati normalizer would leave only '.'"
    assert result.history_pair is None


def test_hold_message_answers_even_mid_call_and_on_the_consent_turn():
    history = _exchange("મારી ગાયને તાવ છે", "Please call a vet.")

    result = _answer(_turn("please stay on the line", history=history, call=_CONSENT_CALL))

    assert result.label == "hold_message"


# ── greeting ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("target_lang, spoken", [("gu", _GREETING_GU), (" GU ", _GREETING_GU), ("", _GREETING_GU), ("en", _GREETING_EN)])
def test_greeting_uses_the_canned_line_without_translating(target_lang, spoken):
    render = _Render()

    result = _answer(_turn("hello", target_lang=target_lang), render)

    assert result.label == "greeting_fast_path"
    assert result.canned_text == spoken
    assert render.calls == []


def test_greeting_in_a_language_without_a_canned_line_is_rendered_from_english():
    render = _Render()

    result = _answer(_turn("namaste", target_lang="hi"), render)

    assert render.calls == [(_GREETING_EN, "hi")]
    assert result.canned_text == f"<hi> {_GREETING_EN}"


def test_greeting_history_stays_in_english():
    result = _answer(_turn("હલો"))

    assert _pair_texts(result) == ["hello", _GREETING_EN]


def test_greeting_after_only_stt_noise_still_greets():
    history = [ModelRequest(parts=[UserPromptPart(content="*No audio/User is speaking softly*")])]

    assert _answer(_turn("hello", history=history)).label == "greeting_fast_path"


def test_greeting_without_call_details_still_greets():
    assert _answer(_turn("hello", call=None)).label == "greeting_fast_path"


def test_affirmative_with_meaningful_history_goes_to_the_agent():
    history = _exchange("ગાય માટે કે ભેંસ માટે?", "ગાય માટે કે ભેંસ માટે?")

    assert _answer(_turn("હા", history=history)) is None
    assert _answer(_turn("hello", history=history)) is None


def test_the_consent_reply_is_left_to_the_consent_gate():
    """"ha bolo" is also a greeting token: answering it with the greeting would
    swallow the farmer's yes to the outbound readout."""
    assert _answer(_turn("ha bolo", call=_CONSENT_CALL)) is None


# ── identity ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("query, target_lang", [
    ("What is your name?", "en"),
    ("who are you", "en"),
    ("તમારું નામ શું છે?", "gu"),
    ("તમે કોણ છો", "gu"),
])
def test_identity_renders_sarlabens_line_for_the_caller(query, target_lang):
    render = _Render()

    result = _answer(_turn(query, target_lang=target_lang), render)

    assert result.label == "identity_fast_path"
    assert render.calls == [(vc._IDENTITY_RESPONSE_EN, target_lang)]
    assert result.canned_text == f"<{target_lang}> {vc._IDENTITY_RESPONSE_EN}"
    assert _pair_texts(result) == ["hello", vc._IDENTITY_RESPONSE_EN]


def test_identity_line_carries_the_configured_creation_date():
    assert "Sarlaben" in vc._IDENTITY_RESPONSE_EN
    assert settings.voice_profile_creation_date_words in vc._IDENTITY_RESPONSE_EN


def test_identity_with_more_to_say_goes_to_the_agent():
    assert _answer(_turn("who are you, and can you help with my cow")) is None


@pytest.mark.parametrize("history, call", [
    (_exchange("મારી ગાયને તાવ છે", "Please call a vet."), _CALL),
    ((), _CONSENT_CALL),
])
def test_identity_stands_down_mid_call_and_on_the_consent_turn(history, call):
    render = _Render()

    assert _answer(_turn("who are you", history=history, call=call), render) is None
    assert render.calls == []


# ── fragment ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("query", ["ok", "", "*", "ઓ"])
def test_fragment_asks_to_repeat(query):
    result = _answer(_turn(query))

    assert result.label == "fragment_fast_path"
    assert result.canned_text == _FRAGMENT_GU
    assert _pair_texts(result) == ["[fragment]", _FRAGMENT_EN]


def test_fragment_in_a_language_without_a_canned_line_is_rendered_from_english():
    render = _Render()

    result = _answer(_turn("ok", target_lang="hi"), render)

    assert render.calls == [(_FRAGMENT_EN, "hi")]
    assert result.canned_text == f"<hi> {_FRAGMENT_EN}"


@pytest.mark.parametrize("history, call", [
    (_exchange("મારી ગાયને તાવ છે", "Please call a vet."), _CALL),
    ((), _CONSENT_CALL),
])
def test_fragment_stands_down_mid_call_and_on_the_consent_turn(history, call):
    assert _answer(_turn("ok", history=history, call=call)) is None


# ── on the seam ────────────────────────────────────────────────────────────


class _Scheduler:
    def schedule(self, fn, /, *args):
        pass


class _Labels:
    """Records the label run_turn hands the surface's telemetry."""

    seen = []

    def __init__(self, turn, *, pipeline_profile, pipeline_trace):
        pass

    @contextlib.contextmanager
    def root(self):
        yield

    def record_output(self, text, label):
        _Labels.seen.append(label)

    def record_outcome(self, outcome):
        pass


def _voice_surface(render=None):
    _Labels.seen = []
    return dataclasses.replace(
        chat_service.CHAT_SURFACE,
        classifiers=voice_classifiers(render or _Render()),
        telemetry=_Labels,
    )


def _run(turn, surface):
    async def _collect():
        return [e async for e in chat_service.run_turn(turn, surface, scheduler=_Scheduler())]

    return asyncio.run(_collect())


def test_run_turn_answers_a_greeting_and_writes_voices_history(monkeypatch):
    seen = patch_turn(monkeypatch)

    emissions = _run(_turn("hello"), _voice_surface())

    assert emissions == [TextEmission(_GREETING_GU)]
    assert _Labels.seen == ["greeting_fast_path"]
    ((key, messages),) = seen["history_writes"]
    assert key == "call-1"
    assert [p.content for m in messages for p in m.parts] == ["hello", _GREETING_EN]
    assert seen["deps"] == [], "the agent ran on a short-circuited turn"


def test_run_turn_hangs_up_on_a_hold_message_raw_and_without_history(monkeypatch):
    seen = patch_turn(monkeypatch)

    emissions = _run(_turn("please remain on the line"), _voice_surface())

    assert emissions == [TextEmission("Goodbye.", raw=True)]
    assert _Labels.seen == ["hold_message"]
    assert seen["history_writes"] == []


def test_run_turn_hands_an_ordinary_question_to_the_agent(monkeypatch):
    seen = patch_turn(monkeypatch)

    _run(_turn("મારી ગાયને તાવ છે"), _voice_surface())

    assert len(seen["deps"]) == 1
    assert _Labels.seen == ["final"]


def test_repeated_stt_failures_hit_the_retry_ceiling_on_the_third_attempt(monkeypatch):
    """voice-oan-api's multi-turn case, with history carried from turn to turn."""
    seen = patch_turn(monkeypatch)
    history = ()
    spoken = []

    for _ in range(3):
        emissions = _run(_turn("No audio/User is speaking softly", history=history), _voice_surface())
        spoken.append(emissions[0].text)
        history = tuple(seen["history_writes"][-1][1])

    assert spoken[0] in stt_signals._FALLBACK_NO_AUDIO["gu"]
    assert spoken[1] in stt_signals._FALLBACK_NO_AUDIO["gu"]
    assert spoken[2] in stt_signals._FINAL_NO_AUDIO["gu"]
    history_text = " ".join(
        part.content for message in history for part in message.parts if isinstance(part.content, str)
    )
    assert "[stt:no-audio]" in history_text
    assert "No audio/User is speaking softly" not in history_text


def test_a_turn_built_without_call_details_has_none():
    turn = Turn(
        query="hello",
        session_id="chat",
        source_lang="gu",
        target_lang="gu",
        user_id="anonymous",
        authenticated_user=MappingProxyType({}),
        history=(),
        history_session_id="chat",
        channel=WEB,
        persona="farmer",
    )
    assert turn.call is None
    assert TelephonyCall(process_id=None).outbound_consent_turn is False
