"""Voice's pretranslation: the caller's words into the English the agent reads.

Voice's own cases for the prompt, the conversation it quotes and the glossary
fixups come first, then the step with the tiers ``llm_core`` picks, then the
step on the seam.
"""
import asyncio
import contextlib
import dataclasses
import json
import os
from types import MappingProxyType, SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest
from openai import AsyncOpenAI
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from agents.tools.terms import get_ambiguity_hints_for_query
from app.channels.chat import WEB
from app.llm_core import config_source, runtime
from app.llm_core.config_model import NamedProfile, PipelineConfig, Provider, Step, StepConfig, Tier
from app.llm_core.execution import ExecutionContext
from app.services import chat as chat_service
from app.turn.types import ClassifierResult, Pretranslated, TelephonyCall, TextEmission, Turn
from app.voice import background as voice_bg
from app.voice import pretranslation as vp
from app.voice.classifiers import _FRAGMENT_RESPONSES
from app.voice.history import HISTORY_MARKERS
from app.voice.moderation import ModerationVerdict
from app.voice.non_meaningful import NonMeaningfulVerdict
from tests.test_chat_turn_contract import patch_turn

SPECIES_Q = "Is this for a cow or a buffalo?"
TECH_Q = (
    "Sangitaben, which technician should I book with? I can book with "
    "Anilbhai Galjibhai Pandar, Narayanbhai Dhulabhai Patel, or "
    "Narendrakumar Narayandas Pandor."
)
CONTEXT = "The assistant's last question was"
RULE = "translate it as the option it clearly corresponds to"
SPECIES_HINT = "Common ASR corruptions"


def _conv(prev):
    return [("Assistant", prev)] if prev is not None else None


def _system(text, prev=None, conversation=None):
    if conversation is None:
        conversation = _conv(prev)
    return vp._build_openai_pretranslation_messages("Gujarati", "gu", text, conversation)[0]["content"]


def _exchange(user, *assistant_parts):
    return [
        ModelRequest(parts=[UserPromptPart(content=user)]),
        ModelResponse(parts=list(assistant_parts)),
    ]


def _q(text):
    """How the agent path stores the farmer's translated turn."""
    return '**User:** "' + text + '"'


# ── voice's own cases: the conversation the translator sees ─────────────────


@pytest.mark.parametrize("prev", [
    SPECIES_Q,
    TECH_Q,
    "Booking artificial insemination for your cow with Mayurbhai Naranbhai Patel. Shall I confirm?",
    "Vikrambhai, what main symptom are you seeing in your animal?",
    "Please wait, I am checking.",
])
def test_previous_turn_is_quoted_whatever_it_asked(prev):
    system = _system("બે બે ચાર ભાઈ", prev)
    assert CONTEXT in system
    assert prev in system
    assert RULE in system


@pytest.mark.parametrize("prev", [None, "", "   "])
def test_no_previous_turn_adds_nothing(prev):
    assert CONTEXT not in _system("બે બે ચાર ભાઈ", prev)


def test_technician_names_reach_the_translator_verbatim():
    system = _system("પંડોર નરેનભાઈ", TECH_Q)
    for name in ("Anilbhai Galjibhai Pandar", "Narayanbhai Dhulabhai Patel", "Narendrakumar Narayandas Pandor"):
        assert name in system
    assert "spelled EXACTLY as that option appears above" in system


def test_rule_refuses_to_guess_between_two_options():
    system = _system("પંડોર નરેનભાઈ નારાયણભાઈ", TECH_Q)
    assert "fits TWO of the options, or none, do NOT choose" in system
    assert "acting on the wrong option is worse than re-asking" in system


def test_species_hint_present_after_the_species_question():
    assert SPECIES_HINT in _system("ગેસ માટે", SPECIES_Q)


@pytest.mark.parametrize("prev", [
    TECH_Q,
    "Vikrambhai, what main symptom are you seeing in your animal?",
    "A cow and a buffalo both need clean water.",
    None,
])
def test_species_hint_absent_otherwise(prev):
    assert SPECIES_HINT not in _system("ગેસ માટે", prev)


def test_general_conservative_rules_survive():
    for prev in (SPECIES_Q, TECH_Q, "Please wait, I am checking."):
        system = _system("ગેસ માટે", prev)
        assert "Do not infer animal species" in system
        assert "Do not repair missing words" in system
        assert "do NOT invent a meaning" in system
    assert "still say 'unclear animal'" in _system("ગેસ માટે", SPECIES_Q)


def test_user_message_still_carries_only_the_utterance():
    messages = vp._build_openai_pretranslation_messages("Gujarati", "gu", " પંડોર ", _conv(TECH_Q))
    assert messages[1]["content"] == "પંડોર"


def test_rule_targets_the_last_assistant_turn_not_an_earlier_one():
    conversation = [("Assistant", SPECIES_Q), ("Farmer", "buffalo"), ("Assistant", TECH_Q)]
    system = _system("ગેસ માટે", conversation=conversation)
    assert f'last question was: "{TECH_Q}"' in system
    assert SPECIES_HINT not in system


def test_session_601f1db4_repeat_requests_no_longer_hide_the_species_question():
    history = [
        *_exchange("hello", TextPart(content="Hello, I am Sarlaben.")),
        *_exchange(_q("I want to do AI booking"), TextPart(content=SPECIES_Q)),
        *_exchange(_q("[unclear token] [unclear token]"),
                   TextPart(content="Please repeat that once. I did not understand you clearly.")),
        *_exchange("[stt:no-audio]",
                   TextPart(content="I could not understand your question. Please ask your question again.")),
    ]
    context = vp._pretranslation_context(history)
    assert context == [
        ("Assistant", "Hello, I am Sarlaben."),
        ("Farmer", "I want to do AI booking"),
        ("Assistant", SPECIES_Q),
    ]
    assert SPECIES_HINT in _system("પસ માટે", conversation=context)


def test_runtime_context_never_reaches_the_translator():
    history = [
        ModelRequest(parts=[UserPromptPart(content="Runtime context for this turn:\n- Union code: 2021")]),
        *_exchange(_q("hi"), TextPart(content="How can I help?")),
    ]
    assert all("Union code" not in text for _, text in vp._pretranslation_context(history))


def test_unclear_farmer_turn_drops_only_the_farmer_side():
    heat_q = "Dhanabhai, how many months ago did the animal last come in heat?"
    history = [
        *_exchange(_q("I want AI"), TextPart(content=SPECIES_Q)),
        *_exchange(_q("that heifer [unclear token]"), TextPart(content=heat_q)),
    ]
    assert vp._pretranslation_context(history)[-1] == ("Assistant", heat_q)


# ── voice's own cases: the prompt and the glossary ──────────────────────────


def test_pretranslation_prompt_preserves_uncertainty():
    prompt = _system("કા પણ બેની કઈ દોરણ ખાવડાવું જોઈએ")
    assert "faithful pretranslation" in prompt
    assert "Preserve uncertainty" in prompt
    assert "Do not infer animal species" in prompt
    assert "unclear animal" in prompt
    assert "Never convert a doubtful token into a specific medicine, feed, disease, animal species, or service term" in prompt


def test_pretranslation_prompt_does_not_turn_address_words_into_caller_gender():
    prompt = _system("બેન મારી ભેંસને તાવ છે")
    assert "Kinship words" in prompt
    assert "Do not turn them into the caller's gender" in prompt
    assert "address marker" in prompt


@pytest.mark.parametrize("query", ["મારી ભેસ્ટને તાવ છે", "ભંચ દૂધ ઓછું આપે છે", "ભેંચને ખાવાનું બંધ છે"])
def test_pretranslation_prompt_maps_buffalo_asr_variants(query):
    prompt = _system(query)
    assert "Domain-specific disambiguation rules" in prompt
    assert "mean buffalo" in prompt.lower()
    assert "NOT sheep" in prompt
    assert "Translate these as Buffalo" in prompt


def test_pretranslation_prompt_requires_glossary_labels_over_transliteration():
    prompt = _system("મારે જિજ્ઞાસા વિશે પૂછવું છે")
    assert "જિજ્ઞાસા = Curiosity" in prompt
    assert "right-hand English label" in prompt
    assert "Do not output the romanized/transliterated form" in prompt


def test_pretranslation_prompt_includes_beech_daan_disambiguation():
    prompt = _system("મારી ગાય માટે બીજ દાન બુક કરાવવું છે")
    assert "Domain-specific disambiguation rules" in prompt
    assert "artificial insemination" in prompt.lower()
    assert "Glossary usage rule" in prompt


def test_the_translator_gets_voices_ambiguity_rules_not_chats():
    """Voice reads an ASR'd નિદાન as insemination, not diagnosis; chat's rules do
    not have that, so voice's prompt must come from voice's own."""
    query = "મારે નિદાન કરાવવું છે"
    assert "Insemination vs diagnosis" in _system(query)
    assert "Insemination vs diagnosis" not in get_ambiguity_hints_for_query(query, include_ask=False)


def test_the_matcher_takes_the_rules_it_is_given():
    rules = [{"gu_terms": ["ઝઝઝ"], "rule": "ઝઝઝ is a test term."}]
    assert get_ambiguity_hints_for_query("આ ઝઝઝ છે", terms=rules) == "- ઝઝઝ is a test term."
    assert get_ambiguity_hints_for_query("આ ઝઝઝ છે", terms=[]) == ""


def test_beech_daan_hints_omitted_for_unrelated_query():
    hints = get_ambiguity_hints_for_query("મારી ગાયને તાવ છે")
    assert "beech daan" not in hints.lower()


def test_gujarati_glossary_hints_skip_empty_transliteration_matches():
    hints = vp._get_glossary_hints_for_gu_query("મારી ભેસ્ટને તાવ છે")
    assert "Buffalo" in hints
    assert "Fever" in hints
    for unrelated in ("Acaricide", "Deworming", "Pesticide", "Pre-Partum Prolapse"):
        assert unrelated not in hints


def test_gujarati_glossary_hints_include_short_buffalo_variant():
    assert "Buffalo" in vp._get_glossary_hints_for_gu_query("ભંચ")


@pytest.mark.parametrize("query, expected", [
    ("મારી ગાય માટે બીજ દાન બુક કરાવવું છે", "Insemination"),
    ("મારી ગાયને બીજદાન કરાવવાનું છે", "Insemination"),
    ("કૃત્રિમ બીજદાન બુક કરવું છે", "insemination"),
])
def test_glossary_hints_surface_insemination_terms(query, expected):
    assert expected in vp._get_glossary_hints_for_gu_query(query, max_results=20)


def test_glossary_hints_do_not_fire_for_unrelated_bija():
    assert "insemination" not in vp._get_glossary_hints_for_gu_query("મારે બીજા વિષય વિશે પૂછવું છે", max_results=20).lower()


def test_pretranslation_replaces_exact_glossary_transliteration():
    translated = vp._apply_exact_glossary_transliteration_replacements(
        "મારે જિજ્ઞાસા વિશે પૂછવું છે", "I want to ask about Jignasa"
    )
    assert translated == "I want to ask about Curiosity"


def test_pretranslation_glossary_transliteration_replacement_requires_source_term():
    translated = vp._apply_exact_glossary_transliteration_replacements(
        "મારે બીજા વિષય વિશે પૂછવું છે", "I want to ask about Jignasa"
    )
    assert translated == "I want to ask about Jignasa"


def test_extract_translation_from_raw_json():
    assert vp._extract_translation_from_raw('{"translation": "the cow has fever", "confidence": "low"}') == "the cow has fever"


# ── the step, on the tiers llm_core picks ───────────────────────────────────


def _turn(query="મારી ગાયને તાવ છે", source_lang="gu", target_lang="gu", history=()):
    return Turn(
        query=query,
        session_id="call-1",
        source_lang=source_lang,
        target_lang=target_lang,
        user_id="anonymous",
        authenticated_user=MappingProxyType({}),
        history=tuple(history),
        history_session_id="call-1",
        channel=WEB,
        persona="farmer",
        call=TelephonyCall(process_id="proc-1"),
    )


_OSS = Tier(provider=Provider.VLLM, model="gemma", endpoint="http://oss:8020/v1")
_MANAGED = Tier(provider=Provider.OPENAI, model="gpt-mini")


def _execution(*tiers):
    step = StepConfig(tiers=list(tiers or (_OSS, _MANAGED)))
    config = PipelineConfig(
        profiles=[NamedProfile(name="oss", weight=100, steps={Step.AGENT: step, Step.PRE_TRANSLATION: step})],
        fallback_enabled=True,
    )
    return ExecutionContext(session_id="call-1", config=config, profile_name="oss")


class _Background:
    """Stands in for voice's background set: records what history keeps and
    answers ``decline`` as told."""

    def __init__(self, declined=None):
        self.history_text = None
        self.declined = declined
        self.asked_to_decline = 0

    def set_history_text(self, text):
        self.history_text = text

    async def decline(self):
        self.asked_to_decline += 1
        return self.declined


async def _render(text_en, target_lang):
    return f"<{target_lang}> {text_en}"


def _pretranslate(monkeypatch, turn, *, replies=(), background=None, tiers=()):
    """Runs voice's step with each attempt answered from ``replies`` in order;
    an exception there is raised by that attempt."""
    calls = []
    answers = list(replies)

    async def _attempt(text, source_lang, **kw):
        calls.append(dict(kw, text=text, source_lang=source_lang))
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(vp, "_translate_to_english_pretranslation", _attempt)
    step = vp.voice_pretranslation(_render)
    result = asyncio.run(step(turn, execution=_execution(*tiers), background=background))
    return result, calls


def test_english_is_asked_as_it_came(monkeypatch):
    background = _Background()

    result, calls = _pretranslate(monkeypatch, _turn("My cow has fever", source_lang="en"), background=background)

    assert result == Pretranslated(query="My cow has fever", lang="en")
    assert calls == []
    assert background.history_text == "My cow has fever"


def test_gujarati_is_translated_on_the_oss_tier_with_the_conversation(monkeypatch):
    background = _Background()
    history = _exchange(_q("I want AI"), TextPart(content=SPECIES_Q))

    result, calls = _pretranslate(
        monkeypatch, _turn("ભેંસ", history=history), replies=["For a buffalo"], background=background
    )

    assert result == Pretranslated(query="For a buffalo", lang="en")
    (call,) = calls
    assert call["text"] == "ભેંસ" and call["source_lang"] == "gu"
    assert isinstance(call["client"], AsyncOpenAI)
    assert str(call["client"].base_url).startswith("http://oss:8020/v1")
    assert call["model"] == "gemma"
    assert call["label"] == "OSS vLLM"
    assert call["translation_provider_label"] == "vllm"
    assert call["extra_metadata"] == {"pipeline_profile": "oss"}
    assert call["conversation"] == [("Farmer", "I want AI"), ("Assistant", SPECIES_Q)]
    assert background.history_text == "For a buffalo"


def test_a_failed_oss_tier_falls_back_to_the_managed_one(monkeypatch):
    result, calls = _pretranslate(
        monkeypatch, _turn(), replies=[TimeoutError("OSS vLLM pretranslation timed out"), "My cow has fever"]
    )

    assert result == Pretranslated(query="My cow has fever", lang="en")
    assert [c["label"] for c in calls] == ["OSS vLLM", "OpenAI"]
    assert calls[1]["model"] == "gpt-mini"
    assert calls[1]["translation_provider_label"] == "openai"
    assert "extra_metadata" not in calls[1]


def test_nothing_usable_asks_the_caller_to_repeat(monkeypatch):
    background = _Background()

    result, _ = _pretranslate(
        monkeypatch, _turn(), replies=[TimeoutError("down"), ConnectionError("down")], background=background
    )

    assert isinstance(result, ClassifierResult)
    assert result.label == "pretranslation_empty"
    assert result.canned_text == _FRAGMENT_RESPONSES["gu"]
    assert not result.raw
    user, assistant = result.history_pair
    assert user.parts[0].content == HISTORY_MARKERS["pretranslation_failed"]
    assert assistant.parts[0].content == _FRAGMENT_RESPONSES["en"]
    assert background.history_text == HISTORY_MARKERS["pretranslation_failed"]
    assert background.asked_to_decline == 1


def test_a_tier_that_answers_nothing_leaves_the_unclear_marker(monkeypatch):
    background = _Background()

    result, _ = _pretranslate(monkeypatch, _turn(), replies=[""], background=background)

    assert result.label == "pretranslation_empty"
    user, _ = result.history_pair
    assert user.parts[0].content == HISTORY_MARKERS["low_confidence"]
    assert background.history_text == HISTORY_MARKERS["low_confidence"]


def test_voice_never_hands_its_call_a_client_that_is_not_openai(monkeypatch):
    """Voice's call is chat.completions. On a tier that only has a native client
    it fails there, and the caller is asked to repeat."""
    result, calls = _pretranslate(
        monkeypatch, _turn(), replies=["My cow has fever"], tiers=(Tier(provider=Provider.ANTHROPIC, model="haiku"),)
    )

    assert calls == []
    assert result.label == "pretranslation_empty"


def test_a_caller_language_without_a_canned_line_gets_it_rendered(monkeypatch):
    result, _ = _pretranslate(
        monkeypatch, _turn(target_lang="hi"), replies=[TimeoutError("down"), ConnectionError("down")]
    )

    assert result.canned_text == f"<hi> {_FRAGMENT_RESPONSES['en']}"


def test_a_rejected_query_is_declined_rather_than_asked_again(monkeypatch):
    decline = ClassifierResult(canned_text="Please ask about your animals.", label="moderation_rejected")

    result, _ = _pretranslate(
        monkeypatch, _turn(), replies=[TimeoutError("down"), ConnectionError("down")],
        background=_Background(declined=decline),
    )

    assert result is decline


def test_an_empty_translation_keeps_the_callers_words(monkeypatch):
    """Voice's call gives the caller's own words back when the model says nothing,
    so the agent still gets them."""
    client = _Client("")
    result = asyncio.run(vp._translate_to_english_pretranslation(
        "મારી ગાયને તાવ છે", "gu", client=client, model="gemma", label="OSS vLLM", translation_provider_label="vllm",
    ))

    assert result == "મારી ગાયને તાવ છે"


class _Client:
    """An OpenAI-compatible client whose chat completion answers ``content``."""

    def __init__(self, content):
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self._content = content

    async def _create(self, **request):
        self.requests.append(request)
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_the_call_asks_for_json_and_fixes_a_transliterated_glossary_term():
    client = _Client(json.dumps({"translation": "I want to ask about Jignasa"}))

    result = asyncio.run(vp._translate_to_english_pretranslation(
        "મારે જિજ્ઞાસા વિશે પૂછવું છે", "gu", client=client, model="gemma",
        label="OSS vLLM", translation_provider_label="vllm", conversation=_conv(TECH_Q),
    ))

    assert result == "I want to ask about Curiosity"
    (request,) = client.requests
    assert request["model"] == "gemma"
    assert request["response_format"] == {"type": "json_object"}
    assert CONTEXT in request["messages"][0]["content"]


# ── llm_core: voice's pretranslation is a bare OpenAI call ──────────────────


def _pretranslation_config(*tiers):
    return PipelineConfig(
        profiles=[NamedProfile(name="p", weight=100, steps={
            Step.AGENT: StepConfig(tiers=[_MANAGED]),
            Step.PRE_TRANSLATION: StepConfig(tiers=list(tiers)),
        })],
    )


def test_pretranslation_on_anthropic_stays_valid_for_chat(monkeypatch):
    monkeypatch.delenv(config_source.CHANNEL_ENV, raising=False)

    runtime.validate_config(_pretranslation_config(Tier(provider=Provider.ANTHROPIC, model="haiku")))


def test_pretranslation_on_the_voice_channel_must_be_a_bare_openai_client(monkeypatch):
    monkeypatch.setenv(config_source.CHANNEL_ENV, "voice")

    runtime.validate_config(_pretranslation_config(_OSS, _MANAGED))
    with pytest.raises(ValueError, match="step=pre_translation provider=anthropic"):
        runtime.validate_config(_pretranslation_config(Tier(provider=Provider.ANTHROPIC, model="haiku")))


# ── the step on the seam ────────────────────────────────────────────────────


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


class _Staleness:
    def __init__(self, stale_at=None):
        self.asked = []
        self._stale_at = stale_at

    async def __call__(self, reason):
        self.asked.append(reason)
        return "stale_request" if reason == self._stale_at else None


def _surface(**fields):
    _Outcomes.seen = []
    return dataclasses.replace(chat_service.CHAT_SURFACE, telemetry=_Outcomes, **fields)


def _run(turn, surface, *, is_stale=None, side_channel=None):
    async def _drain():
        return [
            emission
            async for emission in chat_service.run_turn(
                turn, surface, scheduler=_Scheduler(), is_stale=is_stale, side_channel=side_channel
            )
        ]

    return asyncio.run(asyncio.wait_for(_drain(), timeout=10))


def test_chat_pretranslates_through_the_slot():
    assert chat_service.CHAT_SURFACE.pretranslation is chat_service._chat_pretranslation


def test_a_surface_that_reaches_pretranslation_without_one_is_an_error(monkeypatch):
    patch_turn(monkeypatch)

    with pytest.raises(TypeError, match="reached pretranslation without one"):
        _run(_turn(), _surface(pretranslation=None))


def test_a_stale_request_is_not_pretranslated(monkeypatch):
    seen = patch_turn(monkeypatch)
    asked = []

    async def _step(turn, *, execution, background):
        asked.append(turn)
        return Pretranslated(query="My cow has fever", lang="en")

    staleness = _Staleness(stale_at="before_query_pretranslation")

    emissions = _run(_turn(), _surface(pretranslation=_step), is_stale=staleness)

    assert emissions == []
    assert asked == []
    assert seen["deps"] == [] and seen["history_writes"] == []
    assert _Outcomes.seen == [("outcome", "stale_request")]


def test_the_agent_is_asked_what_pretranslation_gave(monkeypatch):
    seen = patch_turn(monkeypatch)

    async def _step(turn, *, execution, background):
        return Pretranslated(query="My cow has fever", lang="en")

    _run(_turn(), _surface(pretranslation=_step))

    (deps,) = seen["deps"]
    assert deps.query == "My cow has fever"
    assert deps.lang_code == "en"


def test_a_pretranslation_answer_is_said_like_the_gates(monkeypatch):
    seen = patch_turn(monkeypatch)
    log = []
    answer = ClassifierResult(
        canned_text="Please say that again.",
        label="pretranslation_empty",
        history_pair=vp._history_pair(HISTORY_MARKERS["pretranslation_failed"], _FRAGMENT_RESPONSES["en"]),
    )

    async def _step(turn, *, execution, background):
        return answer

    class _Liveness:
        def __init__(self, turn, *, started_at, send, is_stale):
            pass

        async def stop(self):
            log.append("stopped")

    class _Sender:
        async def send(self, emission):
            pass

    async def _drain():
        async for emission in chat_service.run_turn(
            _turn(), _surface(pretranslation=_step, liveness=_Liveness),
            scheduler=_Scheduler(), side_channel=_Sender(), is_stale=_Staleness(),
        ):
            log.append(emission)

    asyncio.run(asyncio.wait_for(_drain(), timeout=10))

    assert log == ["stopped", TextEmission("Please say that again.")]
    assert seen["deps"] == [], "the agent ran"
    ((_, messages),) = seen["history_writes"]
    assert [p.content for m in messages for p in m.parts] == [HISTORY_MARKERS["pretranslation_failed"], _FRAGMENT_RESPONSES["en"]]
    assert ("output", "pretranslation_empty", "Please say that again.") in _Outcomes.seen
    assert _Outcomes.seen[-1] == ("outcome", "success")


def test_a_raw_pretranslation_answer_stays_raw(monkeypatch):
    patch_turn(monkeypatch)

    async def _step(turn, *, execution, background):
        return ClassifierResult(canned_text="Goodbye.", label="hang_up", raw=True)

    assert _run(_turn(), _surface(pretranslation=_step)) == [TextEmission("Goodbye.", raw=True)]


def test_a_stale_pretranslation_answer_is_neither_said_nor_saved(monkeypatch):
    seen = patch_turn(monkeypatch)

    async def _step(turn, *, execution, background):
        return ClassifierResult(canned_text="Please say that again.", label="pretranslation_empty")

    staleness = _Staleness(stale_at="before_pretranslation_empty_response")

    emissions = _run(_turn(), _surface(pretranslation=_step), is_stale=staleness)

    assert emissions == []
    assert seen["history_writes"] == []
    assert _Outcomes.seen[-1] == ("outcome", "stale_request")


def _voice_checks(monkeypatch, *, rejected=False, hang_up=False):
    async def _moderation(**_kw):
        return ModerationVerdict(category="offensive" if rejected else "in_scope", reason="test")

    async def _non_meaningful(**_kw):
        return NonMeaningfulVerdict(five_consecutive_non_meaningful=hang_up, reason="test")

    monkeypatch.setattr(voice_bg, "check_moderation", _moderation)
    monkeypatch.setattr(voice_bg, "check_non_meaningful_streak", _non_meaningful)


def _voice_surface():
    return _surface(
        background=voice_bg.voice_background(_render),
        pretranslation=vp.voice_pretranslation(_render),
    )


def test_a_hang_up_keeps_the_english_pretranslation_gave(monkeypatch):
    seen = patch_turn(monkeypatch)
    _voice_checks(monkeypatch, hang_up=True)

    async def _translated(text, source_lang, **_kw):
        return "My cow has fever"

    monkeypatch.setattr(vp, "_translate_to_english_pretranslation", _translated)
    # The streak is only judged on a full window of five caller turns.
    history = [m for word in ("hmm", "haa", "ok", "hmm") for m in _exchange(word, TextPart(content="Please repeat."))]

    emissions = _run(_turn(history=history), _voice_surface())

    assert emissions == [TextEmission("Goodbye.", raw=True)]
    ((_, messages),) = seen["history_writes"]
    assert [p.content for m in messages[-2:] for p in m.parts] == ["My cow has fever", "Goodbye."]


def test_a_rejected_query_with_nothing_translated_is_declined_on_the_seam(monkeypatch):
    seen = patch_turn(monkeypatch)
    _voice_checks(monkeypatch, rejected=True)

    async def _down(text, source_lang, **_kw):
        raise ConnectionError("pretranslation down")

    monkeypatch.setattr(vp, "_translate_to_english_pretranslation", _down)

    (emission,) = _run(_turn(), _voice_surface())

    assert emission.text.startswith("<gu> ")
    assert seen["deps"] == [], "the agent ran"
    assert ("output", "moderation_rejected", emission.text) in _Outcomes.seen
