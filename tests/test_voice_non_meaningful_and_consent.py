"""Voice's non-meaningful streak check and outbound consent classifier.

The parsing cases are voice-oan-api's own (tests/test_non_meaningful.py and the
consent part of tests/test_outbound_intro.py). The rest pin what moved: both
checks take their client from the turn's ExecutionContext, from the
``non_meaningful`` step, and still never raise.
"""
import asyncio
import os
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest
from openai import AsyncOpenAI

from app.config import settings
from app.llm_core.config_model import (
    NamedProfile,
    PipelineConfig,
    Provider,
    Step,
    StepConfig,
    Tier,
)
from app.llm_core.execution import ExecutionContext
from app.voice import non_meaningful as nm
from app.voice import outbound_consent as oc
from app.voice.outbound_consent import INTENT_AFFIRMATIVE, INTENT_NEGATIVE, INTENT_OTHER

_CLASSIFIER = Tier(provider=Provider.VLLM, model="gemma-classifier", endpoint="http://oss:8020/v1")
_AGENT = Tier(provider=Provider.OPENAI, model="gpt")
_FIVE = ["hmm", "hello?", "ok", "haan", "uh"]


def _execution(*, with_step=True, profile="oss", oss_override=None):
    oss_steps = {Step.AGENT: StepConfig(tiers=[_AGENT])}
    if oss_override is not None:
        oss_steps[Step.NON_MEANINGFUL] = StepConfig(tiers=[oss_override])
    config = PipelineConfig(
        profiles=[
            NamedProfile(name="oss", weight=50, steps=oss_steps),
            NamedProfile(name="managed", weight=50, steps={Step.AGENT: StepConfig(tiers=[_AGENT])}),
        ],
        defaults={Step.NON_MEANINGFUL: StepConfig(tiers=[_CLASSIFIER])} if with_step else {},
    )
    return ExecutionContext(session_id="s", config=config, profile_name=profile)


def _resp(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture(autouse=True)
def _no_langfuse(monkeypatch):
    monkeypatch.setattr(nm, "_get_langfuse", lambda: None)
    monkeypatch.setattr(oc, "_get_langfuse", lambda: None)


# ── non-meaningful ──────────────────────────────────────────────────────────


def test_parse_non_meaningful_valid_true():
    verdict = nm._parse_verdict('{"five_consecutive_non_meaningful": true, "reason": "five filler turns"}')
    assert verdict.five_consecutive_non_meaningful is True
    assert verdict.failed_open is False


def test_parse_non_meaningful_invalid_json_fails_open():
    verdict = nm._parse_verdict("not-json")
    assert verdict.five_consecutive_non_meaningful is False
    assert verdict.failed_open is True


def test_non_meaningful_less_than_five_turns_returns_false():
    verdict = asyncio.run(nm.check_non_meaningful_streak(
        user_turns=["hello", "hmm", "ok"], source_lang="gu", execution=_execution(),
    ))
    assert verdict.five_consecutive_non_meaningful is False
    assert verdict.failed_open is False


def test_non_meaningful_asks_the_classifier_tier_about_the_last_five_turns(monkeypatch):
    seen = {}

    async def fake_create(client, model, user_turns, source_lang):
        seen.update(client=client, model=model, turns=user_turns)
        return _resp('{"five_consecutive_non_meaningful": true, "reason": "filler"}')

    monkeypatch.setattr(nm, "_create_non_meaningful_response", fake_create)

    verdict = asyncio.run(nm.check_non_meaningful_streak(
        user_turns=["first", *_FIVE], source_lang="gu", execution=_execution(),
    ))

    assert verdict.five_consecutive_non_meaningful is True
    assert isinstance(seen["client"], AsyncOpenAI)
    assert seen["model"] == "gemma-classifier"
    assert seen["turns"] == _FIVE


def test_non_meaningful_fails_open_on_timeout(monkeypatch):
    async def slow(client, model, user_turns, source_lang):
        raise asyncio.TimeoutError

    monkeypatch.setattr(nm, "_create_non_meaningful_response", slow)

    verdict = asyncio.run(nm.check_non_meaningful_streak(user_turns=_FIVE, source_lang="gu", execution=_execution()))

    assert verdict.five_consecutive_non_meaningful is False
    assert verdict.failed_open is True
    assert verdict.reason == "classifier timeout"


def test_non_meaningful_without_the_step_fails_open():
    verdict = asyncio.run(nm.check_non_meaningful_streak(
        user_turns=_FIVE, source_lang="gu", execution=_execution(with_step=False),
    ))

    assert verdict.five_consecutive_non_meaningful is False
    assert verdict.failed_open is True
    assert verdict.reason == "classifier client error: ValueError"


class _SlowCompletions:
    async def create(self, **_kw):
        await asyncio.sleep(1)


_SLOW_CLIENT = SimpleNamespace(chat=SimpleNamespace(completions=_SlowCompletions()))


def test_non_meaningful_times_out_on_voices_setting(monkeypatch):
    assert settings.voice_non_meaningful_timeout_seconds == 0.60
    monkeypatch.setattr(settings, "voice_non_meaningful_timeout_seconds", 0.01)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(nm._create_non_meaningful_response(_SLOW_CLIENT, "m", _FIVE, "gu"))


# ── outbound consent ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"intent": "affirmative", "reason": "clear yes"}', INTENT_AFFIRMATIVE),
        ('{"intent": "negative", "reason": "declines"}', INTENT_NEGATIVE),
        ('{"intent": "other", "reason": "asks own question"}', INTENT_OTHER),
        ('{"intent": "AFFIRMATIVE", "reason": "case"}', INTENT_AFFIRMATIVE),
    ],
)
def test_parse_consent_valid(raw, expected):
    verdict = oc._parse_verdict(raw)
    assert verdict.intent == expected
    assert verdict.failed_open is False


@pytest.mark.parametrize("raw", ["not-json", "[]", "", '{"intent": "maybe"}', '{"reason": "x"}'])
def test_parse_consent_bad_output_falls_back_to_other(raw):
    """Never negative on a bad parse: an uncertain verdict must not hang up."""
    verdict = oc._parse_verdict(raw)
    assert verdict.intent == INTENT_OTHER
    assert verdict.is_negative is False
    assert verdict.failed_open is True


def test_classify_consent_empty_reply_is_other():
    verdict = asyncio.run(oc.classify_consent(reply="   ", source_lang="gu", execution=_execution()))
    assert verdict.intent == INTENT_OTHER
    assert verdict.failed_open is False


def test_consent_uses_the_non_meaningful_tier(monkeypatch):
    seen = {}

    async def fake_create(client, model, reply, source_lang):
        seen.update(client=client, model=model, reply=reply)
        return _resp('{"intent": "affirmative", "reason": "yes"}')

    monkeypatch.setattr(oc, "_create_consent_response", fake_create)

    verdict = asyncio.run(oc.classify_consent(reply="ha bolo", source_lang="gu", execution=_execution()))

    assert verdict.is_affirmative
    assert isinstance(seen["client"], AsyncOpenAI)
    assert (seen["model"], seen["reply"]) == ("gemma-classifier", "ha bolo")


@pytest.mark.parametrize("error", [asyncio.TimeoutError(), ConnectionError("down")])
def test_consent_never_raises_and_never_says_no_on_failure(monkeypatch, error):
    async def broken(client, model, reply, source_lang):
        raise error

    monkeypatch.setattr(oc, "_create_consent_response", broken)

    verdict = asyncio.run(oc.classify_consent(reply="nahi", source_lang="gu", execution=_execution()))

    assert verdict.intent == INTENT_OTHER
    assert verdict.failed_open is True


def test_consent_without_the_step_is_other():
    verdict = asyncio.run(oc.classify_consent(reply="ha", source_lang="gu", execution=_execution(with_step=False)))

    assert verdict.intent == INTENT_OTHER
    assert verdict.failed_open is True


def test_consent_times_out_on_voices_setting(monkeypatch):
    assert settings.voice_outbound_consent_timeout_seconds == 0.60
    monkeypatch.setattr(settings, "voice_outbound_consent_timeout_seconds", 0.01)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(oc._create_consent_response(_SLOW_CLIENT, "m", "ha", "gu"))


@pytest.mark.parametrize("profile", ["oss", "managed", "unknown"])
def test_both_checks_read_the_step_the_same_way_on_every_profile(profile):
    """The step is profile-invariant: it is read under managed, whatever profile
    the session is on, so an override on another profile is not used."""
    override = Tier(provider=Provider.OPENAI, model="gpt-override")
    execution = _execution(profile=profile, oss_override=override)

    assert nm._non_meaningful_client_and_model(execution)[1:] == ("gemma-classifier", "vllm")
    assert oc._consent_client_and_model(execution)[1:] == ("gemma-classifier", "vllm")
