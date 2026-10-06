"""Voice's background checks and the gate they feed.

The three multi-turn cases at the end are voice-oan-api's own
(tests/test_voice_regressions_apr11_12.py), run against the gate directly.
"""
import asyncio
import gc
import os
from types import MappingProxyType, SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest
from pydantic_ai.messages import ModelRequest, TextPart, UserPromptPart

from app.channels.chat import WEB
from app.config import settings
from app.turn.types import TelephonyCall, Turn
from app.voice import background as bg
from app.voice.moderation import ModerationVerdict
from app.voice.non_meaningful import NonMeaningfulVerdict

_EXECUTION = object()
_ALLOW = ModerationVerdict(category="in_scope", reason="ok")
_KEEP_GOING = NonMeaningfulVerdict(five_consecutive_non_meaningful=False, reason="fine")
_FIVE_FILLERS = NonMeaningfulVerdict(five_consecutive_non_meaningful=True, reason="five filler turns")


def _exchange(user_text, assistant_text):
    return [
        ModelRequest(parts=[UserPromptPart(content=user_text)]),
        SimpleNamespace(parts=[TextPart(content=assistant_text)]),
    ]


def _filler_history():
    history = []
    for filler in ["હા", "ઓકે", "હમ્મ", "બરાબર"]:
        history.extend(_exchange(filler, "સમજાયું."))
    return history


def _turn(query="My cow has fever", *, history=(), source_lang="en", target_lang="en"):
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


async def _render(text_en, target_lang, *, execution):
    return f"<{target_lang}> {text_en}"


@pytest.fixture
def checks(monkeypatch):
    """Stand-ins for the two checks, recording what they were asked."""
    seen = {"moderation": [], "non_meaningful": [], "moderation_result": _ALLOW, "streak_result": _KEEP_GOING}

    async def _moderation(**kwargs):
        seen["moderation"].append(kwargs)
        result = seen["moderation_result"]
        if callable(result):
            return await result()
        return result

    async def _non_meaningful(**kwargs):
        seen["non_meaningful"].append(kwargs)
        result = seen["streak_result"]
        if callable(result):
            return await result()
        return result

    monkeypatch.setattr(bg, "check_moderation", _moderation)
    monkeypatch.setattr(bg, "check_non_meaningful_streak", _non_meaningful)
    return seen


def _gate(turn, *, close=True):
    """Build the background (which starts the checks), ask the gate, close.

    Whether the streak task was already cancelled is read inside the loop, right
    after the gate: once ``asyncio.run`` returns it has cancelled every task
    itself, so a later look would prove nothing.
    """

    async def _run():
        background = bg.VoiceBackground(turn, execution=_EXECUTION, render=_render)
        try:
            decision = await background.gate()
            background.streak_cancelled_by_gate = background._non_meaningful_task.cancelled()
            return decision, background
        finally:
            if close:
                await background.close()

    return asyncio.run(_run())


def _pair_texts(result):
    return [part.content for message in result.history_pair for part in message.parts]


# ── the non-meaningful window ───────────────────────────────────────────────


def test_the_window_is_the_last_four_turns_plus_this_one():
    history = []
    for text in ["one", "two", "three", "four", "five"]:
        history.extend(_exchange(text, "reply"))

    assert bg._collect_recent_user_turns_for_non_meaningful(history, "six") == ["two", "three", "four", "five", "six"]


def test_system_turns_neither_count_nor_break_the_streak():
    history = [
        *_exchange("હા", "ok"),
        *_exchange("[pretranslation-failed]", "please repeat"),
        *_exchange("[moderation-rejected]", "declined"),
        *_exchange("[outbound-call-started]", "intro"),
        *_exchange("ઓકે", "ok"),
    ]

    assert bg._collect_recent_user_turns_for_non_meaningful(history, "હમ્મ") == ["હા", "ઓકે", "હમ્મ"]


def test_unclear_turn_markers_do_count():
    history = [*_exchange("[fragment]", "repeat"), *_exchange("[stt:no-audio]", "repeat")]

    assert bg._collect_recent_user_turns_for_non_meaningful(history, "હા") == ["[fragment]", "[stt:no-audio]", "હા"]


@pytest.mark.parametrize("stored, expected", [
    ('**User:** "Correct"', "Correct"),
    ("'yes'", "yes"),
    ("  plain  ", "plain"),
    ("", ""),
])
def test_stored_turns_are_unwrapped_before_checking(stored, expected):
    assert bg._normalize_user_turn_for_non_meaningful(stored) == expected


def test_the_classifier_is_consulted_only_with_a_full_window():
    assert bg._should_gate_non_meaningful_llm(["a"] * 5) is True
    assert bg._should_gate_non_meaningful_llm(["a"] * 4) is False


# ── what the checks are asked ───────────────────────────────────────────────


def test_building_it_starts_both_checks_with_voices_inputs(checks):
    history = [
        *_exchange("oldest question", "old answer"),
        *_exchange("મારી ગાયને તાવ છે", "Call a vet."),
        *_exchange("ભેંસ", "Which animal?"),
    ]

    _gate(_turn("હા", history=history, source_lang=" GU "))

    (moderation,) = checks["moderation"]
    assert moderation["text"] == "હા"
    assert moderation["source_lang"] == "gu"
    # Moderation sees the last two exchanges, not the whole call.
    assert "મારી ગાયને તાવ છે" in moderation["recent_history_text"]
    assert "ભેંસ" in moderation["recent_history_text"]
    assert "oldest question" not in moderation["recent_history_text"]
    assert moderation["execution"] is _EXECUTION
    (streak,) = checks["non_meaningful"]
    assert streak["user_turns"] == ["oldest question", "મારી ગાયને તાવ છે", "ભેંસ", "હા"]
    assert streak["source_lang"] == "gu"
    assert streak["execution"] is _EXECUTION


# ── the gate ────────────────────────────────────────────────────────────────


def test_a_clean_turn_goes_through(checks):
    decision, _ = _gate(_turn())

    assert decision is None


def test_a_rejected_query_is_declined_in_the_callers_language(checks):
    checks["moderation_result"] = ModerationVerdict(category="irrelevant", reason="off topic")
    decline_en = "This helpline answers questions about animal health, dairy, and farming. Do you have a question about your animals?"

    decision, _ = _gate(_turn("movie recommendation please", target_lang="gu"))

    assert decision.label == "moderation_rejected"
    assert decision.canned_text == f"<gu> {decline_en}"
    assert decision.raw is False
    assert _pair_texts(decision) == ["[moderation-rejected]", decline_en]


def test_the_decline_is_rendered_on_the_turns_execution(checks):
    checks["moderation_result"] = ModerationVerdict(category="irrelevant", reason="off topic")
    used = []

    async def _render_on(text_en, target_lang, *, execution):
        used.append(execution)
        return text_en

    async def _run():
        background = bg.VoiceBackground(_turn(target_lang="gu"), execution=_EXECUTION, render=_render_on)
        try:
            return await background.gate()
        finally:
            await background.close()

    asyncio.run(_run())

    assert used == [_EXECUTION]


def test_moderation_failing_closed_is_declined_with_the_try_again_line(checks):
    checks["moderation_result"] = ModerationVerdict(category="unavailable", reason="down", failed_closed=True)

    decision, _ = _gate(_turn())

    assert decision.label == "moderation_rejected"
    assert "trouble processing your request" in decision.canned_text


def test_a_moderation_task_that_raises_lets_the_turn_through(checks):
    async def _boom():
        raise RuntimeError("moderation task blew up")

    checks["moderation_result"] = _boom

    decision, _ = _gate(_turn())

    assert decision is None
    assert checks["non_meaningful"], "the streak check must still be consulted"


def test_a_config_without_moderation_is_declined_not_let_through(monkeypatch):
    # The real moderation check, on a config where no profile has a moderation step.
    from app.llm_core.config_model import NamedProfile, PipelineConfig, Provider, Step, StepConfig, Tier
    from app.llm_core.execution import ExecutionContext

    async def _streak(**kwargs):
        return _KEEP_GOING

    monkeypatch.setattr(bg, "check_non_meaningful_streak", _streak)
    agent_only = {Step.AGENT: StepConfig(tiers=[Tier(provider=Provider.OPENAI, model="gpt")])}
    config = PipelineConfig(profiles=[NamedProfile(name="managed", weight=100, steps=agent_only)], fallback_enabled=True)
    execution = ExecutionContext(session_id="call-1", config=config, profile_name="managed")

    async def _run():
        background = bg.VoiceBackground(_turn(), execution=execution, render=_render)
        try:
            return await background.gate()
        finally:
            await background.close()

    decision = asyncio.run(_run())

    assert decision.label == "moderation_rejected"
    assert "trouble processing your request" in decision.canned_text


def test_five_non_meaningful_turns_hang_up_with_the_exact_token(checks):
    checks["streak_result"] = _FIVE_FILLERS

    decision, _ = _gate(_turn("હા", history=_filler_history(), source_lang="gu", target_lang="gu"))

    assert decision.label == "non_meaningful_hangup"
    assert decision.canned_text == "Goodbye."
    assert decision.raw is True
    assert _pair_texts(decision) == ["હા", "Goodbye."]


def test_a_short_window_skips_the_classifier_without_waiting(checks):
    async def _never():
        await asyncio.sleep(10)

    checks["streak_result"] = _never

    decision, background = _gate(_turn(), close=False)

    assert decision is None
    assert background.streak_cancelled_by_gate
    assert background._non_meaningful_verdict.reason == "gate skipped by heuristic"


def test_a_slow_classifier_times_out_and_the_turn_goes_through(checks, monkeypatch):
    monkeypatch.setattr(settings, "voice_non_meaningful_gate_timeout_seconds", 0.05)

    async def _slow():
        await asyncio.sleep(0.3)
        return _FIVE_FILLERS

    checks["streak_result"] = _slow

    decision, background = _gate(_turn("હા", history=_filler_history()), close=False)

    assert decision is None
    assert background.streak_cancelled_by_gate, "the timed-out task was left running"
    assert background._non_meaningful_verdict.reason == "gate timeout"
    assert background._non_meaningful_verdict.failed_open is True


def test_the_gate_waits_as_long_as_the_setting_allows(checks, monkeypatch):
    monkeypatch.setattr(settings, "voice_non_meaningful_gate_timeout_seconds", 1.0)

    async def _takes_a_moment():
        await asyncio.sleep(0.05)
        return _FIVE_FILLERS

    checks["streak_result"] = _takes_a_moment

    decision, _ = _gate(_turn("હા", history=_filler_history()))

    assert decision.label == "non_meaningful_hangup"


def test_a_classifier_error_lets_the_turn_through(checks):
    async def _boom():
        raise ConnectionError("classifier down")

    checks["streak_result"] = _boom

    decision, background = _gate(_turn("હા", history=_filler_history()), close=False)

    assert decision is None
    assert background._non_meaningful_verdict.reason == "task error: ConnectionError"


# ── close ───────────────────────────────────────────────────────────────────


def test_close_cancels_and_reaps_whatever_is_still_running(checks):
    async def _never():
        await asyncio.sleep(10)

    checks["moderation_result"] = _never
    checks["streak_result"] = _never

    async def _run():
        background = bg.VoiceBackground(_turn(), execution=_EXECUTION, render=_render)
        await asyncio.sleep(0)
        await background.close()
        return background._moderation_task.cancelled(), background._non_meaningful_task.cancelled()

    assert asyncio.run(_run()) == (True, True)


def test_close_never_raises_and_can_run_twice(checks):
    async def _boom():
        raise RuntimeError("check failed")

    async def _fails_on_cancel():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            raise RuntimeError("cleanup failed")

    checks["moderation_result"] = _boom
    checks["streak_result"] = _fails_on_cancel

    async def _run():
        background = bg.VoiceBackground(_turn(), execution=_EXECUTION, render=_render)
        await asyncio.sleep(0.01)
        await background.close()
        await background.close()

    asyncio.run(_run())


def test_a_check_that_failed_early_is_not_logged_again_at_shutdown(checks):
    async def _boom():
        raise RuntimeError("check failed")

    checks["moderation_result"] = _boom

    async def _run():
        unhandled = []
        asyncio.get_running_loop().set_exception_handler(lambda _loop, context: unhandled.append(context))
        background = bg.VoiceBackground(_turn(), execution=_EXECUTION, render=_render)
        await asyncio.sleep(0.01)
        await background.close()
        del background
        gc.collect()
        return unhandled

    assert asyncio.run(_run()) == []


# ── voice-oan-api's multi-turn cases ────────────────────────────────────────


def test_non_meaningful_five_turn_streak_emits_goodbye(checks):
    checks["streak_result"] = _FIVE_FILLERS

    decision, _ = _gate(_turn("હા", history=_filler_history(), source_lang="gu", target_lang="en"))

    assert decision.canned_text.strip() == "Goodbye."
    assert len(checks["non_meaningful"][0]["user_turns"]) == 5
    assert "Goodbye." in _pair_texts(decision)


def test_non_meaningful_timeout_fails_open_and_streams_agent_output(checks, monkeypatch):
    monkeypatch.setattr(settings, "voice_non_meaningful_gate_timeout_seconds", 0.01)

    async def _slow():
        await asyncio.sleep(0.2)
        raise AssertionError("should be canceled before finishing")

    checks["streak_result"] = _slow

    decision, _ = _gate(_turn("My cow has fever"))

    assert decision is None


def test_moderation_reject_still_wins_over_non_meaningful(checks):
    checks["moderation_result"] = ModerationVerdict(category="irrelevant", reason="off topic")
    checks["streak_result"] = _FIVE_FILLERS

    decision, _ = _gate(_turn("movie recommendation please", history=_filler_history()))

    assert "Goodbye." not in decision.canned_text
    assert "This helpline answers questions about animal health, dairy, and farming." in decision.canned_text
