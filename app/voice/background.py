"""Voice's background checks for one turn, and the gate before its first emission.

From voice-oan-api (``app/services/voice.py`` at amul-dev ``3b19835``). Moderation
and the non-meaningful streak start as soon as the classifiers have passed, run
alongside everything up to the agent's first chunk, and are resolved only then.
A rejected query is declined and five non-meaningful turns in a row end the
call. A check that errors lets the turn through; moderation's own fail-closed
verdict is a rejection and is declined like one.

A turn that ends before the gate still declines a rejected query: when
pretranslation leaves nothing to ask the agent, voice's pretranslation asks
``decline`` rather than having the caller repeat a rejected question.

The farmer data fetch starts here too, and on the farmer's reply to an outbound
call the consent classifier and the milk prefetch. The agent input reads the
first two. The farmer fetch is left to finish when the turn ends, as voice
leaves it: it fills the cache for the call's next turn.

Each check's latency and verdict go on the turn's trace when the check is
resolved, timed to when it finished rather than to when it was read.
"""
from __future__ import annotations

import asyncio
import re
import time
from functools import partial
from typing import Optional

from agents.voice.tools.farmer import normalize_phone_to_mobile
from app.config import settings
from app.turn.types import BackgroundFactory, ClassifierResult, Turn
from app.utils import format_message_pairs
from app.voice import outbound as _outbound
from app.voice.classifiers import TELEPHONY_TERMINATE_CALL_TOKEN, RenderForCaller
from app.voice.farmer import _collect_farmer_accounts, get_or_fetch_farmer_data
from app.voice.history import HISTORY_MARKERS, history_pair
from app.voice.liveness import nudge_stopped
from app.voice.moderation import ModerationVerdict, check_moderation
from app.voice.non_meaningful import NonMeaningfulVerdict, check_non_meaningful_streak
from app.voice.outbound_consent import ConsentVerdict, classify_consent
from app.voice.trace import current_trace, sanitize_text
from helpers.utils import get_logger

logger = get_logger(__name__)

# Markers that are dropped from the non-meaningful window entirely (neither
# counted nor streak-breaking). Excluded turns are skipped, so "5 consecutive
# non-meaningful turns" means consecutive among *eligible* turns — a
# pretranslation/moderation system turn in the middle does not reset the streak.
# fragment / unclear / no-audio / stt markers are intentionally NOT excluded:
# they represent genuinely unclear caller turns and should count toward a
# hangup (the prompt documents them as non-meaningful system markers).
_NON_MEANINGFUL_EXCLUDED_TURNS = frozenset(
    {
        HISTORY_MARKERS["pretranslation_failed"],
        HISTORY_MARKERS["moderation_reject"],
        # Legacy marker from the removed in-app intro (Raya owns the opener now).
        # Not a caller turn at all, so it must neither count toward a hangup
        # streak nor break one, for as long as such histories survive trimming.
        HISTORY_MARKERS["outbound_intro"],
    }
)

_DEFAULT_DECLINE_EN = "This helpline only handles dairy farming and animal husbandry questions."


def _is_eligible_non_meaningful_turn(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    if cleaned in _NON_MEANINGFUL_EXCLUDED_TURNS:
        return False
    return True


def _normalize_user_turn_for_non_meaningful(text: str) -> str:
    """Normalize stored user turn text before heuristic/classifier checks.

    History can contain wrapped forms like:
      **User:** "Correct"
    Strip wrappers/quotes so comparisons operate on caller content only.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    match = re.match(r'^\*\*User:\*\*\s*"?(.*?)"?$', cleaned, flags=re.IGNORECASE)
    if match:
        cleaned = (match.group(1) or "").strip()
    # Remove balanced outer quotes if still present.
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _collect_recent_user_turns_for_non_meaningful(history: list, current_query: str, limit: int = 5) -> list[str]:
    turns: list[str] = []
    for msg in reversed(history or []):
        for part in getattr(msg, "parts", []) or []:
            if getattr(part, "part_kind", "") != "user-prompt":
                continue
            content = getattr(part, "content", None)
            if not isinstance(content, str):
                continue
            normalized = _normalize_user_turn_for_non_meaningful(content)
            if not _is_eligible_non_meaningful_turn(normalized):
                continue
            turns.append(normalized)
            if len(turns) >= limit - 1:
                break
        if len(turns) >= limit - 1:
            break
    turns.reverse()
    normalized_current = _normalize_user_turn_for_non_meaningful(current_query)
    if _is_eligible_non_meaningful_turn(normalized_current):
        turns.append(normalized_current)
    return turns[-limit:]


def _should_gate_non_meaningful_llm(turns: list[str]) -> bool:
    """Gate on LLM only when we have a full 5-turn window."""
    return len(turns) >= 5


async def _warm_outbound_milk(session_id: str, mobile_number: str) -> None:
    """Warm the milk summary alongside the consent classifier, so an affirmative
    reply does not pay the upstream lookup serially. Failure just means the
    agent fetches it itself."""
    envelope = await get_or_fetch_farmer_data(mobile_number)
    await _outbound.prefetch_milk_summary(session_id, _collect_farmer_accounts(envelope))


async def _cancel_and_reap(task: asyncio.Task) -> None:
    if task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


class VoiceBackground:
    """Moderation, the non-meaningful streak and the farmer data for one call turn.

    Building it starts them all, plus the consent classifier on an outbound
    call's consent turn. ``gate`` resolves the checks in voice's order,
    moderation first.
    """

    def __init__(self, turn: Turn, *, execution, render: RenderForCaller) -> None:
        self._turn = turn
        self._render = render
        self._source_lang = (turn.source_lang or "gu").strip().lower()
        self._target_lang = (turn.target_lang or "gu").strip().lower()
        self._process_id = turn.call.process_id if turn.call is not None else None
        self._history_text = turn.query
        history = list(turn.history)
        self._moderation_recent_history = "\n\n".join(format_message_pairs(history, 2))
        self._non_meaningful_turns = _collect_recent_user_turns_for_non_meaningful(history, turn.query, limit=5)
        self._started_at: dict[str, float] = {}
        self._done_at: dict[str, float] = {}
        self._moderation_task = self._timed(
            "moderation",
            check_moderation(
                text=turn.query,
                source_lang=self._source_lang,
                recent_history_text=self._moderation_recent_history,
                execution=execution,
            ),
        )
        self._non_meaningful_task = self._timed(
            "non_meaningful",
            check_non_meaningful_streak(
                user_turns=self._non_meaningful_turns,
                source_lang=self._source_lang,
                execution=execution,
            ),
        )
        self._mobile = normalize_phone_to_mobile(turn.user_id)
        self._farmer_task = (
            asyncio.create_task(get_or_fetch_farmer_data(self._mobile))
            if self._mobile
            else None
        )
        consent_turn = turn.call is not None and turn.call.outbound_consent_turn
        # It reads the raw native-language reply, so it runs under pretranslation
        # and the farmer fetch rather than after them.
        self._consent_task = (
            asyncio.create_task(
                classify_consent(reply=turn.query, source_lang=self._source_lang, execution=execution)
            )
            if consent_turn
            else None
        )
        if consent_turn and self._mobile:
            _outbound.spawn(
                _warm_outbound_milk(turn.session_id, self._mobile), label="outbound_milk_prefetch"
            )
        self._moderation_resolved = False
        self._moderation_verdict: Optional[ModerationVerdict] = None
        self._non_meaningful_resolved = False
        self._non_meaningful_verdict: Optional[NonMeaningfulVerdict] = None

    def _timed(self, name: str, check) -> asyncio.Task:
        """Start a check, stamping when it really finishes: its verdict is read
        later, and that wait is not the check's latency."""
        self._started_at[name] = time.monotonic()
        task = asyncio.create_task(check)
        task.add_done_callback(lambda _t: self._done_at.__setitem__(name, time.monotonic()))
        return task

    def _duration_ms(self, name: str) -> float:
        return ((self._done_at.get(name) or time.monotonic()) - self._started_at[name]) * 1000.0

    @property
    def mobile(self) -> Optional[str]:
        """The caller's mobile, when the user id is one."""
        return self._mobile

    @property
    def moderation_task(self) -> asyncio.Task:
        """The running moderation check. Booking tools wait on it through
        ``FarmerContext.ensure_in_scope`` before they write anything."""
        return self._moderation_task

    @property
    def history_text(self) -> str:
        """What history keeps for the caller's turn."""
        return self._history_text

    async def farmer_data(self):
        """The farmer fetch's envelope, or None without a mobile or when the
        fetch did not resolve. Raises what the fetch raised."""
        if self._farmer_task is None:
            return None
        return await self._farmer_task

    async def consent(self) -> Optional[ConsentVerdict]:
        """The verdict on an outbound call's consent turn, else None."""
        if self._consent_task is None:
            return None
        return await self._consent_task

    def set_history_text(self, text: str) -> None:
        """What history keeps for the caller's turn: the English pretranslation
        gave, or the marker it left when it gave nothing."""
        self._history_text = text

    async def decline(self) -> Optional[ClassifierResult]:
        """The decline for a query moderation rejected, else None."""
        verdict = await self._resolve_moderation()
        if verdict is not None and verdict.rejected:
            return await self._decline(verdict)
        return None

    async def gate(self) -> Optional[ClassifierResult]:
        declined = await self.decline()
        if declined is not None:
            return declined
        # Resolving here also reaps the classifier task on the normal agent path.
        streak = await self._resolve_non_meaningful()
        if streak is not None and streak.five_consecutive_non_meaningful:
            return self._hang_up(streak)
        return None

    async def close(self) -> None:
        tasks = (self._moderation_task, self._non_meaningful_task, self._consent_task)
        for task in (task for task in tasks if task is not None):
            try:
                if task.done():
                    # A check that failed before anyone asked: take its error,
                    # so asyncio does not log it again when the task is freed.
                    if not task.cancelled():
                        task.exception()
                    continue
                await _cancel_and_reap(task)
            except Exception:
                pass

    async def _resolve_moderation(self) -> Optional[ModerationVerdict]:
        if self._moderation_resolved:
            return self._moderation_verdict
        self._moderation_resolved = True
        trace = current_trace()
        moderation_status = "ok"
        moderation_status_message: Optional[str] = None
        try:
            self._moderation_verdict = await self._moderation_task
            trace.attach_stage_timing(
                "moderation",
                self._duration_ms("moderation"),
                source_lang=self._source_lang,
            )
        except asyncio.CancelledError:
            raise
        except Exception as moderation_error:
            moderation_status = "error"
            moderation_status_message = str(moderation_error)[:300]
            trace.attach_stage_timing(
                "moderation",
                self._duration_ms("moderation"),
                status="error",
                source_lang=self._source_lang,
            )
            logger.error(
                "Moderation task raised unexpectedly for session_id=%s error=%s",
                self._turn.session_id,
                moderation_error,
            )
            self._moderation_verdict = None
        moderation_duration_ms = self._duration_ms("moderation")
        trace.set_moderation(self._moderation_verdict)
        moderation_payload = trace.metadata.get("moderation", {})
        verdict = self._moderation_verdict
        trace.record_child_observation(
            name="moderation",
            as_type="generation",
            input={
                "source_lang": self._source_lang,
                "text": sanitize_text(self._turn.query),
                "recent_history_text": sanitize_text(self._moderation_recent_history),
            },
            output=(
                {
                    "category": getattr(verdict, "category", None),
                    "reason": getattr(verdict, "reason", None),
                    "rejected": getattr(verdict, "rejected", None),
                    "failed_open": getattr(verdict, "failed_open", None),
                    "failed_closed": getattr(verdict, "failed_closed", None),
                }
                if verdict is not None
                else {"available": False}
            ),
            metadata={
                "duration_ms": round(moderation_duration_ms, 2),
                "status": moderation_status,
                "source_lang": self._source_lang,
                "pipeline_profile": trace.metadata.get("pipeline_profile"),
                "requested_tier": moderation_payload.get("requested_tier"),
                "requested_provider": moderation_payload.get("requested_provider"),
                "requested_model": moderation_payload.get("requested_model"),
                "actual_tier": moderation_payload.get("actual_tier"),
                "actual_provider": moderation_payload.get("actual_provider"),
                "actual_model": moderation_payload.get("actual_model"),
                "fallback_used": moderation_payload.get("fallback_used"),
                "attempts": moderation_payload.get("attempts"),
            },
            model=moderation_payload.get("actual_model") or moderation_payload.get("requested_model"),
            level="ERROR" if moderation_status == "error" else "DEFAULT",
            status_message=moderation_status_message,
        )
        if self._moderation_verdict is not None:
            logger.info(
                "Moderation verdict: category=%s rejected=%s failed_open=%s reason=%r session_id=%s process_id=%s",
                self._moderation_verdict.category,
                self._moderation_verdict.rejected,
                self._moderation_verdict.failed_open,
                self._moderation_verdict.reason,
                self._turn.session_id,
                self._process_id,
            )
        return self._moderation_verdict

    async def _resolve_non_meaningful(self) -> Optional[NonMeaningfulVerdict]:
        if self._non_meaningful_resolved:
            return self._non_meaningful_verdict
        self._non_meaningful_resolved = True
        task = self._non_meaningful_task
        trace = current_trace()
        should_gate = _should_gate_non_meaningful_llm(self._non_meaningful_turns)
        try:
            if not should_gate and not task.done():
                await _cancel_and_reap(task)
                self._non_meaningful_verdict = NonMeaningfulVerdict(
                    five_consecutive_non_meaningful=False,
                    reason="gate skipped by heuristic",
                    failed_open=False,
                )
                trace.attach_stage_timing(
                    "non_meaningful",
                    (time.monotonic() - self._started_at["non_meaningful"]) * 1000.0,
                    source_lang=self._source_lang,
                    turn_count=len(self._non_meaningful_turns),
                    gate_skipped=True,
                )
            elif task.done():
                self._non_meaningful_verdict = await task
            else:
                done, _pending = await asyncio.wait(
                    {task},
                    timeout=max(0.0, settings.voice_non_meaningful_gate_timeout_seconds),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    # Reap the cancellation. asyncio.wait() does not await the
                    # pending task, so without this the classifier task lingers
                    # cancelled-but-unawaited on the normal agent path.
                    await _cancel_and_reap(task)
                    self._non_meaningful_verdict = NonMeaningfulVerdict(
                        five_consecutive_non_meaningful=False,
                        reason="gate timeout",
                        failed_open=True,
                    )
                    logger.info(
                        "Non-meaningful gate timed out; fail-open session_id=%s process_id=%s timeout=%.2fs",
                        self._turn.session_id,
                        self._process_id,
                        settings.voice_non_meaningful_gate_timeout_seconds,
                    )
                else:
                    self._non_meaningful_verdict = await task
            trace.attach_stage_timing(
                "non_meaningful",
                self._duration_ms("non_meaningful"),
                source_lang=self._source_lang,
                turn_count=len(self._non_meaningful_turns),
            )
        except asyncio.CancelledError:
            raise
        except Exception as non_meaningful_error:
            trace.attach_stage_timing(
                "non_meaningful",
                self._duration_ms("non_meaningful"),
                status="error",
                source_lang=self._source_lang,
            )
            logger.error(
                "Non-meaningful task raised unexpectedly for session_id=%s error=%s",
                self._turn.session_id,
                non_meaningful_error,
            )
            self._non_meaningful_verdict = NonMeaningfulVerdict(
                five_consecutive_non_meaningful=False,
                reason=f"task error: {type(non_meaningful_error).__name__}",
                failed_open=True,
            )
        trace.metadata["non_meaningful"] = {
            "available": self._non_meaningful_verdict is not None,
            "five_consecutive_non_meaningful": bool(
                getattr(self._non_meaningful_verdict, "five_consecutive_non_meaningful", False)
            ),
            "failed_open": bool(getattr(self._non_meaningful_verdict, "failed_open", False)),
            "reason": getattr(self._non_meaningful_verdict, "reason", ""),
            "turn_count": len(self._non_meaningful_turns),
            "gate_skipped": not should_gate,
        }
        if self._non_meaningful_verdict is not None:
            logger.info(
                "Non-meaningful verdict: five_consecutive_non_meaningful=%s failed_open=%s reason=%r session_id=%s process_id=%s",
                self._non_meaningful_verdict.five_consecutive_non_meaningful,
                self._non_meaningful_verdict.failed_open,
                self._non_meaningful_verdict.reason,
                self._turn.session_id,
                self._process_id,
            )
        return self._non_meaningful_verdict

    async def _decline(self, verdict: ModerationVerdict) -> ClassifierResult:
        """The canned decline for a rejected query. History keeps it in English."""
        nudge_stopped("moderation_rejected")
        decline_en = verdict.decline_text_en() or _DEFAULT_DECLINE_EN
        decline_for_caller = await self._render(decline_en, self._target_lang)
        return ClassifierResult(
            canned_text=decline_for_caller,
            label="moderation_rejected",
            history_pair=history_pair(HISTORY_MARKERS["moderation_reject"], decline_en),
            outcome="moderation_rejected",
        )

    def _hang_up(self, verdict: NonMeaningfulVerdict) -> ClassifierResult:
        """Say goodbye so the telephony provider ends the call."""
        nudge_stopped("non_meaningful_hangup")
        goodbye = TELEPHONY_TERMINATE_CALL_TOKEN.get(
            self._target_lang,
            TELEPHONY_TERMINATE_CALL_TOKEN["en"],
        )
        logger.info(
            "Non-meaningful hangup emitted; session_id=%s process_id=%s reason=%r",
            self._turn.session_id,
            self._process_id,
            verdict.reason,
        )
        # Keep the exact telephony termination token: raw, past the normalizer.
        return ClassifierResult(
            canned_text=goodbye,
            label="non_meaningful_hangup",
            history_pair=history_pair(self._history_text, TELEPHONY_TERMINATE_CALL_TOKEN["en"]),
            raw=True,
            outcome="non_meaningful_hangup",
        )


def voice_background(render: RenderForCaller) -> BackgroundFactory:
    """Voice's background set for ``SurfaceProfile.background``.

    ``render`` puts the English decline into the caller's language, as it does
    for the classifiers.
    """
    return partial(VoiceBackground, render=render)
