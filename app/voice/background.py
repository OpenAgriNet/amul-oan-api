"""Voice's background checks for one turn, and the gate before its first emission.

From voice-oan-api (``app/services/voice.py`` at amul-dev ``3b19835``). Moderation
and the non-meaningful streak start as soon as the classifiers have passed, run
alongside everything up to the agent's first chunk, and are resolved only then.
A rejected query is declined and five non-meaningful turns in a row end the
call. A check that errors lets the turn through; moderation's own fail-closed
verdict is a rejection and is declined like one.

Voice's other background tasks, the farmer data fetch and the outbound consent
classifier, arrive with the code that reads them. So do the nudge cancellation
and the stale-request checks around a decline.
"""
from __future__ import annotations

import asyncio
import re
from functools import partial
from typing import Optional

from app.config import settings
from app.turn.types import BackgroundFactory, ClassifierResult, Turn
from app.utils import format_message_pairs
from app.voice.classifiers import TELEPHONY_TERMINATE_CALL_TOKEN, RenderForCaller
from app.voice.history import HISTORY_MARKERS, history_pair
from app.voice.moderation import ModerationVerdict, check_moderation
from app.voice.non_meaningful import NonMeaningfulVerdict, check_non_meaningful_streak
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


async def _cancel_and_reap(task: asyncio.Task) -> None:
    if task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


class VoiceBackground:
    """Moderation and the non-meaningful streak for one call turn.

    Building it starts both checks. ``gate`` resolves them in voice's order,
    moderation first.
    """

    def __init__(self, turn: Turn, *, execution, render: RenderForCaller) -> None:
        self._turn = turn
        self._execution = execution
        self._render = render
        self._source_lang = (turn.source_lang or "gu").strip().lower()
        self._target_lang = (turn.target_lang or "gu").strip().lower()
        self._process_id = turn.call.process_id if turn.call is not None else None
        history = list(turn.history)
        self._non_meaningful_turns = _collect_recent_user_turns_for_non_meaningful(history, turn.query, limit=5)
        self._moderation_task = asyncio.create_task(
            check_moderation(
                text=turn.query,
                source_lang=self._source_lang,
                recent_history_text="\n\n".join(format_message_pairs(history, 2)),
                execution=execution,
            )
        )
        self._non_meaningful_task = asyncio.create_task(
            check_non_meaningful_streak(
                user_turns=self._non_meaningful_turns,
                source_lang=self._source_lang,
                execution=execution,
            )
        )
        self._moderation_resolved = False
        self._moderation_verdict: Optional[ModerationVerdict] = None
        self._non_meaningful_resolved = False
        self._non_meaningful_verdict: Optional[NonMeaningfulVerdict] = None

    async def gate(self) -> Optional[ClassifierResult]:
        verdict = await self._resolve_moderation()
        if verdict is not None and verdict.rejected:
            return await self._decline(verdict)
        # Resolving here also reaps the classifier task on the normal agent path.
        streak = await self._resolve_non_meaningful()
        if streak is not None and streak.five_consecutive_non_meaningful:
            return self._hang_up(streak)
        return None

    async def close(self) -> None:
        for task in (self._moderation_task, self._non_meaningful_task):
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
        try:
            self._moderation_verdict = await self._moderation_task
        except asyncio.CancelledError:
            raise
        except Exception as moderation_error:
            logger.error(
                "Moderation task raised unexpectedly for session_id=%s error=%s",
                self._turn.session_id,
                moderation_error,
            )
            self._moderation_verdict = None
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
        try:
            if not _should_gate_non_meaningful_llm(self._non_meaningful_turns) and not task.done():
                await _cancel_and_reap(task)
                self._non_meaningful_verdict = NonMeaningfulVerdict(
                    five_consecutive_non_meaningful=False,
                    reason="gate skipped by heuristic",
                    failed_open=False,
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
        except asyncio.CancelledError:
            raise
        except Exception as non_meaningful_error:
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
        decline_en = verdict.decline_text_en() or _DEFAULT_DECLINE_EN
        decline_for_caller = await self._render(decline_en, self._target_lang, execution=self._execution)
        return ClassifierResult(
            canned_text=decline_for_caller,
            label="moderation_rejected",
            history_pair=history_pair(HISTORY_MARKERS["moderation_reject"], decline_en),
        )

    def _hang_up(self, verdict: NonMeaningfulVerdict) -> ClassifierResult:
        """Say goodbye so the telephony provider ends the call."""
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
        # History keeps the caller's words until pretranslation supplies the
        # English text voice stores for the turn.
        return ClassifierResult(
            canned_text=goodbye,
            label="non_meaningful_hangup",
            history_pair=history_pair(self._turn.query, TELEPHONY_TERMINATE_CALL_TOKEN["en"]),
            raw=True,
        )


def voice_background(render: RenderForCaller) -> BackgroundFactory:
    """Voice's background set for ``SurfaceProfile.background``.

    ``render`` puts the English decline into the caller's language, as it does
    for the classifiers.
    """
    return partial(VoiceBackground, render=render)
