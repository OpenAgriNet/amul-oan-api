"""The words just answered, sent again: said once more, not answered again (#582).

Raya sends what the caller said as soon as its speech-to-text has it, and at
times sends the same words again, as a new request, after we have answered
them. Both were answered: on prod, 20 Oct - 6 Nov, 116 times in 105 calls, each
within 5 s of the answered request, while a caller who says the same thing again
does so 8 s or more later. Eight bookings were made by the repeat: the first
answer asked whether to book, and the repeated words read as the yes.

The voice adapter keeps the answer the agent gave a request, as the caller heard
it, and a request with the same words within ``WINDOW_SECONDS`` of that one is
given it again. Raya has not said whether it drops an answer when a newer request
comes in, so the answer is said again rather than nothing: the caller hears it
once or twice, never silence.
"""
from __future__ import annotations

import json
import re
from typing import Optional, Sequence

from app.core.cache import build_cache_key, redis_client
from app.voice.classifiers import TELEPHONY_TERMINATE_CALL_TOKEN
from helpers.utils import get_logger

logger = get_logger(__name__)

#: How long after an answered request the same words are a repeat of it. Prod's
#: repeats came 0-5 s after it; a caller saying it again, from 8 s.
WINDOW_SECONDS = 5.0

_HANG_UP = set(TELEPHONY_TERMINATE_CALL_TOKEN.values())


def _key(session_id: str) -> str:
    return build_cache_key(f"{session_id}_last_answer")


def _words(text: str) -> str:
    """The words, without the punctuation and spacing the speech-to-text varies."""
    return re.sub(r"\s+", " ", re.sub(r"[*.,?!।]", " ", text or "")).strip()


async def answer_to_repeat(session_id: str, query: str, received_at: float) -> Optional[str]:
    """The answer to say again when ``query`` has the words answered on this call
    within ``WINDOW_SECONDS`` before ``received_at``, else None. Never raises."""
    words = _words(query)
    if not words:
        return None
    try:
        kept = await redis_client.get(_key(session_id))
        last = json.loads(kept) if kept else None
    except Exception as exc:  # the turn goes on as if nothing were kept
        logger.warning("Reading the last answer failed - session_id=%s error=%s", session_id, type(exc).__name__)
        return None
    if not last or last.get("words") != words or received_at - last.get("at", 0) > WINDOW_SECONDS:
        return None
    return last.get("answer") or None


async def remember_answer(session_id: str, query: str, heard: Sequence[str], received_at: float) -> None:
    """Keep what the caller heard as the answer to ``query``, for a repeat of it.

    An answer that hung up is not kept: the call is over, and said again as one
    text the hang-up token would not reach the provider on its own. Never raises.
    """
    words, answer = _words(query), "".join(heard)
    if not words or not answer.strip() or heard[-1].strip() in _HANG_UP:
        return
    try:
        await redis_client.set(
            _key(session_id),
            json.dumps({"words": words, "answer": answer, "at": received_at}, ensure_ascii=False),
            # A request this long after the answered one is no longer its repeat.
            ex=int(WINDOW_SECONDS) + 1,
        )
    except Exception as exc:
        logger.warning("Keeping the last answer failed - session_id=%s error=%s", session_id, type(exc).__name__)
