"""The words just answered, sent again: said once more, not answered again (#582).

Raya sends what the caller said as soon as its speech-to-text has it, and at
times sends the same words again, as a new request, after we have answered
them. Both were answered: on prod, 20 Oct - 6 Nov, 116 times in 105 calls. Each
repeat came 0.02-1.52 s after the first answer was sent (p50 0.47 s), while a
caller who says the same thing again does so seconds later, after hearing the
answer. Eight bookings were made by the repeat: the first answer asked whether
to book, and the repeated words read as the yes.

The voice adapter keeps the answer the agent gave a request, as the caller heard
it, for ``WINDOW_SECONDS`` after it went out: Redis expires it. A request with the
same words in that time is given it again; one with other words clears it. Raya
has not said whether it drops an answer when a newer request comes in, so the
answer is said again rather than nothing: the caller hears it once or twice,
never silence.
"""
from __future__ import annotations

import json
import re
from typing import Optional, Sequence

from app.core.cache import build_cache_key, redis_client
from app.voice.classifiers import TELEPHONY_TERMINATE_CALL_TOKEN
from helpers.utils import get_logger

logger = get_logger(__name__)

#: How long after an answer went out the same words are a repeat of it. Prod's
#: repeats all came within 1.52 s.
WINDOW_SECONDS = 2.0

_HANG_UP = set(TELEPHONY_TERMINATE_CALL_TOKEN.values())


def _key(session_id: str) -> str:
    return build_cache_key(f"{session_id}_last_answer")


def _words(text: str) -> str:
    """The words, without the punctuation and spacing the speech-to-text varies."""
    return re.sub(r"\s+", " ", re.sub(r"[*.,?!।]", " ", text or "")).strip()


async def answer_to_repeat(session_id: str, query: str) -> Optional[str]:
    """The answer to say again when ``query`` has the words just answered on this
    call, else None. Other words clear the kept answer. Never raises."""
    words = _words(query)
    try:
        kept = await redis_client.get(_key(session_id))
        last = json.loads(kept) if kept else None
        if not isinstance(last, dict):
            return None
        if words and last.get("words") == words:
            return last.get("answer") or None
        # The caller said something else since: a later repeat of the old words
        # would answer past it.
        await redis_client.delete(_key(session_id))
    except Exception as exc:  # the turn goes on as if nothing were kept
        logger.warning("Reading the last answer failed - session_id=%s error=%s", session_id, type(exc).__name__)
    return None


async def remember_answer(session_id: str, query: str, heard: Sequence[str]) -> None:
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
            json.dumps({"words": words, "answer": answer}, ensure_ascii=False),
            px=int(WINDOW_SECONDS * 1000),
        )
    except Exception as exc:
        logger.warning("Keeping the last answer failed - session_id=%s error=%s", session_id, type(exc).__name__)
