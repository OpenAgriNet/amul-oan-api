"""Voice's pre-turn classifiers: the five ways a call turn is answered without the agent.

Brought over from voice-oan-api (``app/services/voice.py`` at amul-dev
``3b19835``): the short-circuits at the top of ``stream_voice_message`` and the
helpers they use. Detection rules, canned lines and history markers are voice's,
unchanged. The order is voice's too, and it matters: the first match answers.

Each classifier returns the caller's text *before* voice's output normalizer,
which applies to every emission that is not ``raw`` (see ``TextEmission.raw``).
The hang-up line is the one that is raw.

Putting an English line into the caller's language is a translation call, which
belongs to voice's sink. It is passed in as ``render`` rather than brought over
here; voice's own tests stub the same function.
"""
from __future__ import annotations

import re
from functools import partial
from typing import Literal, Optional, Protocol

from app import llm_core
from app.config import settings
from app.turn.types import Classifier, ClassifierResult, Turn
from app.voice.history import HISTORY_MARKERS as _HISTORY_MARKERS
from app.voice.history import history_pair as _history_pair
from app.voice.stt_signals import (
    count_consecutive_stt_signals,
    detect_stt_signal,
    generate_stt_signal_response,
)
from helpers.utils import get_logger

logger = get_logger(__name__)


class RenderForCaller(Protocol):
    """Puts an English line into the caller's language, ready to be spoken, on the
    turn's own models: ``execution`` is the turn's ExecutionContext."""

    async def __call__(self, text_en: str, target_lang: str, *, execution) -> str: ...


# ── Greeting short-circuit helpers ─────────────────────────────────────
_GREETING_TOKENS = {
    # English
    "hello", "hi", "hey", "hlo",
    # Gujarati
    "હલો", "હેલો", "નમસ્તે", "નમસ્કાર",
    # Hindi
    "नमस्ते", "हेलो", "हलो",
    # Transliteration
    "namaste", "halo", "helo",
    # Multi-word greeting combos
    "ha hello", "હા હલો", "ji", "જી", "bolo", "બોલો",
    "ha bolo", "હા બોલો", "ji bolo", "જી બોલો",
}


def _is_bare_greeting(query: str) -> bool:
    """Return True if the query is just a greeting with no real content."""
    cleaned = re.sub(r"[*\s]+", " ", query).strip().lower()
    if not cleaned:
        return False
    # Strip punctuation for matching
    cleaned = re.sub(r"[.,!?।]+$", "", cleaned).strip()
    if cleaned in _GREETING_TOKENS:
        return True
    # Collapse repeated words: "hello hello" → "hello"
    words = cleaned.split()
    if len(words) <= 4:
        deduped = " ".join(dict.fromkeys(words))
        if deduped in _GREETING_TOKENS:
            return True
    return False


_GREETING_RESPONSES = {
    "gu": "નમસ્તે, હું સરલાબેન છું. તમારા પશુ વિશે કોઈ સમસ્યા હોય તો મને જણાવો.",
    "en": "Hello, I am Sarlaben. Please tell me what issue you are facing with your animal.",
}

# ── Fragment detection (garbled / too-short input) ────────────────────────
_FRAGMENT_RESPONSES = {
    "gu": "મને તમારો પ્રશ્ન સમજાયો નથી. કૃપા કરીને તમારો પ્રશ્ન ફરીથી પૂછો.",
    "en": "I could not understand your question. Please ask your question again.",
}

def _is_fragment_query(query: str) -> bool:
    """Return True if query is too short/garbled to be a real question."""
    cleaned = re.sub(r"[*\s.,!?।]+", " ", query).strip()
    if not cleaned:
        return True
    # Single character or very short (≤3 chars) — likely noise
    if len(cleaned) <= 3:
        return True
    return False


# ── Hold message detection ─────────────────────────────────────────────
# Carrier IVR "your call is on hold" messages get picked up by STT and
# sent as user input, creating runaway loops. Detect them and respond
# with "goodbye" so the STT provider cuts the call.
_HOLD_MSG_PATTERNS_GU = [
    "હોલ્ડ પર",            # "on hold" in Gujarati
    "લાઇન પર રહો",        # "stay on the line"
    "લાઈન પર રહો",        # variant spelling
]
_HOLD_MSG_PATTERNS_EN = [
    "put your call on hold",
    "call has been put on hold",
    "call on hold",
    "please stay on the line",
    "please remain on the line",
]
TELEPHONY_TERMINATE_CALL_TOKEN = {
    "gu": "Goodbye.",
    "en": "Goodbye.",
}


def _has_meaningful_history(history) -> bool:
    """Return True when the session already contains non-trivial conversation."""
    for msg in reversed(history or []):
        for part in getattr(msg, "parts", []) or []:
            content = getattr(part, "content", None)
            if not isinstance(content, str):
                continue
            text = content.strip()
            if not text:
                continue
            if detect_stt_signal(text) is not None:
                continue
            return True
    return False


def _is_hold_message(query: str) -> bool:
    """Return True if the query looks like a carrier hold/IVR message."""
    lower = query.lower()
    for pat in _HOLD_MSG_PATTERNS_GU:
        if pat in lower:
            return True
    for pat in _HOLD_MSG_PATTERNS_EN:
        if pat in lower:
            return True
    return False


# ── Identity fast-path ────────────────────────────────────────────────────
_IDENTITY_PHRASES_GU = {
    "તમારું નામ શું છે", "તારું નામ શું છે", "તમે કોણ છો", "આ સેવા શું છે",
    "આ કઈ સેવા છે", "તમે ક્યાંથી બોલો છો", "ક્યાંથી બોલો",
}
_IDENTITY_PHRASES_EN = {
    "what is your name", "who are you", "what is this service", "what service is this",
    "where are you calling from",
}

_IDENTITY_RESPONSE_EN = (
    f"I am Sarlaben from Amul AI. I was created on {settings.voice_profile_creation_date_words}, "
    "and I help dairy farmers with animal health, feed, and breeding guidance."
)


def _fast_path_kind_for_query(text: str) -> Optional[Literal["identity"]]:
    """Return 'identity' if the query is an identity or social-greeting query, else None."""
    cleaned = re.sub(r"[.,!?।\s]+", " ", text).strip().lower()
    if not cleaned:
        return None
    if cleaned in _IDENTITY_PHRASES_GU or cleaned in _IDENTITY_PHRASES_EN:
        return "identity"
    return None


# ── the classifiers ──────────────────────────────────────────────────────


def _caller_lang(turn: Turn) -> str:
    return (turn.target_lang or "gu").strip().lower()


def _process_id(turn: Turn) -> Optional[str]:
    return turn.call.process_id if turn.call is not None else None


def _fast_paths_apply(turn: Turn) -> bool:
    """Whether the greeting, identity and fragment fast paths may answer this turn.

    They answer only while nothing meaningful has been said yet. They also leave
    alone the reply to an outbound call's opening question: "હા બોલો", the most
    likely yes, is itself a greeting, and answering it with the greeting would
    lose the reply.
    """
    if turn.call is not None and turn.call.outbound_consent_turn:
        return False
    return not _has_meaningful_history(turn.history)


async def _canned_for_caller(
    render: RenderForCaller, text_en: str, target_lang: str, canned: dict[str, str], *, execution
) -> str:
    """The pre-written line for the caller's language when there is one, which
    skips a translation round-trip on a fixed reply; otherwise the English line
    rendered live."""
    key = (target_lang or "en").strip().lower()
    if key in canned:
        return canned[key]
    return await render(text_en, target_lang, execution=execution)


async def _session_execution(turn: Turn):
    """The turn's session on llm_core. A classifier is given only the turn, and the
    profile follows the session id, so its fixed reply runs on the turn's models."""
    return await llm_core.context(turn.session_id)


async def _stt_signal_classifier(turn: Turn) -> Optional[ClassifierResult]:
    """The STT sent a no-audio or unclear-speech sentinel instead of words.

    Asks the farmer to repeat; after ``stt_signal_retry_ceiling`` in a row, says
    to call back later. History keeps a marker, never the sentinel.
    """
    stt_signal = detect_stt_signal(turn.query)
    if stt_signal is None:
        return None
    logger.info(
        "STT signal detected; session_id=%s process_id=%s signal=%s",
        turn.session_id,
        _process_id(turn),
        stt_signal,
    )
    prior_stt_failures = count_consecutive_stt_signals(turn.history)
    final_attempt = (prior_stt_failures + 1) >= max(1, settings.stt_signal_retry_ceiling)
    stt_response = await generate_stt_signal_response(
        signal=stt_signal,
        target_lang=_caller_lang(turn),
        final_attempt=final_attempt,
    )
    history_signal = (
        _HISTORY_MARKERS["stt_no_audio"]
        if stt_signal == "No audio/User is speaking softly"
        else _HISTORY_MARKERS["stt_unclear"]
    )
    history_response = (
        _FRAGMENT_RESPONSES["en"]
        if not final_attempt
        else "Sorry, I still could not hear you clearly. Please try again later."
    )
    return ClassifierResult(
        canned_text=stt_response,
        label="stt_signal",
        history_pair=_history_pair(history_signal, history_response),
    )


async def _hold_message_classifier(turn: Turn) -> Optional[ClassifierResult]:
    """A carrier "your call is on hold" message: say goodbye so the provider hangs up.

    Persists nothing, so the hold loop leaves no trace in the conversation.
    """
    if not _is_hold_message(turn.query):
        return None
    logger.info(
        "Hold message detected; responding with goodbye to cut call - session_id=%s process_id=%s query=%r",
        turn.session_id, _process_id(turn), turn.query[:100],
    )
    goodbye = TELEPHONY_TERMINATE_CALL_TOKEN.get(
        _caller_lang(turn),
        TELEPHONY_TERMINATE_CALL_TOKEN["en"],
    )
    # The telephony hang-up token must stay exact ASCII "Goodbye.". The Gujarati
    # normalizer would strip the Latin letters and leave only ".".
    return ClassifierResult(canned_text=goodbye, label="hold_message", raw=True)


async def _greeting_classifier(turn: Turn, *, render: RenderForCaller) -> Optional[ClassifierResult]:
    """A bare "hello" at the start of a call: greet back without the agent.

    History keeps the English greeting, so later turns stay in English.
    """
    if not _is_bare_greeting(turn.query) or not _fast_paths_apply(turn):
        return None
    logger.info(
        "Bare greeting detected; short-circuiting - session_id=%s process_id=%s query=%r",
        turn.session_id, _process_id(turn), turn.query,
    )
    greeting_history = _GREETING_RESPONSES["en"]
    greeting_response = await _canned_for_caller(
        render, greeting_history, _caller_lang(turn), _GREETING_RESPONSES,
        execution=await _session_execution(turn),
    )
    return ClassifierResult(
        canned_text=greeting_response,
        label="greeting_fast_path",
        history_pair=_history_pair(_HISTORY_MARKERS["greeting"], greeting_history),
    )


async def _identity_classifier(turn: Turn, *, render: RenderForCaller) -> Optional[ClassifierResult]:
    """"What is your name?" at the start of a call: the Sarlaben identity line.

    Voice's line, not chat's identity table.
    """
    if _fast_path_kind_for_query(turn.query) != "identity" or not _fast_paths_apply(turn):
        return None
    logger.info(
        "Identity fast-path triggered; session_id=%s process_id=%s query=%r",
        turn.session_id, _process_id(turn), turn.query,
    )
    identity_resp_for_caller = await render(
        _IDENTITY_RESPONSE_EN, _caller_lang(turn), execution=await _session_execution(turn)
    )
    return ClassifierResult(
        canned_text=identity_resp_for_caller,
        label="identity_fast_path",
        history_pair=_history_pair(_HISTORY_MARKERS["greeting"], _IDENTITY_RESPONSE_EN),
    )


async def _fragment_classifier(turn: Turn, *, render: RenderForCaller) -> Optional[ClassifierResult]:
    """Very short or garbled input (≤3 characters) that is not a greeting or an
    STT signal: ask the farmer to repeat instead of sending it to the agent."""
    if not _is_fragment_query(turn.query) or not _fast_paths_apply(turn):
        return None
    logger.info(
        "Fragment query detected; short-circuiting - session_id=%s process_id=%s query=%r",
        turn.session_id, _process_id(turn), turn.query,
    )
    frag_response_for_history = _FRAGMENT_RESPONSES["en"]
    frag_response_for_caller = await _canned_for_caller(
        render, frag_response_for_history, _caller_lang(turn), _FRAGMENT_RESPONSES,
        execution=await _session_execution(turn),
    )
    return ClassifierResult(
        canned_text=frag_response_for_caller,
        label="fragment_fast_path",
        history_pair=_history_pair(_HISTORY_MARKERS["fragment"], frag_response_for_history),
    )


def voice_classifiers(render: RenderForCaller) -> tuple[Classifier, ...]:
    """Voice's chain, in voice's order, for ``SurfaceProfile.classifiers``.

    The labels are voice's route and outcome names, which ``voice_turns``
    already counts as ``non_question``.
    """
    return (
        _stt_signal_classifier,
        _hold_message_classifier,
        partial(_greeting_classifier, render=render),
        partial(_identity_classifier, render=render),
        partial(_fragment_classifier, render=render),
    )
