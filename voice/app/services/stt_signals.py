"""
Handle special STT signals (no-audio, unclear speech) with contextual responses.

When the STT pipeline cannot transcribe the user's audio it sends sentinel
strings instead of real text.  We intercept these early, skip translation
and the main agent, and return a hardcoded 'please repeat' prompt.
"""

import random
import re
from typing import Optional, Sequence

from pydantic_ai.messages import ModelMessage

from voice.helpers.utils import get_logger

logger = get_logger(__name__)

# ── Signal detection ────────────────────────────────────────────────────
# Strip leading/trailing whitespace and surrounding * markers, then match.
_SIGNAL_NO_AUDIO = "No audio/User is speaking softly"
_SIGNAL_UNCLEAR = "Unclear Speech"


def _normalize(text: str) -> str:
    return re.sub(r"[*]", "", text).strip().lower()


_SIGNALS = {
    _normalize(s): s
    for s in (_SIGNAL_NO_AUDIO, _SIGNAL_UNCLEAR)
}
_SIGNALS.update({
    "[stt:no-audio]": _SIGNAL_NO_AUDIO,
    "[stt:unclear-speech]": _SIGNAL_UNCLEAR,
})


def detect_stt_signal(query: str) -> Optional[str]:
    """Return the canonical signal name if *query* is a known STT signal, else None."""
    if not query:
        return None
    return _SIGNALS.get(_normalize(query))


# ── Hardcoded fallback responses ──────────────────────────────────────
# Rotated randomly so repeated failures don't sound robotic.
_FALLBACK_NO_AUDIO: dict[str, list[str]] = {
    "gu": [
        "માફ કરશો, મને તમારો અવાજ સંભળાતો નથી. કૃપા કરીને ફોન નજીક રાખીને ફરીથી બોલો.",
        "હું તમને સાંભળી શકતી નથી. થોડું મોટેથી બોલી શકશો?",
        "તમારો અવાજ આવતો નથી. કૃપા કરીને ફરીથી બોલો.",
    ],
    "en": [
        "Sorry, I can't hear you. Could you please speak a little louder?",
        "I wasn't able to hear that. Could you move closer to the phone and try again?",
        "I didn't catch any audio. Please try speaking again.",
    ],
}

_FALLBACK_UNCLEAR: dict[str, list[str]] = {
    "gu": [
        "માફ કરશો, મને તમારી વાત બરાબર સમજાઈ નહીં. કૃપા કરીને ફરીથી બોલો.",
        "તમે શું કહ્યું એ સ્પષ્ટ સંભળાયું નહીં. થોડું ધીમેથી ફરી કહેશો?",
        "મને બરાબર સમજાયું નહીં. કૃપા કરીને ફરીથી કહો.",
    ],
    "en": [
        "Sorry, I couldn't understand that clearly. Could you please repeat?",
        "That wasn't clear to me. Could you say it again, a bit slowly?",
        "I didn't quite catch what you said. Please try again.",
    ],
}

_FINAL_NO_AUDIO: dict[str, list[str]] = {
    "gu": [
        "માફ કરશો, હજુ તમારો અવાજ સંભળાતો નથી. કૃપા કરીને પછીથી ફરી પ્રયાસ કરો.",
        "હાલમાં અવાજ સ્પષ્ટ નથી. કૃપા કરીને થોડા સમય પછી ફરી કોલ કરો.",
    ],
    "en": [
        "Sorry, I still can't hear you. Please try again later.",
        "The audio is still unclear. Please call again later.",
    ],
}

_FINAL_UNCLEAR: dict[str, list[str]] = {
    "gu": [
        "માફ કરશો, તમારી વાત હજી સ્પષ્ટ સમજાઈ નથી. કૃપા કરીને પછીથી ફરી પ્રયાસ કરો.",
        "હું હજી તમારી વાત સમજી શકતી નથી. કૃપા કરીને થોડા સમય પછી ફરી કોલ કરો.",
    ],
    "en": [
        "Sorry, I still couldn't understand that clearly. Please try again later.",
        "I'm still not able to understand. Please call again later.",
    ],
}


def _pick_fallback(signal: str, target_lang: str) -> str:
    pool = _FALLBACK_NO_AUDIO if signal == _SIGNAL_NO_AUDIO else _FALLBACK_UNCLEAR
    responses = pool.get(target_lang, pool["en"])
    return random.choice(responses)


def _pick_final_fallback(signal: str, target_lang: str) -> str:
    pool = _FINAL_NO_AUDIO if signal == _SIGNAL_NO_AUDIO else _FINAL_UNCLEAR
    responses = pool.get(target_lang, pool["en"])
    return random.choice(responses)


def count_consecutive_stt_signals(messages: Sequence[ModelMessage] | None) -> int:
    """Count consecutive STT signal user messages from the tail of history."""
    count = 0
    for msg in reversed(messages or []):
        user_text = None
        for part in getattr(msg, "parts", []) or []:
            if getattr(part, "part_kind", "") == "user-prompt":
                user_text = getattr(part, "content", None)
                break
        if not user_text:
            continue
        if detect_stt_signal(user_text) is not None:
            count += 1
            continue
        break
    return count


async def generate_stt_signal_response(
    signal: str,
    target_lang: str,
    recent_history_text: str = "",
    final_attempt: bool = False,
) -> str:
    """Return a short 'please repeat' message using hardcoded fallbacks.

    Args:
        signal: canonical signal name (from detect_stt_signal).
        target_lang: language code for the response (e.g. "gu", "en").
        recent_history_text: unused, kept for API compatibility.
        final_attempt: if True, return a final disconnect message.

    Returns:
        A single-sentence response string.
    """
    if final_attempt:
        return _pick_final_fallback(signal, target_lang)

    return _pick_fallback(signal, target_lang)
