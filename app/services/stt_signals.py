"""
Handle special STT signals (no-audio, unclear speech) with contextual responses.
"""

import random
import re
from typing import Optional

from openai import AsyncOpenAI

from app.config import settings
from helpers.utils import get_logger

logger = get_logger(__name__)

_SIGNAL_NO_AUDIO = "No audio/User is speaking softly"
_SIGNAL_UNCLEAR = "Unclear Speech"


def _normalize(text: str) -> str:
    return re.sub(r"[*]", "", text).strip().lower()


_SIGNALS = {_normalize(s): s for s in (_SIGNAL_NO_AUDIO, _SIGNAL_UNCLEAR)}


def detect_stt_signal(query: str) -> Optional[str]:
    if not query:
        return None
    return _SIGNALS.get(_normalize(query))


STT_RESPONSE_MODEL = "gpt-5-mini"

_SYSTEM_PROMPT = """\
You are a friendly voice assistant on a phone call with a farmer.
The speech-to-text system reported a problem with the caller's audio.

Your ONLY job is to produce a single short sentence (in the target language)
politely asking the user to repeat or speak louder.
"""

_openai_client: Optional[AsyncOpenAI] = None

_FALLBACK_NO_AUDIO = {
    "gu": [
        "માફ કરશો, મને તમારો અવાજ સંભળાતો નથી. કૃપા કરીને ફોન નજીક રાખીને ફરીથી બોલો.",
        "હું તમને સાંભળી શકતી નથી. થોડું મોટેથી બોલી શકશો?",
    ],
    "en": [
        "Sorry, I can't hear you. Could you please speak a little louder?",
        "I didn't catch any audio. Please try speaking again.",
    ],
}

_FALLBACK_UNCLEAR = {
    "gu": [
        "માફ કરશો, મને તમારી વાત બરાબર સમજાઈ નહીં. કૃપા કરીને ફરીથી બોલો.",
        "મને બરાબર સમજાયું નહીં. કૃપા કરીને ફરીથી કહો.",
    ],
    "en": [
        "Sorry, I couldn't understand that clearly. Could you please repeat?",
        "I didn't quite catch what you said. Please try again.",
    ],
}


def _pick_fallback(signal: str, target_lang: str) -> str:
    pool = _FALLBACK_NO_AUDIO if signal == _SIGNAL_NO_AUDIO else _FALLBACK_UNCLEAR
    responses = pool.get(target_lang, pool["en"])
    return random.choice(responses)


def _get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = settings.openai_api_key
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for STT signal responses")
        _openai_client = AsyncOpenAI(api_key=api_key)
    return _openai_client


async def generate_stt_signal_response(
    signal: str,
    target_lang: str,
    recent_history_text: str = "",
) -> str:
    lang_label = {"gu": "Gujarati", "en": "English"}.get(target_lang, target_lang)
    user_content = f"STT signal: {signal}\nTarget language: {lang_label}\n"
    if recent_history_text:
        user_content += f"\nRecent conversation:\n{recent_history_text}\n"

    try:
        client = _get_openai_client()
        response = await client.chat.completions.create(
            model=STT_RESPONSE_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_completion_tokens=150,
            temperature=0.7,
        )
        text = (response.choices[0].message.content or "").strip()
        if text:
            return text
    except Exception:
        logger.warning(
            "STT signal LLM call failed, using hardcoded fallback - signal=%s lang=%s model=%s",
            signal,
            target_lang,
            STT_RESPONSE_MODEL,
            exc_info=True,
        )

    return _pick_fallback(signal, target_lang)

