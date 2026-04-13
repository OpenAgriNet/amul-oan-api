"""Feedback flow helpers for voice conversations."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Literal, Optional

from app.config import settings
from helpers.utils import get_logger

logger = get_logger(__name__)

_FEEDBACK_MESSAGES_PATH = (
    Path(__file__).resolve().parent.parent.parent / "assets" / "feedback_messages.json"
)
with open(_FEEDBACK_MESSAGES_PATH, "r", encoding="utf-8") as _f:
    _FEEDBACK_MESSAGES: dict = json.load(_f)

RATING_MAP = {
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "૧": 1,
    "૨": 2,
    "૩": 3,
    "૪": 4,
    "૫": 5,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "એક": 1,
    "બે": 2,
    "ત્રણ": 3,
    "ચાર": 4,
    "પાંચ": 5,
    "પંચ": 5,
}


def get_feedback_question(lang_code: str = "gu") -> str:
    return _FEEDBACK_MESSAGES["question"].get(lang_code, _FEEDBACK_MESSAGES["question"]["gu"])


def get_feedback_ack(rating: int, lang_code: str = "gu") -> str:
    key = "ack_low" if rating <= 3 else "ack_high"
    return _FEEDBACK_MESSAGES[key].get(lang_code, _FEEDBACK_MESSAGES[key]["gu"])


def parse_rating_from_text(text: str) -> Optional[int]:
    if not text or not isinstance(text, str):
        return None
    cleaned = re.sub(r"\s+", " ", text.strip()).lower()
    if not cleaned:
        return None
    if cleaned in RATING_MAP:
        return RATING_MAP[cleaned]
    for token in re.split(r"[\s,]+", cleaned):
        if token in RATING_MAP:
            return RATING_MAP[token]
    match = re.search(r"[1-5૧૨૩૪૫]", text)
    if match:
        return RATING_MAP.get(match.group())
    return None


FEEDBACK_PARSE_SYSTEM = """You classify the user's reply after a voice assistant asked:
"On a scale of 1 to 5, how helpful was this call?"
Respond with JSON only: {"is_feedback": true|false, "rating": 1-5 or null}."""

FEEDBACK_PARSE_USER_TEMPLATE = """User's reply: "{text}"
Respond with exactly one JSON object:
{"is_feedback": <true|false>, "rating": <1-5|null>}."""


async def parse_feedback_with_llm(text: str, lang_code: str = "gu") -> dict:
    if not text or not isinstance(text, str) or not text.strip():
        return {"is_feedback": False, "rating": None}

    regex_rating = parse_rating_from_text(text)
    if regex_rating is not None:
        return {"is_feedback": True, "rating": regex_rating}

    api_key = settings.openai_api_key
    if not api_key:
        logger.warning("No OPENAI_API_KEY for feedback parse; treating as non-feedback")
        return {"is_feedback": False, "rating": None}

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model=settings.feedback_parse_model if hasattr(settings, "feedback_parse_model") else "gpt-5-mini",
            messages=[
                {"role": "system", "content": FEEDBACK_PARSE_SYSTEM},
                {"role": "user", "content": FEEDBACK_PARSE_USER_TEMPLATE.format(text=text.strip()[:500])},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=80,
        )
        raw = (response.choices[0].message.content or "").strip()
        if not raw:
            return {"is_feedback": False, "rating": None}
        data = json.loads(raw)
        is_feedback = data.get("is_feedback") is True
        rating = data.get("rating")
        if rating is not None and isinstance(rating, (int, float)):
            r = int(rating)
            if 1 <= r <= 5:
                return {"is_feedback": is_feedback, "rating": r}
        return {"is_feedback": False, "rating": None}
    except Exception as e:
        logger.warning("Feedback parse LLM failed: %s", e)
        return {"is_feedback": False, "rating": None}


async def send_feedback(
    session_id: str,
    user_id: str,
    process_id: Optional[str],
    rating: int,
    trigger: Literal["conversation_closing", "user_frustration"],
    source_lang: str,
    target_lang: str,
    message_history_summary: Optional[dict] = None,
    farmer_info: Optional[dict] = None,
    raw_input: Optional[str] = None,
) -> None:
    comment_parts = [f"trigger={trigger}", f"source_lang={source_lang}", f"target_lang={target_lang}"]
    if user_id and user_id != "anonymous":
        comment_parts.append(f"user_id={user_id}")
    if process_id:
        comment_parts.append(f"process_id={process_id}")
    if raw_input:
        raw_preview = (raw_input[:200] + "…") if len(raw_input) > 200 else raw_input
        comment_parts.append(f"raw_input={raw_preview}")
    comment = "; ".join(comment_parts)

    try:
        from app.observability import langfuse_client

        if langfuse_client is not None:
            def _create_and_flush() -> None:
                langfuse_client.create_score(
                    session_id=session_id,
                    name="user-feedback",
                    value=float(rating),
                    comment=comment,
                    data_type="NUMERIC",
                    score_id=f"{session_id}-user-feedback",
                )
                langfuse_client.flush()

            await asyncio.to_thread(_create_and_flush)
    except Exception as e:
        logger.warning("Failed to store feedback in Langfuse: %s", e)

    logger.info("send_feedback: session_id=%s rating=%s trigger=%s", session_id, rating, trigger)

