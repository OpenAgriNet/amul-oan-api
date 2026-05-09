"""
Voice content moderation.

Runs an LLM classifier in parallel with query pretranslation to catch
inputs that are out-of-scope for the Amul dairy helpline — irrelevant,
offensive, culturally sensitive, or aberrant usage of the helpline.

Fail-open: on timeout, parse error, or any exception we allow the query
through (label `in_scope`). A flaky moderation call must not drop a
legitimate farmer call.
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Literal, Optional

from openai import AsyncOpenAI

from voice.app.config import settings
from voice.app.services.translation import (
    OPENAI_PRETRANSLATION_MODEL,
    _get_langfuse,
    _get_openai_client,
)
from voice.helpers.utils import get_logger, get_prompt

logger = get_logger(__name__)


ModerationCategory = Literal[
    "in_scope",
    "irrelevant",
    "offensive",
    "cultural_sensitivity",
    "aberration",
]


REJECT_CATEGORIES: frozenset[str] = frozenset(
    {"irrelevant", "offensive", "cultural_sensitivity", "aberration"}
)


DECLINE_MESSAGES_EN: dict[str, str] = {
    "irrelevant": (
        "This helpline answers questions about animal health, dairy, and farming. "
        "Do you have a question about your animals?"
    ),
    "offensive": (
        "This is a service for farmers. Please keep the conversation respectful, "
        "otherwise I will have to end the call."
    ),
    "cultural_sensitivity": (
        "I cannot discuss this topic. "
        "Do you have a question about your animals or farming?"
    ),
    "aberration": (
        "This helpline only handles dairy farming and animal husbandry questions. "
        "For other matters, please contact the appropriate service."
    ),
}


MODERATION_PROMPT_NAME = "voice_moderation_en"
_STATIC_MODERATION_SYSTEM_PROMPT = get_prompt(MODERATION_PROMPT_NAME)


@dataclass(frozen=True)
class ModerationVerdict:
    category: ModerationCategory
    reason: str
    raw_output: Optional[str] = None
    failed_open: bool = False

    @property
    def rejected(self) -> bool:
        return self.category in REJECT_CATEGORIES

    def decline_text_en(self) -> Optional[str]:
        return DECLINE_MESSAGES_EN.get(self.category)


def _allow(reason: str, *, raw_output: Optional[str] = None, failed_open: bool = False) -> ModerationVerdict:
    return ModerationVerdict(
        category="in_scope",
        reason=reason,
        raw_output=raw_output,
        failed_open=failed_open,
    )


def _parse_verdict(raw: str) -> ModerationVerdict:
    """Parse the model's JSON output. Fail-open on any parsing issue."""
    stripped = (raw or "").strip()
    if not stripped:
        logger.warning("Moderation returned empty output; failing open")
        return _allow("empty model output", raw_output=raw, failed_open=True)

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        logger.warning("Moderation returned non-JSON output; failing open - raw=%r", stripped[:200])
        return _allow("non-json model output", raw_output=raw, failed_open=True)

    if not isinstance(data, dict):
        logger.warning("Moderation returned non-object JSON; failing open - raw=%r", stripped[:200])
        return _allow("non-object model output", raw_output=raw, failed_open=True)

    category = (data.get("category") or "").strip().lower()
    reason = (data.get("reason") or "").strip()[:200]

    valid = {"in_scope", "irrelevant", "offensive", "cultural_sensitivity", "aberration"}
    if category not in valid:
        logger.warning(
            "Moderation returned unknown category=%r; failing open - raw=%r",
            category,
            stripped[:200],
        )
        return _allow(f"unknown category: {category}", raw_output=raw, failed_open=True)

    return ModerationVerdict(category=category, reason=reason, raw_output=raw)  # type: ignore[arg-type]


def _build_messages(
    text: str,
    source_lang: str,
    recent_history_text: str = "",
) -> list[dict[str, str]]:
    user_parts = [f"Source language: {source_lang}"]
    if recent_history_text.strip():
        user_parts.append(f"Recent conversation context:\n{recent_history_text.strip()}")
    user_parts.append(f"Caller utterance:\n{text.strip()}")
    user_content = "\n\n".join(user_parts)
    return [
        {"role": "system", "content": _STATIC_MODERATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


async def _create_moderation_response(
    client: AsyncOpenAI,
    text: str,
    source_lang: str,
    recent_history_text: str = "",
):
    return await asyncio.wait_for(
        client.chat.completions.create(
            model=OPENAI_PRETRANSLATION_MODEL,
            messages=_build_messages(text, source_lang, recent_history_text),
            max_completion_tokens=200,
            response_format={"type": "json_object"},
        ),
        timeout=settings.openai_pretranslation_timeout_seconds,
    )


async def check_moderation(
    text: str,
    source_lang: str,
    recent_history_text: str = "",
) -> ModerationVerdict:
    """Classify a caller utterance. Returns a ModerationVerdict.

    Fails open on any error — the returned verdict will have
    `category == "in_scope"` and `failed_open == True` so the caller can
    still log the failure but will not block the farmer.
    """
    if not text or not text.strip():
        return _allow("empty input", failed_open=False)

    client = _get_openai_client()
    langfuse = _get_langfuse()

    try:
        if not langfuse:
            response = await _create_moderation_response(
                client,
                text,
                source_lang,
                recent_history_text,
            )
            raw = (response.choices[0].message.content or "").strip()
            return _parse_verdict(raw)

        with langfuse.start_as_current_observation(
            name="query_moderation",
            as_type="generation",
            input={
                "source_lang": source_lang,
                "text": text,
                "recent_history_text": recent_history_text,
            },
            model=OPENAI_PRETRANSLATION_MODEL,
            metadata={
                "pipeline_stage": "query_moderation",
                "moderation_provider": "openai",
            },
        ) as observation:
            response = await _create_moderation_response(
                client,
                text,
                source_lang,
                recent_history_text,
            )
            raw = (response.choices[0].message.content or "").strip()
            verdict = _parse_verdict(raw)
            observation.update(
                output={
                    "category": verdict.category,
                    "reason": verdict.reason,
                    "failed_open": verdict.failed_open,
                },
                metadata={"rejected": verdict.rejected},
            )
            return verdict
    except asyncio.TimeoutError:
        logger.error(
            "Moderation timed out - source_lang=%s model=%s timeout=%.2fs query_chars=%s query_preview=%r",
            source_lang,
            OPENAI_PRETRANSLATION_MODEL,
            settings.openai_pretranslation_timeout_seconds,
            len(text),
            text[:160],
        )
        return _allow("moderation timeout", failed_open=True)
    except Exception as e:
        logger.error(
            "Moderation failed - source_lang=%s error=%s query_preview=%r",
            source_lang,
            e,
            text[:160],
        )
        return _allow(f"moderation error: {type(e).__name__}", failed_open=True)
