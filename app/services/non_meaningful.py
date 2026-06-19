"""
Non-meaningful turn streak classifier for voice calls.

This classifier is intentionally independent from moderation. It decides whether
the latest five user turns are all non-meaningful for progressing support.

Fail-open: on timeout, parse error, or any exception, return False so normal
conversation flow is not blocked.
"""

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI

from app.config import settings
from app.services.translation import (
    OPENAI_PRETRANSLATION_MODEL,
    OSS_PRETRANSLATION_MODEL,
    _get_langfuse,
    _get_openai_client,
    _get_oss_pretranslation_client,
)
from helpers.utils import get_logger, get_prompt

logger = get_logger(__name__)

NON_MEANINGFUL_PROMPT_NAME = "voice_non_meaningful_en"
_STATIC_NON_MEANINGFUL_SYSTEM_PROMPT = get_prompt(NON_MEANINGFUL_PROMPT_NAME)
_NON_MEANINGFUL_PROVIDER = (os.getenv("VOICE_NON_MEANINGFUL_PROVIDER", "vllm") or "vllm").strip().lower()


def _non_meaningful_client_and_model() -> tuple[AsyncOpenAI, str, str]:
    if _NON_MEANINGFUL_PROVIDER == "openai":
        return _get_openai_client(), OPENAI_PRETRANSLATION_MODEL, "openai"
    return _get_oss_pretranslation_client(), OSS_PRETRANSLATION_MODEL, "vllm"


@dataclass(frozen=True)
class NonMeaningfulVerdict:
    five_consecutive_non_meaningful: bool
    reason: str
    raw_output: Optional[str] = None
    failed_open: bool = False


def _allow(reason: str, *, raw_output: Optional[str] = None, failed_open: bool = False) -> NonMeaningfulVerdict:
    return NonMeaningfulVerdict(
        five_consecutive_non_meaningful=False,
        reason=reason,
        raw_output=raw_output,
        failed_open=failed_open,
    )


def _parse_verdict(raw: str) -> NonMeaningfulVerdict:
    stripped = (raw or "").strip()
    if not stripped:
        logger.warning("Non-meaningful classifier returned empty output; failing open")
        return _allow("empty model output", raw_output=raw, failed_open=True)

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        logger.warning("Non-meaningful classifier returned non-JSON output; failing open - raw=%r", stripped[:200])
        return _allow("non-json model output", raw_output=raw, failed_open=True)

    if not isinstance(data, dict):
        logger.warning("Non-meaningful classifier returned non-object JSON; failing open - raw=%r", stripped[:200])
        return _allow("non-object model output", raw_output=raw, failed_open=True)

    flag = data.get("five_consecutive_non_meaningful")
    if not isinstance(flag, bool):
        logger.warning(
            "Non-meaningful classifier returned invalid flag=%r; failing open - raw=%r",
            flag,
            stripped[:200],
        )
        return _allow("invalid non-meaningful flag", raw_output=raw, failed_open=True)

    reason = (data.get("reason") or "").strip()[:200]
    return NonMeaningfulVerdict(
        five_consecutive_non_meaningful=flag,
        reason=reason or ("five non-meaningful turns" if flag else "not five non-meaningful turns"),
        raw_output=raw,
        failed_open=False,
    )


def _build_messages(user_turns: list[str], source_lang: str) -> list[dict[str, str]]:
    turns = [turn.strip() for turn in user_turns if isinstance(turn, str) and turn.strip()]
    numbered_turns = "\n".join(f"{idx}. {turn}" for idx, turn in enumerate(turns, start=1))
    user_content = (
        f"Source language: {source_lang}\n\n"
        f"Recent 5 caller turns (oldest to newest):\n{numbered_turns}"
    )
    return [
        {"role": "system", "content": _STATIC_NON_MEANINGFUL_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


async def _create_non_meaningful_response(
    client: AsyncOpenAI,
    model: str,
    user_turns: list[str],
    source_lang: str,
):
    return await asyncio.wait_for(
        client.chat.completions.create(
            model=model,
            messages=_build_messages(user_turns, source_lang),
            max_completion_tokens=120,
            response_format={"type": "json_object"},
        ),
        timeout=settings.voice_non_meaningful_timeout_seconds,
    )


async def check_non_meaningful_streak(
    user_turns: list[str],
    source_lang: str,
) -> NonMeaningfulVerdict:
    turns = [turn for turn in user_turns if isinstance(turn, str) and turn.strip()]
    if len(turns) < 5:
        return _allow("fewer than five user turns", failed_open=False)

    try:
        client, model, provider = _non_meaningful_client_and_model()
    except Exception as e:
        logger.error("Non-meaningful client init failed (%s); failing open", e)
        return _allow(f"classifier client error: {type(e).__name__}", failed_open=True)
    langfuse = _get_langfuse()

    try:
        if not langfuse:
            response = await _create_non_meaningful_response(client, model, turns[-5:], source_lang)
            raw = (response.choices[0].message.content or "").strip()
            return _parse_verdict(raw)

        with langfuse.start_as_current_observation(
            name="non_meaningful_classifier",
            as_type="generation",
            input={
                "source_lang": source_lang,
                "turn_count": len(turns[-5:]),
                "user_turns": turns[-5:],
            },
            model=model,
            metadata={
                "pipeline_stage": "non_meaningful_classifier",
                "provider": provider,
            },
        ) as observation:
            response = await _create_non_meaningful_response(client, model, turns[-5:], source_lang)
            raw = (response.choices[0].message.content or "").strip()
            verdict = _parse_verdict(raw)
            observation.update(
                output={
                    "five_consecutive_non_meaningful": verdict.five_consecutive_non_meaningful,
                    "reason": verdict.reason,
                    "failed_open": verdict.failed_open,
                }
            )
            return verdict
    except asyncio.TimeoutError:
        logger.error(
            "Non-meaningful classifier timed out - source_lang=%s timeout=%.2fs turn_count=%s",
            source_lang,
            settings.voice_non_meaningful_timeout_seconds,
            len(turns[-5:]),
        )
        return _allow("classifier timeout", failed_open=True)
    except Exception as e:
        logger.error(
            "Non-meaningful classifier failed - source_lang=%s error=%s turn_count=%s",
            source_lang,
            e,
            len(turns[-5:]),
        )
        return _allow(f"classifier error: {type(e).__name__}", failed_open=True)
