"""Outbound-call consent classifier.

On an outbound call we open with one scripted question ("may I read out your last
7 days of milk deposits?"). The farmer's first reply is not a yes/no token — it is
free speech through a noisy ASR, and the single most valuable reply ("my cow isn't
eating") is neither yes nor no. So this is a small LLM gate, not a matcher.

Three-way verdict: ``affirmative`` / ``negative`` / ``other``.

Fail direction differs from the non-meaningful classifier (which fails open to
"allow"). This one fails to ``other`` on timeout, parse error, or any exception:
``other`` hands the turn to the agent, so an uncertain verdict can never hang up
on a farmer who said yes, and can never read account data to one who said no.

Brought over from voice-oan-api (``app/services/outbound_consent.py`` at amul-dev
``3b19835``). Only the client's source changed: the turn's ``ExecutionContext``
instead of voice's resolver.
"""

import asyncio
import json
from dataclasses import dataclass, replace
from typing import Optional

from openai import AsyncOpenAI

from app.config import settings
from app.llm_core.config_model import Step
from app.llm_core.execution import ExecutionContext
from app.services.translation import _get_langfuse
from helpers.utils import get_logger, get_prompt

logger = get_logger(__name__)

OUTBOUND_CONSENT_PROMPT_NAME = "voice_outbound_consent_en"
_STATIC_OUTBOUND_CONSENT_SYSTEM_PROMPT = get_prompt(OUTBOUND_CONSENT_PROMPT_NAME)

INTENT_AFFIRMATIVE = "affirmative"
INTENT_NEGATIVE = "negative"
INTENT_OTHER = "other"
_VALID_INTENTS = {INTENT_AFFIRMATIVE, INTENT_NEGATIVE, INTENT_OTHER}


def _consent_client_and_model(execution: ExecutionContext) -> tuple[AsyncOpenAI, str, str]:
    """(client, model, provider_label) for the consent classifier.

    Reuses the NON_MEANINGFUL step tier rather than introducing a new pipeline
    step: it is the same small ``RAW_OPENAI`` classifier call, and a new Step
    would ripple through the profile schema, the Redis config (M2) and the
    parity checks for no behavioural gain. Promote it to its own step only if
    evaluation shows this gate wants a different model.
    """
    mt = replace(execution, profile_name="managed").target(Step.NON_MEANINGFUL)  # profile-invariant (defaults)
    return mt.handle, mt.model_name, mt.provider


@dataclass(frozen=True)
class ConsentVerdict:
    intent: str
    reason: str
    raw_output: Optional[str] = None
    failed_open: bool = False

    @property
    def is_affirmative(self) -> bool:
        return self.intent == INTENT_AFFIRMATIVE

    @property
    def is_negative(self) -> bool:
        return self.intent == INTENT_NEGATIVE


def _other(reason: str, *, raw_output: Optional[str] = None, failed_open: bool = False) -> ConsentVerdict:
    return ConsentVerdict(
        intent=INTENT_OTHER,
        reason=reason,
        raw_output=raw_output,
        failed_open=failed_open,
    )


def _parse_verdict(raw: str) -> ConsentVerdict:
    stripped = (raw or "").strip()
    if not stripped:
        logger.warning("Consent classifier returned empty output; falling back to 'other'")
        return _other("empty model output", raw_output=raw, failed_open=True)

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        logger.warning("Consent classifier returned non-JSON output; falling back to 'other' - raw=%r", stripped[:200])
        return _other("non-json model output", raw_output=raw, failed_open=True)

    if not isinstance(data, dict):
        logger.warning("Consent classifier returned non-object JSON; falling back to 'other' - raw=%r", stripped[:200])
        return _other("non-object model output", raw_output=raw, failed_open=True)

    intent = data.get("intent")
    if not isinstance(intent, str) or intent.strip().lower() not in _VALID_INTENTS:
        logger.warning(
            "Consent classifier returned invalid intent=%r; falling back to 'other' - raw=%r",
            intent,
            stripped[:200],
        )
        return _other("invalid intent value", raw_output=raw, failed_open=True)

    intent = intent.strip().lower()
    reason = (data.get("reason") or "").strip()[:200]
    return ConsentVerdict(
        intent=intent,
        reason=reason or intent,
        raw_output=raw,
        failed_open=False,
    )


def _build_messages(reply: str, source_lang: str) -> list[dict[str, str]]:
    user_content = (
        f"Source language: {source_lang}\n\n"
        f"Farmer's reply to the consent question:\n{(reply or '').strip()}"
    )
    return [
        {"role": "system", "content": _STATIC_OUTBOUND_CONSENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


async def _create_consent_response(client: AsyncOpenAI, model: str, reply: str, source_lang: str):
    return await asyncio.wait_for(
        client.chat.completions.create(
            model=model,
            messages=_build_messages(reply, source_lang),
            max_completion_tokens=80,
            response_format={"type": "json_object"},
        ),
        timeout=settings.voice_outbound_consent_timeout_seconds,
    )


async def classify_consent(reply: str, source_lang: str, *, execution: ExecutionContext) -> ConsentVerdict:
    """Classify the farmer's first reply on an outbound call. Never raises."""
    if not (reply or "").strip():
        return _other("empty caller reply", failed_open=False)

    try:
        client, model, provider = _consent_client_and_model(execution)
    except Exception as e:
        logger.error("Consent client init failed (%s); falling back to 'other'", e)
        return _other(f"classifier client error: {type(e).__name__}", failed_open=True)
    langfuse = _get_langfuse()

    try:
        if not langfuse:
            response = await _create_consent_response(client, model, reply, source_lang)
            return _parse_verdict((response.choices[0].message.content or "").strip())

        with langfuse.start_as_current_observation(
            name="outbound_consent_classifier",
            as_type="generation",
            input={"source_lang": source_lang, "reply": reply},
            model=model,
            metadata={
                "pipeline_stage": "outbound_consent_classifier",
                "provider": provider,
            },
        ) as observation:
            response = await _create_consent_response(client, model, reply, source_lang)
            verdict = _parse_verdict((response.choices[0].message.content or "").strip())
            observation.update(
                output={
                    "intent": verdict.intent,
                    "reason": verdict.reason,
                    "failed_open": verdict.failed_open,
                }
            )
            return verdict
    except asyncio.TimeoutError:
        logger.error(
            "Consent classifier timed out - source_lang=%s timeout=%.2fs",
            source_lang,
            settings.voice_outbound_consent_timeout_seconds,
        )
        return _other("classifier timeout", failed_open=True)
    except Exception as e:
        logger.error("Consent classifier failed - source_lang=%s error=%s", source_lang, e)
        return _other(f"classifier error: {type(e).__name__}", failed_open=True)
