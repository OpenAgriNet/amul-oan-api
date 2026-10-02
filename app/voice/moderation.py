"""
Voice content moderation.

Runs an LLM classifier in parallel with query pretranslation to catch
inputs that are out-of-scope for the Amul dairy helpline — irrelevant,
offensive, culturally sensitive, or aberrant usage of the helpline.

Fail-open: on timeout, parse error, or any exception we allow the query
through (label `in_scope`). A flaky moderation call must not drop a
legitimate farmer call.

Brought over from voice-oan-api (``app/services/moderation.py`` at amul-dev
``3b19835``). Categories, prompt, parsing, fail directions and timeouts are
voice's. What changed is where the tiers come from: the turn's
``ExecutionContext`` instead of voice's resolver, built as bare OpenAI clients
(``StepClientKind.RAW_OPENAI``), and the fallback switch is the pipeline
config's ``fallback_enabled``.
"""

import asyncio
import json
import os
from dataclasses import dataclass, replace
from typing import Literal, Optional

from openai import AsyncOpenAI

from app.config import settings
from app.llm_core.config_model import Step, StepClientKind
from app.llm_core.execution import ExecutionContext, classify
from app.services.translation import _get_langfuse
from helpers.utils import get_logger, get_prompt

logger = get_logger(__name__)


ModerationCategory = Literal[
    "in_scope",
    "irrelevant",
    "offensive",
    "cultural_sensitivity",
    "aberration",
    # Internal-only: not a model output. Marks a fail-CLOSED verdict (the
    # moderation gate could not produce a trustworthy result on any tier).
    "unavailable",
]


REJECT_CATEGORIES: frozenset[str] = frozenset(
    {"irrelevant", "offensive", "cultural_sensitivity", "aberration", "unavailable"}
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
    "unavailable": (
        "I'm having trouble processing your request right now. "
        "Please try again in a moment."
    ),
}


MODERATION_PROMPT_NAME = "voice_moderation_en"
_STATIC_MODERATION_SYSTEM_PROMPT = get_prompt(MODERATION_PROMPT_NAME)

# Voice moderation runs on the self-hosted gemma vLLM endpoint by default: it is
# fast (~0.2s, measured in the load sweep) and in-cluster, vs ~1.5s for the
# OpenAI gpt path that previously served it — which doubled as the long pole on
# warm-cache voice turns. Override with VOICE_MODERATION_PROVIDER=openai to fall
# back to the gpt model (OPENAI_PRETRANSLATION_MODEL).
_MODERATION_PROVIDER = (os.getenv("VOICE_MODERATION_PROVIDER", "vllm") or "vllm").strip().lower()


def _moderation_tier(execution: ExecutionContext, profile_name: str):
    """The primary MODERATION tier of a named profile, as a bare OpenAI client.

    A name the config does not have ("legacy") resolves to ``managed``, then to
    the first profile, as voice's resolver did."""
    return replace(execution, profile_name=profile_name).target(
        Step.MODERATION, client_kind=StepClientKind.RAW_OPENAI
    )


def _moderation_client_and_model(execution: ExecutionContext) -> tuple[AsyncOpenAI, str, str]:
    """Return (client, model, provider_label) for the configured moderation backend
    (legacy fail-open path: single global VOICE_MODERATION_PROVIDER).

    VOICE_MODERATION_PROVIDER governs the tier independent of session variant:
    openai -> the managed tier; else -> the OSS (vLLM) tier. The provider label
    is preserved exactly."""
    variant = "legacy" if _MODERATION_PROVIDER == "openai" else "oss"
    mt = _moderation_tier(execution, variant)
    return mt.handle, mt.model_name, ("openai" if _MODERATION_PROVIDER == "openai" else "vllm")


def _client_model_for_kind(execution: ExecutionContext, kind: str) -> tuple[AsyncOpenAI, str, str]:
    """Return (client, model, provider_label) for one fallback-chain attempt:
    'oss' -> self-hosted vLLM, anything else -> managed OpenAI.

    The client + model come from the MODERATION tier for the matching variant
    ('oss' tier vs managed tier), keeping the provider label byte-identical."""
    variant = "oss" if kind == "oss" else "legacy"
    mt = _moderation_tier(execution, variant)
    return mt.handle, mt.model_name, ("vllm" if kind == "oss" else "openai")


def _legacy_requested_model(tier: str) -> str:
    """The model the legacy path reports when its client could not be built: the
    pretranslation model moderation shares, from the same env and defaults as
    voice-oan-api's translation.py."""
    if tier == "oss":
        return os.getenv("OSS_PRETRANSLATION_MODEL", os.getenv("OSS_LLM_MODEL_NAME", "gemma-4-31b-it"))
    provider = os.getenv("PRETRANSLATION_PROVIDER", os.getenv("LLM_PROVIDER", "openai")).lower()
    default = os.getenv("LLM_MODEL_NAME", "gemma-4-31b-it") if provider == "vllm" else "gpt-5.1"
    return os.getenv("PRETRANSLATION_MODEL", os.getenv("OPENAI_PRETRANSLATION_MODEL", default))


@dataclass(frozen=True)
class ModerationVerdict:
    category: ModerationCategory
    reason: str
    raw_output: Optional[str] = None
    failed_open: bool = False
    failed_closed: bool = False
    requested_tier: Optional[str] = None
    requested_provider: Optional[str] = None
    requested_model: Optional[str] = None
    actual_tier: Optional[str] = None
    actual_provider: Optional[str] = None
    actual_model: Optional[str] = None
    fallback_used: Optional[bool] = None
    attempts: Optional[list[dict[str, object]]] = None

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


def _block_unavailable(reason: str, *, raw_output: Optional[str] = None) -> ModerationVerdict:
    """Fail-CLOSED verdict: blocks the turn with a generic 'try again' decline
    when moderation can't produce a trustworthy result."""
    return ModerationVerdict(
        category="unavailable",
        reason=reason,
        raw_output=raw_output,
        failed_closed=True,
    )


def _with_telemetry(
    verdict: ModerationVerdict,
    *,
    requested_tier: Optional[str],
    requested_provider: Optional[str],
    requested_model: Optional[str],
    actual_tier: Optional[str],
    actual_provider: Optional[str],
    actual_model: Optional[str],
    fallback_used: Optional[bool],
    attempts: Optional[list[dict[str, object]]],
) -> ModerationVerdict:
    return replace(
        verdict,
        requested_tier=requested_tier,
        requested_provider=requested_provider,
        requested_model=requested_model,
        actual_tier=actual_tier,
        actual_provider=actual_provider,
        actual_model=actual_model,
        fallback_used=fallback_used,
        attempts=attempts,
    )


def _parse_verdict(raw: str, *, fail_closed: bool = False) -> ModerationVerdict:
    """Parse the model's JSON output into a ModerationVerdict.

    ONE parser for both moderation policies (collapse of the former
    ``_parse_verdict`` / ``_parse_verdict_strict`` twins, which were line-identical
    bar the terminal on malformed/untrustworthy output). The ``fail_closed`` flag
    selects that terminal WITHOUT changing any classification:

      * ``fail_closed=False`` (default) — fail OPEN: malformed/unknown output
        allows the turn (``in_scope``, ``failed_open=True``). Today's behaviour on
        the ``FALLBACK_ENABLED``-off legacy path.
      * ``fail_closed=True`` — fail CLOSED: malformed/unknown output blocks the
        turn (``unavailable`` reject, ``failed_closed=True``). Today's behaviour on
        the fallback path so a garbage response blocks rather than waves through.

    A VALID verdict (including a reject category) is returned unchanged under both
    policies. This is a behaviour-preserving de-duplication only — pinned by
    tests/test_moderation_characterization.py."""
    disposition = "closed" if fail_closed else "open"

    def _reject(reason: str) -> ModerationVerdict:
        if fail_closed:
            return _block_unavailable(reason, raw_output=raw)
        return _allow(reason, raw_output=raw, failed_open=True)

    stripped = (raw or "").strip()
    if not stripped:
        logger.warning("Moderation returned empty output; failing %s", disposition)
        return _reject("empty model output")

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        logger.warning("Moderation returned non-JSON output; failing %s - raw=%r", disposition, stripped[:200])
        return _reject("non-json model output")

    if not isinstance(data, dict):
        logger.warning("Moderation returned non-object JSON; failing %s - raw=%r", disposition, stripped[:200])
        return _reject("non-object model output")

    category = (data.get("category") or "").strip().lower()
    reason = (data.get("reason") or "").strip()[:200]

    valid = {"in_scope", "irrelevant", "offensive", "cultural_sensitivity", "aberration"}
    if category not in valid:
        logger.warning(
            "Moderation returned unknown category=%r; failing %s - raw=%r",
            category, disposition, stripped[:200],
        )
        return _reject(f"unknown category: {category}")

    return ModerationVerdict(category=category, reason=reason, raw_output=raw)  # type: ignore[arg-type]


def _parse_verdict_strict(raw: str) -> ModerationVerdict:
    """Fail-CLOSED parser — thin alias over ``_parse_verdict(raw, fail_closed=True)``.
    Kept as a name for the fallback-path call site + existing tests."""
    return _parse_verdict(raw, fail_closed=True)


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
    model: str,
    text: str,
    source_lang: str,
    recent_history_text: str = "",
):
    return await asyncio.wait_for(
        client.chat.completions.create(
            model=model,
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
    *,
    execution: ExecutionContext,
) -> ModerationVerdict:
    """Classify a caller utterance. Returns a ModerationVerdict.

    With the pipeline's ``fallback_enabled`` (standard path): route by the session
    *variant* (OSS for OSS sessions, managed for legacy) through the OSS->managed
    fallback chain, and **fail CLOSED** — return an ``unavailable`` reject when no
    tier produces a trustworthy verdict. Mirrors amul-oan-api moderation. Failing
    closed only triggers when both OSS and managed fail, so it does not drop calls
    on a single-provider blip.

    Without it (legacy path): a single global provider
    (``VOICE_MODERATION_PROVIDER``) and **fail OPEN** — today's behaviour, so the
    kill-switch reverts exactly.
    """
    if not text or not text.strip():
        return _with_telemetry(
            _allow("empty input", failed_open=False),
            requested_tier="none",
            requested_provider="none",
            requested_model="none",
            actual_tier="none",
            actual_provider="none",
            actual_model="none",
            fallback_used=False,
            attempts=[],
        )

    if not execution.config.fallback_enabled:
        return await _check_moderation_legacy(
            text,
            source_lang,
            recent_history_text,
            execution=execution,
        )

    # Requested (primary) tier for this session's profile — resolved by NAME from the
    # unified config, so a 3rd profile's kind is honoured (vllm -> "oss"; managed
    # provider -> "managed"), not collapsed via a variant string.
    #
    # Finding voice#1 (fail-CLOSED): this resolution MUST NOT escape check_moderation.
    # It sat outside the fail-closed try below, so a resolver error (e.g. an N-way
    # config whose profile omits the ``moderation`` step -> ``ValueError("no config
    # for step=moderation")``) propagated to voice's ``_resolve_moderation`` handler,
    # which fails OPEN -> moderation bypassed. Default to a managed kind on ANY
    # error, and walk the managed profile's chain as voice's resolver degraded to: a
    # managed default NEVER bypasses, and the walk below still fail-CLOSEs if every
    # tier is unavailable. The client is built here, as voice's resolver built it,
    # so a tier that cannot be built takes the same managed default.
    walk = execution
    try:
        primary = execution.target(Step.MODERATION, client_kind=StepClientKind.RAW_OPENAI)
        primary.handle
        requested_kind = primary.kind
    except Exception:
        requested_kind = "managed"
        walk = replace(execution, profile_name="managed")
    _, requested_model, requested_provider = _client_model_for_kind(execution, requested_kind)
    attempts: list[dict[str, object]] = []
    actual_tier = requested_kind
    actual_provider = requested_provider
    actual_model = requested_model

    async def _run(attempt):
        nonlocal actual_tier, actual_provider, actual_model
        client, model, provider = _client_model_for_kind(execution, attempt.kind)
        attempt_info: dict[str, object] = {
            "tier": attempt.kind,
            "provider": provider,
            "model": model,
            "endpoint": attempt.endpoint,
            "status": "started",
        }
        attempts.append(attempt_info)
        try:
            response = await _create_moderation_response(client, model, text, source_lang, recent_history_text)
            raw = (response.choices[0].message.content or "").strip()
            verdict = _parse_verdict_strict(raw)
            attempt_info["status"] = "ok"
            actual_tier = attempt.kind
            actual_provider = provider
            actual_model = model
            return verdict
        except Exception as exc:
            attempt_info["status"] = "error"
            attempt_info["error_class"] = type(exc).__name__
            attempt_info["error_reason"] = classify(exc).value
            raise

    try:
        verdict = await walk.run_adapter(
            Step.MODERATION, _run, client_kind=StepClientKind.RAW_OPENAI
        )
        fallback_used = len(attempts) > 1 and attempts[0].get("status") == "error"
        return _with_telemetry(
            verdict,
            requested_tier=requested_kind,
            requested_provider=requested_provider,
            requested_model=requested_model,
            actual_tier=actual_tier,
            actual_provider=actual_provider,
            actual_model=actual_model,
            fallback_used=fallback_used,
            attempts=attempts,
        )
    except Exception as e:
        logger.error(
            "Moderation failed on all tiers; failing closed - source_lang=%s error=%s",
            source_lang,
            type(e).__name__,
        )
        fallback_used = len(attempts) > 1 and attempts[0].get("status") == "error"
        return _with_telemetry(
            _block_unavailable(f"moderation unavailable: {type(e).__name__}"),
            requested_tier=requested_kind,
            requested_provider=requested_provider,
            requested_model=requested_model,
            actual_tier="failed",
            actual_provider="failed",
            actual_model="failed",
            fallback_used=fallback_used,
            attempts=attempts,
        )


async def _check_moderation_legacy(
    text: str,
    source_lang: str,
    recent_history_text: str = "",
    *,
    execution: ExecutionContext,
) -> ModerationVerdict:
    """Legacy moderation: single global provider (VOICE_MODERATION_PROVIDER),
    fails OPEN. Used when the pipeline's ``fallback_enabled`` is false."""
    requested_tier = "oss" if _MODERATION_PROVIDER != "openai" else "managed"
    requested_provider = "vllm" if requested_tier == "oss" else "openai"
    requested_model = _legacy_requested_model(requested_tier)
    attempts: list[dict[str, object]] = [
        {
            "tier": requested_tier,
            "provider": requested_provider,
            "model": requested_model,
            "status": "started",
        }
    ]
    try:
        client, model, provider = _moderation_client_and_model(execution)
        requested_tier = "managed" if provider == "openai" else "oss"
        attempts[0]["tier"] = requested_tier
        requested_provider = provider
        requested_model = model
        attempts[0]["provider"] = requested_provider
        attempts[0]["model"] = requested_model
    except Exception as e:
        logger.error("Moderation client init failed (%s); failing open", e)
        attempts[0]["status"] = "error"
        attempts[0]["error_class"] = type(e).__name__
        attempts[0]["error_reason"] = classify(e).value
        return _with_telemetry(
            _allow(f"moderation client error: {type(e).__name__}", failed_open=True),
            requested_tier=requested_tier,
            requested_provider=requested_provider,
            requested_model=requested_model,
            actual_tier="failed",
            actual_provider="failed",
            actual_model="failed",
            fallback_used=False,
            attempts=attempts,
        )
    langfuse = _get_langfuse()

    try:
        if not langfuse:
            response = await _create_moderation_response(
                client,
                model,
                text,
                source_lang,
                recent_history_text,
            )
            raw = (response.choices[0].message.content or "").strip()
            verdict = _parse_verdict(raw)
            attempts[0]["status"] = "ok"
            return _with_telemetry(
                verdict,
                requested_tier=requested_tier,
                requested_provider=requested_provider,
                requested_model=requested_model,
                actual_tier=requested_tier,
                actual_provider=requested_provider,
                actual_model=requested_model,
                fallback_used=False,
                attempts=attempts,
            )

        with langfuse.start_as_current_observation(
            name="query_moderation",
            as_type="generation",
            input={
                "source_lang": source_lang,
                "text": text,
                "recent_history_text": recent_history_text,
            },
            model=model,
            metadata={
                "pipeline_stage": "query_moderation",
                "moderation_provider": provider,
            },
        ) as observation:
            response = await _create_moderation_response(
                client,
                model,
                text,
                source_lang,
                recent_history_text,
            )
            raw = (response.choices[0].message.content or "").strip()
            verdict = _parse_verdict(raw)
            attempts[0]["status"] = "ok"
            observation.update(
                output={
                    "category": verdict.category,
                    "reason": verdict.reason,
                    "failed_open": verdict.failed_open,
                },
                metadata={"rejected": verdict.rejected},
            )
            return _with_telemetry(
                verdict,
                requested_tier=requested_tier,
                requested_provider=requested_provider,
                requested_model=requested_model,
                actual_tier=requested_tier,
                actual_provider=requested_provider,
                actual_model=requested_model,
                fallback_used=False,
                attempts=attempts,
            )
    except asyncio.TimeoutError:
        logger.error(
            "Moderation timed out - source_lang=%s model=%s timeout=%.2fs query_chars=%s query_preview=%r",
            source_lang,
            model,
            settings.openai_pretranslation_timeout_seconds,
            len(text),
            text[:160],
        )
        attempts[0]["status"] = "error"
        attempts[0]["error_class"] = "TimeoutError"
        attempts[0]["error_reason"] = "timeout"
        return _with_telemetry(
            _allow("moderation timeout", failed_open=True),
            requested_tier=requested_tier,
            requested_provider=requested_provider,
            requested_model=requested_model,
            actual_tier="failed",
            actual_provider="failed",
            actual_model="failed",
            fallback_used=False,
            attempts=attempts,
        )
    except Exception as e:
        logger.error(
            "Moderation failed - source_lang=%s error=%s query_preview=%r",
            source_lang,
            e,
            text[:160],
        )
        attempts[0]["status"] = "error"
        attempts[0]["error_class"] = type(e).__name__
        attempts[0]["error_reason"] = classify(e).value
        return _with_telemetry(
            _allow(f"moderation error: {type(e).__name__}", failed_open=True),
            requested_tier=requested_tier,
            requested_provider=requested_provider,
            requested_model=requested_model,
            actual_tier="failed",
            actual_provider="failed",
            actual_model="failed",
            fallback_used=False,
            attempts=attempts,
        )
