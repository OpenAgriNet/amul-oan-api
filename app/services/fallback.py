"""Standard OSS -> managed fallback for LLM pipelines.

See docs/oss-fallback-design.md. One mechanism, used by every unary pipeline
(pretranslation, moderation, suggestions): a resolved pipeline *variant* becomes
an ordered *attempt chain* — ``[oss, managed]`` for OSS sessions, ``[managed]``
otherwise — and ``execute_with_fallback`` walks it, classifying each failure,
falling back on infrastructure errors, and recording every failure for the
``oss_fallback`` metric.

Gated by ``settings.fallback_enabled`` (default off): when disabled, callers keep
their existing code path, so merging this changes nothing until it is flipped on.

Core-chat streaming uses ``stream_with_fallback`` (first-token commit): an OSS
failure *before* the first token swaps to managed transparently; once the first
token has reached the caller a swap is impossible, so the error propagates.
``with_first_token_deadline`` bounds time-to-first-token only (mid-stream tool
round-trips / slow generation are unaffected) by isolating the agent stream in its
own task + queue, so it stays disconnect-safe (see its docstring).
"""

from __future__ import annotations

import asyncio
import time

import anyio
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from app.config import settings
from agents.models import (
    LLM_MODEL,
    LLM_MODEL_NAME,
    LLM_PROVIDER,
    OSS_INFERENCE_ENDPOINT_URL,
    OSS_LLM_MODEL,
    OSS_LLM_MODEL_NAME,
    oss_model_available,
)
from app.llm_core.config_model import Step
from app.llm_core import health
from helpers.utils import get_logger

logger = get_logger(__name__)

# Optional Sentry breadcrumbs — best-effort, never a hard dependency.
try:  # pragma: no cover - import guard
    import sentry_sdk as _sentry
except Exception:  # pragma: no cover
    _sentry = None

# Optional Langfuse trace tagging — best-effort.
try:  # pragma: no cover - import guard
    from langfuse import get_client as _get_langfuse_client
except Exception:  # pragma: no cover
    _get_langfuse_client = None


class FallbackReason(str, Enum):
    """Why an OSS attempt failed. Drives the ``oss_fallback`` rate, sliced by
    pipeline x reason x endpoint — the lever to reduce fallbacks over time."""

    TIMEOUT = "timeout"            # asyncio/HTTP read timeout
    CONNECTION = "connection"      # connect refused / DNS / reset
    HTTP_5XX = "http_5xx"          # vLLM server error
    RATE_LIMITED = "rate_limited"  # 429 / queue full
    OOM = "oom"                    # 5xx whose body marks CUDA OOM
    BAD_OUTPUT = "bad_output"      # schema/validation exhausted (pydantic-ai) — NOT fallbackable
    CANCELLED = "cancelled"        # caller hung up — NOT fallbackable
    UNKNOWN = "unknown"


# We fall back on infrastructure failures only. ``bad_output`` stays on the same
# model (pydantic-ai already retries it; once exhausted it is a model-quality
# problem to fix, not mask) and ``cancelled`` means the caller is gone.
FALLBACKABLE = {
    FallbackReason.TIMEOUT,
    FallbackReason.CONNECTION,
    FallbackReason.HTTP_5XX,
    FallbackReason.RATE_LIMITED,
    FallbackReason.OOM,
    FallbackReason.UNKNOWN,
}

_OOM_MARKERS = ("out of memory", "cuda", "oom", "kv cache", "no available memory")


def classify(exc: BaseException) -> FallbackReason:
    """Map an exception raised by an OSS attempt to a FallbackReason.

    Defensive by design: we inspect status codes, attribute and type names, and
    message text rather than importing every provider's exception hierarchy, so
    this keeps working across openai/httpx/aiohttp/pydantic-ai version churn.
    """
    if isinstance(exc, asyncio.CancelledError):
        return FallbackReason.CANCELLED

    name = type(exc).__name__
    msg = str(exc).lower()

    # Timeouts (asyncio.TimeoutError, httpx.ReadTimeout, openai.APITimeoutError, ...)
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)) or "timeout" in name.lower():
        return FallbackReason.TIMEOUT

    # HTTP status carried by openai.APIStatusError / httpx responses / pydantic-ai ModelHTTPError.
    status = getattr(exc, "status_code", None)
    if status is None:
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None) or getattr(resp, "status", None)
    if isinstance(status, int):
        if status == 429:
            return FallbackReason.RATE_LIMITED
        if 500 <= status <= 599:
            if any(m in msg for m in _OOM_MARKERS):
                return FallbackReason.OOM
            return FallbackReason.HTTP_5XX

    # Connection-level failures.
    if isinstance(exc, ConnectionError) or any(
        tok in name.lower() for tok in ("connect", "connection")
    ) or "connection" in msg or "refused" in msg:
        return FallbackReason.CONNECTION

    # pydantic-ai schema/validation exhaustion — explicitly not fallbackable.
    if any(tok in name for tok in ("UnexpectedModelBehavior", "Validation", "Unexpected")):
        return FallbackReason.BAD_OUTPUT

    if any(m in msg for m in _OOM_MARKERS):
        return FallbackReason.OOM

    return FallbackReason.UNKNOWN


@dataclass(frozen=True)
class Attempt:
    """One tier in a fallback chain. Carries the pydantic-ai model object (agent
    pipelines pass it as ``model=``), telemetry labels, and a per-attempt
    timeout. ``timeout=None`` means no deadline (used when fallback is disabled)."""

    kind: str                 # "oss" | "managed"
    model: Any                # pydantic-ai model object
    model_name: str
    provider: str             # "vllm" | "openai" | "anthropic"
    endpoint: str             # OSS endpoint URL, or "managed"
    timeout: Optional[float]  # seconds


def _oss_timeout_for(pipeline: str) -> float:
    return {
        "chat": settings.fallback_chat_oss_timeout_ms,
        "moderation": settings.fallback_moderation_oss_timeout_ms,
        "pretranslation": settings.fallback_pretranslation_oss_timeout_ms,
        "suggestions": settings.fallback_suggestions_oss_timeout_ms,
    }.get(pipeline, settings.fallback_moderation_oss_timeout_ms) / 1000.0


def _managed_attempt() -> Attempt:
    return Attempt(
        kind="managed",
        model=LLM_MODEL,
        model_name=LLM_MODEL_NAME,
        provider=LLM_PROVIDER,
        endpoint="managed",
        timeout=settings.fallback_managed_timeout_ms / 1000.0,
    )


def attempt_chain(variant: str, pipeline: str) -> list[Attempt]:
    """Ordered tiers for a resolved variant.

    * fallback disabled -> single tier = the variant's own model, no deadline
      (behaviour identical to today; only used if a caller routes through here).
    * OSS session with OSS configured -> ``[oss, managed]``.
    * everything else -> ``[managed]`` (nothing to fall back to).
    """
    if not settings.fallback_enabled:
        # Single tier matching the variant; no fallback, no added deadline.
        if variant == "oss" and oss_model_available():
            return [Attempt("oss", OSS_LLM_MODEL, OSS_LLM_MODEL_NAME or "oss",
                            "vllm", OSS_INFERENCE_ENDPOINT_URL or "oss", None)]
        return [Attempt("managed", LLM_MODEL, LLM_MODEL_NAME, LLM_PROVIDER, "managed", None)]

    if variant == "oss" and oss_model_available():
        oss = Attempt(
            kind="oss",
            model=OSS_LLM_MODEL,
            model_name=OSS_LLM_MODEL_NAME or "oss",
            provider="vllm",
            endpoint=OSS_INFERENCE_ENDPOINT_URL or "oss",
            timeout=_oss_timeout_for(pipeline),
        )
        return [oss, _managed_attempt()]
    return [_managed_attempt()]


# ── config-driven chain (llm_core P1) ─────────────────────────────────────────
# Maps the pipeline label the walkers receive to an llm_core Step, so a
# PROFILES_ENABLED session materializes the resolved profile's tiers instead of
# the hardwired attempt_chain. The materialized chain (list[MaterializedTier]) is
# a drop-in for list[Attempt]: the walkers only read .kind/.model/.model_name/
# .provider/.endpoint/.timeout, all of which MaterializedTier carries.
_PIPELINE_TO_STEP = {
    "chat": Step.AGENT,
    "moderation": Step.MODERATION,
    "pretranslation": Step.PRE_TRANSLATION,
    "suggestions": Step.SUGGESTIONS,
}


async def _resolve_chain(*, pipeline: str, session_id: str, variant: str) -> list:
    """How the walkers receive their chain (the ONLY thing P1 changes about them).

    Composition of the two P0/P1 kill-switches:
      * flags off (default) -> legacy ``attempt_chain(variant, pipeline)`` — the
        result is byte-identical to today, so the walker bodies are unchanged;
      * ``PROFILES_ENABLED`` AND ``LLM_CORE_ENABLED`` -> the config-driven chain:
        the session's weighted profile's step tiers, materialized via the P0
        factory. Both flags are required because the materialized chain carries
        factory-built handles, which are P0's (``LLM_CORE_ENABLED``) domain.

    Any failure resolving the config chain degrades to ``attempt_chain`` — the
    fallback path must never be broken by a config/Redis edge case.

    P2 health prune (a pre-flight FILTER) is applied on BOTH branches, so the
    breaker/poller work independent of PROFILES: the config path is pruned inside
    ``split.resolve_chain`` (before materialize — the documented P3 seam); the
    legacy ``attempt_chain`` path is pruned here on the resulting ``Attempt`` s
    (both expose ``.endpoint``). No-op unless a HEALTH_* flag is on; never empties
    the chain."""
    step = _PIPELINE_TO_STEP.get(pipeline)
    if settings.profiles_enabled and settings.llm_core_enabled and step is not None:
        try:
            from app.llm_core import split
            # (C) Honor the variant the router already resolved from the FULL
            # session id. The walkers receive a 200-char-capped session id, so
            # re-bucketing here on that capped id could pick a different profile
            # than the primary path — threading the variant selects the same
            # profile without a second (divergent) bucket.
            return await split.resolve_chain(
                session_id, step, variant=variant
            )  # health-pruned inside
        except Exception as exc:  # never break the fallback path on a config edge
            logger.warning(
                "fallback: profiles chain resolve failed (pipeline=%s): %s; "
                "using legacy attempt_chain", pipeline, exc,
            )

    chain = attempt_chain(variant, pipeline)
    if settings.health_breaker_enabled or settings.health_poller_enabled:
        try:
            chain = health.prune_unhealthy(step, chain)
        except Exception as exc:  # a prune bug must never break the fallback path
            logger.warning("fallback: health prune failed (pipeline=%s): %s", pipeline, exc)
    return chain


@dataclass
class FallbackEvent:
    """Recorded for every classified OSS failure — both fallbacks (``fell_back=True``)
    and non-fallbackable failures (``fell_back=False``), so dashboards see the full
    picture, not just the fallbacks."""

    pipeline: str
    session_id: str
    from_variant: str
    to_variant: Optional[str]
    reason: FallbackReason
    error_class: str
    error_detail: str
    oss_endpoint: str
    oss_model: str
    latency_ms: int
    fell_back: bool
    committed: bool = False


def emit(event: FallbackEvent) -> None:
    """Record a fallback event.

    Canonical sink is a structured log line (always available, greppable, and the
    source for the ``oss_fallback`` rate metric). Langfuse trace-tagging and a
    Sentry breadcrumb are added best-effort. NOTE: this intentionally does not use
    the canonical telemetry queue — that pipeline is for farmer Q&A analytics, not
    ops metrics; revisit if a fallback event type is added there."""
    logger.warning(
        "oss_fallback pipeline=%s reason=%s fell_back=%s from=%s to=%s "
        "endpoint=%s model=%s latency_ms=%s error_class=%s committed=%s session=%s detail=%s",
        event.pipeline,
        event.reason.value,
        event.fell_back,
        event.from_variant,
        event.to_variant,
        event.oss_endpoint,
        event.oss_model,
        event.latency_ms,
        event.error_class,
        event.committed,
        event.session_id,
        event.error_detail,
    )

    if _sentry is not None:
        try:  # pragma: no cover - best effort
            _sentry.add_breadcrumb(
                category="oss_fallback",
                level="warning",
                message=f"{event.pipeline} {event.reason.value} fell_back={event.fell_back}",
                data={
                    "pipeline": event.pipeline,
                    "reason": event.reason.value,
                    "endpoint": event.oss_endpoint,
                    "error_class": event.error_class,
                    "latency_ms": event.latency_ms,
                },
            )
        except Exception:
            pass

    if _get_langfuse_client is not None:
        try:  # pragma: no cover - best effort
            client = _get_langfuse_client()
            if client is not None:
                client.update_current_trace(
                    tags=[f"oss_fallback:{event.pipeline}:{event.reason.value}"],
                    metadata={
                        "oss_fallback": {
                            "pipeline": event.pipeline,
                            "reason": event.reason.value,
                            "fell_back": event.fell_back,
                            "endpoint": event.oss_endpoint,
                            "model": event.oss_model,
                            "latency_ms": event.latency_ms,
                        }
                    },
                )
        except Exception:
            pass


async def execute_with_fallback(
    *,
    pipeline: str,
    session_id: str,
    variant: str,
    run: Callable[[Attempt], Awaitable[Any]],
) -> Any:
    """Run ``run(attempt)`` against each tier of the chain, falling back on a
    classified infrastructure failure and recording every failure via ``emit``.

    ``run`` receives the active :class:`Attempt` and returns the awaitable for that
    tier (e.g. ``agent.run(..., model=attempt.model)``). Returns whatever ``run``
    returns. Re-raises when the failure is non-fallbackable or the chain is
    exhausted, so the caller's existing degrade path (moderation fail-closed,
    pretranslation safe-default, suggestions ``[]``) stays the terminal net.
    """
    chain = await _resolve_chain(pipeline=pipeline, session_id=session_id, variant=variant)
    for i, attempt in enumerate(chain):
        t0 = time.monotonic()
        try:
            if attempt.timeout is None:
                result = await run(attempt)
            else:
                with anyio.fail_after(attempt.timeout):
                    result = await run(attempt)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            reason = classify(exc)
            is_last = i == len(chain) - 1
            will_fall_back = reason in FALLBACKABLE and not is_last
            # P2 passive breaker feed: an infrastructure (FALLBACKABLE) failure is
            # evidence this endpoint is down — count it toward the trip. Self-gated:
            # a no-op unless HEALTH_BREAKER_ENABLED, so flag-off behaviour is identical.
            if reason in FALLBACKABLE:
                health.record_failure(attempt.endpoint)
            emit(
                FallbackEvent(
                    pipeline=pipeline,
                    session_id=session_id,
                    from_variant=attempt.kind,
                    to_variant=chain[i + 1].kind if will_fall_back else None,
                    reason=reason,
                    error_class=type(exc).__name__,
                    error_detail=str(exc)[:500],
                    oss_endpoint=attempt.endpoint,
                    oss_model=attempt.model_name,
                    latency_ms=int((time.monotonic() - t0) * 1000),
                    fell_back=will_fall_back,
                )
            )
            if not will_fall_back:
                raise
        else:
            # Clean success resets the breaker for this endpoint (P2). No-op unless
            # HEALTH_BREAKER_ENABLED.
            health.record_success(attempt.endpoint)
            return result


async def with_first_token_deadline(attempt: Attempt, agen: AsyncIterator[Any]) -> AsyncIterator[Any]:
    """Bound time-to-first-token only — safely across client disconnects.

    ``attempt.timeout`` bounds only the wait for the FIRST chunk; mid-stream gaps
    after it (tool round-trips, slow generation) are NOT bounded — that is why we
    can't just shorten the model's httpx read-timeout, which can't tell a silent
    pre-first-token hang from a normal inter-token gap.

    The agent stream (which carries pydantic-ai's ``run_stream`` anyio cancel
    scope) is driven entirely inside a dedicated task and forwarded chunk-by-chunk
    through a queue; only the first ``queue.get`` is bounded (``asyncio.wait_for``).
    This is deliberate: ``run_stream``'s cancel scope is opened, advanced AND closed
    within that one task, while the consumer only ever awaits a plain queue — so
    THIS generator can be ``aclose()``'d from another task (a client disconnect /
    mid-stream ``GeneratorExit``) without ever touching an anyio scope. An earlier
    version wrapped the stream in an ``anyio.move_on_after`` scope that spanned the
    ``yield``s; that crashed on every disconnect with "exit cancel scope in a
    different task" / "aclose: generator already running". Validated against real
    ``run_stream`` incl. the disconnect path.

    A pre-first-token timeout raises ``TimeoutError`` (``classify`` -> ``TIMEOUT``
    -> fallbackable, so ``stream_with_fallback`` swaps before any token reached the
    caller). Stream exceptions propagate unchanged. No-op when ``attempt.timeout``
    is None (fallback disabled).
    """
    ttft = attempt.timeout
    queue: asyncio.Queue = asyncio.Queue()
    _CHUNK, _END, _ERR = 0, 1, 2

    async def _drain() -> None:
        try:
            async for item in agen:
                await queue.put((_CHUNK, item))
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # forward pre-/mid-stream failures to the consumer
            await queue.put((_ERR, exc))
            return
        await queue.put((_END, None))

    task = asyncio.create_task(_drain())
    try:
        try:
            if ttft is not None:
                kind, val = await asyncio.wait_for(queue.get(), ttft)
            else:
                kind, val = await queue.get()
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"OSS first-token deadline exceeded ({ttft}s) [{attempt.kind}/{attempt.endpoint}]"
            )
        while True:
            if kind == _END:
                return
            if kind == _ERR:
                raise val
            yield val
            kind, val = await queue.get()
    finally:
        # Single-task teardown: cancelling _drain unwinds run_stream's scope in its
        # OWN task (incl. agen.aclose via the async-for cleanup). The consumer never
        # entered an anyio scope, so an outer aclose (disconnect) is safe.
        if not task.done():
            task.cancel()
        try:
            await task
        except BaseException:
            pass


async def _aclose(agen) -> None:
    close = getattr(agen, "aclose", None)
    if close is not None:
        try:  # pragma: no cover - best effort cleanup
            await close()
        except Exception:
            pass


async def stream_with_fallback(
    *,
    pipeline: str,
    session_id: str,
    variant: str,
    make_stream: Callable[[Attempt], AsyncIterator[Any]],
) -> AsyncIterator[Any]:
    """Stream a chain tier with *first-token commit* semantics.

    ``make_stream(attempt)`` returns an async iterator of chunks (e.g. English
    text deltas from an agent run on ``attempt.model``). The first yielded chunk
    is the **commit point**:

    * Failure BEFORE the first chunk, on a fallbackable reason and not the last
      tier -> silently swap to the next tier (the client has seen nothing).
    * Failure before the first chunk that is non-fallbackable or on the last tier
      -> re-raise (no tokens sent; caller handles it as today).
    * Failure AFTER the first chunk -> the client already has partial output, so a
      transparent swap is impossible; the exception propagates (no worse than
      today). The per-attempt timeout therefore bounds time-to-first-token only.

    Every classified failure is recorded via ``emit`` (``committed`` distinguishes
    pre- from post-commit).
    """
    chain = await _resolve_chain(pipeline=pipeline, session_id=session_id, variant=variant)
    last_exc: Optional[BaseException] = None
    for i, attempt in enumerate(chain):
        t0 = time.monotonic()
        committed = False

        # IMPORTANT: consume make_stream here with a plain `async for` and never
        # wrap THIS loop in an external timeout/cancel scope — pydantic-ai's
        # run_stream opens an anyio cancel scope inside make_stream that stays open
        # across the `yield`, so any scope spanning these yields unwinds out of order
        # on aclose/disconnect ("cancel scope in a different task"). The
        # time-to-first-token deadline is applied by callers wrapping make_stream in
        # `with_first_token_deadline`, which isolates run_stream in its own task +
        # queue precisely so no anyio scope is ever open at a `yield` (disconnect-safe).
        try:
            async for chunk in make_stream(attempt):
                committed = True
                yield chunk
            # Clean stream finish resets the breaker for this endpoint (P2).
            health.record_success(attempt.endpoint)
            return  # stream finished cleanly
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            reason = classify(exc)
            is_last = i == len(chain) - 1
            if committed:
                # Client already received output — a transparent swap is impossible.
                emit(
                    FallbackEvent(
                        pipeline=pipeline,
                        session_id=session_id,
                        from_variant=attempt.kind,
                        to_variant=None,
                        reason=reason,
                        error_class=type(exc).__name__,
                        error_detail=str(exc)[:500],
                        oss_endpoint=attempt.endpoint,
                        oss_model=attempt.model_name,
                        latency_ms=int((time.monotonic() - t0) * 1000),
                        fell_back=False,
                        committed=True,
                    )
                )
                raise
            will_fall_back = reason in FALLBACKABLE and not is_last
            # P2 passive breaker feed: only PRE-commit infrastructure failures are
            # evidence the endpoint is down. A post-commit failure (handled above)
            # is NOT — the box answered and streamed tokens — so it must not trip
            # the breaker. Self-gated (no-op unless HEALTH_BREAKER_ENABLED).
            if reason in FALLBACKABLE:
                health.record_failure(attempt.endpoint)
            emit(
                FallbackEvent(
                    pipeline=pipeline,
                    session_id=session_id,
                    from_variant=attempt.kind,
                    to_variant=chain[i + 1].kind if will_fall_back else None,
                    reason=reason,
                    error_class=type(exc).__name__,
                    error_detail=str(exc)[:500],
                    oss_endpoint=attempt.endpoint,
                    oss_model=attempt.model_name,
                    latency_ms=int((time.monotonic() - t0) * 1000),
                    fell_back=will_fall_back,
                    committed=False,
                )
            )
            last_exc = exc
            if will_fall_back:
                continue
            raise

    # All tiers failed before commit (every fallbackable tier swapped, last raised).
    if last_exc is not None:
        raise last_exc
