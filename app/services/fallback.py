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
import os
import random
import time

import anyio
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from app.config import settings
from app.llm_core.config_model import Step
from app.llm_core.factory import MaterializedTier
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

# (G) Breaker evidence — the subset of FALLBACKABLE that genuinely indicts the
# ENDPOINT (the box is down / erroring / overloaded), as distinct from a
# caller-side or context problem. We still fall to the next tier on ANY
# FALLBACKABLE reason, but only feed the health breaker on BREAKER_EVIDENCE — so a
# caller ``TypeError`` (-> UNKNOWN) or a 4xx context-overflow (-> UNKNOWN) can no
# longer trip the OSS breaker and shift everyone to the managed tier. UNKNOWN is
# deliberately excluded; it is fallbackable but is NOT endpoint evidence.
BREAKER_EVIDENCE = {
    FallbackReason.CONNECTION,
    FallbackReason.HTTP_5XX,
    FallbackReason.OOM,
    FallbackReason.RATE_LIMITED,
    FallbackReason.TIMEOUT,
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


# ── config-driven chain (the only path after P4) ──────────────────────────────
# Maps the pipeline label the walkers receive to an llm_core Step, so a session's
# resolved weighted profile materializes that step's tiers (the config-driven
# successor to the removed hardwired ``attempt_chain([oss, managed])``). The
# materialized chain (list[MaterializedTier]) is what the walkers read: they only
# touch .kind/.model/.model_name/.provider/.endpoint/.timeout, all carried by
# MaterializedTier (the drop-in successor to the removed ``Attempt``).
_PIPELINE_TO_STEP = {
    "chat": Step.AGENT,
    "moderation": Step.MODERATION,
    "pretranslation": Step.PRE_TRANSLATION,
    "suggestions": Step.SUGGESTIONS,
}


async def _resolve_chain(*, pipeline: str, session_id: str, variant: str) -> list:
    """How the walkers receive their chain — always the config-driven pipeline.

    Resolves the session's sticky weighted profile, looks up the step's tiers, and
    materializes them (primary first, never empty). Health pruning (the pre-flight
    FILTER) and concurrency reordering happen inside ``split.resolve_chain`` — a
    no-op unless a HEALTH_* / CONCURRENCY_GAUGE flag is on.

    ``variant`` is kept in the signature (walker call sites + telemetry) but the
    chain is driven by the session's sticky profile, not the variant string.

    A config/Redis edge case must never break the fallback path: on any failure
    (or an unmapped pipeline) it degrades to the resolver's managed-tier chain for
    the step, so callers still get a non-empty chain and their terminal net
    (moderation fail-closed, pretranslation safe-default, suggestions ``[]``)
    remains the last line of defence."""
    step = _PIPELINE_TO_STEP.get(pipeline)
    if step is not None:
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
                "fallback: config chain resolve failed (pipeline=%s): %s; "
                "degrading to managed tier", pipeline, exc,
            )

    from app.llm_core import resolver as _resolver
    degrade_step = step or Step.AGENT
    return _resolver.resolve_chain(degrade_step, "legacy")


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
    reason_value = event.reason.value
    from app import metrics
    metrics.record_fallback(event.pipeline, reason_value, event.fell_back, event.committed)

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

    # NOTE: the fallback event lands via the structured log line above + the Sentry
    # breadcrumb. A prior ``client.update_current_trace(...)`` call was removed here:
    # this Langfuse SDK has no ``update_current_trace`` (it always raised and was
    # swallowed, so the tag/metadata never landed). Re-landing fallback-event tags
    # via a supported API (update_current_span) is a tracked follow-up.


def _record_served(pipeline: str, kind: str, index: int) -> None:
    """Tracing-only: thread the tier that actually served (kind + 0-based chain
    index) back to the current turn's pipeline-trace, keyed by the pipeline's
    Step. No-op when no trace context is active; never breaks the request path."""
    try:  # pragma: no cover - best effort
        from app.llm_core import trace as _trace
        _trace.record_served(_PIPELINE_TO_STEP.get(pipeline, pipeline), kind, index)
    except Exception:
        pass


# (D) Internal commit sentinel. Agent producers yield this the instant the agent
# does ANY work — the FIRST pydantic-ai model event, which is a tool-call part that
# pydantic-ai emits BEFORE it runs the tools and long before the first TEXT delta.
# ``with_first_token_deadline`` treats the sentinel as the first-token commit, so a
# turn that has begun executing tools can never trip the TTFT deadline and force a
# cross-tier re-run of side-effecting tools (duplicate bookings / SMS) or poison the
# OSS breaker. The sentinel is consumed by the deadline wrapper and is NEVER
# forwarded to the caller. (Liveness is preserved: a truly hung endpoint emits no
# event, so no sentinel arrives and the deadline still fires -> swap.)
class _AgentActivity:
    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug aid only
        return "<AGENT_ACTIVITY>"


AGENT_ACTIVITY = _AgentActivity()


# (F) Bound concurrent MANAGED-tier admission + honor 429 on the last tier.
# During a full GPU outage ~100% of turns overflow to the managed tier; with no cap
# we drive it into its own 429 with nothing behind it. A per-process semaphore caps
# concurrent managed calls (size from ``MANAGED_MAX_CONCURRENCY``); OSS tiers are
# uncapped. Acquisition is FAIL-OPEN — if a slot can't be had within a short
# timeout we proceed anyway rather than deadlock a farmer's turn.
def _managed_max_concurrency() -> int:
    try:
        return max(1, int(os.getenv("MANAGED_MAX_CONCURRENCY", "64")))
    except Exception:
        return 64


_MANAGED_ACQUIRE_TIMEOUT = 5.0   # fail-open cap on waiting for a managed slot (s)
_RATE_LIMIT_MAX_WAIT = 5.0       # cap on honoring a last-tier 429 Retry-After (s)

# The semaphore is (re)bound to the RUNNING loop lazily: a module-level Semaphore
# binds to whatever loop imported this module and then explodes when awaited from a
# different loop (per-test ``asyncio.run`` loops, a reloaded app loop). Keyed on the
# identity of the running loop, so production creates it exactly once and tests get
# a fresh one per loop.
_managed_sem_state: dict = {"loop": None, "sem": None}


def _get_managed_sem() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    if _managed_sem_state["sem"] is None or _managed_sem_state["loop"] is not loop:
        _managed_sem_state["loop"] = loop
        _managed_sem_state["sem"] = asyncio.Semaphore(_managed_max_concurrency())
    return _managed_sem_state["sem"]


async def _acquire_managed_slot(sem: asyncio.Semaphore) -> bool:
    """Acquire a managed-tier slot; FAIL-OPEN on timeout so a turn never deadlocks.

    Returns True if the slot was acquired (caller must ``release``), False if it
    proceeded uncapped after the acquire timed out."""
    try:
        await asyncio.wait_for(sem.acquire(), _MANAGED_ACQUIRE_TIMEOUT)
        return True
    except asyncio.TimeoutError:
        logger.warning(
            "fallback: managed-tier slot acquire timed out (%.1fs); proceeding uncapped",
            _MANAGED_ACQUIRE_TIMEOUT,
        )
        return False


def _retry_after_seconds(exc: BaseException) -> float:
    """Best-effort ``Retry-After`` (seconds) from a 429, capped + jittered.

    Reads a numeric ``Retry-After`` header off the exception's response when
    present; otherwise a small default backoff. Always bounded by
    ``_RATE_LIMIT_MAX_WAIT`` so a hostile/huge value can't stall the turn."""
    delay: Optional[float] = None
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) or getattr(exc, "headers", None)
    if headers is not None:
        try:
            getter = getattr(headers, "get", None)
            raw = (getter("retry-after") or getter("Retry-After")) if getter else None
            if raw is not None:
                delay = float(str(raw).strip())
        except Exception:
            delay = None
    if delay is None or delay < 0:
        delay = 0.5
    return min(delay, _RATE_LIMIT_MAX_WAIT) + random.uniform(0.0, 0.25)


async def execute_with_fallback(
    *,
    pipeline: str,
    session_id: str,
    variant: str,
    run: Callable[[MaterializedTier], Awaitable[Any]],
) -> Any:
    """Run ``run(attempt)`` against each tier of the chain, falling back on a
    classified infrastructure failure and recording every failure via ``emit``.

    ``run`` receives the active :class:`MaterializedTier` and returns the awaitable for that
    tier (e.g. ``agent.run(..., model=attempt.model)``). Returns whatever ``run``
    returns. Re-raises when the failure is non-fallbackable or the chain is
    exhausted, so the caller's existing degrade path (moderation fail-closed,
    pretranslation safe-default, suggestions ``[]``) stays the terminal net.
    """
    chain = await _resolve_chain(pipeline=pipeline, session_id=session_id, variant=variant)
    for i, attempt in enumerate(chain):
        is_last = i == len(chain) - 1
        # (F) Cap concurrent MANAGED-tier admission; OSS tiers stay uncapped.
        sem = _get_managed_sem() if attempt.kind == "managed" else None
        retried_rate_limit = False
        while True:
            t0 = time.monotonic()
            acquired = False
            try:
                if sem is not None:
                    acquired = await _acquire_managed_slot(sem)
                if attempt.timeout is None:
                    result = await run(attempt)
                else:
                    with anyio.fail_after(attempt.timeout):
                        result = await run(attempt)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                reason = classify(exc)
                will_fall_back = reason in FALLBACKABLE and not is_last
                # (G) FALLBACKABLE decides fall-to-next-tier; only BREAKER_EVIDENCE
                # (never UNKNOWN) feeds the health breaker, so a caller error / 4xx
                # overflow can't trip the OSS breaker. Self-gated: a no-op unless
                # HEALTH_BREAKER_ENABLED, so flag-off behaviour is identical.
                if reason in BREAKER_EVIDENCE:
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
                # (F) Last tier hit 429 with nothing behind it: ONE bounded retry
                # honoring Retry-After (capped + jittered) before giving up.
                if is_last and reason is FallbackReason.RATE_LIMITED and not retried_rate_limit:
                    retried_rate_limit = True
                    await asyncio.sleep(_retry_after_seconds(exc))
                    continue
                if not will_fall_back:
                    raise
                break  # fall to the next tier
            else:
                # Clean success resets the breaker for this endpoint (P2). No-op unless
                # HEALTH_BREAKER_ENABLED.
                health.record_success(attempt.endpoint)
                _record_served(pipeline, attempt.kind, i)
                from app import metrics
                metrics.record_served(pipeline, attempt.kind, attempt.provider, attempt.model_name)
                return result
            finally:
                if acquired:
                    sem.release()


async def with_first_token_deadline(attempt: MaterializedTier, agen: AsyncIterator[Any]) -> AsyncIterator[Any]:
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
            # (D) The AGENT_ACTIVITY sentinel satisfies the first-token deadline (the
            # endpoint is alive and working — a tool-call event precedes any text) but
            # is INTERNAL: it commits the turn without being forwarded to the caller.
            if val is not AGENT_ACTIVITY:
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
    make_stream: Callable[[MaterializedTier], AsyncIterator[Any]],
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
        is_last = i == len(chain) - 1
        # (F) Cap concurrent MANAGED-tier admission; OSS tiers stay uncapped. The slot
        # is held for the whole managed stream and released in `finally` — including on
        # a client disconnect / aclose, which unwinds through this frame's finally.
        sem = _get_managed_sem() if attempt.kind == "managed" else None
        retried_rate_limit = False
        while True:
            t0 = time.monotonic()
            committed = False
            acquired = False

            # IMPORTANT: consume make_stream here with a plain `async for` and never
            # wrap THIS loop in an external timeout/cancel scope — pydantic-ai's
            # run_stream opens an anyio cancel scope inside make_stream that stays open
            # across the `yield`, so any scope spanning these yields unwinds out of order
            # on aclose/disconnect ("cancel scope in a different task"). The plain
            # semaphore `finally` below is NOT a cancel scope, so it is disconnect-safe.
            # The time-to-first-token deadline is applied by callers wrapping make_stream
            # in `with_first_token_deadline`, which isolates run_stream in its own task +
            # queue precisely so no anyio scope is ever open at a `yield`.
            try:
                if sem is not None:
                    acquired = await _acquire_managed_slot(sem)
                async for chunk in make_stream(attempt):
                    committed = True
                    yield chunk
                # Clean stream finish resets the breaker for this endpoint (P2).
                health.record_success(attempt.endpoint)
                _record_served(pipeline, attempt.kind, i)
                from app import metrics
                metrics.record_served(pipeline, attempt.kind, attempt.provider, attempt.model_name)
                return  # stream finished cleanly
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                reason = classify(exc)
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
                # (G) Only PRE-commit BREAKER_EVIDENCE feeds the breaker: a post-commit
                # failure (handled above) is NOT evidence — the box answered and
                # streamed tokens — and neither is UNKNOWN (a caller/context problem).
                # Self-gated (no-op unless HEALTH_BREAKER_ENABLED).
                if reason in BREAKER_EVIDENCE:
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
                # (F) Last tier hit 429 pre-commit with nothing behind it: ONE bounded
                # retry honoring Retry-After. Safe because committed is False (no output
                # reached the caller), so the retried stream can't duplicate anything.
                if is_last and reason is FallbackReason.RATE_LIMITED and not retried_rate_limit:
                    retried_rate_limit = True
                    await asyncio.sleep(_retry_after_seconds(exc))
                    continue
                if will_fall_back:
                    break  # fall to the next tier
                raise
            finally:
                if acquired:
                    sem.release()

    # All tiers failed before commit (every fallbackable tier swapped, last raised).
    if last_exc is not None:
        raise last_exc
