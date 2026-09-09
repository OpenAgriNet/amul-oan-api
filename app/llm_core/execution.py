"""LLM tier execution policy.

See docs/oss-fallback-design.md. One mechanism, used by every unary pipeline
(pretranslation, moderation, suggestions): a resolved pipeline *variant* becomes
an ordered *attempt chain* — ``[oss, managed]`` for OSS sessions, ``[managed]``
otherwise — and ``execute_with_fallback`` walks it, classifying each failure,
falling back on infrastructure errors, and recording every failure for the
``oss_fallback`` metric.

This module is deliberately inside :mod:`app.llm_core`: application services do
not choose tiers, inspect fallback flags, or implement retry loops.

Core-chat streaming uses ``stream_with_fallback`` (first-token commit): an OSS
failure *before* the first token swaps to managed transparently; once the first
token has reached the caller a swap is impossible, so the error propagates.
The common stream state machine owns both time-to-first-token and commit state.
"""

from __future__ import annotations

import asyncio
import random
import time

import anyio
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from app.config import settings
from app.llm_core.config_model import (
    AdmissionPolicy,
    PipelineConfig,
    Provider,
    Step,
    StepClientKind,
    Tier,
)
from app.llm_core.factory import build_handle
from app.llm_core import health
from helpers.utils import get_logger

logger = get_logger(__name__)

# Optional Sentry breadcrumbs — best-effort, never a hard dependency.
try:  # pragma: no cover - import guard
    import sentry_sdk as _sentry
except Exception:  # pragma: no cover
    _sentry = None

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


@dataclass(frozen=True)
class ExecutionTarget:
    """An inert resolved tier. Its client is built only when the walker reaches it."""

    tier: Tier
    client_kind: StepClientKind

    @cached_property
    def handle(self) -> Any:
        return build_handle(self.tier, self.client_kind)

    @property
    def kind(self) -> str:
        return "oss" if self.tier.provider is Provider.VLLM else "managed"

    @property
    def provider(self) -> str:
        return self.tier.provider.value

    @property
    def model_name(self) -> str:
        return self.tier.model

    @property
    def endpoint(self) -> str:
        return self.tier.endpoint or "managed"

    @property
    def timeout(self) -> Optional[float]:
        return self.tier.timeout_ms / 1000.0 if self.tier.timeout_ms is not None else None

    @property
    def ttft(self) -> Optional[float]:
        return self.tier.ttft_ms / 1000.0 if self.tier.ttft_ms is not None else None

    @property
    def admission(self) -> AdmissionPolicy:
        return self.tier.admission


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


def _record_served(step: Step, kind: str, index: int) -> None:
    """Tracing-only: thread the tier that actually served (kind + 0-based chain
    index) back to the current turn's pipeline-trace, keyed by the pipeline's
    Step. No-op when no trace context is active; never breaks the request path."""
    try:  # pragma: no cover - best effort
        from app.llm_core import trace as _trace
        _trace.record_served(step, kind, index)
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
        return max(1, settings.managed_max_concurrency)
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
    step: Step,
    session_id: str,
    run: Callable[[ExecutionTarget], Awaitable[Any]],
    chain: list[ExecutionTarget],
) -> Any:
    """Run ``run(attempt)`` against each tier of the chain, falling back on a
    classified infrastructure failure and recording every failure via ``emit``.

    ``run`` receives the active :class:`ExecutionTarget` and returns the awaitable for that
    tier (e.g. ``agent.run(..., model=attempt.model)``). Returns whatever ``run``
    returns. Re-raises when the failure is non-fallbackable or the chain is
    exhausted, so the caller's existing degrade path (moderation fail-closed,
    pretranslation safe-default, suggestions ``[]``) stays the terminal net.
    """
    pipeline = step.value
    try:
        for i, attempt in enumerate(chain):
            is_last = i == len(chain) - 1
            # (F) Cap concurrent MANAGED-tier admission; OSS tiers stay uncapped.
            sem = _get_managed_sem() if attempt.admission is AdmissionPolicy.MANAGED else None
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
                    _record_served(step, attempt.kind, i)
                    from app import metrics
                    metrics.record_served(pipeline, attempt.kind, attempt.provider, attempt.model_name)
                    return result
                finally:
                    if acquired:
                        sem.release()
    finally:
        # (Finding #3) Free any half-open probe token granted during chain resolution
        # (``prune_unhealthy`` -> ``is_open``) for EVERY tier in the chain, no matter
        # how the walk ended: a tier that succeeded / failed on any classified reason,
        # a caller cancellation, OR a tier that never executed because an earlier tier
        # returned/raised (a concurrency reorder can move the just-probed tier off
        # index 0). ``release_probe`` is idempotent + self-gated (no-op unless a health
        # flag is on) and only frees the probe slot — it never perturbs the
        # record_success/record_failure breaker transitions above.
        for _tier in chain:
            health.release_probe(_tier.endpoint)


async def _attempt_events(
    attempt: ExecutionTarget,
    source: AsyncIterator[Any],
) -> AsyncIterator[Any]:
    """Drive one attempt in its own task and bound only its first event.

    ``attempt.ttft`` bounds only the wait for the FIRST event; mid-stream gaps
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

    The event is returned unchanged, including ``AGENT_ACTIVITY``. The common
    walker owns commit state and decides whether an event is externally visible.
    """
    queue: asyncio.Queue = asyncio.Queue()
    _CHUNK, _END, _ERR = 0, 1, 2

    async def _drain() -> None:
        try:
            async for item in source:
                await queue.put((_CHUNK, item))
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # forward pre-/mid-stream failures to the consumer
            await queue.put((_ERR, exc))
            return
        await queue.put((_END, None))

    task = asyncio.create_task(_drain())
    try:
        deadline = attempt.ttft if attempt.ttft is not None else attempt.timeout
        try:
            if deadline is not None:
                kind, val = await asyncio.wait_for(queue.get(), deadline)
            else:
                kind, val = await queue.get()
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"first-token deadline exceeded ({deadline}s) "
                f"[{attempt.provider}/{attempt.endpoint}]"
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
async def stream_with_fallback(
    *,
    step: Step,
    session_id: str,
    make_stream: Callable[[ExecutionTarget], AsyncIterator[Any]],
    chain: list[ExecutionTarget],
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
    pipeline = step.value
    try:
        for i, attempt in enumerate(chain):
            is_last = i == len(chain) - 1
            sem = _get_managed_sem() if attempt.admission is AdmissionPolicy.MANAGED else None
            retried_rate_limit = False
            while True:
                t0 = time.monotonic()
                committed = False
                acquired = False

                try:
                    if sem is not None:
                        acquired = await _acquire_managed_slot(sem)
                    async for chunk in _attempt_events(attempt, make_stream(attempt)):
                        committed = True
                        if chunk is not AGENT_ACTIVITY:
                            yield chunk
                    # Clean stream finish resets the breaker for this endpoint (P2).
                    health.record_success(attempt.endpoint)
                    _record_served(step, attempt.kind, i)
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

    finally:
        # (Finding #3) Free any half-open probe token granted during chain
        # resolution (``prune_unhealthy`` -> ``is_open``) for EVERY tier, no matter
        # how the stream ended: a clean finish, any classified pre/post-commit
        # failure, a client disconnect / aclose unwinding through here, OR a tier
        # that never ran because an earlier tier committed/returned (a concurrency
        # reorder can move the just-probed tier off index 0). ``release_probe`` is
        # idempotent + self-gated and only frees the probe slot — it never perturbs
        # the record_success/record_failure breaker transitions above.
        for _tier in chain:
            health.release_probe(_tier.endpoint)


# ── application API ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelInfo:
    provider: str
    model_name: str
    kind: str


@dataclass(frozen=True)
class ExecutionContext:
    """Immutable model config and profile snapshot for one turn/session."""

    session_id: str
    config: PipelineConfig
    profile_name: str

    @property
    def profile(self):
        return (
            self.config.by_name(self.profile_name)
            or self.config.by_name("managed")
            or self.config.profiles[0]
        )

    @property
    def capabilities(self):
        return self.profile.capabilities

    def _step_config(self, step: Step):
        configured = self.config.step_config(self.profile, step)
        if configured is None:
            raise ValueError(
                f"no config for step={step.value} in profile={self.profile_name}"
            )
        return configured

    def _target(self, step: Step, tier: Tier) -> ExecutionTarget:
        from app.llm_core.factory import STEP_CLIENT_KIND, tier_client_kind

        return ExecutionTarget(tier, tier_client_kind(STEP_CLIENT_KIND[step], tier))

    def info(self, step: Step) -> ModelInfo:
        target = self._target(step, self._step_config(step).tiers[0])
        return ModelInfo(target.provider, target.model_name, target.kind)

    def begin_trace(self):
        from app.llm_core import trace

        current = trace.begin(self.profile_name)
        trace.set_profile(current, self.profile.name, self.profile.weight)
        for step in Step:
            try:
                trace.set_step_primary(
                    current, step, self._target(step, self._step_config(step).tiers[0])
                )
            except ValueError:
                pass
        return current

    async def _chain(self, step: Step) -> list[ExecutionTarget]:
        configured = self._step_config(step)
        tiers = list(configured.tiers)
        if self.config.fallback_enabled:
            from app.llm_core import split

            return await split.resolve_chain(
                self.session_id,
                step,
                self.config,
                profile_name=self.profile_name,
            )
        else:
            tiers = [tiers[0].model_copy(update={"timeout_ms": None, "ttft_ms": None})]
        chain = [self._target(step, tier) for tier in tiers]
        from app.llm_core import trace

        trace.record_profile(self.profile.name, self.profile.weight)
        trace.record_step_chain(step, chain)
        return chain

    async def run_adapter(
        self,
        step: Step,
        invoke: Callable[[ExecutionTarget], Awaitable[Any]],
    ) -> Any:
        return await execute_with_fallback(
            step=step,
            session_id=self.session_id[:200],
            run=invoke,
            chain=await self._chain(step),
        )

    async def run(self, step: Step, agent: Any, prompt: str, **run_kwargs: Any) -> Any:
        return await self.run_adapter(
            step,
            lambda target: agent.run(prompt, model=target.handle, **run_kwargs),
        )

    async def stream_adapter(
        self,
        step: Step,
        make_stream: Callable[[ExecutionTarget], AsyncIterator[Any]],
    ) -> AsyncIterator[Any]:
        async for chunk in stream_with_fallback(
            step=step,
            session_id=self.session_id[:200],
            make_stream=make_stream,
            chain=await self._chain(step),
        ):
            yield chunk

    async def stream(
        self,
        agent: Any,
        prompt: str,
        *,
        message_history: list,
        deps: Any,
        new_messages: list,
    ) -> AsyncIterator[str]:
        """Stream Agent text, committing on its first model activity event."""

        async def raw(tier: ExecutionTarget) -> AsyncIterator[Any]:
            activity_signaled = False
            async with agent.iter(
                user_prompt=prompt,
                message_history=message_history,
                deps=deps,
                model=tier.handle,
            ) as agent_run:
                async for node in agent_run:
                    if type(node).__name__ != "ModelRequestNode":
                        continue
                    async with node.stream(agent_run.ctx) as request_stream:
                        async for event in request_stream:
                            if not activity_signaled:
                                activity_signaled = True
                                yield AGENT_ACTIVITY
                            event_type = type(event).__name__
                            if event_type == "PartStartEvent" and type(event.part).__name__ == "TextPart":
                                if event.part.content:
                                    yield event.part.content
                            elif event_type == "PartDeltaEvent" and type(event.delta).__name__ == "TextPartDelta":
                                if event.delta.content_delta:
                                    yield event.delta.content_delta
                new_messages.extend(agent_run.result.new_messages())

        async for chunk in self.stream_adapter(Step.AGENT, raw):
            yield chunk


async def context(session_id: str) -> ExecutionContext:
    """Snapshot current config and resolve the session profile exactly once."""
    from app.llm_core import runtime, split

    config = runtime.get_pipeline().model_copy(deep=True)
    profile_name = await split.resolve_profile(session_id, config)
    return ExecutionContext(session_id=session_id, config=config, profile_name=profile_name)
