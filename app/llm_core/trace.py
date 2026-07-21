"""Per-turn pipeline-trace recorder — the RESOLVED config + routing decisions of
one turn, surfaced on the Langfuse trace as a handful of COMPACT flat metadata
keys.

TRACING ONLY — this module changes NO pipeline behaviour.

Landing path (important): this Langfuse SDK has **no** ``update_current_trace``,
and the working ``propagate_attributes(metadata=...)`` path maps to OTEL span
attributes whose values are SIZE-CAPPED (~128-256 chars) — a big nested blob is
silently dropped. So the request path builds ``pt`` (via :func:`begin` +
:func:`populate`) and merges :func:`compact_metadata` (short flat keys —
``pipeline_profile`` / ``pipeline_flags`` / ``pc_<step>``) into the SAME metadata
dict it already hands to ``propagate_attributes`` / ``VoiceTrace.metadata``. The
COMPLETE static config is logged once at boot by :func:`log_full_config`
(``grep llm_core.full_config``).

The ``pt`` instance is threaded EXPLICITLY (not read from the ContextVar at the
emit site): the ContextVar does not survive Starlette's StreamingResponse
async-generator boundary. The ContextVar is kept only for the best-effort deep
recorders (health/concurrency/served-tier) that mutate ``pt`` mid-turn.

SECRETS: records endpoints (already in logs), providers, model names and timeouts
— and the *name* of a tier's api-key env var, never its value. It never reads or
stores an api key.

Kept import-clean (stdlib + ``app.config`` + ``config_model`` only) so the voice
repo mirrors this file byte-for-byte and the eventual repo-merge stays mechanical
(this file is one of the convergent cores).
"""

from __future__ import annotations

import contextvars
import json
from dataclasses import dataclass, field
from typing import Any, Optional

from helpers.utils import get_logger

logger = get_logger(__name__)


# ── per-step + per-turn records ───────────────────────────────────────────────
@dataclass
class StepRecord:
    """The resolved config + routing outcome for ONE executed pipeline step."""

    step: str
    provider: Optional[str] = None       # primary tier's provider ("vllm"/"openai"/...)
    model: Optional[str] = None          # primary tier's model name
    endpoint: Optional[str] = None       # primary tier's endpoint URL (or "managed")
    timeout_ms: Optional[int] = None     # primary tier's per-attempt timeout
    tier_chain: list[dict] = field(default_factory=list)  # ordered, primary-first
    tier_served_kind: Optional[str] = None   # kind of the tier that actually served
    tier_served_index: Optional[int] = None  # 0 = primary, 1 = first fallback, ...
    health: Optional[dict] = None        # {"pruned": [...], "breaker_states": {...}}
    concurrency: Optional[dict] = None   # {"gauge", "max_concurrency", "deprioritized", ...}

    def to_dict(self) -> dict:
        served = None
        if self.tier_served_kind is not None or self.tier_served_index is not None:
            served = {"kind": self.tier_served_kind, "index": self.tier_served_index}
        triggers: dict = {}
        if self.health is not None:
            triggers["health"] = self.health
        if self.concurrency is not None:
            triggers["concurrency"] = self.concurrency
        out: dict = {
            "provider": self.provider,
            "model": self.model,
            "endpoint": self.endpoint,
            "timeout_ms": self.timeout_ms,
            "tier_served": served,
            "chain": self.tier_chain,
        }
        if triggers:
            out["triggers"] = triggers
        return out


@dataclass
class PipelineTrace:
    """Accumulates one turn's resolved profile, per-step tiers and trigger
    outcomes. Built by :func:`begin`; drained by :meth:`to_metadata`."""

    variant: Optional[str] = None
    profile_name: Optional[str] = None
    profile_weight: Optional[int] = None
    steps: dict[str, StepRecord] = field(default_factory=dict)
    flags: dict = field(default_factory=dict)

    def step(self, name: str) -> StepRecord:
        """Get-or-create the record for a step (health/concurrency may touch it
        before the materialized chain is recorded)."""
        rec = self.steps.get(name)
        if rec is None:
            rec = StepRecord(step=name)
            self.steps[name] = rec
        return rec

    def to_metadata(self) -> dict:
        """The ``pipeline`` metadata object (schema shared chat<->voice)."""
        profile = None
        if self.profile_name is not None:
            profile = {"name": self.profile_name, "weight": self.profile_weight}
        return {
            "profile": profile,
            "variant": self.variant,
            "flags": self.flags,
            "steps": {name: rec.to_dict() for name, rec in self.steps.items()},
        }


_CTX: contextvars.ContextVar[Optional[PipelineTrace]] = contextvars.ContextVar(
    "llm_core_pipeline_trace", default=None
)


def snapshot_flags() -> dict:
    """The operational trigger flags that gate the pipeline — recorded so a turn's
    trace shows which machinery was even eligible to fire. The llm_core/profiles
    kill-switches were removed in P4 (the unified pipeline is now the only path),
    leaving only the health-breaker/poller and concurrency-gauge triggers."""
    from app.config import settings

    return {
        "health_breaker_enabled": bool(getattr(settings, "health_breaker_enabled", False)),
        "health_poller_enabled": bool(getattr(settings, "health_poller_enabled", False)),
        "concurrency_gauge_enabled": bool(getattr(settings, "concurrency_gauge_enabled", False)),
    }


def begin(variant: Optional[str] = None) -> PipelineTrace:
    """Open a fresh per-turn recorder and install it in the context. Idempotent
    per turn: the chat/voice request path calls this once, near the top, as soon
    as the resolved variant is known."""
    pt = PipelineTrace(variant=variant, flags=snapshot_flags())
    _CTX.set(pt)
    return pt


def current() -> Optional[PipelineTrace]:
    return _CTX.get()


def clear() -> None:
    _CTX.set(None)


# ── record hooks (all no-op when no context is active) ────────────────────────
def record_profile(name: str, weight: Optional[int]) -> None:
    pt = _CTX.get()
    if pt is None:
        return
    pt.profile_name = name
    pt.profile_weight = weight


# ── EXPLICIT-instance API (contextvar-independent) ────────────────────────────
# The ContextVar does NOT survive Starlette's StreamingResponse async-generator
# consumption (each __anext__ step can run under a different context snapshot), so
# an ``emit_to_trace()`` that read the contextvar got a fresh EMPTY PipelineTrace.
# The request path therefore holds the ``pt`` returned by ``begin()`` and threads
# it explicitly: populate the static must-have fields on it here, and pass it to
# ``emit_to_trace(pt)``. Deep trigger/served recording via the contextvar stays
# best-effort on top.
def set_profile(pt: Optional[PipelineTrace], name: str, weight: Optional[int]) -> None:
    if pt is None:
        return
    pt.profile_name = name
    pt.profile_weight = weight


def set_step_primary(pt: Optional[PipelineTrace], step: Any, tier: Any) -> None:
    """Set a step's PRIMARY resolved tier (provider/model/endpoint/timeout) on an
    EXPLICIT pt, from a resolved ``MaterializedTier``/``Attempt``. Independent of
    the contextvar. Preserves any served/trigger fields a deep recorder may have
    already set on the same step."""
    if pt is None or tier is None:
        return
    name = getattr(step, "value", step)
    rec = pt.step(name)
    rec.provider = getattr(tier, "provider", None)
    rec.model = getattr(tier, "model_name", None)
    rec.endpoint = getattr(tier, "endpoint", None)
    rec.timeout_ms = _timeout_ms(tier)
    if not rec.tier_chain:
        rec.tier_chain = [_tier_summary(tier)]
    if rec.tier_served_index is None:
        rec.tier_served_kind = getattr(tier, "kind", None)
        rec.tier_served_index = 0


def populate(
    pt: Optional[PipelineTrace],
    pipeline: Any,
    primary_tier_fn: Any,
    variant: Optional[str],
    steps: Any,
) -> None:
    """Explicitly (contextvar-independent) populate the fields the emit MUST carry:
    the resolved profile (from the pipeline + variant) and each step's PRIMARY tier
    (resolved via ``primary_tier_fn(step, variant)``). Best-effort per step; a
    resolve failure for one step is skipped, never raised into the request path.

    ``pipeline`` and ``primary_tier_fn`` are passed in (duck-typed) so this module
    stays import-clean — it never imports resolver/runtime itself."""
    if pt is None:
        return
    try:
        name = "oss" if variant == "oss" else "managed"
        prof = pipeline.by_name(name) or pipeline.by_name("managed") or pipeline.profiles[0]
        set_profile(pt, prof.name, prof.weight)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("llm_core.trace: profile populate skipped: %s", e)
    for step in steps:
        try:
            set_step_primary(pt, step, primary_tier_fn(step, variant))
        except Exception:
            continue


def _tier_summary(tier: Any) -> dict:
    """A secret-free summary of a resolved tier (``MaterializedTier``/``Attempt``)."""
    return {
        "kind": getattr(tier, "kind", None),
        "provider": getattr(tier, "provider", None),
        "model": getattr(tier, "model_name", None),
        "endpoint": getattr(tier, "endpoint", None),
    }


def _timeout_ms(tier: Any) -> Optional[int]:
    t = getattr(tier, "timeout", None)
    return int(round(t * 1000)) if t is not None else None


def record_step_chain(step: Any, chain: list) -> None:
    """Record the resolved (materialized) tier chain for a step: primary tier's
    provider/model/endpoint/timeout + the ordered chain. Defaults ``tier_served``
    to the primary (index 0) — the fallback walker overwrites it if a later tier
    actually serves."""
    pt = _CTX.get()
    if pt is None or not chain:
        return
    name = getattr(step, "value", step)
    rec = pt.step(name)
    primary = chain[0]
    rec.provider = getattr(primary, "provider", None)
    rec.model = getattr(primary, "model_name", None)
    rec.endpoint = getattr(primary, "endpoint", None)
    rec.timeout_ms = _timeout_ms(primary)
    rec.tier_chain = [_tier_summary(t) for t in chain]
    # Default served = primary; a real fallback overwrites via record_served.
    if rec.tier_served_index is None:
        rec.tier_served_kind = getattr(primary, "kind", None)
        rec.tier_served_index = 0


def record_served(step: Any, kind: Optional[str], index: int) -> None:
    """The fallback walker's success hook: which tier (kind + 0-based index in the
    chain) actually produced the answer."""
    pt = _CTX.get()
    if pt is None:
        return
    name = getattr(step, "value", step)
    rec = pt.step(name)
    rec.tier_served_kind = kind
    rec.tier_served_index = index


def record_health_prune(step: Any, pruned: list, breaker_states: dict) -> None:
    """Health-filter outcome for a step: the endpoints pruned (breaker ``open``)
    and the breaker state consulted per endpoint. Best-effort — mutates ``pt`` on
    the current turn's context; the data surfaces via the compact metadata keys."""
    pt = _CTX.get()
    if pt is None:
        return
    name = getattr(step, "value", step)
    pt.step(name).health = {"pruned": list(pruned), "breaker_states": dict(breaker_states)}


def record_concurrency(
    step: Any,
    *,
    gauge: Optional[int],
    max_concurrency: Optional[int],
    deprioritized: bool,
    metrics_url: Optional[str],
) -> None:
    """Concurrency-gauge outcome for a step: the gauge read, the threshold, and
    whether the vLLM tier was deprioritized. Best-effort — mutates ``pt`` on the
    current turn's context; the data surfaces via the compact metadata keys."""
    pt = _CTX.get()
    if pt is None:
        return
    name = getattr(step, "value", step)
    pt.step(name).concurrency = {
        "gauge": gauge,
        "max_concurrency": max_concurrency,
        "deprioritized": bool(deprioritized),
        "metrics_url": metrics_url,
    }


# ── compact trace-metadata keys (the path that actually lands) ────────────────
# The Langfuse SDK here has NO ``update_current_trace``; the working path is the
# ``propagate_attributes(metadata=...)`` dict (chat) / ``VoiceTrace.metadata``
# (voice), which maps to OTEL span attributes — and those SIZE-CAP each value
# (~128-256 chars), so a big nested blob is silently dropped. So we serialize the
# resolved config into a handful of SHORT flat string keys that fit under the cap
# and land reliably. The COMPLETE static config still goes to the boot log via
# ``log_full_config`` (``grep llm_core.full_config``).
#
# Every emitted value is HARD-CAPPED to ``_ATTR_CAP`` chars so a long configured
# endpoint/model/profile can never push a key over the OTEL cap and silently drop
# it (the exact failure this compact path exists to avoid). Truncation is safe:
# the complete untruncated config is always in the boot ``llm_core.full_config``.
_ATTR_CAP = 240


def compact_metadata(pt: Optional[PipelineTrace]) -> dict:
    """Flatten ``pt`` into short, cap-safe metadata keys:

    * ``pipeline_profile`` = resolved profile name;
    * ``pipeline_flags``   = comma-joined enabled guard flags (``_enabled`` stripped);
    * ``pc_<step>``        = ``"<provider>:<model>@<endpoint>#<kind>(<timeout_ms>ms)"``
                             per step, where ``<kind>`` is the CONFIGURED PRIMARY
                             tier (oss/managed) — NOT necessarily the tier that
                             actually served. Health-prune / concurrency-reorder /
                             failure fallback can route a given request to a
                             different tier; that is visible only in the pydantic-ai
                             GENERATION observations on the same trace (tracked
                             follow-up to surface the served tier here).

    Returns ``{}`` on any error / empty pt (never raises into the request path).
    None-valued keys are dropped (propagate_attributes wants string values)."""
    out: dict = {}
    if pt is None:
        return out
    try:
        pc = pt.to_metadata()
        out["pipeline_profile"] = (pc.get("profile") or {}).get("name")
        out["pipeline_flags"] = ",".join(
            k.replace("_enabled", "") for k, v in (pc.get("flags") or {}).items() if v
        )
        for step_name, sv in (pc.get("steps") or {}).items():
            sv = sv or {}
            served = (sv.get("tier_served") or {}).get("kind")
            out[f"pc_{step_name}"] = (
                f'{sv.get("provider")}:{sv.get("model")}@{sv.get("endpoint")}'
                f'#{served}({sv.get("timeout_ms")}ms)'
            )
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("llm_core.trace: compact_metadata failed: %s", e)
    # Hard-cap every value (a long endpoint/model can't exceed the OTEL cap and
    # silently drop the key) and drop None values (propagate_attributes expects
    # string attribute values — a None profile on the populate-failure path must
    # never reach it).
    return {k: v[:_ATTR_CAP] for k, v in out.items() if isinstance(v, str)}


def add_compact_metadata(pt: Optional[PipelineTrace], metadata: dict) -> None:
    """Merge :func:`compact_metadata` into an existing metadata dict IN PLACE — the
    same dict the request path already hands to ``propagate_attributes`` (chat) /
    ``VoiceTrace.metadata`` (voice). No-op-safe (never raises)."""
    if metadata is None:
        return
    try:
        metadata.update(compact_metadata(pt))
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("llm_core.trace: add_compact_metadata failed: %s", e)


# ── startup full-config dump ──────────────────────────────────────────────────
def config_to_dict(pipeline: Any) -> dict:
    """The COMPLETE loaded ``PipelineConfig`` as a plain dict — all profiles, all
    step tiers, all triggers — for one greppable structured boot log line.

    Secret-free by construction: a ``Tier`` only ever names its api-key env var
    (``api_key_env``); the value is never on the model, so this can't leak a key.
    """

    def _tier(t: Any) -> dict:
        return {
            "provider": getattr(getattr(t, "provider", None), "value", getattr(t, "provider", None)),
            "model": getattr(t, "model", None),
            "endpoint": getattr(t, "endpoint", None),
            "timeout_ms": getattr(t, "timeout_ms", None),
            "api_key_env": getattr(t, "api_key_env", None),  # NAME only, never the value
            "api_version": getattr(t, "api_version", None),
        }

    def _triggers(tr: Any) -> dict:
        gate = getattr(tr, "concurrency_gate", None)
        return {
            "ttft_deadline_ms": getattr(tr, "ttft_deadline_ms", None),
            "health_check": getattr(tr, "health_check", None),
            "concurrency_gate": (
                {
                    "metrics_url": getattr(gate, "metrics_url", None),
                    "max_concurrency": getattr(gate, "max_concurrency", None),
                }
                if gate is not None
                else None
            ),
        }

    def _step_cfg(sc: Any) -> dict:
        return {
            "tiers": [_tier(t) for t in getattr(sc, "tiers", [])],
            "triggers": _triggers(getattr(sc, "triggers", None)),
        }

    def _steps(steps: Any) -> dict:
        return {
            getattr(step, "value", step): _step_cfg(sc) for step, sc in (steps or {}).items()
        }

    return {
        "sticky_ttl_s": getattr(pipeline, "sticky_ttl_s", None),
        "fallback_enabled": getattr(pipeline, "fallback_enabled", None),
        "defaults": _steps(getattr(pipeline, "defaults", {})),
        "profiles": [
            {
                "name": p.name,
                "weight": p.weight,
                "steps": _steps(getattr(p, "steps", {})),
            }
            for p in getattr(pipeline, "profiles", [])
        ],
    }


def log_full_config(pipeline: Any) -> None:
    """Emit the complete resolved config as ONE structured log line at boot, so the
    full wiring is greppable in logs even before any turn arrives (``grep
    llm_core.full_config``). Best-effort — never breaks startup."""
    try:
        payload = config_to_dict(pipeline)
        payload["flags"] = snapshot_flags()
        logger.info("llm_core.full_config %s", json.dumps(payload, sort_keys=True))
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("llm_core.trace: log_full_config failed: %s", e)
