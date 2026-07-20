"""Per-turn pipeline-trace recorder — the full RESOLVED config + routing
decisions of one turn, emitted as a single ``pipeline`` object into the Langfuse
trace metadata.

TRACING ONLY — this module changes NO pipeline behaviour. Every recorder is a
no-op unless a :class:`PipelineTrace` context is active (opened per-turn by the
chat/voice request path via :func:`begin`), so the core seams
(``split``/``health``/``concurrency``/``fallback``) can call the ``record_*``
hooks unconditionally and cheaply — a bare ``ContextVar.get()`` when nothing is
listening. Nothing here is on any poll loop; it only ever runs on the request
path.

SECRETS: this records endpoints (already in logs), providers, model names and
timeouts — and the *name* of a tier's api-key env var, never its value. It never
reads or stores an api key. See :func:`_no_secret_assert` in the tests.

Kept import-clean (stdlib + ``app.config`` + ``config_model`` + a best-effort
Langfuse import) so the voice repo mirrors this file byte-for-byte and the
eventual repo-merge stays mechanical (this file is one of the convergent cores).
"""

from __future__ import annotations

import contextvars
import json
from dataclasses import dataclass, field
from typing import Any, Optional

from helpers.utils import get_logger

logger = get_logger(__name__)

# Best-effort Langfuse trace tagging (same import shape as services.fallback).
try:  # pragma: no cover - import guard
    from langfuse import get_client as _get_langfuse_client
except Exception:  # pragma: no cover
    _get_langfuse_client = None


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
    """The five guard flags that gate the pipeline — recorded so a turn's trace
    shows which machinery was even eligible to fire."""
    from app.config import settings

    return {
        "llm_core_enabled": bool(getattr(settings, "llm_core_enabled", False)),
        "profiles_enabled": bool(getattr(settings, "profiles_enabled", False)),
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
    and the breaker state consulted per endpoint. Adds a cheap trace tag when a
    tier was actually pruned so a turn's routing decision is visible."""
    pt = _CTX.get()
    if pt is None:
        return
    name = getattr(step, "value", step)
    pt.step(name).health = {"pruned": list(pruned), "breaker_states": dict(breaker_states)}
    if pruned:
        _tag_trace([f"health_prune:{name}"])


def record_concurrency(
    step: Any,
    *,
    gauge: Optional[int],
    max_concurrency: Optional[int],
    deprioritized: bool,
    metrics_url: Optional[str],
) -> None:
    """Concurrency-gauge outcome for a step: the gauge read, the threshold, and
    whether the vLLM tier was deprioritized. Adds a cheap trace tag when a tier
    was actually deprioritized."""
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
    if deprioritized:
        _tag_trace([f"concurrency_deprioritize:{name}"])


# ── emission ──────────────────────────────────────────────────────────────────
def _tag_trace(tags: list) -> None:
    if _get_langfuse_client is None:
        return
    try:  # pragma: no cover - best effort
        client = _get_langfuse_client()
        if client is not None:
            client.update_current_trace(tags=tags)
    except Exception:
        pass


# The trace-metadata key the resolved-config object is emitted under. It is
# deliberately NOT ``pipeline`` — chat already sets ``metadata["pipeline"]`` to
# the pipeline NAME string ("translation"/"default") via propagate_attributes, so
# reusing that key would collide (the string shadows the object). ``pipeline_config``
# is a distinct namespace; the existing ``pipeline``/``variant`` keys are untouched.
METADATA_KEY = "pipeline_config"


def emit_to_trace() -> None:
    """Flush the accumulated resolved-config object onto the current Langfuse
    trace's metadata under the ``pipeline_config`` key. Best-effort and merge-only:
    it never overwrites the existing ``pipeline``/``variant``/``pipeline_variant``
    keys or scores (those stay for dashboard continuity). No-op when no context is
    active or Langfuse is unavailable."""
    pt = _CTX.get()
    if pt is None or _get_langfuse_client is None:
        return
    try:  # pragma: no cover - best effort
        client = _get_langfuse_client()
        if client is not None:
            client.update_current_trace(metadata={METADATA_KEY: pt.to_metadata()})
    except Exception as e:
        logger.debug("llm_core.trace: emit_to_trace failed: %s", e)


# ── startup full-config dump (item 4) ─────────────────────────────────────────
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
