"""Sticky per-session pipeline-variant router.

Routes a configurable percentage of chat sessions to the OSS pipeline
(vLLM gemma agent + translategemma pre/post-translation) while the rest stay
on the legacy pipeline. The assignment is:

* deterministic   - bucketed from a stable hash of session_id, so the same
                     session always lands the same way even without Redis;
* sticky          - the resolved variant is persisted in the shared Redis
                     cache, so a session keeps its variant for its lifetime
                     even if OSS_PIPELINE_PCT is changed afterwards;
* shared          - all API instances read the same Redis state;
* fail-safe       - any Redis error degrades to the deterministic hash, never
                     raising into the request path. With OSS_PIPELINE_PCT=0 (or
                     OSS model unconfigured) every session resolves to
                     ``legacy`` and behaviour is identical to today.
"""
from __future__ import annotations

import hashlib

from app.config import settings
from app.core.cache import cache
from agents.models import oss_model_available
from helpers.utils import get_logger

logger = get_logger(__name__)

LEGACY = "legacy"
OSS = "oss"

_KEY_PREFIX = "pipeline_variant:"


def _deterministic_variant(session_id: str) -> str:
    """Stable 0-99 bucket from session_id; OSS when bucket < configured pct."""
    pct = max(0, min(100, settings.oss_pipeline_pct))
    if pct <= 0 or not oss_model_available():
        return LEGACY
    digest = hashlib.sha256((session_id or "").encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return OSS if bucket < pct else LEGACY


async def resolve_pipeline_variant(session_id: str) -> str:
    """Return the sticky pipeline variant ('oss' | 'legacy') for a session."""
    if not session_id:
        return _deterministic_variant(session_id)

    key = f"{_KEY_PREFIX}{session_id}"

    try:
        stored = await cache.get(key)
        if stored in (OSS, LEGACY):
            return stored
    except Exception as e:  # Redis down / timeout -> deterministic fallback
        logger.warning("pipeline_router: cache read failed for %s: %s", session_id, e)
        return _deterministic_variant(session_id)

    variant = _deterministic_variant(session_id)

    try:
        await cache.set(key, variant, ttl=settings.oss_variant_ttl)
    except Exception as e:  # persistence is best-effort; assignment still stable
        logger.warning("pipeline_router: cache write failed for %s: %s", session_id, e)

    # Note: the per-session Langfuse `pipeline_variant` score is emitted from
    # `app/services/chat.py` after `set_current_trace_io`, where a Langfuse
    # trace context is active. Emitting from here previously silently no-op'd
    # ("Operations that depend on an active span will be skipped").
    return variant
