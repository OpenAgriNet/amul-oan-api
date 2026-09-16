"""Shared cache-aside policy for agent-facing read tools.

The Beckn adapters remain the public interface.  This module only handles
successful response reuse and deliberately fails open when Redis is unavailable
or a cached value can no longer be decoded.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

from app.config import settings
from app.core.cache import cache
from helpers.utils import get_logger

logger = get_logger(__name__)

_CACHE_NAMESPACE = "agent-tool-response-v1"
_CACHE_VERSION = 1
T = TypeVar("T")


@dataclass(frozen=True)
class ResponseCachePolicy:
    name: str
    ttl_seconds: int
    negative_ttl_seconds: int


AI_TECHNICIAN_CACHE_POLICY = ResponseCachePolicy(
    name="ai-technician",
    ttl_seconds=settings.agent_ai_technician_cache_ttl_seconds,
    negative_ttl_seconds=settings.agent_negative_cache_ttl_seconds,
)
FARMER_CACHE_POLICY = ResponseCachePolicy(
    name="farmer",
    ttl_seconds=settings.agent_farmer_cache_ttl_seconds,
    negative_ttl_seconds=settings.agent_negative_cache_ttl_seconds,
)
ANIMAL_CACHE_POLICY = ResponseCachePolicy(
    name="animal",
    ttl_seconds=settings.agent_animal_cache_ttl_seconds,
    negative_ttl_seconds=settings.agent_negative_cache_ttl_seconds,
)
CVCC_CACHE_POLICY = ResponseCachePolicy(
    name="cvcc",
    ttl_seconds=settings.agent_cvcc_cache_ttl_seconds,
    negative_ttl_seconds=settings.agent_negative_cache_ttl_seconds,
)


def _cache_key(policy: ResponseCachePolicy, identity: Mapping[str, object]) -> str:
    material = json.dumps(
        identity,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return f"{policy.name}:{digest}"


async def _discard_invalid_entry(key: str) -> None:
    try:
        await cache.delete(key, namespace=_CACHE_NAMESPACE)
    except Exception as exc:
        logger.warning("Agent response cache cleanup failed for %s: %s", key, exc)


async def get_or_load(
    *,
    policy: ResponseCachePolicy,
    identity: Mapping[str, object],
    loader: Callable[[], Awaitable[T]],
    encode: Callable[[T], Any],
    decode: Callable[[Any], T],
    is_negative: Callable[[T], bool],
    force_refresh: bool = False,
) -> T:
    """Return a validated cached response or load and cache a fresh success.

    ``force_refresh`` bypasses the read but still replaces the entry after a
    successful live call.  Exceptions from ``loader`` propagate and therefore
    can never become cached responses.
    """
    key = _cache_key(policy, identity)
    if not force_refresh:
        try:
            cached = await cache.get(key, namespace=_CACHE_NAMESPACE)
        except Exception as exc:
            logger.warning("Agent response cache read failed for %s: %s", policy.name, exc)
        else:
            if cached is not None:
                try:
                    if not isinstance(cached, dict) or cached.get("version") != _CACHE_VERSION:
                        raise ValueError("unsupported cache envelope")
                    return decode(cached["value"])
                except Exception as exc:
                    logger.warning(
                        "Ignoring invalid %s response cache entry: %s",
                        policy.name,
                        exc,
                    )
                    await _discard_invalid_entry(key)

    value = await loader()
    ttl = policy.negative_ttl_seconds if is_negative(value) else policy.ttl_seconds
    try:
        await cache.set(
            key,
            {"version": _CACHE_VERSION, "value": encode(value)},
            ttl=ttl,
            namespace=_CACHE_NAMESPACE,
        )
    except Exception as exc:
        logger.warning("Agent response cache write failed for %s: %s", policy.name, exc)
    return value
