"""Private, session-scoped Soil Health Card context for conversational follow-ups."""

from __future__ import annotations

import hashlib
import re

from app.config import settings
from app.core.cache import cache
from helpers.utils import get_logger

logger = get_logger(__name__)

SHC_CONTEXT_NAMESPACE = "shc-agent-context"


def _cache_key(session_id: str, mobile: str) -> str:
    # The tool uses E.164 (+91...) while authenticated chat stores the last ten
    # digits. Canonicalize here so both representations resolve to one owner.
    digits = re.sub(r"\D", "", mobile)[-10:]
    owner = hashlib.sha256(digits.encode("utf-8")).hexdigest()
    return f"{session_id}:{owner}"


async def get_session_shc_context(session_id: str | None, mobile: str | None) -> str | None:
    """Load agronomic SHC facts only for the same authenticated session owner."""
    if not session_id or not mobile:
        return None
    if len(re.sub(r"\D", "", mobile)) < 10:
        return None
    try:
        value = await cache.get(
            _cache_key(session_id, mobile),
            namespace=SHC_CONTEXT_NAMESPACE,
        )
    except Exception as exc:  # noqa: BLE001 - a cache blip must not break chat
        logger.warning("session SHC context read failed: %s", exc)
        return None
    return value if isinstance(value, str) and value.strip() else None


async def set_session_shc_context(
    session_id: str | None,
    mobile: str | None,
    context: str,
) -> None:
    """Store bounded agronomic facts, never the raw HTML report or identity fields."""
    if not session_id or not mobile or not context.strip():
        return
    if len(re.sub(r"\D", "", mobile)) < 10:
        return
    try:
        await cache.set(
            _cache_key(session_id, mobile),
            context,
            ttl=settings.history_cache_ttl_seconds,
            namespace=SHC_CONTEXT_NAMESPACE,
        )
    except Exception as exc:  # noqa: BLE001 - current-turn summary still works
        logger.warning("session SHC context write failed: %s", exc)
