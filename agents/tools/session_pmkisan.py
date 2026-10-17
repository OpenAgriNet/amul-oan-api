"""Short-lived, session-owned state for the two-turn PM-KISAN OTP flow."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

from app.config import settings
from app.core.cache import cache

PMKISAN_SESSION_NAMESPACE = "pmkisan-session"


def _owner_token(mobile: str | None) -> str:
    digits = re.sub(r"\D", "", mobile or "")[-10:]
    owner = digits or "anonymous"
    return hashlib.sha256(owner.encode("utf-8")).hexdigest()


def _cache_key(session_id: str, mobile: str | None) -> str:
    return f"{session_id}:{_owner_token(mobile)}"


async def get_pm_kisan_session(
    session_id: str | None, mobile: str | None
) -> dict[str, Any] | None:
    if not session_id:
        return None
    value = await cache.get(
        _cache_key(session_id, mobile), namespace=PMKISAN_SESSION_NAMESPACE
    )
    return dict(value) if isinstance(value, Mapping) else None


async def set_pm_kisan_session(
    session_id: str | None,
    mobile: str | None,
    state: Mapping[str, Any],
) -> None:
    if not session_id:
        raise ValueError("session_id is required for PM-KISAN status checks")
    await cache.set(
        _cache_key(session_id, mobile),
        dict(state),
        ttl=settings.vistaar_pmkisan_session_ttl_seconds,
        namespace=PMKISAN_SESSION_NAMESPACE,
    )


async def clear_pm_kisan_session(
    session_id: str | None, mobile: str | None
) -> None:
    if not session_id:
        return
    await cache.delete(
        _cache_key(session_id, mobile), namespace=PMKISAN_SESSION_NAMESPACE
    )
