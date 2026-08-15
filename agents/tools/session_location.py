"""Session-sticky market location for the Bharat Vistaar mandi/weather tools.

When a farmer says *"prices in Junagadh"*, the next question in the same
conversation is almost always about the same place — *"and cotton?"*. Without
stickiness we would silently drop back to their profile district (or Anand)
between turns, so a two-turn exchange would answer from two different markets
and never say why.

Scope and TTL
-------------
Keyed on `ctx.deps.session_id` only, in the shared Redis via the existing
`app.core.cache` (same client, same prefix). **TTL = 1 hour**
(`MANDI_LOCATION_TTL_S`, override `MANDI_LOCATION_TTL_S` in env): long enough to
cover any single chat session or voice call end-to-end, short enough that a
farmer who returns tomorrow gets their own district back rather than a stale
override they have forgotten stating. It stores the *district key* (e.g.
`"junagadh"`), never coordinates — the table stays the single source of truth
for where a district is, so a corrected coordinate takes effect immediately
instead of being pinned in a cache entry.

Failure policy: **fail soft, both ways.** A Redis blip must never break a price
lookup, so a read error reads as "no override" and a write error is logged and
dropped. The cost of the failure is one un-sticky turn, and the farmer can
restate the location; the cost of raising would be a dead tool.
"""
from __future__ import annotations

import os

from agents.tools.districts import DISTRICTS
from app.core.cache import cache
from helpers.utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "MANDI_LOCATION_NAMESPACE",
    "MANDI_LOCATION_TTL_S",
    "get_session_district_key",
    "set_session_district_key",
]

MANDI_LOCATION_NAMESPACE = "mandi-location"
MANDI_LOCATION_TTL_S = int(os.getenv("MANDI_LOCATION_TTL_S", "3600"))


async def get_session_district_key(session_id: str | None) -> str | None:
    """Return the district key this session last stated, or None.

    An unknown key (a table entry removed since it was written) reads as None
    rather than as a lookup failure downstream.
    """
    if not session_id:
        return None
    try:
        value = await cache.get(str(session_id), namespace=MANDI_LOCATION_NAMESPACE)
    except Exception as exc:  # noqa: BLE001 - a cache blip must not break the tool
        logger.warning("session location read failed session=%s: %s", session_id, exc)
        return None
    if not isinstance(value, str) or value not in DISTRICTS:
        return None
    return value


async def set_session_district_key(session_id: str | None, district_key: str) -> None:
    """Remember the district this session explicitly asked for (best effort)."""
    if not session_id or district_key not in DISTRICTS:
        return
    try:
        await cache.set(
            str(session_id),
            district_key,
            ttl=MANDI_LOCATION_TTL_S,
            namespace=MANDI_LOCATION_NAMESPACE,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("session location write failed session=%s: %s", session_id, exc)
