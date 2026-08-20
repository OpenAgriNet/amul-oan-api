"""
Beckn network client for union scheme discovery.

When `settings.enable_network` is true, the union-scheme tool routes through
Amul's Beckn network instead of the Redis cache. This module wraps the seeker
call and formats Beckn `on_search` payloads into the same JSON string shape
the direct tool returns.

Topology:
  - discovery → seeker BAP  {AMUL_NETWORK_URL}/search  (sync aggregation,
                 scoped by a `legs` filter)

The seeker returns `{ results: { <leg>: on_search|null }, traces, errors }`.
Leg names: schemes:amul-union → "amulschemes"; Vistaar central schemes use the
configured `VISTAAR_LEG`.

One seeker request carries ONE query string for every leg it fans out to (see
`seeker.js`: `input.query` is built once and passed to each `callOne`). The legs
do not share a query vocabulary — `amulschemes` does free-text title matching
while the Bharat Vistaar BPP matches exact scheme codes and nothing else — so
scheme discovery issues one concurrent request PER query instead. Same wall
time, different words down each leg.
"""
import asyncio
from typing import Any, Optional

import httpx

from agents.tools.scheme_codes import resolve_scheme_code
from app.config import settings
from helpers.utils import get_logger

logger = get_logger(__name__)

SCHEMES_LEG = "amulschemes"
# Bharat Vistaar government schemes (KCC etc.). Use the same configured leg as
# the dedicated Vistaar tools: dev and production intentionally route through
# different seeker legs, and hard-coding the legacy MOA leg sends production
# scheme discovery back to the playground sandbox.
VISTAAR_LEG = settings.vistaar_leg


def _providers(on_search: Any) -> list[dict]:
    """Extract providers[] from a Beckn on_search body (handles both
    `catalog.providers` and the 1.x `catalog.bpp/providers`)."""
    catalog = (on_search or {}).get("message", {}).get("catalog", {}) if isinstance(on_search, dict) else {}
    return catalog.get("providers") or catalog.get("bpp/providers") or []


def _items(on_search: Any) -> list[dict]:
    return [it for prov in _providers(on_search) for it in prov.get("items", [])]


def _tag_map(item: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for t in item.get("tags", []) or []:
        if "code" in t and "value" in t:
            out[t["code"]] = t["value"]
    return out


async def _seeker_search_legs(
    query: str, legs: list[str], user_id: Optional[str] = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """POST the seeker BAP for one or more legs sharing ONE query string.

    Returns `(results, errors)` — both keyed by leg. The seeker only includes
    `errors` when a leg actually failed, and a failed leg's `results` entry is
    null, which is indistinguishable from an empty catalogue unless you read
    `errors`. Callers that ignore the second element will report a dead leg as
    "nothing found"; that is the bug that hid the `moa` timeout flap.
    """
    url = f"{settings.amul_network_url.rstrip('/')}/search"
    payload: dict[str, Any] = {"query": query, "legs": legs}
    if user_id:
        payload["user_id"] = user_id
    async with httpx.AsyncClient(timeout=settings.amul_network_timeout_s) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
    return (data.get("results") or {}), (data.get("errors") or {})


async def _seeker_search_per_leg(
    queries: dict[str, str], user_id: Optional[str] = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Search several legs, each with its OWN query string.

    The seeker has no per-leg query field, so this is one concurrent request per
    leg rather than one fan-out request. Wall time is unchanged (they run under
    a single `asyncio.gather`); what changes is that each leg finally receives
    words it can match. A leg whose request raises is reported in `errors`
    rather than taking the whole tool down — a dead central leg must not cost
    the farmer their union schemes.
    """
    legs = list(queries)
    outcomes = await asyncio.gather(
        *(_seeker_search_legs(queries[leg], [leg], user_id) for leg in legs),
        return_exceptions=True,
    )
    results: dict[str, Any] = {}
    errors: dict[str, Any] = {}
    for leg, outcome in zip(legs, outcomes):
        if isinstance(outcome, BaseException):
            logger.warning("seeker leg=%s request failed: %r", leg, outcome)
            errors[leg] = str(outcome) or outcome.__class__.__name__
            continue
        leg_results, leg_errors = outcome
        results[leg] = leg_results.get(leg)
        if leg_errors.get(leg):
            errors[leg] = leg_errors[leg]
    return results, errors


async def network_union_schemes(query: str, union: Optional[str] = None) -> str:
    """Scheme discovery via the network — Amul union schemes (schemes:amul-union)
    AND Bharat Vistaar central schemes (via the configured Vistaar leg), merged so the farmer
    sees both in one answer.

    Each leg gets its OWN query. The union leg takes the farmer's free text (it
    matches scheme titles); the BV leg takes a resolved scheme CODE, because its
    BPP matches nothing else — measured on dev, `query="schemes"` returns 15
    union items and 0 central, while `query="kcc"` returns 0 union and 1
    central. One shared string could never have produced both, which is why this
    tool returned zero `bharat-vistaar` records every time.

    When no code resolves the farmer is not asking about a central scheme, so
    the BV leg is SKIPPED rather than sent a word that always returns nothing —
    saving a guaranteed-empty ~2.2s round trip.
    """
    union_query = (query or "").strip() or "schemes"
    scheme_code = resolve_scheme_code(query)
    queries = {SCHEMES_LEG: union_query}
    if scheme_code:
        queries[VISTAAR_LEG] = scheme_code
    logger.info(
        "network scheme discovery union_query=%r vistaar_code=%s legs=%s",
        union_query, scheme_code, list(queries),
    )
    results, errors = await _seeker_search_per_leg(queries)

    def _leg_failed(leg: str) -> bool:
        """A leg failed if the seeker said so, or if it answered with no
        on_search body at all. Only a leg we never asked for is 'not failed'."""
        if leg not in queries:
            return False
        return bool(errors.get(leg)) or not isinstance(results.get(leg), dict)

    union_failed = _leg_failed(SCHEMES_LEG)
    vistaar_failed = _leg_failed(VISTAAR_LEG)

    union_items = _items(results.get(SCHEMES_LEG))
    if union:
        u = union.strip().lower()
        union_items = [it for it in union_items if _tag_map(it).get("union", "").lower() == u] or union_items
    vistaar_items = _items(results.get(VISTAAR_LEG))

    out: list[dict] = []
    for it in union_items:
        d = it.get("descriptor", {})
        tags = _tag_map(it)
        out.append({
            "scheme_title": d.get("name"),
            "description": d.get("long_desc") or d.get("short_desc"),
            "union": tags.get("union"),
            "category": tags.get("category"),
            "source": tags.get("source"),
            "source_network": "amul-union",
        })
    for it in vistaar_items:
        d = it.get("descriptor", {})
        tags = _tag_map(it)
        out.append({
            "scheme_title": d.get("name"),
            "description": d.get("long_desc") or d.get("short_desc"),
            "category": tags.get("category"),
            "source": tags.get("source"),
            "source_network": "bharat-vistaar",
        })
    # A leg that FAILED must never be reported as a leg that found nothing.
    # These two sentences are the whole point of reading `errors`.
    unavailable = [
        name
        for name, failed in (
            ("Amul union schemes", union_failed),
            ("central government schemes", vistaar_failed),
        )
        if failed
    ]
    if not out:
        if unavailable:
            return (
                "Scheme information ("
                + " and ".join(unavailable)
                + ") is temporarily unavailable. Please try again shortly."
            )
        return "No scheme data was found on the network for this query."
    import json
    payload = json.dumps(out, ensure_ascii=False, indent=2)
    if unavailable:
        payload += (
            "\n\nNOTE: " + " and ".join(unavailable) + " could not be reached for this "
            "query, so the list above may be incomplete. Tell the farmer that part is "
            "temporarily unavailable — do NOT state that no such schemes exist."
        )
    return payload
