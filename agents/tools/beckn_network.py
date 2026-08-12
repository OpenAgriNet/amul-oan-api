"""
Beckn network client for the agent tools.

When `settings.enable_network` is true, the vet-KB search, union-scheme, and
AI-call-booking tools route through Amul's own Beckn network instead of their
direct integrations (Marqo / Redis / PashuGPT). This module wraps those network
calls and formats the Beckn `on_search` / `on_confirm` payloads into the same
string shapes the direct tools return, so the agent sees a consistent contract
regardless of the flag.

Topology (all reachable over HTTP):
  - discovery  → seeker BAP  {AMUL_NETWORK_URL}/search  (sync aggregation,
                 scoped by a `legs` filter so a vet query hits only the vet leg)
  - booking    → booking BPP {AMUL_BOOKING_BPP_URL}/confirm

The seeker returns `{ results: { <leg>: on_search|null }, traces, errors }`.
Leg names: advisory:amul-vet → "amulvet", schemes:amul-union → "amulschemes".
"""
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from app.config import settings
from helpers.utils import get_logger

logger = get_logger(__name__)

VET_LEG = "amulvet"
SCHEMES_LEG = "amulschemes"
# Bharat Vistaar government schemes (KCC etc.), reached via the seeker's MOA leg.
# Merged into scheme discovery so a farmer's scheme query surfaces Amul union
# schemes AND Bharat Vistaar central schemes together.
VISTAAR_LEG = "moa"


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


async def _seeker_search(query: str, leg: str, user_id: Optional[str] = None) -> Any:
    """POST the seeker BAP, scoped to a single leg; return that leg's on_search."""
    return (await _seeker_search_legs(query, [leg], user_id)).get(leg)


async def _seeker_search_legs(
    query: str, legs: list[str], user_id: Optional[str] = None
) -> dict[str, Any]:
    """POST the seeker BAP for one or more legs; return the {leg: on_search} map.
    One fan-out call aggregates every requested leg (the seeker mixes sync and
    async legs itself), so callers can merge across sources."""
    url = f"{settings.amul_network_url.rstrip('/')}/search"
    payload: dict[str, Any] = {"query": query, "legs": legs}
    if user_id:
        payload["user_id"] = user_id
    async with httpx.AsyncClient(timeout=settings.amul_network_timeout_s) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
    return data.get("results") or {}


async def network_search_documents(query: str, top_k: int = 12) -> str:
    """Vet-KB discovery via the network (advisory:amul-vet). Formats items the
    same way the direct Marqo tool does: numbered snippets with source."""
    on_search = await _seeker_search(query, VET_LEG)
    items = _items(on_search)[:top_k]
    if not items:
        logger.info("network vet search returned no items query=%s", query)
        return "No relevant documents were found on the network for this query."
    lines: list[str] = []
    for i, it in enumerate(items, 1):
        d = it.get("descriptor", {})
        tags = _tag_map(it)
        text = d.get("long_desc") or d.get("short_desc") or d.get("name", "")
        src = tags.get("source", "")
        lines.append(f"{i}. {d.get('name', 'Document')}\n{text}" + (f"\n(source: {src})" if src else ""))
    return "\n\n".join(lines)


async def network_union_schemes(query: str, union: Optional[str] = None) -> str:
    """Scheme discovery via the network — Amul union schemes (schemes:amul-union)
    AND Bharat Vistaar central schemes (via the MOA leg), merged so the farmer
    sees both in one answer."""
    results = await _seeker_search_legs(query or "schemes", [SCHEMES_LEG, VISTAAR_LEG])
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
    if not out:
        return "No scheme data was found on the network for this query."
    import json
    return json.dumps(out, ensure_ascii=False, indent=2)


@dataclass(frozen=True)
class NetworkBookingResult:
    """Outcome of a network booking confirm.

    The caller (agents/tools/ai_call.py) must know whether a booking actually
    happened — it releases its session reservation on failure and marks the
    session booked on success — so the outcome is structured rather than
    inferred from the message string.

    `ok=False` alone is not enough to justify releasing the reservation: a
    booking that did happen can still come back as a failure to us. Hence
    `authoritative_no_booking`, which is only true when the BPP told us in so
    many words that it did not book (a NACK). It defaults to False so that any
    new or sloppily-constructed failure result takes the safe branch — hold the
    reservation — rather than risking a second SMS to a real farmer.
    """
    ok: bool
    ticket: Optional[str]
    message: str
    authoritative_no_booking: bool = False


async def network_create_ai_call_result(
    union_code: str,
    society_code: str,
    farmer_code: str,
    user_id: str,
    species: str,
) -> NetworkBookingResult:
    """AI-call booking via the network (services:amul-vet-booking) — POSTs a
    Beckn confirm to the booking BPP, which calls PashuGPT CreateAICall.

    Raises on transport/HTTP errors (raise_for_status), exactly as before; the
    caller decides how to surface them.
    """
    order = {
        "provider": {"id": "amul-ai-service"},
        "items": [{"id": f"ait:{user_id}"}],
        "fulfillment": {
            "type": "TECHNICIAN_VISIT",
            "customer": {"tags": [
                {"code": "farmer_id", "value": farmer_code},
                {"code": "species", "value": species},
            ]},
            "stops": [{"location": {"descriptor": {"code": f"society:{society_code}"}}}],
            "tags": [{"code": "union", "value": union_code}],
        },
    }
    context = {
        "domain": "services:amul-vet-booking", "action": "confirm", "version": "1.1.0",
        "transaction_id": f"agent-{society_code}-{user_id}", "message_id": "agent-confirm",
        "timestamp": "1970-01-01T00:00:00.000Z",
    }
    url = f"{settings.amul_booking_bpp_url.rstrip('/')}/confirm"
    async with httpx.AsyncClient(timeout=settings.amul_network_timeout_s) as client:
        r = await client.post(url, json={"context": context, "message": {"order": order}})
        r.raise_for_status()
        body = r.json()
    if (body.get("message", {}).get("ack", {}) or {}).get("status") == "NACK":
        err = body.get("error", {})
        logger.info("network AI call NACK code=%s msg=%s", err.get("code"), err.get("message"))
        return NetworkBookingResult(
            ok=False,
            ticket=None,
            message=(
                "Artificial insemination call booking failed on the network: "
                f"{err.get('message', 'unknown error')}"
            ),
            # A NACK is the BPP stating it did not accept the order. Provably no
            # booking, so the caller may release the reservation and let the
            # farmer retry immediately.
            authoritative_no_booking=True,
        )
    ticket = body.get("message", {}).get("order", {}).get("id")
    if not ticket:
        # A 200 that is neither a NACK nor a confirmed order. `ok` used to
        # default to True here, so `200 {}` told the farmer "booked
        # successfully. Ticket: None" and locked the session for the whole TTL
        # with nothing booked. Not success.
        #
        # It is also NOT authoritative-no-booking: the BPP answered without
        # saying it refused, so it may well have called PashuGPT and sent the
        # SMS before losing the order id. Treated as ambiguous — the caller
        # holds the reservation, same as a read timeout.
        logger.warning(
            "network AI call: 200 without NACK and without order.id; treating as unconfirmed body=%s",
            body,
        )
        return NetworkBookingResult(
            ok=False,
            ticket=None,
            message=(
                "The artificial insemination call booking could not be confirmed. "
                "Please check with your society whether the visit is booked before booking again."
            ),
            authoritative_no_booking=False,
        )
    return NetworkBookingResult(
        ok=True,
        ticket=ticket,
        message=f"Artificial insemination call booked successfully via the Beckn network. Ticket: {ticket}",
    )


async def network_create_ai_call(
    union_code: str,
    society_code: str,
    farmer_code: str,
    user_id: str,
    species: str,
) -> str:
    """String-returning wrapper kept for callers that only need the message."""
    result = await network_create_ai_call_result(
        union_code, society_code, farmer_code, user_id, species
    )
    return result.message
