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
from typing import Any, Optional

import httpx

from app.config import settings
from helpers.utils import get_logger

logger = get_logger(__name__)

VET_LEG = "amulvet"
SCHEMES_LEG = "amulschemes"


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
    url = f"{settings.amul_network_url.rstrip('/')}/search"
    payload = {"query": query, "legs": [leg]}
    if user_id:
        payload["user_id"] = user_id
    async with httpx.AsyncClient(timeout=settings.amul_network_timeout_s) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
    return (data.get("results") or {}).get(leg)


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
    """Union-scheme discovery via the network (schemes:amul-union)."""
    on_search = await _seeker_search(query or "schemes", SCHEMES_LEG)
    items = _items(on_search)
    if union:
        u = union.strip().lower()
        items = [it for it in items if _tag_map(it).get("union", "").lower() == u] or items
    if not items:
        return "No union scheme data was found on the network for this query."
    out: list[dict] = []
    for it in items:
        d = it.get("descriptor", {})
        tags = _tag_map(it)
        out.append({
            "scheme_title": d.get("name"),
            "description": d.get("long_desc") or d.get("short_desc"),
            "union": tags.get("union"),
            "category": tags.get("category"),
            "source": tags.get("source"),
        })
    import json
    return json.dumps(out, ensure_ascii=False, indent=2)


async def network_create_ai_call(
    union_code: str,
    society_code: str,
    farmer_code: str,
    user_id: str,
    species: str,
) -> str:
    """AI-call booking via the network (services:amul-vet-booking) — POSTs a
    Beckn confirm to the booking BPP, which calls PashuGPT CreateAICall."""
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
        return f"Artificial insemination call booking failed on the network: {err.get('message', 'unknown error')}"
    ticket = body.get("message", {}).get("order", {}).get("id")
    return f"Artificial insemination call booked successfully via the Beckn network. Ticket: {ticket}"
