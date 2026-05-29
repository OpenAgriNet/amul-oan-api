"""Beckn "network of networks" government-scheme discovery tool (demo stack).

Calls the Amul BAP (the netofnet-beckn-poc orchestrator) which fans a single
query out across two Beckn networks in parallel — the real Bharat Vistaar Dev
schemes network (Government of India sandbox, "MOA" leg) and a Maharashtra
network ("MH" leg) — and returns the aggregated catalogs.

Discovery only: this lists schemes/services. It does NOT order, apply, or move
any money. Gated behind ``settings.beckn_enabled`` and only registered in the
demo image (see ``agents/tools/__init__.py``).
"""

import json

import httpx

from app.config import settings
from app.core.cache import (
    build_api_cache_key,
    get_cached_api_response,
    set_cached_api_response,
)
from helpers.utils import get_logger

logger = get_logger(__name__)

# The BAP holds the request up to ~30s server-side while it awaits the async
# on_search callbacks, so the client timeout sits just above that.
_BECKN_TIMEOUT_S = 35.0

# The Vistaar Dev sandbox catalog matches on near-exact keys (e.g. "KCC",
# "PMKISAN") and returns nothing for natural phrasings ("Kisan Credit Card",
# "PM-KISAN", "credit"). The agent phrases queries freely, so we translate common
# phrasings to the keys Vistaar actually matches and try those first.
_SCHEME_ALIASES = {
    "KCC": ("kisan credit card", "kcc", "kisan credit", "credit card", "farm credit", "crop loan"),
    "PMKISAN": ("pm-kisan", "pmkisan", "pm kisan", "samman nidhi", "kisan samman", "income support"),
    "PMFBY": ("pmfby", "fasal bima", "crop insurance", "fasal", "pradhan mantri fasal"),
    "PMKSY": ("pmksy", "sinchayee", "irrigation", "krishi sinchayee", "drip", "micro irrigation"),
    "PKVY": ("pkvy", "paramparagat", "organic farming", "organic"),
}


def _candidate_queries(query: str) -> list[str]:
    """Map a free-text scheme query to the keys Vistaar matches, then the raw query."""
    raw = (query or "").strip()
    low = raw.casefold()
    candidates: list[str] = []
    for key, aliases in _SCHEME_ALIASES.items():
        if key.casefold() in low or any(alias in low for alias in aliases):
            candidates.append(key)
    if raw and raw not in candidates:
        candidates.append(raw)
    return candidates or ([raw] if raw else [])


async def _call_bap(url: str, query: str) -> dict | None:
    """One best-effort POST to the BAP. Returns the parsed payload or None on failure."""
    try:
        async with httpx.AsyncClient(timeout=_BECKN_TIMEOUT_S) as client:
            response = await client.post(
                url,
                headers={"Content-Type": "application/json"},
                json={"query": query},
            )
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, dict) else None
    except httpx.HTTPStatusError as e:
        logger.error("[Beckn] search failed status=%s query=%s", e.response.status_code, query)
    except httpx.TimeoutException:
        logger.error("[Beckn] search timed out query=%s", query)
    except Exception as e:  # noqa: BLE001 - degrade to cache, surface clean text
        logger.error("[Beckn] search error query=%s: %s", query, str(e))
    return None


def _extract_items(leg: dict | None) -> list[dict]:
    """Flatten one leg's Beckn on_search catalog into a list of scheme items.

    Normalizes the two catalog shapes we see in practice:
      - Vistaar uses ``message.catalog.providers[]``
      - the MH mock uses the Beckn 1.x slash convention
        ``message.catalog["bpp/providers"][]``
    """
    if not isinstance(leg, dict):
        return []
    catalog = (leg.get("message") or {}).get("catalog") or {}
    providers = catalog.get("providers") or catalog.get("bpp/providers") or []
    items: list[dict] = []
    for provider in providers:
        provider = provider or {}
        provider_name = (provider.get("descriptor") or {}).get("name") or ""
        for item in provider.get("items") or []:
            item = item or {}
            descriptor = item.get("descriptor") or {}
            items.append(
                {
                    "provider": provider_name,
                    "name": descriptor.get("name") or "",
                    "description": descriptor.get("short_desc")
                    or descriptor.get("long_desc")
                    or "",
                    "id": item.get("id") or "",
                }
            )
    return items


async def search_government_schemes(query: str) -> str:
    """Discover Indian government agriculture schemes, subsidies and benefits.

    Use this when the farmer asks about **government** schemes, subsidies,
    benefits, agricultural credit (e.g. Kisan Credit Card / KCC), PM-KISAN, crop
    insurance, or eligibility for central/state agri programmes. Results are
    discovered live from the Bharat Vistaar (Government of India) Beckn network
    and a Maharashtra network. This is discovery only — it lists schemes; it does
    not apply for them or move any money. Distinct from `get_union_scheme_data`,
    which covers the farmer's Amul milk-union schemes.

    Args:
        query: The scheme / subsidy / benefit the farmer is asking about, written
            in English (e.g. "Kisan Credit Card", "crop insurance",
            "dairy subsidy").

    Returns:
        A JSON-formatted string of discovered schemes grouped by source network,
        or a clear message when the networks return nothing / are unavailable.
    """
    if not settings.beckn_enabled or not settings.amul_bap_url:
        return "Government scheme discovery is not enabled in this environment."

    cache_key = build_api_cache_key("beckn_schemes", (query or "").strip().casefold())
    url = settings.amul_bap_url.rstrip("/") + "/search"

    # --- live call: try mapped scheme keys then the raw query, take first with items ---
    for cand in _candidate_queries(query):
        payload = await _call_bap(url, cand)
        if not isinstance(payload, dict):
            continue
        errors = payload.get("errors") or {}
        vistaar_items = _extract_items(payload.get("moa"))
        mh_items = _extract_items(payload.get("mh"))
        if vistaar_items or mh_items:
            result = {
                "query_matched": cand,
                "vistaar_goi_schemes": vistaar_items,
                "maharashtra_network": mh_items,
                "unavailable_networks": {
                    "vistaar_goi": errors.get("moa"),
                    "maharashtra": errors.get("mh"),
                },
            }
            logger.info(
                "[Beckn] live query=%s matched=%s vistaar=%s mh=%s errors=%s",
                query, cand, len(vistaar_items), len(mh_items), errors,
            )
            # Last-known-good for the next time the BAP / Vistaar is flaky.
            await set_cached_api_response(cache_key, result)
            return json.dumps(result, indent=2, ensure_ascii=False)
    logger.warning("[Beckn] no items for any candidate of query=%s; trying cache", query)

    # --- fallback: serve last-known-good so a flaky BAP doesn't break the demo ---
    hit, cached = await get_cached_api_response(cache_key)
    if hit and isinstance(cached, dict):
        logger.info("[Beckn] served from cache query=%s", query)
        return json.dumps(
            {**cached, "note": "served from cache (live network unavailable)"},
            indent=2,
            ensure_ascii=False,
        )

    return "Government scheme discovery is temporarily unavailable. Please try again."
