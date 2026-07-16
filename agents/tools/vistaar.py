"""
Bharat Vistaar (BV) discovery tools — weather, mandi prices, and scheme info.

These call Bharat Vistaar's Beckn BAP directly (the sandbox today) with the
right intent per use case, all on domain `schemes:vistaar`, differentiated by
category:
  - schemes  -> category `schemes-agri`, item.descriptor.name = <scheme_code>
  - weather  -> category `Weather-Forecast-Mausamgram` / code `WFC`
  - mandi    -> category `price-discovery`, item.descriptor.name = <commodity>

The BAP runs in sync mode, so `/search` returns the on_search catalog inline.
Endpoint is overridable via VISTAAR_BAP_URL (default: the Vistaar sandbox).
Advisory (ICAR/NPSS) is NOT here — on BV that's document search, not Beckn.
"""
import os
import uuid
from typing import Any, Optional

import httpx

from helpers.utils import get_logger

logger = get_logger(__name__)

VISTAAR_BAP_URL = os.getenv(
    "VISTAAR_BAP_URL", "https://bap-client-playground-sandbox-vistaar.da.gov.in"
).rstrip("/")
VISTAAR_TIMEOUT_S = float(os.getenv("VISTAAR_TIMEOUT_S", "40"))
# Default location (Anand, Gujarat — Amul region) when the caller has no coords.
DEFAULT_LAT = float(os.getenv("VISTAAR_DEFAULT_LAT", "22.55"))
DEFAULT_LON = float(os.getenv("VISTAAR_DEFAULT_LON", "72.93"))

# BV's get_scheme_info codes (domain schemes:vistaar, category schemes-agri).
SCHEME_CODES = {
    "kcc", "pmkisan", "pmfby", "shc", "pmksy", "sathi", "pmasha", "aif",
    "smam", "pdmc", "pkvy", "nfsm", "rad", "ffs", "nbhm",
}


def _context() -> dict[str, Any]:
    return {
        "domain": "schemes:vistaar",
        "action": "search",
        "version": "1.1.0",
        "bap_id": os.getenv("VISTAAR_BAP_ID", "amul-dev"),
        "bap_uri": os.getenv("VISTAAR_BAP_URI", "https://bap-network-playground-sandbox-vistaar.da.gov.in"),
        "bpp_id": os.getenv("VISTAAR_BPP_ID", "bpp-network-playground-sandbox-vistaar.da.gov.in"),
        "bpp_uri": os.getenv("VISTAAR_BPP_URI", "https://bpp-network-playground-sandbox-vistaar.da.gov.in"),
        "transaction_id": str(uuid.uuid4()),
        "message_id": str(uuid.uuid4()),
        "timestamp": "1970-01-01T00:00:00.000Z",
        "ttl": "PT10M",
        "location": {"country": {"code": "IND"}, "city": {"code": "*"}},
    }


def _items(body: dict) -> list[dict]:
    """Pull items[] out of the (sync) on_search — handles the `responses[]`
    wrapper and a bare `message` body."""
    responses = body.get("responses")
    catalogs = (
        [r.get("message", {}).get("catalog", {}) for r in responses]
        if isinstance(responses, list)
        else [body.get("message", {}).get("catalog", {})]
    )
    out: list[dict] = []
    for cat in catalogs:
        for prov in cat.get("providers", []) or []:
            out.extend(prov.get("items", []) or [])
    return out


async def _vistaar_search(intent: dict) -> list[dict]:
    payload = {"context": _context(), "message": {"intent": intent}}
    async with httpx.AsyncClient(timeout=VISTAAR_TIMEOUT_S) as client:
        r = await client.post(f"{VISTAAR_BAP_URL}/search", json=payload)
        r.raise_for_status()
        return _items(r.json())


def _fmt_tag_group(tag: dict) -> str:
    header = (tag.get("descriptor", {}) or {}).get("code") or (tag.get("descriptor", {}) or {}).get("name") or ""
    rows = []
    for li in tag.get("list", []) or []:
        k = (li.get("descriptor", {}) or {}).get("code") or (li.get("descriptor", {}) or {}).get("name") or ""
        v = li.get("value", "")
        if k or v:
            rows.append(f"  - {k}: {v}")
    body = "\n".join(rows)
    return (f"**{header}**\n{body}" if header else body).strip()


def _format_items(items: list[dict], max_items: int = 20) -> str:
    blocks = []
    for it in items[:max_items]:
        d = it.get("descriptor", {}) or {}
        parts = []
        name = d.get("name")
        if name:
            parts.append(f"### {name}")
        desc = d.get("long_desc") or d.get("short_desc")
        if desc:
            parts.append(desc)
        for tag in it.get("tags", []) or []:
            g = _fmt_tag_group(tag)
            if g:
                parts.append(g)
        block = "\n".join(parts).strip()
        if block:
            blocks.append(block)
    return "\n\n".join(blocks)


async def get_vistaar_weather(latitude: float | None = None, longitude: float | None = None) -> str:
    """Get the weather forecast for a location from Bharat Vistaar (Mausamgram).

    Args:
        latitude: location latitude. Defaults to the Amul region if omitted.
        longitude: location longitude. Defaults to the Amul region if omitted.
    Returns a day-wise forecast (rainfall, min/max temp, humidity, etc.).
    """
    lat = latitude if latitude is not None else DEFAULT_LAT
    lon = longitude if longitude is not None else DEFAULT_LON
    intent = {
        "category": {"descriptor": {"name": "Weather-Forecast-Mausamgram", "code": "WFC"}},
        "fulfillment": {"stops": [{"location": {"lat": lat, "lon": lon}}]},
    }
    try:
        items = await _vistaar_search(intent)
    except Exception:
        logger.exception("vistaar weather failed lat=%s lon=%s", lat, lon)
        return "Weather is temporarily unavailable from Bharat Vistaar."
    if not items:
        return "No weather forecast was returned for this location."
    return _format_items(items)


async def get_vistaar_mandi_prices(
    commodity_name: str,
    latitude: float | None = None,
    longitude: float | None = None,
    location_name: str = "",
) -> str:
    """Get live mandi (market) prices for a commodity from Bharat Vistaar.

    Args:
        commodity_name: e.g. "Onion", "Wheat", "Cotton".
        latitude: location latitude. Defaults to the Amul region if omitted.
        longitude: location longitude. Defaults to the Amul region if omitted.
        location_name: optional human-readable location.
    Returns nearby market prices (min / max / modal, market, arrival date).
    """
    lat = latitude if latitude is not None else DEFAULT_LAT
    lon = longitude if longitude is not None else DEFAULT_LON
    intent = {
        "category": {"descriptor": {"code": "price-discovery"}},
        "item": {"descriptor": {"name": commodity_name}},
        "fulfillment": {"end": {"location": {"descriptor": {"name": location_name}, "gps": f"{lat},{lon}"}}},
    }
    try:
        items = await _vistaar_search(intent)
    except Exception:
        logger.exception("vistaar mandi failed commodity=%s", commodity_name)
        return "Mandi prices are temporarily unavailable from Bharat Vistaar."
    if not items:
        return f"No mandi prices were found for '{commodity_name}' near this location."
    return _format_items(items)


async def get_vistaar_scheme_info(scheme_code: str) -> str:
    """Get information about a central agriculture scheme from Bharat Vistaar.

    Args:
        scheme_code: one of kcc, pmkisan, pmfby, shc, pmksy, sathi, pmasha, aif,
            smam, pdmc, pkvy, nfsm, rad, ffs, nbhm.
    Returns the scheme's eligibility, benefits, and application details.
    """
    code = (scheme_code or "").strip().lower()
    if code not in SCHEME_CODES:
        return f"Unknown scheme code '{scheme_code}'. Valid codes: {', '.join(sorted(SCHEME_CODES))}."
    intent = {
        "category": {"descriptor": {"code": "schemes-agri"}},
        "item": {"descriptor": {"name": code}},
    }
    try:
        items = await _vistaar_search(intent)
    except Exception:
        logger.exception("vistaar scheme failed code=%s", code)
        return "Scheme information is temporarily unavailable from Bharat Vistaar."
    if not items:
        return f"No information was found for scheme '{scheme_code}'."
    return _format_items(items)
