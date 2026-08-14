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
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import httpx

from agents.tools.scheme_codes import (
    SCHEME_CODES,
    SCHEME_LABELS,
    SchemeCode,
    resolve_scheme_code,
    scheme_names_sentence,
)
from app.config import settings
from helpers.utils import get_logger

logger = get_logger(__name__)

VISTAAR_BAP_URL = settings.vistaar_bap_url.rstrip("/")
# Timeout: ONE knob, shared with the rest of the Beckn network tools
# (AMUL_NETWORK_TIMEOUT_S, default 35s). There used to be a second, larger
# VISTAAR_TIMEOUT_S (40s) governing exactly these calls, and a real 40.11s stall
# was observed against the 60s nginx window that has previously dropped voice
# calls. The larger knob is gone rather than merely lowered: two timeouts for
# one hop only ever meant the tighter budget was not the one in force.
# Default location (Anand, Gujarat — Amul region) when the caller has no coords.
DEFAULT_LAT = settings.vistaar_default_lat
DEFAULT_LON = settings.vistaar_default_lon

# Route BV searches through Amul's Beckn seeker (canonical N-N: the seeker signs
# as bap.amul-net.internal -> gateway -> BH BPP, then returns the on_search it
# gets back). Set VISTAAR_SEEKER_URL="" to fall back to calling the BV sandbox
# BAP directly (VISTAAR_BAP_URL, sync inline).
VISTAAR_SEEKER_URL = settings.vistaar_seeker_url.rstrip("/")
VISTAAR_LEG = settings.vistaar_leg

# Mandi price-discovery is a *range* API: it answers with one row per arrival
# date inside [from_date, to_date]. Send no range and it falls back to a single
# row from whichever market it considers nearest, which is how "no prices found"
# and "only today, wrong market" happened. Tags are FLAT ({"code","value"}), not
# Beckn tag groups — the BPP silently ignores the group form. Windows wider than
# MANDI_MAX_RANGE_DAYS are rejected upstream, so we clamp.
MANDI_MAX_RANGE_DAYS = 30
_IST = timezone(timedelta(hours=5, minutes=30))
_DATE_FMT = "%d-%m-%Y"

# BV's get_scheme_info codes and the farmer-phrasing alias map live in
# agents/tools/scheme_codes.py — ONE copy, shared with beckn_network.py.
__all__ = [
    "SCHEME_CODES",
    "VistaarLegUnavailable",
    "get_vistaar_weather",
    "get_vistaar_mandi_prices",
    "get_vistaar_scheme_info",
]


class VistaarLegUnavailable(RuntimeError):
    """The seeker reported this leg as failed (or returned no on_search at all).

    Distinct from "the catalogue is empty": an empty catalogue is an answer, a
    failed leg is not. Conflating the two is what let the `moa` leg's timeout
    flap render as a confident "No mandi prices were found…" — infrastructure
    failure presented to the farmer as fact, byte-identical to a real miss.
    """


def _context() -> dict[str, Any]:
    return {
        "domain": "schemes:vistaar",
        "action": "search",
        "version": "1.1.0",
        "bap_id": settings.vistaar_bap_id,
        "bap_uri": settings.vistaar_bap_uri,
        "bpp_id": settings.vistaar_bpp_id,
        "bpp_uri": settings.vistaar_bpp_uri,
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
    """Run one BV intent and return its items.

    Raises VistaarLegUnavailable when the seeker reports the leg as failed, or
    hands back no on_search body for it. Previously this read only
    `results.<leg>` and dropped `errors` on the floor, so a dead leg returned
    `[]` and every caller announced "none found".
    """
    async with httpx.AsyncClient(timeout=settings.amul_network_timeout_s) as client:
        if VISTAAR_SEEKER_URL:
            # Canonical Beckn path: hand the raw intent to the seeker, which
            # signs + routes it to BH and returns the on_search it gets back
            # under results.<leg>.
            r = await client.post(
                f"{VISTAAR_SEEKER_URL}/search",
                json={"intent": intent, "legs": [VISTAAR_LEG]},
            )
            r.raise_for_status()
            body = r.json()
            leg_error = (body.get("errors") or {}).get(VISTAAR_LEG)
            on_search = (body.get("results") or {}).get(VISTAAR_LEG)
            if leg_error or not isinstance(on_search, dict):
                logger.warning(
                    "vistaar leg unavailable leg=%s error=%s on_search_type=%s",
                    VISTAAR_LEG, leg_error, type(on_search).__name__,
                )
                raise VistaarLegUnavailable(str(leg_error or "no on_search returned"))
            return _items(on_search)
        # Fallback: call the BV sandbox BAP directly (sync inline on_search).
        r = await client.post(
            f"{VISTAAR_BAP_URL}/search",
            json={"context": _context(), "message": {"intent": intent}},
        )
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


def _format_items(items: list[dict], max_items: int = settings.vistaar_max_items) -> str:
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


def _parse_date(value: Optional[str], warn: bool = True) -> Optional[date]:
    """Parse a DD-MM-YYYY (or YYYY-MM-DD) date; None if absent/unparseable.

    `warn=False` for per-row BPP data: a catalog of odd arrival dates would emit
    one warning per row, and the row-level fallback ("date n/a") is already
    graceful. The caller-supplied window keeps the warning — that one is worth
    knowing about and happens at most twice per call.
    """
    text = (value or "").strip()
    if not text:
        return None
    for fmt in (_DATE_FMT, "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    (logger.warning if warn else logger.debug)(
        "vistaar mandi: unparseable date %r, ignoring", value
    )
    return None


def _resolve_date_range(
    price_date: Optional[str] = None, price_date_to: Optional[str] = None
) -> tuple[str, str]:
    """Resolve the from_date/to_date window (DD-MM-YYYY) for the mandi intent.

    `to_date` is never null — it defaults to today (IST) even when a single date
    was asked for, so the BPP still has newer arrivals to fall back to. With no
    date at all we look back the full window, which is what makes "what is the
    price of X" return a real local market instead of one stale row. Reversed
    ranges are swapped, a future end is capped at today, and the window is
    clamped to MANDI_MAX_RANGE_DAYS because the API rejects wider ones.
    """
    today = datetime.now(_IST).date()
    start = _parse_date(price_date)
    end = _parse_date(price_date_to)

    if start and end and start > end:
        start, end = end, start

    to_date = min(end, today) if end else today
    from_date = min(start, to_date) if start else to_date - timedelta(days=MANDI_MAX_RANGE_DAYS)
    from_date = max(from_date, to_date - timedelta(days=MANDI_MAX_RANGE_DAYS))
    return from_date.strftime(_DATE_FMT), to_date.strftime(_DATE_FMT)


def _tag_values(item: dict) -> dict[str, str]:
    """Flatten an item's tag groups into {code: value}."""
    out: dict[str, str] = {}
    for tag in item.get("tags", []) or []:
        for li in tag.get("list", []) or []:
            code = (li.get("descriptor", {}) or {}).get("code")
            if code:
                out[code] = li.get("value", "")
    return out


def _format_mandi_items(items: list[dict], max_rows: int = 40) -> str:
    """One line per arrival date, newest first — a 30-day window is ~25 rows, and
    the generic _format_items would both truncate at 20 and bury the series in
    repeated tag dumps."""
    rows = []
    for it in items:
        t = _tag_values(it)
        arrival = t.get("Arrival Date", "")
        parsed = _parse_date(arrival, warn=False)
        # An unparseable arrival date sorts last (date.min) AND renders as
        # "date n/a" — echoing the raw junk back at the farmer is worse than
        # admitting we don't know the date.
        rows.append((parsed or date.min, arrival if parsed else "", t))
    rows.sort(key=lambda r: r[0], reverse=True)

    lines = []
    for _, arrival, t in rows[:max_rows]:
        market = t.get("Market") or t.get("District") or "market n/a"
        unit = t.get("Price Unit", "Rs./Qtl")
        modal, lo, hi = t.get("Modal Price"), t.get("Min Price"), t.get("Max Price")
        price = f"modal {modal}" if modal else "modal n/a"
        if lo and hi:
            price += f" (min {lo} – max {hi})"
        extra = ", ".join(v for v in (t.get("Variety"), t.get("Grade")) if v and v != "----")
        lines.append(f"- {arrival or 'date n/a'} — {market}: {price} {unit}"
                     + (f" [{extra}]" if extra else ""))

    if len(rows) > max_rows:
        lines.append(f"- …{len(rows) - max_rows} older rows omitted.")
    return "\n".join(lines)


async def get_vistaar_weather() -> str:
    """Get the current weather forecast for the Amul region (Anand, Gujarat) from
    Bharat Vistaar (Mausamgram). Call this directly for any weather question — no
    location argument is needed; the location is fixed to Anand.

    Returns a day-wise forecast (rainfall, min/max temp, humidity, wind, etc.).
    """
    lat = DEFAULT_LAT
    lon = DEFAULT_LON
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
    price_date: Optional[str] = None,
    price_date_to: Optional[str] = None,
) -> str:
    """Get mandi (market) prices for a commodity near the Amul region (Anand,
    Gujarat) from Bharat Vistaar. Call this directly with just the commodity —
    do NOT ask the user for a market or city; the location is fixed to Anand.

    Returns one row per arrival date, so it answers history questions ("prices
    for the last 10 days") as well as "what is the price today". Markets do not
    trade every commodity every day, so gaps between dates are normal.

    Args:
        commodity_name: the commodity to price, e.g. "Tomato", "Onion", "Wheat".
            Use the English Agmarknet name; invented variants ("Onion Big")
            return nothing.
        price_date: optional START of the date window, DD-MM-YYYY. Omit for the
            latest available prices.
        price_date_to: optional END of the date window, DD-MM-YYYY. Pass BOTH
            ends for a range ("last 10 days"); never send only one end. Windows
            wider than 30 days are trimmed to the most recent 30.
    Returns market prices per date (min / max / modal, market, arrival date).
    """
    lat = DEFAULT_LAT
    lon = DEFAULT_LON
    from_date, to_date = _resolve_date_range(price_date, price_date_to)
    intent = {
        "category": {"descriptor": {"code": "price-discovery"}},
        "item": {"descriptor": {"name": commodity_name}},
        "fulfillment": {"end": {"location": {"descriptor": {"name": "Anand"}, "gps": f"{lat},{lon}"}}},
        # Flat tags — the BPP ignores the Beckn tag-group form.
        "tags": [
            {"code": "from_date", "value": from_date},
            {"code": "to_date", "value": to_date},
        ],
    }
    try:
        items = await _vistaar_search(intent)
    except VistaarLegUnavailable:
        # A failed leg is NOT an empty market. Never fall through to the
        # "No mandi prices were found…" line below on infrastructure failure.
        logger.warning("vistaar mandi leg unavailable commodity=%s", commodity_name)
        return "Mandi prices are temporarily unavailable from Bharat Vistaar. Please try again shortly."
    except Exception:
        logger.exception(
            "vistaar mandi failed commodity=%s from=%s to=%s", commodity_name, from_date, to_date
        )
        return "Mandi prices are temporarily unavailable from Bharat Vistaar."
    logger.info(
        "vistaar mandi commodity=%s from=%s to=%s items=%d",
        commodity_name, from_date, to_date, len(items),
    )
    if not items:
        return (
            f"No mandi prices were found for '{commodity_name}' near Anand "
            f"between {from_date} and {to_date}."
        )
    return f"Mandi prices for {commodity_name} near Anand ({from_date} to {to_date}):\n" + _format_mandi_items(items)


async def get_vistaar_scheme_info(scheme_code: SchemeCode) -> str:
    """Get information about a CENTRAL / national government agriculture scheme
    (e.g. Kisan Credit Card, PM-KISAN, crop insurance, Soil Health Card) from
    Bharat Vistaar. Use this for ANY central government scheme a farmer asks
    about. For the farmer's Amul dairy-union welfare schemes, use
    get_union_scheme_data instead.

    Args:
        scheme_code: which central scheme to look up. Pick the closest match:
            kcc = Kisan Credit Card; pmkisan = PM-KISAN income support;
            pmfby = crop insurance; shc = Soil Health Card;
            pmksy = irrigation; sathi = seed authentication;
            pmasha = price support / MSP; aif = agriculture infrastructure fund;
            smam = farm mechanisation; pdmc = micro / drip irrigation;
            pkvy = organic farming; nfsm = food security mission;
            rad = rainfed area development; ffs = fertilizer sales;
            nbhm = beekeeping and honey.
    Returns the scheme's eligibility, benefits, and application details.
    """
    # The signature is a Literal, so pydantic-ai publishes a JSON-schema enum
    # and the model can no longer invent a code (verified: the generated schema
    # carries `"enum": [...]` under docstring_format='auto'). The alias map is
    # still applied because this function is also called with free farmer
    # phrasing from the merged discovery path.
    code = resolve_scheme_code(scheme_code)
    if code is None:
        raw = (scheme_code or "").strip().casefold()
        code = raw if raw in SCHEME_CODES else None
    if code is None:
        # NEVER echo the internal code list at the farmer — agrinet_system.md
        # ("Do not mention internal tool mechanics"). Name schemes, not codes.
        logger.info("vistaar scheme: unresolvable scheme_code=%r", scheme_code)
        return (
            "I could not match that to a central government scheme. I can look up "
            f"schemes such as {scheme_names_sentence()}."
        )
    intent = {
        "category": {"descriptor": {"code": "schemes-agri"}},
        "item": {"descriptor": {"name": code}},
    }
    try:
        items = await _vistaar_search(intent)
    except VistaarLegUnavailable:
        logger.warning("vistaar scheme leg unavailable code=%s", code)
        return "Scheme information is temporarily unavailable from Bharat Vistaar. Please try again shortly."
    except Exception:
        logger.exception("vistaar scheme failed code=%s", code)
        return "Scheme information is temporarily unavailable from Bharat Vistaar."
    if not items:
        return f"No information was found for the {SCHEME_LABELS.get(code, code)} scheme."
    return _format_items(items)
