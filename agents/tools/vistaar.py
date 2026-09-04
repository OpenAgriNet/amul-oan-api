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
import re
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Optional

import httpx
from pydantic_ai import RunContext

from agents.deps import FarmerContext
from agents.tools.districts import (
    DEFAULT_LOCATION,
    DISTRICTS,
    Candidate,
    DistrictLocation,
    resolve_place,
    unknown_place_message,
)
from agents.tools.session_location import (
    get_session_district_key,
    set_session_district_key,
)
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
# Default location (Anand, Gujarat — Amul region) when nothing else resolves.
# This is the LAST resort now, not the only option: see _resolve_search_location.
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
# DEFAULT window when the farmer names no dates. The BPP caps every response at
# **10 rows**, ordered by arrival date descending across all in-range markets, so
# 1 day → 2 rows, 3 → 6, 7 → 10, 15 → 10, 30 → 10: past ~7 days a wider window
# returns byte-identical data. Asking for 30 was paying upstream for nothing.
# ⚠️ This is the DEFAULT only. An explicit price_date/price_date_to is still
# honoured out to MANDI_MAX_RANGE_DAYS — clamping those to 7 would silently trim
# a farmer's "last 20 days" and regress the shipped date-range feature.
MANDI_DEFAULT_RANGE_DAYS = 7
_IST = timezone(timedelta(hours=5, minutes=30))
_DATE_FMT = "%d-%m-%Y"

# How many district candidates to try before giving up. Walked STRICTLY IN
# SEQUENCE and only on zero rows: each attempt costs ~2.2 s, and the upstream is
# a single non-redundant sandbox, so fanning out in parallel would double its
# load on every happy path to save time only on the rare failure path.
MANDI_MAX_CANDIDATES = int(os.getenv("MANDI_MAX_CANDIDATES", "3"))

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
        body = r.json()
        # The sync BAP client wraps each received on_search in `responses`.
        # An empty wrapper means nobody replied before its internal timeout; it
        # is infrastructure failure, not a genuine empty catalogue. A present
        # response whose catalog has zero providers/items remains a valid miss.
        if body.get("responses") == []:
            logger.warning("vistaar direct BAP returned no on_search responses")
            raise VistaarLegUnavailable("no on_search returned")
        return _items(body)


# ── Location resolution ──────────────────────────────────────────────────────

# Yard markers that mean the farmer named a *specific market*, not just a
# district/town. Checked BEFORE `resolve_place` / `normalize_place`. Those strip
# "apmc"/"mandi" for GPS lookup and would otherwise erase the distinction
# between "Anand" and "Anand APMC". "yard" is also treated as yard intent here
# even though the district normaliser does not strip it.
_YARD_WORD = re.compile(r"(?i)\b(apmc|mandi|yard)\b")
_YARD_SUFFIXES = ("apmc", "mandi", "yard")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _extract_requested_market_name(location: Optional[str]) -> Optional[str]:
    """Return the farmer's explicit yard phrase, or None for a plain place name.

    "Anand APMC" / "Nadiad mandi" → the cleaned phrase (intent to pin a yard).
    "Anand" / "Junagadh" → None (district/town search; nearby markets OK).
    Must run on the raw argument: normalization strips these suffixes for the
    district table lookup and would make every APMC ask look like a town ask.
    """
    asked = " ".join((location or "").split())
    if not asked:
        return None
    squeezed = _NON_ALNUM.sub("", asked.casefold())
    has_marker = bool(_YARD_WORD.search(asked)) or any(
        squeezed.endswith(suffix) and len(squeezed) > len(suffix)
        for suffix in _YARD_SUFFIXES
    )
    if not has_marker:
        return None
    return asked


@dataclass(frozen=True)
class SearchLocation:
    """Where we are about to search, and why.

    `source` is not decoration — it decides whether the farmer is told the
    location was assumed. Telling someone in Bhuj that Anand's onion price is
    "your local mandi" is the failure this field exists to prevent.

    `requested_market_name` is set only when the farmer named a yard (e.g.
    "Anand APMC"). GPS still resolves via the district table; this field is the
    preserved yard intent for later row filtering. None for district/town asks,
    session reuse, profile district, and the Anand default.
    """

    location: DistrictLocation
    source: str  # "explicit" | "session" | "farmer" | "default"
    requested_market_name: Optional[str] = None

    @property
    def assumed(self) -> bool:
        return self.source == "default"

    @property
    def explicit_yard(self) -> bool:
        return self.requested_market_name is not None


def _deps(ctx: Optional[RunContext[FarmerContext]]) -> Optional[FarmerContext]:
    """Tolerate a missing/`None` ctx so these stay callable outside an agent run."""
    return getattr(ctx, "deps", None)


async def _resolve_search_location(
    ctx: Optional[RunContext[FarmerContext]], location: Optional[str]
) -> tuple[Optional[SearchLocation], Optional[str]]:
    """Resolve where to search, most specific first.

    Order: explicit argument → sticky session override → the farmer's own
    district → the Anand default. Returns `(resolved, refusal)`; exactly one is
    non-None.

    Two rules that look like details and are not:

    * **An explicit place must resolve, or we refuse.** If the farmer names a
      place we do not cover we say so and name what we do — we never quietly
      answer with their profile district instead. Someone who asked about
      Junagadh must not be handed Anand's prices without being told.
    * **The model never supplies coordinates.** The argument is a place *name*,
      resolved here against the static table. A hallucinated lat/lon fails
      silently as zero rows — indistinguishable from a genuinely empty market —
      so there is no parameter through which one can arrive.

    Explicit yard phrases ("Anand APMC") still resolve GPS via the district
    table (suffixes stripped), but `requested_market_name` keeps the yard
    phrase for mandi row matching. Session stickiness stores only the district
    key, so a follow-up without a new location does not keep yard intent.
    """
    deps = _deps(ctx)
    session_id = getattr(deps, "session_id", None)

    asked = (location or "").strip()
    if asked:
        # Capture yard intent before resolve_place strips apmc/mandi suffixes.
        requested_market = _extract_requested_market_name(asked)
        resolved = resolve_place(asked)
        if resolved is None:
            logger.info("vistaar location unresolved asked=%r", asked)
            return None, unknown_place_message(asked)
        # Sticky: the next turn is usually about the same place ("and cotton?").
        await set_session_district_key(session_id, resolved.key)
        return SearchLocation(resolved, "explicit", requested_market), None

    session_key = await get_session_district_key(session_id)
    if session_key:
        return SearchLocation(DISTRICTS[session_key], "session"), None

    district = deps.get_farmer_district() if deps is not None else None
    if district:
        resolved = resolve_place(district)
        if resolved is not None:
            return SearchLocation(resolved, "farmer"), None
        # A district string we cannot map is a table gap worth seeing in logs —
        # it is the silent-fallthrough failure mode the normaliser exists for.
        logger.warning("vistaar: unmapped farmer district=%r, using default", district)

    return SearchLocation(DEFAULT_LOCATION, "default"), None


def _location_phrase(where: SearchLocation, candidate: Candidate) -> str:
    """"Deesa, Banaskantha" — or just "Anand" where town and district agree."""
    if candidate.town.casefold() == where.location.display.casefold():
        return candidate.town
    return f"{candidate.town}, {where.location.display}"


def _assumed_location_note(where: SearchLocation) -> str:
    """Answer-then-invite, for a farmer whose profile carries no district.

    Deliberately NOT a blocking clarification question. A farmer who asked for
    onion prices should get onion prices; on voice an extra round-trip is
    expensive and a stalled tool call has previously run past the 60 s nginx
    window and dropped the call outright. So: answer, then say plainly whose
    market this is and invite a correction — once, in one sentence.
    """
    return (
        f"\n\nThese are {where.location.display}-area prices, used because I do not "
        "have your district on file. Tell me your district and I will use your "
        "local market next time."
    )


async def _search_candidates(
    build_intent: Callable[[Candidate], dict], where: SearchLocation
) -> tuple[list[dict], Candidate]:
    """Try each candidate in order until one returns rows.

    Zero rows can legitimately mean "that commodity, that market, that day", so
    this is a fallback and not a retry: it exists because a ~50 km catchment does
    not cover a large district (Bhuj → Rapar is 106 km). A failed *leg* is not
    zero rows and is left to propagate — walking coordinates during an upstream
    outage would burn 3 × 2.2 s to reach the same "temporarily unavailable".
    """
    candidates = where.location.candidates[:MANDI_MAX_CANDIDATES] or (
        DEFAULT_LOCATION.primary,
    )
    tried = candidates[0]
    for candidate in candidates:
        tried = candidate
        items = await _vistaar_search(build_intent(candidate))
        if items:
            return items, candidate
        if candidate is not candidates[-1]:
            logger.info(
                "vistaar: zero rows at %s (%s), trying next candidate",
                candidate.town, where.location.display,
            )
    return [], tried


async def _search_candidates_for_yard(
    build_intent: Callable[[Candidate], dict],
    where: SearchLocation,
    requested_market_name: str,
) -> tuple[list[dict], list[dict], Candidate]:
    """Walk candidates until one returns rows matching the requested yard.

    Like _search_candidates but yard-aware: a candidate that returns rows from
    *nearby* markets but not the farmer's named yard is not "good enough" — the
    walk continues.  Nearby items are accumulated across all candidates so the
    miss message can list every market that *did* have data.

    Returns (matched_items, all_nearby_items, last_tried_candidate).
    """
    candidates = where.location.candidates[:MANDI_MAX_CANDIDATES] or (
        DEFAULT_LOCATION.primary,
    )
    all_nearby: list[dict] = []
    tried = candidates[0]
    for candidate in candidates:
        tried = candidate
        items = await _vistaar_search(build_intent(candidate))
        if items:
            matched, nearby = _partition_items_by_requested_market(
                items, requested_market_name
            )
            all_nearby.extend(nearby)
            if matched:
                return matched, all_nearby, candidate
        if candidate is not candidates[-1]:
            logger.info(
                "vistaar: yard %r not in %s (%s) rows, trying next candidate",
                requested_market_name, candidate.town, where.location.display,
            )
    return [], all_nearby, tried


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
    date at all we look back MANDI_DEFAULT_RANGE_DAYS, which is what makes "what
    is the price of X" return a real local market instead of one stale row.
    Reversed ranges are swapped, a future end is capped at today, and the window
    is clamped to MANDI_MAX_RANGE_DAYS because the API rejects wider ones.

    ⚠️ The 7-day DEFAULT and the 30-day CAP are different numbers on purpose. The
    default is 7 because the BPP's 10-row cap makes anything wider identical; the
    cap stays 30 so an explicit "last 20 days" is still answered in full.
    """
    today = datetime.now(_IST).date()
    start = _parse_date(price_date)
    end = _parse_date(price_date_to)

    if start and end and start > end:
        start, end = end, start

    to_date = min(end, today) if end else today
    from_date = min(start, to_date) if start else to_date - timedelta(days=MANDI_DEFAULT_RANGE_DAYS)
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


def _market_label(t: dict[str, str]) -> str:
    """"Jetpur APMC, Rajkot, Gujarat" — market, district AND state.

    All three, always, because the BPP does a ~50 km radius search and crossing
    a district or even a state line is NORMAL, not an error: Palanpur legitimately
    returns Abu Road APMC in Rajasthan, and Junagadh + Cotton returns Jetpur in
    Rajkot district. Printing only the market name is how a farmer ends up told
    that a market two districts away is "your local mandi".
    """
    parts = [t.get("Market") or t.get("District") or "market n/a"]
    for key in ("District", "State"):
        value = (t.get(key) or "").strip()
        if value and value.casefold() not in {p.casefold() for p in parts}:
            parts.append(value)
    return ", ".join(parts)


# BPP market names often carry a parenthetical qualifier after the town:
#   "Anand(Veg,Yard,Anand) APMC", "Khambhat(Veg Yard Khambhat) APMC".
_PAREN_RE = re.compile(r"\(.*?\)")
# Whole-word filler removed before squeezing so "APMC HALVAD" / "Deesa Veg Yard"
# collapse to the town, not "apmchalvad" / "deesaveg".
_MARKET_FILLER = re.compile(
    r"(?i)\b(apmc|mandi|yard|veg|vegetable|market)\b"
)
# Glued tokens still need prefix/suffix stripping after non-alnum collapse
# ("AnandAPMC", "APMCHALVAD").
_GLUED_YARD_TOKENS = _YARD_SUFFIXES + ("veg", "vegetable", "market")

# Same-yard spelling / transliteration variants only. NEVER map a district name
# onto a yard town (e.g. sabarkantha↛himatnagar, kheda↛nadiad) — that would
# reintroduce nearby-price substitution for explicit yard asks.
_MARKET_SPELLING_ALIASES: dict[str, str] = {
    "nadiyad": "nadiad",
    "bodeliu": "bodeli",
    "dhragradhra": "dhrangadhra",
    "khambalia": "khambhalia",
    "jamkhambalia": "khambhalia",
    "jamkhambhalia": "khambhalia",
    "sanad": "sanand",
    "vadhvan": "wadhwan",
    "vankaner": "wankaner",
}


def _market_match_key(text: str) -> str:
    """Collapse a market/yard phrase for equality checks.

    Handles both farmer phrasing and live BPP label quirks:
      "Anand APMC" / "AnandAPMC" / "Anand(Veg,Yard,Anand) APMC" → "anand"
      "APMC HALVAD" / "Halvad APMC" → "halvad"
      "Deesa Veg Yard" → "deesa"
      "Nadiyad(Piplag) APMC" → "nadiad" (via spelling alias)
    """
    no_paren = _PAREN_RE.sub(" ", text or "")
    cleaned = _MARKET_FILLER.sub(" ", no_paren)
    squeezed = _NON_ALNUM.sub("", cleaned.casefold())
    changed = True
    while changed and squeezed:
        changed = False
        for token in _GLUED_YARD_TOKENS:
            if squeezed.startswith(token) and len(squeezed) > len(token):
                squeezed = squeezed[len(token) :]
                changed = True
            if squeezed.endswith(token) and len(squeezed) > len(token):
                squeezed = squeezed[: -len(token)]
                changed = True
    return _MARKET_SPELLING_ALIASES.get(squeezed, squeezed)


def _markets_match(requested: str, market_tag: str) -> bool:
    """True when a BPP Market tag is the yard the farmer named.

    Compares canonical match keys after parenthetical / filler stripping and
    same-yard spelling aliases. Does not map district names onto yard towns.
    """
    req = _market_match_key(requested)
    got = _market_match_key(market_tag)
    return bool(req) and bool(got) and req == got


def _partition_items_by_requested_market(
    items: list[dict], requested_market_name: str
) -> tuple[list[dict], list[dict]]:
    """Split catalog rows into matching yard vs other in-range markets."""
    matched: list[dict] = []
    nearby: list[dict] = []
    for item in items:
        market = (_tag_values(item).get("Market") or "").strip()
        if market and _markets_match(requested_market_name, market):
            matched.append(item)
        else:
            nearby.append(item)
    return matched, nearby


def _unique_nearby_market_labels(items: list[dict]) -> list[str]:
    """Distinct market/district/state labels — names only, never prices."""
    labels: list[str] = []
    seen: set[str] = set()
    for item in items:
        label = _market_label(_tag_values(item))
        key = label.casefold()
        if label and key not in seen and label.casefold() != "market n/a":
            seen.add(key)
            labels.append(label)
    return labels


def _explicit_yard_miss_message(
    *,
    commodity_name: str,
    requested_market_name: str,
    from_date: str,
    to_date: str,
    nearby_items: list[dict],
) -> str:
    """Strict no-substitute reply when the named yard has no rows.

    Nearby markets may be named so the farmer knows data exists elsewhere, but
    their prices are never quoted as a stand-in for the requested APMC.
    """
    lines = [
        f"No rates were reported for {requested_market_name} for '{commodity_name}' "
        f"between {from_date} and {to_date}."
    ]
    nearby_names = _unique_nearby_market_labels(nearby_items)
    if nearby_names:
        lines.append(
            "Nearby markets with data (not a substitute for the requested yard): "
            + "; ".join(nearby_names)
            + "."
        )
    return "\n".join(lines)


def _format_mandi_items(items: list[dict], max_rows: int = 40) -> str:
    """One line per arrival date, newest first — a 30-day window is ~25 rows, and
    the generic _format_items would both truncate at 20 and bury the series in
    repeated tag dumps."""
    rows = []
    for it in items:
        t = _tag_values(it)
        arrival = t.get("Arrival Date", "")
        parsed = _parse_date(arrival, warn=False)
        # The BPP's own item name, not the string we asked for. Commodity
        # matching is fuzzy — asking for "Onion" can return "Onion Green", a
        # different, real commodity — and omitting the name made that invisible.
        name = ((it.get("descriptor") or {}).get("name") or "").strip()
        # An unparseable arrival date sorts last (date.min) AND renders as
        # "date n/a" — echoing the raw junk back at the farmer is worse than
        # admitting we don't know the date.
        rows.append((parsed or date.min, arrival if parsed else "", name, t))
    rows.sort(key=lambda r: r[0], reverse=True)

    lines = []
    for _, arrival, name, t in rows[:max_rows]:
        market = _market_label(t)
        unit = t.get("Price Unit", "Rs./Qtl")
        modal, lo, hi = t.get("Modal Price"), t.get("Min Price"), t.get("Max Price")
        price = f"modal {modal}" if modal else "modal n/a"
        if lo and hi:
            price += f" (min {lo} – max {hi})"
        extra = ", ".join(v for v in (t.get("Variety"), t.get("Grade")) if v and v != "----")
        lines.append(f"- {arrival or 'date n/a'} — {market}: "
                     + (f"{name} " if name else "")
                     + f"{price} {unit}"
                     + (f" [{extra}]" if extra else ""))

    if len(rows) > max_rows:
        lines.append(f"- …{len(rows) - max_rows} older rows omitted.")
    return "\n".join(lines)


async def get_vistaar_weather(
    ctx: RunContext[FarmerContext],
    location: Optional[str] = None,
) -> str:
    """Get the weather forecast for the farmer's area from Bharat Vistaar
    (Mausamgram). Call this directly for any weather question. By default it uses
    the farmer's own district, so do NOT ask them where they are — only pass
    `location` if they explicitly name a different place in their question.

    Returns a day-wise forecast (rainfall, min/max temp, humidity, wind, etc.).
    If the named place is not covered, the reply says so and names places that
    are — pass that back to the farmer instead of guessing another location.

    Args:
        ctx: authenticated farmer context used to resolve the default district.
        location: optional district or town in Gujarat the farmer explicitly
            named, e.g. "Junagadh", "Bhuj", "Banaskantha". Omit it to use the
            farmer's own district. Pass a place NAME only — never coordinates.
    """
    where, refusal = await _resolve_search_location(ctx, location)
    if refusal is not None:
        return refusal
    assert where is not None

    def build(candidate: Candidate) -> dict:
        return {
            "category": {"descriptor": {"name": "Weather-Forecast-Mausamgram", "code": "WFC"}},
            "fulfillment": {"stops": [{"location": {"lat": candidate.lat, "lon": candidate.lon}}]},
        }

    try:
        items, used = await _search_candidates(build, where)
    except Exception:
        logger.exception("vistaar weather failed district=%s", where.location.key)
        return "Weather is temporarily unavailable from Bharat Vistaar."
    if not items:
        return f"No weather forecast was returned for {_location_phrase(where, used)}."
    header = f"Weather forecast for {_location_phrase(where, used)}:\n"
    body = header + _format_items(items)
    return body + (_assumed_location_note(where) if where.assumed else "")


async def get_vistaar_mandi_prices(
    ctx: RunContext[FarmerContext],
    commodity_name: str,
    location: Optional[str] = None,
    price_date: Optional[str] = None,
    price_date_to: Optional[str] = None,
) -> str:
    """Get mandi (market) prices for a commodity from Bharat Vistaar. By default
    it uses the farmer's own district, so call it with just the commodity and do
    NOT ask them which market or city they want. Only pass `location` when the
    farmer explicitly names a place in their question ("prices in Junagadh").

    Returns one row per arrival date, so it answers history questions ("prices
    for the last 10 days") as well as "what is the price today". Markets do not
    trade every commodity every day, so gaps between dates are normal. Each row
    names the market, district and state it came from: nearby markets in another
    district — or another state — are normal for a district/town ask, so report
    the market as given rather than describing it as the farmer's own.

    If the farmer names a specific yard ("Anand APMC", "Nadiad mandi"), only
    rows from that yard are returned. Nearby markets are never quoted as that
    yard's price; if the yard has no arrivals, say so and optionally name
    nearby markets without their prices. Pass that back; do not retry with a
    different location.

    If the named place is not covered, the reply says so and names places that
    are. Pass that back to the farmer; do not retry with a different location.

    Args:
        ctx: authenticated farmer context used to resolve the default district.
        commodity_name: the commodity to price, e.g. "Tomato", "Onion", "Wheat".
            Use the English Agmarknet name; invented variants ("Onion Big")
            return nothing.
        location: optional district, town, or yard in Gujarat the farmer
            explicitly named, e.g. "Junagadh", "Anand APMC", "Nadiad APMC".
            Omit it to use the farmer's own district. Pass a place NAME only —
            never coordinates.
        price_date: optional START of the date window, DD-MM-YYYY. Omit for the
            latest available prices.
        price_date_to: optional END of the date window, DD-MM-YYYY. Pass BOTH
            ends for a range ("last 10 days"); never send only one end. Windows
            wider than 30 days are trimmed to the most recent 30.
    Returns market prices per date (min / max / modal, market, arrival date).
    """
    where, refusal = await _resolve_search_location(ctx, location)
    if refusal is not None:
        return refusal
    assert where is not None
    from_date, to_date = _resolve_date_range(price_date, price_date_to)

    def build(candidate: Candidate) -> dict:
        return {
            "category": {"descriptor": {"code": "price-discovery"}},
            "item": {"descriptor": {"name": commodity_name}},
            # `descriptor.name` here is DECORATIVE — measured: name=Junagadh with
            # Anand's gps returns nothing, name=Anand with Junagadh's gps returns
            # Junagadh APMC. The gps is what selects the market. It is sent
            # anyway because it costs nothing and reads correctly in a trace.
            "fulfillment": {"end": {"location": {
                "descriptor": {"name": candidate.town},
                "gps": f"{candidate.lat},{candidate.lon}",
            }}},
            # Flat tags — the BPP ignores the Beckn tag-group form.
            "tags": [
                {"code": "from_date", "value": from_date},
                {"code": "to_date", "value": to_date},
            ],
        }

    try:
        if where.explicit_yard:
            assert where.requested_market_name is not None
            items, nearby, used = await _search_candidates_for_yard(
                build, where, where.requested_market_name
            )
        else:
            items, used = await _search_candidates(build, where)
            nearby = []
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
    place = _location_phrase(where, used)

    matched_count = len(items)
    nearby_count = len(nearby)
    if where.explicit_yard and not items:
        logger.info(
            "vistaar mandi yard miss commodity=%s requested_market=%s "
            "nearby_markets=%d from=%s to=%s",
            commodity_name,
            where.requested_market_name,
            nearby_count,
            from_date,
            to_date,
        )
        return _explicit_yard_miss_message(
            commodity_name=commodity_name,
            requested_market_name=where.requested_market_name,
            from_date=from_date,
            to_date=to_date,
            nearby_items=nearby,
        )

    logger.info(
        "vistaar mandi commodity=%s place=%s source=%s requested_market=%s "
        "from=%s to=%s items=%d matched=%d nearby=%d",
        commodity_name,
        place,
        where.source,
        where.requested_market_name,
        from_date,
        to_date,
        len(items),
        matched_count,
        nearby_count,
    )
    if not items:
        return (
            f"No mandi prices were found for '{commodity_name}' near {place} "
            f"between {from_date} and {to_date}."
            + (_assumed_location_note(where) if where.assumed else "")
        )
    if where.explicit_yard and where.requested_market_name:
        header = (
            f"Mandi prices for {commodity_name} at {where.requested_market_name} "
            f"({from_date} to {to_date}):\n"
        )
    else:
        header = (
            f"Mandi prices for {commodity_name} near {place} "
            f"({from_date} to {to_date}):\n"
        )
    return header + _format_mandi_items(items) + (
        _assumed_location_note(where) if where.assumed else ""
    )


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
