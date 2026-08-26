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

One seeker request carries ONE query string for every leg it fans out to (see
`seeker.js`: `input.query` is built once and passed to each `callOne`). The legs
do not share a query vocabulary — `amulschemes` does free-text title matching
while the Bharat Vistaar BPP matches exact scheme codes and nothing else — so
scheme discovery issues one concurrent request PER query instead. Same wall
time, different words down each leg.
"""
import asyncio
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from agents.tools.scheme_codes import resolve_scheme_code
from app.config import settings
from helpers.utils import get_logger

logger = get_logger(__name__)

VET_LEG = "amulvet"
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


async def _seeker_search(query: str, leg: str, user_id: Optional[str] = None) -> Any:
    """POST the seeker BAP, scoped to a single leg; return that leg's on_search."""
    return (await _seeker_search_legs(query, [leg], user_id))[0].get(leg)


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


async def network_search_documents(query: str, top_k: int = 12) -> str:
    """Vet-KB discovery via the network (advisory:amul-vet). Formats items the
    same way the direct Marqo tool does: numbered snippets with source."""
    results, errors = await _seeker_search_per_leg({VET_LEG: query})
    if errors.get(VET_LEG) or not isinstance(results.get(VET_LEG), dict):
        # Same distinction as everywhere else: a dead leg is not an empty KB.
        logger.warning("network vet search leg failed error=%s", errors.get(VET_LEG))
        return "The document search is temporarily unavailable. Please try again shortly."
    items = _items(results.get(VET_LEG))[:top_k]
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
    *,
    session_id: Optional[str] = None,
    tool_call_id: Optional[str] = None,
) -> NetworkBookingResult:
    """AI-call booking via the network (services:amul-vet-booking) — POSTs a
    Beckn confirm to the booking BPP, which calls PashuGPT CreateAICall.

    Raises on transport/HTTP errors (raise_for_status), exactly as before; the
    caller decides how to surface them.
    """
    if settings.beckn_callback_transactions_enabled:
        return await _network_callback_booking_result(
            service="ai-call",
            union_code=union_code,
            society_code=society_code,
            farmer_code=farmer_code,
            species=species,
            session_id=session_id,
            tool_call_id=tool_call_id,
            technician_id=user_id,
        )

    # Legacy synchronous adapter. Kept behind the new callback feature gate so
    # ENABLE_NETWORK alone remains a no-op deployment-wise until ONIX and the
    # booking BPP have the directed confirm/on_confirm routes described in the
    # integration contract.
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
        "transaction_id": str(uuid.uuid4()), "message_id": str(uuid.uuid4()),
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


async def network_create_health_call_result(
    union_code: str,
    society_code: str,
    farmer_code: str,
    species: str,
    case_type: str,
    remark: Optional[str],
    *,
    session_id: Optional[str] = None,
    tool_call_id: Optional[str] = None,
) -> NetworkBookingResult:
    """Health-call booking through the durable confirm/on_confirm facade."""
    if not settings.beckn_callback_transactions_enabled:
        raise RuntimeError("Health-call Beckn callbacks are not enabled")
    return await _network_callback_booking_result(
        service="health-call",
        union_code=union_code,
        society_code=society_code,
        farmer_code=farmer_code,
        species=species,
        session_id=session_id,
        tool_call_id=tool_call_id,
        case_type=case_type,
        remark=remark,
    )


async def _network_callback_booking_result(
    *,
    service: str,
    union_code: str,
    society_code: str,
    farmer_code: str,
    species: str,
    session_id: Optional[str],
    tool_call_id: Optional[str],
    technician_id: Optional[str] = None,
    case_type: Optional[str] = None,
    remark: Optional[str] = None,
) -> NetworkBookingResult:
    from app.services.beckn_operations import OperationState, get_beckn_operation_client

    action_result = await get_beckn_operation_client().confirm_booking(
        service=service,
        union_code=union_code,
        society_code=society_code,
        farmer_code=farmer_code,
        species=species,
        session_id=session_id,
        tool_call_id=tool_call_id,
        technician_id=technician_id,
        case_type=case_type,
        remark=remark,
    )
    operation = action_result.operation
    label = "Artificial insemination call" if service == "ai-call" else "Health call"

    if operation.state is OperationState.NACKED:
        error = (action_result.payload or {}).get("error") or {}
        return NetworkBookingResult(
            ok=False,
            ticket=None,
            message=f"{label} booking failed on the network: {error.get('message', 'request rejected')}",
            authoritative_no_booking=True,
        )
    if operation.state is OperationState.BUSINESS_FAILED:
        error = (action_result.payload or {}).get("error") or {}
        return NetworkBookingResult(
            ok=False,
            ticket=None,
            message=f"{label} booking could not be completed: {error.get('message', 'provider error')}",
            # Conservative: an accepted confirm that later errors is not enough
            # evidence to automatically submit another irreversible booking.
            authoritative_no_booking=False,
        )
    if operation.state is OperationState.TIMED_OUT_PENDING:
        return NetworkBookingResult(
            ok=False,
            ticket=None,
            message=(
                f"{label} booking is still pending confirmation. Please check with your "
                "society before trying again."
            ),
            authoritative_no_booking=False,
        )

    order = ((action_result.payload or {}).get("message") or {}).get("order") or {}
    ticket = order.get("id")
    if operation.state is not OperationState.SUCCEEDED or not ticket:
        return NetworkBookingResult(
            ok=False,
            ticket=None,
            message=f"{label} booking callback did not contain a provider ticket.",
            authoritative_no_booking=False,
        )
    return NetworkBookingResult(
        ok=True,
        ticket=str(ticket),
        message=f"{label} booked successfully via the Beckn network. Ticket: {ticket}",
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
