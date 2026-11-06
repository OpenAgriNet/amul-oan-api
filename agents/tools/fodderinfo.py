"""Read-only access to current Gujarat fodder listings."""

import asyncio
import re
from datetime import datetime, timezone
from typing import Annotated, Literal

import httpx
from pydantic import Field
from pydantic_ai import RunContext

from agents.deps import FarmerContext
from agents.tools.telemetry import report_tool
from app.config import settings

ORIGIN = "https://fodderinfo.amul.in"
_TIMEOUT_SECONDS = 5.0
Form = Literal["GREEN", "DRY", "HAY", "SILAGE", "CONCENTRATE"]
PlaceLevel = Literal["state", "district", "subdistrict", "village"]
ListingId = Annotated[int, Field(gt=0)]
_CONTACT_WORDS = re.compile(
    r"\b(?:phone|mobile|telephone|contact|whatsapp|seller['’]s number|the number)\b"
    r"|ફોન|મોબાઇલ|મોબાઈલ|નંબર|સંપર્ક|फोन|मोबाइल|नंबर|संपर्क",
    re.IGNORECASE,
)

FODDERINFO_INSTRUCTIONS = """For current fodder stock, sellers or asking rates in Gujarat, use
search_fodder_availability. General feeding, nutrition and growing advice use the usual tools.
Ask for a location for a local search if missing. Use the farmer's original fodder wording in q
and only use supplied or confirmed filters. For an unclear crop or form, ask which they mean.
For an unresolved or potentially ambiguous place, use lookup_fodder_locations and confirm the
place with the farmer if needed. Taluka and village lookups use English names.
Answer from current results and the notice. Give the seller's asking rate in its original unit,
location and last confirmation date, plus quantity and pickup or delivery terms when available.
Identify each option by its listing ID or public listing link so it can be selected later.
Ask the farmer to confirm stock with the seller. Distinguish no matches from a failed lookup.
Never estimate stock, rates or distance. Recheck a selected listing with get_fodder_listing.
Only if the current user explicitly asks for the phone number of one listing, call
request_fodder_seller_phone with its ID instead of get_fodder_listing. Ask the user to choose if
the listing is unclear. Never guess an ID or request multiple numbers. On delivery_pending,
tell the user the seller's number follows and let the application deliver it. If contact is
unavailable, offer the returned public listing link when present."""


async def prepare_fodder_tool(ctx, tool_def):
    return tool_def if settings.fodderinfo_search_api_key else None


async def _get(path: str, key: str, params: dict | None = None) -> dict:
    if key:
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS, follow_redirects=False) as client:
                response = await asyncio.wait_for(
                    client.get(f"{ORIGIN}/api/v1/{path}", params=params,
                               headers={"Authorization": f"Bearer {key}"}),
                    timeout=_TIMEOUT_SECONDS,
                )
            if response.status_code == 404 and path.startswith("listings/"):
                report_tool("no_result")
                return {"status": "not_found"}
            response.raise_for_status()
            body = response.json()
            if isinstance(body, dict):
                return body
        except httpx.HTTPStatusError as exc:
            report_tool("unavailable", f"FodderInfo HTTP {exc.response.status_code}")
            return {"status": "unavailable"}
        except (asyncio.TimeoutError, httpx.RequestError, ValueError):
            pass
    report_tool("unavailable", "FodderInfo lookup failed")
    return {"status": "unavailable"}


def _fields(data: object, names: str) -> dict:
    return {name: data[name] for name in names.split() if name in data} if isinstance(data, dict) else {}


def _current(item: object) -> bool:
    if not isinstance(item, dict) or item.get("is_current") is not True or item.get("is_test_data") is not False:
        return False
    valid_until = _fields(item.get("freshness"), "valid_until").get("valid_until")
    if not isinstance(valid_until, str):
        return False
    try:
        expires_at = datetime.fromisoformat(valid_until.replace("Z", "+00:00"))
        return expires_at.tzinfo is not None and expires_at > datetime.now(timezone.utc)
    except ValueError:
        return False


def _listing(item: dict) -> dict:
    result = _fields(item, "listing_id summary_en")
    if type(item.get("listing_id")) is int:
        result["listing_url"] = f"{ORIGIN}/listing/{item['listing_id']}"
    for key, fields in (
        ("material", "material_code canonical_name_en name_gu form comparison_group processing packaging"),
        ("price", "kind rate_inr per_unit per_unit_name_en per_unit_name_gu price_terms normalized_rate_per_kg_inr conversion_basis weight_is_estimated"),
        ("quantity", "available unit unit_name_en approx_kg min_purchase stock_type available_from"),
        ("delivery", "option transport_charges loading_charges"),
        ("freshness", "last_confirmed_at valid_until checked_at"),
    ):
        result[key] = _fields(item.get(key), fields)
    location = _fields(item.get("location"), "state district subdistrict village")
    result["location"] = {key: _fields(value, "id name_en name_gu") for key, value in location.items() if value}
    return result


async def search_fodder_availability(
    q: str | None = None,
    form: Form | None = None,
    district: str | None = None,
    subdistrict: str | None = None,
    village: str | None = None,
    comparison_group: Literal["green", "dry", "hay", "silage", "concentrate"] | None = None,
    delivery: Literal["delivery", "pickup"] | None = None,
    max_rate_per_kg: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None,
    min_qty_kg: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None,
    sort: Literal["recent", "rate", "qty"] | None = None,
    limit: Annotated[int, Field(ge=1, le=20)] = 5,
) -> dict:
    """Find current seller listings in Gujarat. Use the farmer's wording for q, and English names for subdistrict (taluka) and village."""
    params = {name: value for name, value in locals().items() if value is not None}
    body = await _get("listings", settings.fodderinfo_search_api_key, params)
    status = body.get("status")
    if status == "ok":
        rows = body.get("listings")
        if not isinstance(rows, list):
            report_tool("unavailable", "Invalid FodderInfo search response")
            return {"status": "unavailable"}
        listings = [_listing(row) for row in rows if _current(row)][:limit]
        if listings:
            return {"status": "ok", "notice": body.get("notice"), "listings": listings}
        status = "no_current_listings"
    if status in ("no_current_listings", "location_not_found", "name_not_resolved"):
        report_tool("no_result")
    elif status not in ("clarification_needed", "unavailable"):
        report_tool("unavailable", "Invalid FodderInfo search status")
        status = "unavailable"
    return {"status": status, "notice": body.get("notice")}


async def lookup_fodder_locations(
    q: Annotated[str, Field(min_length=2)], level: PlaceLevel | None = None,
    parent_id: ListingId | None = None,
) -> dict:
    """Resolve a Gujarat place name by prefix. Use English for village and taluka. Use a known parent ID to narrow the lookup."""
    params = {name: value for name, value in locals().items() if value is not None}
    body = await _get("locations", settings.fodderinfo_search_api_key, params)
    if body.get("status") == "no_match":
        report_tool("no_result")
        return {"status": "no_match"}
    rows = body.get("locations")
    if body.get("status") == "ok" and isinstance(rows, list):
        return {"status": "ok", "locations": [_fields(row, "id level name_en name_gu within") for row in rows]}
    report_tool("unavailable", "FodderInfo location lookup failed")
    return {"status": "unavailable"}


async def _detail(listing_id: int, key: str) -> tuple[str, dict]:
    body = await _get(f"listings/{listing_id}", key)
    status = body.get("status")
    if status == "ok":
        item = body.get("listing")
        if isinstance(item, dict) and item.get("listing_id") == listing_id:
            if _current(item):
                return "ok", item
            status = "not_current"
    if status in ("not_found", "not_current"):
        report_tool("no_result")
        return status, {}
    report_tool("unavailable", "FodderInfo listing lookup failed")
    return "unavailable", {}


async def get_fodder_listing(listing_id: ListingId) -> dict:
    """Recheck a specific FodderInfo listing before describing it. Omits contact details."""
    status, item = await _detail(listing_id, settings.fodderinfo_search_api_key)
    return {"status": status, "listing": _listing(item)} if item else {"status": status}


async def request_fodder_seller_phone(ctx: RunContext[FarmerContext], listing_id: ListingId) -> dict:
    """Recheck and arrange private phone delivery only for an explicit request about one identified listing."""
    deps = ctx.deps
    if not _CONTACT_WORDS.search(deps.query) or not await deps.ensure_in_scope():
        return {"status": "explicit_request_required"}
    if not deps.reserve_fodder_contact():
        return {"status": "already_requested"}

    contact_key = settings.fodderinfo_contact_api_key
    status, item = await _detail(listing_id, contact_key or settings.fodderinfo_search_api_key)
    if not item:
        return {"status": status}
    mobile = _fields(item.get("contact"), "mobile").get("mobile")
    if contact_key and item.get("contact_status") == "shared" and isinstance(mobile, str) and re.fullmatch(r"[0-9]{10}", mobile):
        deps.set_fodder_contact(mobile)
        return {"status": "delivery_pending"}
    return {"status": "contact_unavailable", "listing_url": f"{ORIGIN}/listing/{listing_id}"}


FODDER_TOOLS = (
    search_fodder_availability,
    lookup_fodder_locations,
    get_fodder_listing,
    request_fodder_seller_phone,
)
