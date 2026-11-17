"""Read-only access to current Gujarat fodder listings."""

import asyncio
import re
from datetime import datetime, timezone
from typing import Annotated, Literal

import httpx
from pydantic import Field

from agents.tools.telemetry import report_tool
from app.config import settings

ORIGIN = "https://fodderinfo.amul.in"
_TIMEOUT_SECONDS = 5.0
Form = Literal["GREEN", "DRY", "HAY", "SILAGE", "CONCENTRATE"]
PlaceLevel = Literal["state", "district", "subdistrict", "village"]
ListingId = Annotated[int, Field(gt=0)]


async def prepare_fodder_tool(ctx, tool_def):
    return tool_def if settings.fodderinfo_api_key else None


async def _get(path: str, params: dict | None = None) -> dict:
    key = settings.fodderinfo_api_key
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
    contact = _fields(item.get("contact"), "name mobile")
    if item.get("contact_status") == "shared" and re.fullmatch(r"[0-9]{10}", str(contact.get("mobile", ""))):
        result["seller"] = contact
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
    """
    Find fodder offered for sale in Gujarat on FodderInfo, each listing with its asking rate, place, last confirmation and seller's name and mobile.

    Use it when the farmer wants to buy fodder or asks who has fodder, how much is available or the
    asking rate. Use the usual tools for feeding or growing advice and get_vistaar_mandi_prices for
    mandi crop prices. Ask for the place if the farmer gave none. Give the rate in its own unit, the
    place, the last confirmation date and the seller's name and mobile exactly as returned, without
    assuming the seller's gender, and ask the farmer to confirm stock with the seller. Never estimate
    stock, rates or distance.

    Args:
        q: Fodder name as the farmer said it, e.g. "green bajra".
        form: GREEN, DRY, HAY, SILAGE or CONCENTRATE, only when the farmer said it.
        district: District, only as the farmer named it; leave empty when they named only a taluka or village.
        subdistrict: Taluka name in English; add district when the name can match several places.
        village: Village name in English; add district or subdistrict to narrow it.
        comparison_group: Fodder group, only when the farmer named a group rather than a fodder.
        delivery: "delivery" when the seller must deliver, "pickup" when the farmer will collect.
        max_rate_per_kg: Highest rate in rupees per kg the farmer will pay.
        min_qty_kg: Least quantity in kg the farmer needs.
        sort: "recent" (last confirmed first), "rate" (cheapest first) or "qty" (largest first).
        limit: How many listings to return.

    Returns:
        Current listings with the total match count, or a status with a message saying what to ask
        the farmer (for example green or dry when the fodder name is unclear).
    """
    params = {name: value for name, value in locals().items() if value is not None}
    body = await _get("listings", params)
    status = body.get("status")
    if status == "ok":
        rows = body.get("listings")
        if not isinstance(rows, list):
            report_tool("unavailable", "Invalid FodderInfo search response")
            return {"status": "unavailable"}
        listings = [_listing(row) for row in rows if _current(row)][:limit]
        if listings:
            return {"status": "ok", "notice": body.get("notice"), "total": body.get("total"), "listings": listings}
        status = "no_current_listings"
    if status in ("no_current_listings", "location_not_found", "name_not_resolved"):
        report_tool("no_result")
    elif status not in ("clarification_needed", "unavailable"):
        report_tool("unavailable", "Invalid FodderInfo search status")
        status = "unavailable"
    return {"status": status, "notice": body.get("notice"), **_fields(body, "message resolution")}


async def lookup_fodder_locations(
    q: Annotated[str, Field(min_length=2)], level: PlaceLevel | None = None,
    parent_id: ListingId | None = None,
) -> dict:
    """
    Resolve a Gujarat place name for search_fodder_availability when the place is unclear or could match several places.

    Args:
        q: Start of the place name; English for taluka and village.
        level: state, district, subdistrict (taluka) or village.
        parent_id: ID of a place already resolved, to search only inside it.
    """
    params = {name: value for name, value in locals().items() if value is not None}
    body = await _get("locations", params)
    if body.get("status") == "no_match":
        report_tool("no_result")
        return {"status": "no_match"}
    rows = body.get("locations")
    if body.get("status") == "ok" and isinstance(rows, list):
        return {"status": "ok", "locations": [_fields(row, "id level name_en name_gu within") for row in rows]}
    report_tool("unavailable", "FodderInfo location lookup failed")
    return {"status": "unavailable"}


async def get_fodder_listing(listing_id: ListingId) -> dict:
    """
    Recheck one FodderInfo listing the farmer picked, with its seller's name and mobile.

    Args:
        listing_id: listing_id from search_fodder_availability.
    """
    body = await _get(f"listings/{listing_id}")
    status = body.get("status")
    if status == "ok":
        item = body.get("listing")
        if isinstance(item, dict) and item.get("listing_id") == listing_id:
            if _current(item):
                return {"status": "ok", "listing": _listing(item)}
            status = "not_current"
    if status in ("not_found", "not_current"):
        report_tool("no_result")
        return {"status": status}
    report_tool("unavailable", "FodderInfo listing lookup failed")
    return {"status": "unavailable"}


FODDER_TOOLS = (
    search_fodder_availability,
    lookup_fodder_locations,
    get_fodder_listing,
)
