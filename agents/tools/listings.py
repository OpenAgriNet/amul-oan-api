"""Stage read-only listing workflows; source calls happen after the accepted turn."""
from pydantic_ai import RunContext
from agents.deps import FarmerContext
from agents.tools.models.listings import CattleFilters, FodderFilters
from agents.tools.vistaar import _selectors_available, prepare_text_selector_tool
from app.config import settings


async def prepare_fodder_request(ctx, tool_def):
    return tool_def if _selectors_available(ctx.deps) and settings.fodderinfo_api_key else None


async def prepare_legacy_fodder(ctx, tool_def):
    return tool_def if settings.fodderinfo_api_key and await prepare_text_selector_tool(ctx, tool_def) else None


async def request_cattle_listings(ctx: RunContext[FarmerContext], filters: CattleFilters,
                                  listing_ref: str | None = None) -> str:
    """Request verified cattle cards with the existing filters, or details for a displayed listing_ref.

    Pass the complete active filters when narrowing a search; omitted fields clear filters.
    Use the existing English names. Never invent listing references or source records.
    """
    if not _selectors_available(ctx.deps):
        return "Interactive cattle listings are unavailable for this turn."
    return ctx.deps._selector_ui.stage(ctx.deps, dict(kind="cattle", filters=filters.model_dump(),
        listing_ref=listing_ref, district=ctx.deps.get_farmer_district()))


async def request_fodder_listings(ctx: RunContext[FarmerContext], filters: FodderFilters,
                                  listing_ref: str | None = None) -> str:
    """Request FodderInfo cards with existing search filters, or recheck a displayed listing_ref.

    Pass the complete active filters. Ask for missing place or unclear fodder names as before.
    Numeric rate/quantity filters use kg; cards retain each seller's original price unit.
    """
    if not _selectors_available(ctx.deps) or not settings.fodderinfo_api_key:
        return "Interactive fodder listings are unavailable for this turn."
    return ctx.deps._selector_ui.stage(ctx.deps, dict(kind="fodder", filters=filters.model_dump(), listing_ref=listing_ref))
