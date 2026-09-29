"""Beckn-backed dairy-union scheme tool."""

from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition

from agents.deps import FarmerContext
from agents.tools.beckn.network import network_union_schemes
from app.config import settings
from agents.tools.models.union import UnionName, resolve_supported_unions
from helpers.utils import get_logger

SUPPORTED_SCHEME_UNIONS = {
    UnionName.BANAS.value,
    UnionName.KUTCH.value,
    UnionName.SUMUL.value,
    UnionName.SURENDRANAGAR.value,
    UnionName.SABAR.value,
    UnionName.BHARUCH.value,
}

logger = get_logger(__name__)


async def prepare_get_union_scheme_data(
    ctx: RunContext[FarmerContext], tool_def: ToolDefinition
) -> ToolDefinition | None:
    """Hide get_union_scheme_data from the LLM unless the farmer is in a supported union.

    Prevents wasted tool calls and the misleading "union could not be determined"
    bail-out for farmers from unions whose scheme catalog isn't ingested
    (e.g., dudhsagar). The LLM won't see the tool in its schema this turn, so it can't call it.
    """
    farmer_unions = [u for u in (ctx.deps.farmer_unions or []) if u]
    supported_farmer_unions = resolve_supported_unions(farmer_unions, SUPPORTED_SCHEME_UNIONS)
    if supported_farmer_unions:
        return tool_def
    logger.info(
        "Hiding get_union_scheme_data tool because farmer_unions=%s resolved_supported_unions=%s has no supported union",
        farmer_unions,
        supported_farmer_unions,
    )
    return None


async def get_union_scheme_data(ctx: RunContext[FarmerContext], scheme_name: str | None = None) -> str:
    """
    Get scheme information for the farmer, starting from their Amul MILK-UNION /
    dairy-cooperative welfare schemes — the benefits their specific dairy union
    (e.g. Banas, Sarhad) offers, such as accident insurance or cattle/producer
    welfare.

    If `scheme_name` names a CENTRAL / national government agriculture scheme
    (Kisan Credit Card, PM-KISAN, crop insurance, Soil Health Card, …), this
    tool may also return that central scheme alongside the union ones, each
    record labelled with its source. It is therefore safe — not wrong — to call
    it for a mixed "what schemes can I get?" question. If a dedicated central-
    scheme tool (get_vistaar_scheme_info) is listed among your tools, prefer it
    for a question that is purely about one central scheme.

    Args:
        scheme_name: Optional scheme the user named, in their own words
            ("cattle insurance", "Kisan Credit Card", "પાક વીમો"). Omit for the
            farmer's full union scheme list.

    Returns:
        A JSON-formatted string of scheme records, or a clear no-data message.
    """
    farmer_unions = [union_name for union_name in (ctx.deps.farmer_unions or []) if union_name]
    supported_farmer_unions = resolve_supported_unions(farmer_unions, SUPPORTED_SCHEME_UNIONS)
    normalized_union_name = supported_farmer_unions[0] if supported_farmer_unions else None
    normalized_scheme_name = scheme_name.strip() if scheme_name else None
    require_union_auth = settings.scheme_require_union_auth
    logger.info(
        "Union scheme tool invoked farmer_unions=%s resolved_supported_unions=%s selected_union=%s scheme_name=%s require_union_auth=%s",
        farmer_unions,
        supported_farmer_unions,
        normalized_union_name,
        normalized_scheme_name,
        require_union_auth,
    )
    target_unions: list[str] = []
    if require_union_auth:
        if not normalized_union_name:
            logger.warning(
                "Union scheme tool could not infer a supported union from farmer context farmer_unions=%s",
                farmer_unions,
            )
            return "Scheme data is unavailable because the farmer union could not be determined from the current farmer context."
        target_unions = [normalized_union_name]
    else:
        if normalized_union_name:
            target_unions = [normalized_union_name]
        else:
            target_unions = sorted(SUPPORTED_SCHEME_UNIONS)
            logger.info(
                "Union scheme tool bypassed union auth for testing; using supported unions=%s",
                target_unions,
            )

    primary_union = target_unions[0] if len(target_unions) == 1 else None
    logger.info(
        "Union schemes via Beckn union=%s scheme=%s",
        primary_union,
        normalized_scheme_name,
    )
    try:
        return await network_union_schemes(normalized_scheme_name or "", union=primary_union)
    except Exception:
        logger.exception(
            "Union scheme tool failed via Beckn union=%s scheme_name=%s",
            primary_union,
            normalized_scheme_name,
        )
        return "Scheme data is temporarily unavailable due to an unexpected error."
