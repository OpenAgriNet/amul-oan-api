"""Tool for reading cached union scheme data from Redis."""

import json
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition

from agents.deps import FarmerContext
from app.config import settings
from app.models.union import UnionName, resolve_supported_unions
from app.services.scheme_ingestion import (
    SchemeCacheError,
    SchemeDependencyError,
    get_cached_scheme_records_for_union,
)
from helpers.utils import get_logger

SUPPORTED_SCHEME_UNIONS = {
    UnionName.BANAS.value,
    UnionName.KUTCH.value,
    UnionName.SUMUL.value,
    UnionName.SURENDRANAGAR.value,
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


def _filter_scheme_records(records: list[dict[str, Any]], scheme_name: str) -> list[dict[str, Any]]:
    normalized_filter = scheme_name.strip().casefold()
    if not normalized_filter:
        return records

    filtered_records = []
    for record in records:
        title = str(record.get("scheme_title") or "")
        if normalized_filter in title.casefold():
            filtered_records.append(record)
    return filtered_records


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
    # ⚠️ The docstring above is the LLM-visible contract and must stay true for
    # BOTH states of settings.enable_network, because the flag changes what this
    # function returns:
    #   flag OFF → union schemes only, read from the Redis cache below.
    #   flag ON  → union schemes from the Beckn network, MERGED with Bharat
    #              Vistaar central schemes when `scheme_name` resolves to a
    #              central scheme code.
    # It previously said "ONLY for these dairy-union schemes / Do NOT use this
    # for KCC, PM-KISAN, PMFBY" while, with the flag on, doing exactly that.
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

    # Feature flag: route through the Amul Beckn network (schemes:amul-union)
    # instead of the direct Redis cache when enabled. `settings` is imported at
    # module level — do NOT re-import it inside the function, or it becomes a
    # function-local name and earlier settings.* uses raise UnboundLocalError.
    if settings.enable_network:
        from agents.tools.beckn_network import network_union_schemes
        primary_union = target_unions[0] if len(target_unions) == 1 else None
        logger.info(
            "enable_network=on → union schemes via Beckn network union=%s scheme=%s",
            primary_union,
            normalized_scheme_name,
        )
        # The network call must degrade exactly like the direct Redis path
        # below: a seeker timeout / HTTP error / malformed on_search is an
        # infrastructure failure, not a tool failure. Without this the early
        # return skips the try/except that follows and the exception escapes
        # the tool entirely.
        try:
            return await network_union_schemes(normalized_scheme_name or "", union=primary_union)
        except Exception:
            logger.exception(
                "Union scheme tool failed via Beckn network union=%s scheme_name=%s",
                primary_union,
                normalized_scheme_name,
            )
            return "Scheme data is temporarily unavailable due to an unexpected error."

    records: list[dict[str, Any]] = []
    for union_name in target_unions:
        try:
            UnionName(union_name)
        except ValueError:
            logger.warning("Union scheme tool failed enum validation union_name=%s", union_name)
            continue

        try:
            union_records = await get_cached_scheme_records_for_union(union_name)
        except SchemeDependencyError:
            logger.exception("Union scheme tool failed because Redis dependency is unavailable")
            return "Scheme data is temporarily unavailable because the cache dependency is not installed."
        except SchemeCacheError:
            logger.exception("Union scheme tool failed because scheme cache access failed")
            return "Scheme data is temporarily unavailable because the cache could not be read."
        except Exception:
            logger.exception("Union scheme tool failed due to unexpected error for union=%s", union_name)
            return "Scheme data is temporarily unavailable due to an unexpected error."

        records.extend(union_records)

    if normalized_scheme_name:
        records = _filter_scheme_records(records, normalized_scheme_name)
        logger.info(
            "Union scheme tool applied scheme_name filter union=%s scheme_name=%s record_count=%s",
            normalized_union_name,
            normalized_scheme_name,
            len(records),
        )

    if not records:
        logger.info(
            "Union scheme tool found no cached data for unions=%s scheme_name=%s",
            target_unions,
            normalized_scheme_name,
        )
        if normalized_scheme_name:
            return f"Scheme data for '{normalized_scheme_name}' is not available yet for supported unions."
        return "Scheme data is not available yet for supported unions."

    logger.info("Union scheme tool returning cached data for unions=%s record_count=%s", target_unions, len(records))
    return json.dumps(records, indent=2, ensure_ascii=False)
