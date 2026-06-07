"""Tool for reading cached union scheme data from Redis."""

import json
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition

from agents.deps import FarmerContext
from app.config import settings
from app.models.union import UnionName, canonical_union_name
from app.services.scheme_ingestion import (
    SchemeCacheError,
    SchemeDependencyError,
    get_cached_scheme_records_for_union,
)
from helpers.utils import get_logger

SUPPORTED_SCHEME_UNIONS = {
    UnionName.BANAS.value,
    UnionName.KUTCH.value,
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
    farmer_unions = [u.strip().lower() for u in (ctx.deps.farmer_unions or []) if u]
    if any(u in SUPPORTED_SCHEME_UNIONS for u in farmer_unions):
        return tool_def
    logger.info(
        "Hiding get_union_scheme_data tool because farmer_unions=%s has no supported union",
        farmer_unions,
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
    Get cached milk producer scheme data for the union inferred from farmer context.

    Args:
        scheme_name: Optional scheme title filter. Use this when the user asks about a specific scheme.

    Returns:
        A JSON-formatted string of cached scheme records, or a clear no-data message.
    """
    # Normalize each raw union name (dairy brand / spelling variant, e.g. "sarhad"
    # for Kutch) to its canonical UnionName value before matching the supported set.
    farmer_unions = [canonical_union_name(union_name) for union_name in ctx.deps.farmer_unions if union_name]
    normalized_union_name = next((union_name for union_name in farmer_unions if union_name in SUPPORTED_SCHEME_UNIONS), None)
    normalized_scheme_name = scheme_name.strip() if scheme_name else None
    require_union_auth = settings.scheme_require_union_auth
    logger.info(
        "Union scheme tool invoked farmer_unions=%s selected_union=%s scheme_name=%s require_union_auth=%s",
        farmer_unions,
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
