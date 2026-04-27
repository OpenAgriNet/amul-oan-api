"""Tool for reading cached union scheme data from Redis."""

import json
from typing import Any

from pydantic_ai import RunContext

from agents.deps import FarmerContext
from app.models.union import UnionName
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
    farmer_unions = [union_name.strip().lower() for union_name in ctx.deps.farmer_unions if union_name]
    normalized_union_name = next((union_name for union_name in farmer_unions if union_name in SUPPORTED_SCHEME_UNIONS), None)
    normalized_scheme_name = scheme_name.strip() if scheme_name else None
    logger.info(
        "Union scheme tool invoked farmer_unions=%s selected_union=%s scheme_name=%s",
        farmer_unions,
        normalized_union_name,
        normalized_scheme_name,
    )
    if not normalized_union_name:
        logger.warning("Union scheme tool could not infer a supported union from farmer context farmer_unions=%s", farmer_unions)
        return "Scheme data is unavailable because the farmer union could not be determined from the current farmer context."

    if normalized_union_name not in SUPPORTED_SCHEME_UNIONS:
        logger.warning("Union scheme tool received unsupported union normalized_union_name=%s", normalized_union_name)
        return (
            "Scheme data is only available for supported unions: "
            f"{', '.join(sorted(SUPPORTED_SCHEME_UNIONS))}."
        )

    try:
        UnionName(normalized_union_name)
    except ValueError:
        logger.warning("Union scheme tool failed enum validation normalized_union_name=%s", normalized_union_name)
        return (
            "Scheme data is only available for supported unions: "
            f"{', '.join(sorted(SUPPORTED_SCHEME_UNIONS))}."
        )

    try:
        records = await get_cached_scheme_records_for_union(normalized_union_name)
    except SchemeDependencyError:
        logger.exception("Union scheme tool failed because Redis dependency is unavailable")
        return "Scheme data is temporarily unavailable because the cache dependency is not installed."
    except SchemeCacheError:
        logger.exception("Union scheme tool failed because scheme cache access failed")
        return "Scheme data is temporarily unavailable because the cache could not be read."
    except Exception:
        logger.exception("Union scheme tool failed due to unexpected error for union=%s", normalized_union_name)
        return "Scheme data is temporarily unavailable due to an unexpected error."

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
            "Union scheme tool found no cached data for union=%s scheme_name=%s",
            normalized_union_name,
            normalized_scheme_name,
        )
        if normalized_scheme_name:
            return (
                f"Scheme data for '{normalized_scheme_name}' is not available yet "
                f"for union '{normalized_union_name}'."
            )
        return f"Scheme data is not available yet for union '{normalized_union_name}'."

    logger.info("Union scheme tool returning cached data for union=%s record_count=%s", normalized_union_name, len(records))
    return json.dumps(records, indent=2, ensure_ascii=False)
