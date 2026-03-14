"""
Tool for fetching completed operated veterinary visit records by animal tag.
Uses the Banas mobile API GetOperatedVisit endpoint and normalizes nested
JSON-string fields like MedicinesJson and LabReportsJson for cleaner tool output.
"""
import json
import os
from typing import Any, Dict, List, Optional

import httpx
from pydantic_ai import ModelRetry

from agents.tools.farmer_animal_backends import normalize_tag
from helpers.utils import get_logger

logger = get_logger(__name__)

BANAS_OPERATED_VISIT_URL = "https://banasmobileapi.amnex.com/api/FarmerVisitAPIKOS/GetOperatedVisit"


def _parse_nested_json(value: Any) -> Any:
    """Decode JSON strings when the API returns nested arrays as serialized text."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _normalize_visit_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return a normalized visit record with parsed nested JSON fields."""
    normalized = dict(record)
    if "MedicinesJson" in normalized:
        normalized["MedicinesJson"] = _parse_nested_json(normalized.get("MedicinesJson"))
    if "LabReportsJson" in normalized:
        normalized["LabReportsJson"] = _parse_nested_json(normalized.get("LabReportsJson"))
    return normalized


async def get_operated_visit_by_tag(
    tag_no: str,
    str_api_key: Optional[str] = None,
) -> str:
    """
    Fetch completed operated veterinary visit history for a tagged animal.
    Use this tool when the user asks for visit history, operated visit records,
    doctor visit details, medicines given during a visit, or lab reports for a
    specific tagged animal.

    Args:
        tag_no: The animal tag number to look up (required).
        str_api_key: API key for the Banas operated visit API (optional). Defaults
            to BANAS_OPERATED_VISIT_API_KEY from the environment when omitted.

    Returns:
        str: A formatted JSON string containing operated visit history for the tag,
            including visit metadata, disease/ailment details, doctor details,
            payment details, and parsed MedicinesJson/LabReportsJson arrays.
            Returns a clear no-data message when no visits are found.
    """
    tag = normalize_tag(tag_no)
    if not tag:
        return "Please provide a valid tag number."

    api_key = str_api_key or os.getenv("BANAS_OPERATED_VISIT_API_KEY")
    if not api_key:
        logger.error("No API key available for Banas operated visit API")
        raise ModelRetry("Banas operated visit API key is not configured")

    payload = {
        "strApiKey": api_key,
        "tagId": tag,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                BANAS_OPERATED_VISIT_URL,
                headers={"Content-Type": "application/json"},
                json=payload,
            )

        if response.status_code != 200:
            logger.error(
                "Banas operated visit API error for tag %s: %s - %s",
                tag,
                response.status_code,
                response.text[:500],
            )
            raise ModelRetry(
                f"Failed to fetch operated visit history: {response.status_code}"
            )

        text = response.text.strip()
        if not text:
            logger.info("No operated visit data returned for tag %s", tag)
            return f"Operated visit history for tag {tag}:\n\nNo operated visit data found for this tag number."

        data = json.loads(text)
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            visit_records = data["data"]
        elif isinstance(data, list):
            visit_records = data
        else:
            visit_records = []

        if not visit_records:
            logger.info("No operated visit records found for tag %s", tag)
            return f"Operated visit history for tag {tag}:\n\nNo operated visit data found for this tag number."

        normalized_records: List[Dict[str, Any]] = [
            _normalize_visit_record(record)
            for record in visit_records
            if isinstance(record, dict)
        ]
        formatted = json.dumps(normalized_records, indent=2, ensure_ascii=False)
        return f"Operated visit history for tag {tag}:\n\n{formatted}"
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON from Banas operated visit API for tag %s: %s", tag, e)
        raise ModelRetry("Operated visit API returned invalid data, please try again")
    except ModelRetry:
        raise
    except Exception as e:
        logger.error("Error fetching operated visit history for tag %s: %s", tag, e)
        raise ModelRetry("Error fetching operated visit history, please try again")
