"""
Tool for fetching AI technicians by union and society code from PashuGPT.
"""
import json
import os

from voice.helpers.utils import get_logger

from voice.agents.tools.farmer_animal_backends import (
    GetAITechniciansBySocietyQueryParams,
    get_ai_technicians_by_society_api,
)

logger = get_logger(__name__)


async def get_ai_technicians_by_society(union_code: str, society_code: str) -> str:
    """
    Fetch AI technician details for a specific union and society.
    Use this when the caller asks which AI technician is assigned to a society
    or wants technician contact details for AI service.

    Args:
        union_code: Union code for the society.
        society_code: Society code to look up.

    Returns:
        str: Formatted JSON string with AI technician details, or a clear message if unavailable.
    """
    token = os.getenv("PASHUGPT_TOKEN")
    if not token:
        logger.error("PASHUGPT_TOKEN is not set")
        return "AI technician lookup failed. Service is not configured."

    query = GetAITechniciansBySocietyQueryParams(
        unionCode=union_code,
        societyCode=society_code,
    )
    technicians = await get_ai_technicians_by_society_api(query, token)
    if not technicians:
        logger.info(
            "No AI technicians found for union=%s society=%s",
            union_code,
            society_code,
        )
        return (
            f"AI technician details for union {union_code} and society {society_code}:\n\n"
            "No AI technician data found for this society."
        )

    formatted = json.dumps(
        [technician.model_dump() for technician in technicians],
        indent=2,
        ensure_ascii=False,
    )
    return (
        f"AI technician details for union {union_code} and society {society_code}:\n\n"
        f"{formatted}"
    )
