"""
Tool for booking an artificial insemination call for a farmer.
"""
import json
import os

from agents.tools.farmer_animal_backends import create_ai_call_api
from app.models.ai_call import AICallRequestModel, AISpecies
from helpers.utils import get_logger

logger = get_logger(__name__)


async def create_ai_call(
    union_code: str,
    society_code: str,
    farmer_code: str,
    species: AISpecies,
) -> str:
    """
    Book an artificial insemination call for a farmer.

    Args:
        union_code: Union code for the farmer from farmer context.
        society_code: Society code for the farmer from farmer context.
        farmer_code: Farmer code for the farmer from farmer context.
        species: Species to book the AI call for. Use `cow` or `buffalo`.

    Returns:
        str: Formatted JSON string with assigned AIT details and ticket number,
             or a clear message if booking fails.
    """
    logger.info(
        "Create AI call tool invoked for union=%s society=%s farmer=%s species=%s",
        union_code,
        society_code,
        farmer_code,
        species.value,
    )

    token = os.getenv("PASHUGPT_TOKEN")
    if not token:
        logger.error("PASHUGPT_TOKEN is not set")
        return (
            "Artificial insemination call booking failed.\n\n"
            "PASHUGPT_TOKEN is not configured."
        )

    request = AICallRequestModel(
        unionCode=union_code,
        societyCode=society_code,
        farmerCode=farmer_code,
        species=species,
    )
    response = await create_ai_call_api(request, token)
    if response is None:
        logger.info(
            "Create AI call failed for union=%s society=%s farmer=%s species=%s",
            union_code,
            society_code,
            farmer_code,
            species.value,
        )
        return (
            "Artificial insemination call booking failed.\n\n"
            "Unable to create AI call at the moment."
        )

    formatted = json.dumps(response.model_dump(), indent=2, ensure_ascii=False)
    logger.info(
        "Create AI call succeeded for union=%s society=%s farmer=%s species=%s ticket=%s",
        union_code,
        society_code,
        farmer_code,
        species.value,
        response.ticket_number,
    )
    return f"Artificial insemination call booked successfully:\n\n{formatted}"
