"""
Tool for booking a health call for a farmer.
"""
import os

from agents.tools.farmer_animal_backends import create_health_call_api
from app.models.ai_call import AISpecies
from app.models.health_call import HealthCallRequestModel, HealthCaseType
from helpers.utils import get_logger

logger = get_logger(__name__)


async def create_health_call(
    union_code: str,
    society_code: str,
    farmer_code: str,
    species: AISpecies,
    case_type: HealthCaseType,
    remark: str | None = None,
) -> str:
    """
    Book a health call for a farmer and return the generated ticket number.

    Args:
        union_code: Union code for the farmer from farmer context.
        society_code: Society code for the farmer from farmer context.
        farmer_code: Farmer code for the farmer from farmer context.
        species: Species for the call (`cow` or `buffalo`).
        case_type: Case type (`normal` or `emergency`).
        remark: Optional concise issue summary.

    Returns:
        str: Success message containing the ticket number, or a clear failure message.
    """
    logger.info(
        "Create health call tool invoked for union=%s society=%s farmer=%s species=%s case_type=%s",
        union_code,
        society_code,
        farmer_code,
        species.value,
        case_type.value,
    )

    logger.info("[YASHhhhhhhhhhh]begin create_health_call")

    token = os.getenv("PASHUGPT_TOKEN")
    if not token:
        logger.error("PASHUGPT_TOKEN is not set")
        return "Health call booking failed.\n\nPASHUGPT_TOKEN is not configured."

    request = HealthCallRequestModel(
        unionCode=union_code,
        societyCode=society_code,
        farmerCode=farmer_code,
        species=species,
        caseType=case_type,
        remark=remark,
    )

    response = await create_health_call_api(request, token)
    if response is None:
        logger.info(
            "Create health call failed for union=%s society=%s farmer=%s species=%s case_type=%s",
            union_code,
            society_code,
            farmer_code,
            species.value,
            case_type.value,
        )
        return "Health call booking failed.\n\nUnable to create health call at the moment."

    ticket_number = response.ticket_number
    logger.info(
        "Create health call succeeded for union=%s society=%s farmer=%s species=%s case_type=%s ticket=%s",
        union_code,
        society_code,
        farmer_code,
        species.value,
        case_type.value,
        ticket_number,
    )

    if ticket_number:
        return f"Health call booked successfully. Ticket number: {ticket_number}"
    return "Health call booked successfully, but ticket number was not returned."
