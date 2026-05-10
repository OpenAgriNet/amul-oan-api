"""
Tool for booking a health call for a farmer.
"""
import os

from pydantic_ai import RunContext

from voice.agents.deps import FarmerContext
from voice.agents.models.ai_call import AISpecies
from voice.agents.models.health_call import HealthCallRequestModel, HealthCaseType
from voice.agents.tools.farmer_animal_backends import create_health_call_api
from voice.helpers.utils import get_logger

logger = get_logger(__name__)


async def create_health_call(
    ctx: RunContext[FarmerContext],
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
        ctx: The run context (automatically provided).
        union_code: Union code for the farmer from farmer context.
        society_code: Society code for the farmer from farmer context.
        farmer_code: Farmer code for the farmer from farmer context.
        species: Species for the call (`cow` or `buffalo`).
        case_type: Case type (`normal` or `emergency`).
        remark: Optional concise issue summary.

    Returns:
        str: Success message containing the ticket number, or a clear failure message.
    """
    session_id = ctx.deps.session_id
    logger.info(
        "Health call tool invoked: session=%s union=%s society=%s farmer=%s species=%s case_type=%s",
        session_id,
        union_code,
        society_code,
        farmer_code,
        species.value,
        case_type.value,
    )

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
            "Health call API failed: session=%s union=%s society=%s farmer=%s species=%s case_type=%s",
            session_id,
            union_code,
            society_code,
            farmer_code,
            species.value,
            case_type.value,
        )
        return "Health call booking failed.\n\nUnable to create health call at the moment."

    ticket_number = response.ticket_number
    logger.info(
        "Health call booked: session=%s ticket=%s",
        session_id,
        ticket_number,
    )
    if ticket_number:
        return f"Health call booked successfully. Ticket number: {ticket_number}"
    return "Health call booked successfully, but ticket number was not returned."
