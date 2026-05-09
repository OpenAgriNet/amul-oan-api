"""
Tool for booking an artificial insemination (beech daan) call for a farmer.
One booking per session (30-min cooldown via Redis).
"""
import json
import os

from pydantic_ai import RunContext

from voice.agents.deps import FarmerContext
from voice.agents.models.ai_call import AICallRequestModel, AISpecies
from voice.agents.tools.farmer_animal_backends import create_ai_call_api
from voice.app.core.cache import cache
from voice.helpers.utils import get_logger

logger = get_logger(__name__)

AI_CALL_COOLDOWN_TTL = 60 * 30  # 30 minutes
AI_CALL_CACHE_NAMESPACE = "ai_call_booked"


async def create_ai_call(
    ctx: RunContext[FarmerContext],
    union_code: str,
    society_code: str,
    farmer_code: str,
    user_id: str,
    species: AISpecies,
) -> str:
    """
    Book an artificial insemination (beech daan / બીજ દાન) call for a farmer.
    Extract union_code, society_code, farmer_code, and the selected AI technician user_id
    from the farmer context in the system prompt.
    If these details are not available, tell the farmer their details are not available right now.
    Ask the farmer whether the booking is for a cow (ગાય) or buffalo (ભેંસ) before calling this tool.
    Never ask the farmer to speak an internal technician ID. Use the selected technician option
    already present in farmer context.

    Args:
        ctx: The run context (automatically provided).
        union_code: Union code for the farmer from farmer context.
        society_code: Society code for the farmer from farmer context.
        farmer_code: Farmer code for the farmer from farmer context.
        user_id: Selected AI technician user ID mapped from farmer context.
        species: Species to book the AI call for. Use `cow` or `buffalo`.

    Returns:
        str: Formatted result with assigned AIT details and ticket number,
             or a message if booking fails or was already done this session.
    """
    session_id = ctx.deps.session_id
    logger.info(
        "AI call tool invoked: session=%s union=%s society=%s farmer=%s user_id=%s species=%s",
        session_id, union_code, society_code, farmer_code, user_id, species.value,
    )

    # Session-based cooldown: one booking per session per 30 minutes
    if session_id:
        cache_key = session_id
        try:
            existing = await cache.get(cache_key, namespace=AI_CALL_CACHE_NAMESPACE)
            if existing:
                logger.info("AI call already booked for session %s, skipping", session_id)
                return (
                    "This session already has an active artificial insemination booking. "
                    "Please try again later or contact your society for assistance."
                )
        except Exception as e:
            logger.warning("Failed to check AI call cooldown: %s", e)

    token = os.getenv("PASHUGPT_TOKEN")
    if not token:
        logger.error("PASHUGPT_TOKEN is not set")
        return "Artificial insemination call booking failed. Service is not configured."

    request = AICallRequestModel(
        unionCode=union_code,
        societyCode=society_code,
        farmerCode=farmer_code,
        userId=user_id,
        species=species,
    )
    response = await create_ai_call_api(request, token)
    if response is None:
        logger.info("AI call API failed for session=%s", session_id)
        return "Artificial insemination call booking failed. Unable to create booking at the moment."

    # Mark session as booked
    if session_id:
        try:
            await cache.set(
                session_id,
                {"ticket": response.ticket_number, "species": species.value},
                ttl=AI_CALL_COOLDOWN_TTL,
                namespace=AI_CALL_CACHE_NAMESPACE,
            )
        except Exception as e:
            logger.warning("Failed to set AI call cooldown: %s", e)

    formatted = json.dumps(response.model_dump(), indent=2, ensure_ascii=False)
    logger.info(
        "AI call booked: session=%s ticket=%s ait=%s",
        session_id, response.ticket_number, response.ait_name,
    )
    return f"Artificial insemination call booked successfully:\n\n{formatted}"
