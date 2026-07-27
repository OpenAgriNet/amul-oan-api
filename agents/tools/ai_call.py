"""
Tool for booking an artificial insemination call for a farmer.
"""
import json
import os

from pydantic_ai import RunContext

from agents.deps import FarmerContext
from agents.tools.farmer_animal_backends import create_ai_call_api
from app.core.cache import cache, try_reserve, release_reservation
from app.models.ai_call import AICallRequestModel, AISpecies
from app.observability import start_observation, set_trace_io
from helpers.utils import get_logger

logger = get_logger(__name__)

# One booking per (session, species, farmer, technician) per 30 min. Distinct
# species (a cow AND a buffalo) can both be booked in one session, but a repeat
# for the SAME species short-circuits — which also makes this tool idempotent
# against an agent re-run (e.g. the OSS->managed streaming fallback re-executes
# tool calls) and against concurrent duplicate submits (Redis SET NX, shared
# across containers). The upstream CreateAICall API has no server-side dedup.
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
             or a message if booking fails or the same species was already
             booked in this session.
    """
    logger.info(
        "Create AI call tool invoked for union=%s society=%s farmer=%s user_id=%s species=%s",
        union_code,
        society_code,
        farmer_code,
        user_id,
        species.value,
    )

    session_id = ctx.deps.session_id if ctx and ctx.deps else None

    # A booking is IRREVERSIBLE, so block on the moderation verdict before writing.
    # On the voice path moderation runs concurrently with the agent; this refuses
    # the booking if the query was rejected. No-op on the chat path (no moderation
    # task attached → returns True), so chat behaviour is unchanged.
    if not await ctx.deps.ensure_in_scope():
        logger.info("AI call blocked: query failed moderation; session=%s", session_id)
        return "This helpline only handles dairy farming and animal husbandry questions."

    _ai_tool_input = {
        "union_code": union_code,
        "society_code": society_code,
        "farmer_code": farmer_code,
        "user_id": user_id,
        "species": species.value,
    }

    with start_observation(
        "ai_call_booking",
        as_type="generation",
        input=_ai_tool_input,
        metadata={"tool_name": "create_ai_call"},
    ) as ai_tool_obs:
        set_trace_io(input=_ai_tool_input)
        token = os.getenv("PASHUGPT_TOKEN")
        if not token:
            logger.error("PASHUGPT_TOKEN is not set")
            failure_message = (
                "Artificial insemination call booking failed.\n\n"
                "PASHUGPT_TOKEN is not configured."
            )
            if ai_tool_obs is not None:
                ai_tool_obs.update(output={"success": False, "message": failure_message})
            set_trace_io(
                input=_ai_tool_input,
                output={"success": False, "message": failure_message},
            )
            return failure_message

        request = AICallRequestModel(
            unionCode=union_code,
            societyCode=society_code,
            farmerCode=farmer_code,
            userId=user_id,
            species=species,
        )
        # Atomic per-(session, species, farmer, technician) reservation immediately
        # before the write: distinct species proceed independently, but a duplicate
        # submit or a fallback re-run for the SAME species short-circuits instead of
        # double-booking. Released below if the booking API itself fails.
        reservation_key = None
        if session_id:
            reservation_key = f"{session_id}:{species.value}:{farmer_code}:{user_id}"
            if not await try_reserve(reservation_key, AI_CALL_CACHE_NAMESPACE, AI_CALL_COOLDOWN_TTL):
                logger.info(
                    "AI call already booked/in-flight for session=%s species=%s farmer=%s, skipping",
                    session_id,
                    species.value,
                    farmer_code,
                )
                return (
                    f"An artificial insemination booking for your {species.value} is already "
                    "active in this session. Please try again later or contact your society "
                    "for assistance."
                )
        response = await create_ai_call_api(request, token)
        if response is None:
            if reservation_key:
                await release_reservation(reservation_key, AI_CALL_CACHE_NAMESPACE)
            logger.info(
                "Create AI call failed for union=%s society=%s farmer=%s species=%s",
                union_code,
                society_code,
                farmer_code,
                species.value,
            )
            failure_message = (
                "Artificial insemination call booking failed.\n\n"
                "Unable to create AI call at the moment."
            )
            if ai_tool_obs is not None:
                ai_tool_obs.update(output={"success": False, "message": failure_message})
            set_trace_io(
                input=_ai_tool_input,
                output={"success": False, "message": failure_message},
            )
            return failure_message

        # Mark this (session, species, farmer, technician) as booked so a re-run or
        # retry does not double-book. Distinct species remain independently bookable.
        if reservation_key:
            try:
                await cache.set(
                    reservation_key,
                    {"ticket": response.ticket_number, "species": species.value},
                    ttl=AI_CALL_COOLDOWN_TTL,
                    namespace=AI_CALL_CACHE_NAMESPACE,
                )
            except Exception as e:
                logger.warning("Failed to set AI call cooldown: %s", e)

        formatted = json.dumps(response.model_dump(), indent=2, ensure_ascii=False)
        logger.info(
            "Create AI call succeeded for union=%s society=%s farmer=%s species=%s ticket=%s",
            union_code,
            society_code,
            farmer_code,
            species.value,
            response.ticket_number,
        )
        success_message = f"Artificial insemination call booked successfully:\n\n{formatted}"
        if ai_tool_obs is not None:
            ai_tool_obs.update(
                output={
                    "success": True,
                    "ticket_number": response.ticket_number,
                    "ait_name": response.ait_name,
                    "message": success_message,
                }
            )
        set_trace_io(
            input=_ai_tool_input,
            output={
                "success": True,
                "ticket_number": response.ticket_number,
                "ait_name": response.ait_name,
            },
        )
        return success_message
