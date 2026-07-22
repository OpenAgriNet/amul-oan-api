"""
Tool for booking an artificial insemination call for a farmer.
"""
import json
import os

from pydantic_ai import RunContext

from agents.deps import FarmerContext
from agents.tools.farmer_animal_backends import create_ai_call_api
from app.models.ai_call import AICallRequestModel, AISpecies
from app.observability import start_observation, set_trace_io
from helpers.utils import get_logger

logger = get_logger(__name__)


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
             or a message if booking fails.
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
        response = await create_ai_call_api(request, token)
        if response is None:
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
