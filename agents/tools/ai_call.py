"""
Tool for booking an artificial insemination call for a farmer.
"""
import json
import os
from contextlib import nullcontext

from agents.tools.farmer_animal_backends import create_ai_call_api
from app.models.ai_call import AICallRequestModel, AISpecies
from helpers.utils import get_logger

try:
    from langfuse import get_client as get_langfuse_client
except ImportError:
    get_langfuse_client = None

logger = get_logger(__name__)


async def create_ai_call(
    union_code: str,
    society_code: str,
    farmer_code: str,
    user_id: str,
    species: AISpecies,
) -> str:
    """
    Book an artificial insemination call for a farmer.

    Args:
        union_code: Union code for the farmer from farmer context.
        society_code: Society code for the farmer from farmer context.
        farmer_code: Farmer code for the farmer from farmer context.
        user_id: Selected AI technician user identifier to be sent as userId to external CreateAICall API.
        species: Species to book the AI call for. Use `cow` or `buffalo`.

    Returns:
        str: Formatted JSON string with assigned AIT details and ticket number,
             or a clear message if booking fails.
    """
    logger.info(
        "Create AI call tool invoked for union=%s society=%s farmer=%s user_id=%s species=%s",
        union_code,
        society_code,
        farmer_code,
        user_id,
        species.value,
    )
    _lf = get_langfuse_client() if get_langfuse_client else None
    _ai_tool_input = {
        "union_code": union_code,
        "society_code": society_code,
        "farmer_code": farmer_code,
        "user_id": user_id,
        "species": species.value,
    }
    _ai_tool_obs_ctx = (
        _lf.start_as_current_observation(
            name="ai_call_booking",
            as_type="generation",
            input=_ai_tool_input,
            metadata={"tool_name": "create_ai_call"},
        )
        if _lf
        else nullcontext()
    )

    with _ai_tool_obs_ctx as ai_tool_obs:
        if _lf:
            try:
                _lf.set_current_trace_io(input=_ai_tool_input)
            except Exception:
                pass
        token = os.getenv("PASHUGPT_TOKEN")
        if not token:
            logger.error("PASHUGPT_TOKEN is not set")
            failure_message = (
                "Artificial insemination call booking failed.\n\n"
                "PASHUGPT_TOKEN is not configured."
            )
            if ai_tool_obs is not None:
                ai_tool_obs.update(output={"success": False, "message": failure_message})
            if _lf:
                try:
                    _lf.set_current_trace_io(
                        input=_ai_tool_input,
                        output={"success": False, "message": failure_message},
                    )
                except Exception:
                    pass
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
            if _lf:
                try:
                    _lf.set_current_trace_io(
                        input=_ai_tool_input,
                        output={"success": False, "message": failure_message},
                    )
                except Exception:
                    pass
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
        if _lf:
            try:
                _lf.set_current_trace_io(
                    input=_ai_tool_input,
                    output={
                        "success": True,
                        "ticket_number": response.ticket_number,
                        "ait_name": response.ait_name,
                    },
                )
            except Exception:
                pass
        return success_message
