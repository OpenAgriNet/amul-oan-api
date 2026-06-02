"""
Tool for booking an artificial insemination call for a farmer.
"""
import json
import os
from contextlib import nullcontext

from pydantic_ai import RunContext

from agents.deps import FarmerContext
from agents.tools.farmer_animal_backends import create_ai_call_api
from app.core.cache import cache
from app.models.ai_call import AICallRequestModel, AISpecies
from helpers.utils import get_logger

try:
    from langfuse import get_client as get_langfuse_client
except ImportError:
    get_langfuse_client = None

logger = get_logger(__name__)

# One booking per session per 30 min. Also makes this tool idempotent against an
# agent re-run (e.g. the OSS->managed streaming fallback re-executes tool calls):
# a second invocation in the same session short-circuits instead of double-booking.
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

    # Idempotency / cooldown: one booking per session. Guards against an agent
    # re-run (OSS->managed streaming fallback) re-firing this write tool.
    session_id = ctx.deps.session_id if ctx and ctx.deps else None
    if session_id:
        try:
            existing = await cache.get(session_id, namespace=AI_CALL_CACHE_NAMESPACE)
            if existing:
                logger.info("AI call already booked for session %s, skipping", session_id)
                return (
                    "This session already has an active artificial insemination booking. "
                    "Please try again later or contact your society for assistance."
                )
        except Exception as e:
            logger.warning("Failed to check AI call cooldown: %s", e)

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

        # Mark this session as booked so a re-run (or retry) does not double-book.
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
