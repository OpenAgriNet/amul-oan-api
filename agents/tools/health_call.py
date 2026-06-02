"""
Tool for booking a health call for a farmer.
"""
import os
from contextlib import nullcontext

from pydantic_ai import RunContext

from agents.deps import FarmerContext
from agents.tools.farmer_animal_backends import create_health_call_api
from app.core.cache import cache
from app.models.ai_call import AISpecies
from app.models.health_call import HealthCallRequestModel, HealthCaseType
from helpers.utils import get_logger

try:
    from langfuse import get_client as get_langfuse_client
except ImportError:
    get_langfuse_client = None

logger = get_logger(__name__)

# One booking per session per 30 min. Also makes this tool idempotent against an
# agent re-run (OSS->managed streaming fallback re-executes tool calls): a second
# invocation in the same session short-circuits instead of double-booking.
HEALTH_CALL_COOLDOWN_TTL = 60 * 30  # 30 minutes
HEALTH_CALL_CACHE_NAMESPACE = "health_call_booked"


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

    # Idempotency / cooldown: one booking per session. Guards against an agent
    # re-run (OSS->managed streaming fallback) re-firing this write tool.
    session_id = ctx.deps.session_id if ctx and ctx.deps else None
    if session_id:
        try:
            existing = await cache.get(session_id, namespace=HEALTH_CALL_CACHE_NAMESPACE)
            if existing:
                logger.info("Health call already booked for session %s, skipping", session_id)
                return (
                    "This session already has an active health call booking. "
                    "Please try again later or contact your society for assistance."
                )
        except Exception as e:
            logger.warning("Failed to check health call cooldown: %s", e)

    _lf = get_langfuse_client() if get_langfuse_client else None
    _health_tool_input = {
        "union_code": union_code,
        "society_code": society_code,
        "farmer_code": farmer_code,
        "species": species.value,
        "case_type": case_type.value,
        "remark": remark,
    }
    _health_tool_obs_ctx = (
        _lf.start_as_current_observation(
            name="health_call_booking",
            as_type="generation",
            input=_health_tool_input,
            metadata={"tool_name": "create_health_call"},
        )
        if _lf
        else nullcontext()
    )

    with _health_tool_obs_ctx as health_tool_obs:
        if _lf:
            try:
                _lf.set_current_trace_io(input=_health_tool_input)
            except Exception:
                pass
        token = os.getenv("PASHUGPT_TOKEN")
        if not token:
            logger.error("PASHUGPT_TOKEN is not set")
            failure_message = "Health call booking failed.\n\nPASHUGPT_TOKEN is not configured."
            if health_tool_obs is not None:
                health_tool_obs.update(output={"agent_response": failure_message})
            if _lf:
                try:
                    _lf.set_current_trace_io(
                        input=_health_tool_input,
                        output={"agent_response": failure_message},
                    )
                except Exception:
                    pass
            return failure_message

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
            failure_message = "Health call booking failed.\n\nUnable to create health call at the moment."
            if health_tool_obs is not None:
                health_tool_obs.update(output={"agent_response": failure_message})
            if _lf:
                try:
                    _lf.set_current_trace_io(
                        input=_health_tool_input,
                        output={"agent_response": failure_message},
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
                    ttl=HEALTH_CALL_COOLDOWN_TTL,
                    namespace=HEALTH_CALL_CACHE_NAMESPACE,
                )
            except Exception as e:
                logger.warning("Failed to set health call cooldown: %s", e)

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
            success_message = f"Health call booked successfully. Ticket number: {ticket_number}"
            if health_tool_obs is not None:
                health_tool_obs.update(output={"agent_response": success_message})
            if _lf:
                try:
                    _lf.set_current_trace_io(
                        input=_health_tool_input,
                        output={"agent_response": success_message},
                    )
                except Exception:
                    pass
            return success_message
        success_message = "Health call booked successfully, but ticket number was not returned."
        if health_tool_obs is not None:
            health_tool_obs.update(output={"agent_response": success_message})
        if _lf:
            try:
                _lf.set_current_trace_io(
                    input=_health_tool_input,
                    output={"agent_response": success_message},
                )
            except Exception:
                pass
        return success_message
