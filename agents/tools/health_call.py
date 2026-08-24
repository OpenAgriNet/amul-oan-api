"""
Tool for booking a health call for a farmer.
"""
import os

import httpx
from pydantic_ai import RunContext

from agents.deps import FarmerContext
from agents.tools.farmer_animal_backends import create_health_call_api
from app.config import settings
from app.core.cache import cache, reserve, ReservationOutcome, release_reservation
from app.models.ai_call import AISpecies
from app.models.health_call import HealthCallRequestModel, HealthCaseType
from app.observability import start_observation
from helpers.utils import get_logger

logger = get_logger(__name__)

# One booking per session per 30 min. Also makes this tool idempotent against an
# agent re-run (OSS->managed streaming fallback re-executes tool calls): a second
# invocation in the same session short-circuits instead of double-booking.
HEALTH_CALL_COOLDOWN_TTL = settings.health_call_cooldown_ttl_seconds
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
    logger.info(
        "Create health call tool invoked for union=%s society=%s farmer=%s species=%s case_type=%s",
        union_code,
        society_code,
        farmer_code,
        species.value,
        case_type.value,
    )

    # Per-session id for the atomic booking reservation (placed just before the
    # write call below).
    session_id = ctx.deps.session_id if ctx and ctx.deps else None

    # A booking is IRREVERSIBLE, so block on the moderation verdict before writing.
    # On voice, moderation runs concurrently with the agent; this refuses the
    # booking if the query was rejected. No-op on chat (no moderation task attached
    # → returns True), so chat behaviour is unchanged. See create_ai_call.
    if not await ctx.deps.ensure_in_scope():
        logger.info("Health call blocked: query failed moderation; session=%s", session_id)
        return "This helpline only handles dairy farming and animal husbandry questions."

    _health_tool_input = {
        "union_code": union_code,
        "society_code": society_code,
        "farmer_code": farmer_code,
        "species": species.value,
        "case_type": case_type.value,
        "remark": remark,
    }

    if settings.enable_network and settings.beckn_callback_transactions_enabled:
        return await _book_health_via_network(
            union_code=union_code,
            society_code=society_code,
            farmer_code=farmer_code,
            species=species,
            case_type=case_type,
            remark=remark,
            session_id=session_id,
            tool_call_id=getattr(ctx, "tool_call_id", None),
            tool_input=_health_tool_input,
        )

    with start_observation(
        "health_call_booking",
        as_type="generation",
        input=_health_tool_input,
        metadata={"tool_name": "create_health_call"},
    ) as health_tool_obs:
        token = os.getenv("PASHUGPT_TOKEN")
        if not token:
            logger.error("PASHUGPT_TOKEN is not set")
            failure_message = "Health call booking failed.\n\nPASHUGPT_TOKEN is not configured."
            if health_tool_obs is not None:
                health_tool_obs.update(output={"agent_response": failure_message})
            return failure_message

        request = HealthCallRequestModel(
            unionCode=union_code,
            societyCode=society_code,
            farmerCode=farmer_code,
            species=species,
            caseType=case_type,
            remark=remark,
        )

        # Atomic reservation immediately before the write: first caller wins; a
        # concurrent/duplicate submit OR a fallback re-run for the same session
        # short-circuits instead of double-booking (Redis SET NX, shared across
        # containers). Released below if the booking API itself fails.
        _reserved = False
        if session_id:
            reservation = await reserve(
                session_id, HEALTH_CALL_CACHE_NAMESPACE, HEALTH_CALL_COOLDOWN_TTL
            )
            if reservation is ReservationOutcome.TAKEN:
                logger.info("Health call already booked/in-flight for session %s, skipping", session_id)
                return (
                    "This session already has an active health call booking. "
                    "Please try again later or contact your society for assistance."
                )
            # A fail-open reservation is not ours to release later.
            _reserved = reservation is ReservationOutcome.ACQUIRED

        response = await create_health_call_api(request, token)
        if response is None:
            if _reserved:
                await release_reservation(session_id, HEALTH_CALL_CACHE_NAMESPACE)
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
            return success_message
        success_message = "Health call booked successfully, but ticket number was not returned."
        if health_tool_obs is not None:
            health_tool_obs.update(output={"agent_response": success_message})
        return success_message


def _is_provably_pre_send(exc: BaseException) -> bool:
    return isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.UnsupportedProtocol,
            httpx.InvalidURL,
        ),
    )


async def _book_health_via_network(
    *,
    union_code: str,
    society_code: str,
    farmer_code: str,
    species: AISpecies,
    case_type: HealthCaseType,
    remark: str | None,
    session_id: str | None,
    tool_call_id: str | None,
    tool_input: dict,
) -> str:
    """Book a health visit through directed confirm/on_confirm.

    The feature is default-off until ONIX and the provider BPP routes exist.
    Ambiguous timeouts retain the reservation because a late callback may still
    prove that the upstream created a real visit.
    """
    from agents.tools.beckn_network import network_create_health_call_result

    with start_observation(
        "health_call_booking",
        as_type="generation",
        input=tool_input,
        metadata={"tool_name": "create_health_call", "route": "beckn_callback"},
    ) as health_tool_obs:
        owned = False
        if session_id:
            reservation = await reserve(
                session_id, HEALTH_CALL_CACHE_NAMESPACE, HEALTH_CALL_COOLDOWN_TTL
            )
            if reservation is ReservationOutcome.TAKEN:
                return (
                    "This session already has an active health call booking. "
                    "Please try again later or contact your society for assistance."
                )
            owned = reservation is ReservationOutcome.ACQUIRED

        try:
            result = await network_create_health_call_result(
                union_code,
                society_code,
                farmer_code,
                species.value,
                case_type.value,
                remark,
                session_id=session_id,
                tool_call_id=tool_call_id,
            )
        except Exception as exc:
            pre_send = _is_provably_pre_send(exc)
            if owned and pre_send and session_id:
                await release_reservation(session_id, HEALTH_CALL_CACHE_NAMESPACE)
            logger.warning(
                "Network health call errored session=%s pre_send=%s: %r",
                session_id,
                pre_send,
                exc,
            )
            message = (
                "Health call booking failed because the booking network could not be reached."
                if pre_send
                else "Health call booking is unconfirmed. Please check with your society before trying again."
            )
            if health_tool_obs is not None:
                health_tool_obs.update(output={"success": False, "pre_send_failure": pre_send, "message": message})
            return message

        if not result.ok:
            if owned and result.authoritative_no_booking and session_id:
                await release_reservation(session_id, HEALTH_CALL_CACHE_NAMESPACE)
            if health_tool_obs is not None:
                health_tool_obs.update(
                    output={
                        "success": False,
                        "authoritative_no_booking": result.authoritative_no_booking,
                        "message": result.message,
                    }
                )
            return result.message

        if session_id:
            try:
                await cache.set(
                    session_id,
                    {"ticket": result.ticket, "species": species.value},
                    ttl=HEALTH_CALL_COOLDOWN_TTL,
                    namespace=HEALTH_CALL_CACHE_NAMESPACE,
                )
            except Exception as exc:
                logger.warning("Failed to set health call cooldown: %s", exc)
        if health_tool_obs is not None:
            health_tool_obs.update(output={"success": True, "ticket_number": result.ticket, "message": result.message})
        return result.message
