"""
Tool for booking a health call for a farmer.
"""
import os

from pydantic_ai import RunContext

from agents.deps import FarmerContext
from agents.voice.models.ai_call import AISpecies
from agents.voice.models.health_call import HealthCallRequestModel, HealthCaseType
from agents.voice.services.farmer_identity import invalid_identity_code_field
from agents.voice.tools.farmer_animal_backends import create_health_call_api
from app.core.cache import cache, try_reserve, release_reservation
from helpers.utils import get_logger

logger = get_logger(__name__)

# Health call had no identifier validation at all, which is why it was the worst
# of the three tools: 20.2% of voice calls carried invented codes, against 10.4%
# for create_ai_call. The dbc2d23 guard covered AI booking only, and what did the
# real work there was its technician-id check — health call has no technician id,
# so nothing stopped `MISSING` or `F12345` reaching the partner API. See #282.
INVALID_IDENTIFIERS_MESSAGE = (
    "Health call booking failed. The farmer details are not available."
)

# One booking per session per 30 min (mirrors create_ai_call). Also makes this
# tool idempotent against an agent re-run (OSS->managed streaming fallback).
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

    # Moderation runs concurrently with the agent, so this booking write must
    # block on the verdict: a rejected query must never create a real booking.
    if not await ctx.deps.ensure_in_scope():
        logger.info("Health call blocked: query failed moderation; session=%s", session_id)
        return "This helpline only handles dairy farming and animal husbandry questions."

    # Backstop. The tool is withheld entirely on a turn with no resolved identity
    # (agents.voice.services.farmer_identity); this catches a call that reaches the tool
    # by some other path, before it becomes a partner write.
    invalid_field = invalid_identity_code_field(
        union_code,
        society_code,
        farmer_code,
        getattr(ctx.deps, "farmer_accounts", None) if ctx and ctx.deps else None,
    )
    if invalid_field is not None:
        logger.warning(
            "Health call blocked: invalid %s; session=%s union=%s society=%s farmer=%s",
            invalid_field, session_id, union_code, society_code, farmer_code,
        )
        return INVALID_IDENTIFIERS_MESSAGE

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

    # Atomic reservation immediately before the write: first caller wins; a
    # concurrent/duplicate submit OR a fallback re-run for the same session
    # short-circuits instead of double-booking (Redis SET NX, shared across
    # containers). Released below if the booking API itself fails.
    _reserved = False
    if session_id:
        if not await try_reserve(session_id, HEALTH_CALL_CACHE_NAMESPACE, HEALTH_CALL_COOLDOWN_TTL):
            logger.info("Health call already booked/in-flight for session %s, skipping", session_id)
            return (
                "This session already has an active health call booking. "
                "Please try again later or contact your society for assistance."
            )
        _reserved = True

    response = await create_health_call_api(request, token)
    if response is None:
        if _reserved:
            await release_reservation(session_id, HEALTH_CALL_CACHE_NAMESPACE)
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
        "Health call booked: session=%s ticket=%s",
        session_id,
        ticket_number,
    )
    if ticket_number:
        return f"Health call booked successfully. Ticket number: {ticket_number}"
    return "Health call booked successfully, but ticket number was not returned."
