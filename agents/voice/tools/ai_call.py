"""
Tool for booking an artificial insemination (beech daan) call for a farmer.
One booking per session (30-min cooldown via Redis).
"""
import json
import os
import re
from typing import Optional

from pydantic_ai import RunContext

from agents.deps import FarmerContext
from agents.voice.models.ai_call import AICallRequestModel, AISpecies
from agents.voice.tools.farmer_animal_backends import create_ai_call_api
from app.core.cache import cache, try_reserve, release_reservation
from agents.voice.services.farmer_identity import (
    CODE_PATTERN,
    invalid_identity_code_field,
)
from app.voice.models.union import UNION_BANNED_MESSAGE, any_union_banned_from_ai_calls
from helpers.utils import get_logger

logger = get_logger(__name__)

AI_CALL_COOLDOWN_TTL = 60 * 30  # 30 minutes
AI_CALL_CACHE_NAMESPACE = "ai_call_booked"

# With no farmer/technician context the model does not stop — it invents
# identifiers and books anyway (U11223/S67890/F12345/T55667,
# UNION_CODE_FROM_CONTEXT, farmer names in farmerCode), which the partner API
# always 500s. Patterns validated on 9,945 successful prod bookings (30d to
# 2026-09-08): every real code matches _CODE_PATTERN — they are NOT always
# numeric, M001 and NA4192 book fine — and every real technician id is 24
# base64 chars ending "==".
_CODE_PATTERN = CODE_PATTERN  # canonical rule lives in agents.voice.services.farmer_identity
_TECHNICIAN_ID_PATTERN = re.compile(r"^[A-Za-z0-9+/]{22}==$")
INVALID_IDENTIFIERS_MESSAGE = (
    "Artificial insemination call booking failed. "
    "The farmer or technician details are not available."
)


def _invalid_booking_identifier(
    union_code: str,
    society_code: str,
    farmer_code: str,
    user_id: str,
    accounts=None,
) -> Optional[str]:
    """Name of the first identifier that cannot be real, else None.

    The code rules are shared with create_health_call so the two cannot drift;
    only the technician id is specific to AI booking. `accounts` cross-checks the
    triple against the caller's own accounts when they are known.
    """
    invalid_field = invalid_identity_code_field(
        union_code, society_code, farmer_code, accounts
    )
    if invalid_field is not None:
        return invalid_field
    if not _TECHNICIAN_ID_PATTERN.match((user_id or "").strip()):
        return "user_id"
    return None


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
    If Farmer Profile says AI call booking is not allowed for this union, tell the farmer
    exactly: Kindly contact your Milk Society to book the service. Do not ask which
    technician and do not book.

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

    # Moderation runs concurrently with the agent, so this booking write must
    # block on the verdict: a rejected query must never create a real booking.
    if not await ctx.deps.ensure_in_scope():
        logger.info("AI call blocked: query failed moderation; session=%s", session_id)
        return "This helpline only handles dairy farming and animal husbandry questions."

    # Union ban is a policy gate, not a booking write: refuse before Redis
    # reservation and before PashuGPT. farmer_unions may be missing on test
    # stubs and on unsigned-in turns — those are not banned.
    farmer_unions = getattr(ctx.deps, "farmer_unions", []) if ctx and ctx.deps else []
    if any_union_banned_from_ai_calls(farmer_unions):
        logger.info(
            "AI call blocked: union banned from AI-call booking unions=%s session=%s",
            farmer_unions,
            session_id,
        )
        return UNION_BANNED_MESSAGE

    invalid_field = _invalid_booking_identifier(
        union_code,
        society_code,
        farmer_code,
        user_id,
        getattr(ctx.deps, "farmer_accounts", None) if ctx and ctx.deps else None,
    )
    if invalid_field is not None:
        logger.warning(
            "AI call blocked: invalid %s; session=%s union=%s society=%s farmer=%s user_id=%s",
            invalid_field, session_id, union_code, society_code, farmer_code, user_id,
        )
        return INVALID_IDENTIFIERS_MESSAGE

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

    # Atomic reservation immediately before the write: first caller wins; a
    # concurrent/duplicate submit OR a fallback re-run for the same session
    # short-circuits instead of double-booking (Redis SET NX, shared across
    # containers). Released below if the booking API itself fails.
    _reserved = False
    if session_id:
        if not await try_reserve(session_id, AI_CALL_CACHE_NAMESPACE, AI_CALL_COOLDOWN_TTL):
            logger.info("AI call already booked/in-flight for session %s, skipping", session_id)
            return (
                "This session already has an active artificial insemination booking. "
                "Please try again later or contact your society for assistance."
            )
        _reserved = True

    response = await create_ai_call_api(request, token)
    if response is None:
        if _reserved:
            await release_reservation(session_id, AI_CALL_CACHE_NAMESPACE)
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
