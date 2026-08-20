"""
Tool for booking an artificial insemination call for a farmer.
"""
import json
import os

from pydantic_ai import RunContext

from agents.deps import FarmerContext
from agents.tools.farmer_animal_backends import create_ai_call_api
from app.config import settings
from app.core.cache import cache, reserve, ReservationOutcome, release_reservation
from app.models.ai_call import AICallRequestModel, AISpecies
from app.models.union import any_union_banned_from_ai_calls, union_banned_message
from app.observability import start_observation
from helpers.utils import get_logger

logger = get_logger(__name__)

# Redis key namespace for booking reservations. Deliberately NOT config: changing
# it at runtime orphans every in-flight reservation, so a redeploy mid-booking
# would lose the protection it exists to give.
AI_CALL_CACHE_NAMESPACE = "ai_call_booked"

ALREADY_BOOKED_MESSAGE = (
    "This session already has an active artificial insemination booking. "
    "Please try again later or contact your society for assistance."
)
OUT_OF_SCOPE_MESSAGE = "This helpline only handles dairy farming and animal husbandry questions."


async def _reserve_booking_slot(session_id: str | None) -> tuple[bool, bool]:
    """Atomic per-session reservation, taken immediately before a write.

    First caller wins; a concurrent submit OR a fallback re-run for the same
    session short-circuits instead of double-booking. Returns
    ``(allowed, owned)``: ``allowed`` False means refuse the booking, ``owned``
    True means we genuinely wrote the reservation key and may release it if the
    booking itself fails.

    ``owned`` is deliberately narrower than ``allowed``. `reserve` fails OPEN:
    if Redis is unavailable it lets the booking through without holding
    anything. Recording that as "reserved" made a later failure delete a key we
    never wrote — quite possibly this session's marker from an EARLIER
    successful booking — voiding the guard for the rest of its TTL. The
    fail-open itself is unchanged (booking proceeds when the cache is down);
    only the bogus release is gone.

    Flag-gated (see the trade-off note in create_ai_call). With the guard off,
    or with no session id, this is a no-op that allows the booking.
    """
    if not (settings.ai_call_booking_guard_enabled and session_id):
        return True, False
    outcome = await reserve(session_id, AI_CALL_CACHE_NAMESPACE, settings.ai_call_cooldown_ttl_seconds)
    if outcome is ReservationOutcome.TAKEN:
        logger.info("AI call already booked/in-flight for session %s, skipping", session_id)
        return False, False
    if outcome is ReservationOutcome.UNGUARDED:
        logger.warning(
            "AI call proceeding UNGUARDED for session %s (cache unavailable); "
            "no reservation held, so none will be released",
            session_id,
        )
        return True, False
    return True, True


async def _mark_session_booked(session_id: str | None, ticket: str | None, species_value: str) -> None:
    """Mark this session as booked so a re-run (or retry) does not double-book."""
    if settings.ai_call_booking_guard_enabled and session_id:
        try:
            await cache.set(
                session_id,
                {"ticket": ticket, "species": species_value},
                ttl=settings.ai_call_cooldown_ttl_seconds,
                namespace=AI_CALL_CACHE_NAMESPACE,
            )
        except Exception as e:
            logger.warning("Failed to set AI call cooldown: %s", e)


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
    Kindly contact your Milk Society to book the service. Do not ask which technician and do not book.

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

    # Booking idempotency is a PRODUCT TRADE-OFF, so it is a config flag rather
    # than a code decision.
    #
    # OFF (default): a farmer can book multiple AI visits in one session, including
    # the same species with the same technician (two cows in heat is a real case).
    # Cost: an OSS->managed fallback re-run can re-fire this tool and duplicate the
    # booking — accepted, with the upstream CreateAICall API as the backstop.
    #
    # ON: first caller wins per session (Redis SET NX, shared across containers),
    # so no duplicate booking and no duplicate SMS. Cost: a legitimate second
    # booking inside the TTL is refused. amul-prod has historically run this way.
    #
    # See health_call.py, which keeps an unconditional guard for a different contract.

    # A booking is IRREVERSIBLE, so block on the moderation verdict before writing.
    # On the voice path moderation runs concurrently with the agent; this refuses
    # the booking if the query was rejected. No-op on the chat path (no moderation
    # task attached → returns True), so chat behaviour is unchanged.
    if not await ctx.deps.ensure_in_scope():
        logger.info("AI call blocked: query failed moderation; session=%s", session_id)
        return OUT_OF_SCOPE_MESSAGE

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
        lang_code = getattr(ctx.deps, "lang_code", None) if ctx and ctx.deps else None
        return union_banned_message(lang_code)

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
        token = os.getenv("PASHUGPT_TOKEN")
        if not token:
            logger.error("PASHUGPT_TOKEN is not set")
            failure_message = (
                "Artificial insemination call booking failed.\n\n"
                "PASHUGPT_TOKEN is not configured."
            )
            if ai_tool_obs is not None:
                ai_tool_obs.update(output={"success": False, "message": failure_message})
            return failure_message

        request = AICallRequestModel(
            unionCode=union_code,
            societyCode=society_code,
            farmerCode=farmer_code,
            userId=user_id,
            species=species,
        )
        # Atomic reservation immediately before the write: first caller wins; a
        # concurrent submit OR a fallback re-run for the same session short-circuits
        # instead of double-booking. Released below if the booking API itself
        # fails AND we actually hold the reservation (see _reserve_booking_slot).
        _allowed, _owned = await _reserve_booking_slot(session_id)
        if not _allowed:
            return ALREADY_BOOKED_MESSAGE

        response = await create_ai_call_api(request, token)
        if response is None:
            if _owned:
                await release_reservation(session_id, AI_CALL_CACHE_NAMESPACE)
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
            return failure_message

        # Mark this session as booked so a re-run (or retry) does not double-book.
        await _mark_session_booked(session_id, response.ticket_number, species.value)

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
        return success_message
