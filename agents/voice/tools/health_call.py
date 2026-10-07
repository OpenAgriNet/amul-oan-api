"""
Tool for booking a health call for a farmer.
"""
from pydantic_ai import RunContext

from agents.deps import FarmerContext
from agents.tools.health_call import _book_health_via_network
from agents.voice.models.ai_call import AISpecies
from agents.voice.models.health_call import HealthCaseType
from agents.voice.services.farmer_identity import invalid_identity_code_field
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

    # Chat's Beckn booking, with its one booking per session.
    return await _book_health_via_network(
        union_code=union_code,
        society_code=society_code,
        farmer_code=farmer_code,
        species=species,
        case_type=case_type,
        remark=remark,
        session_id=session_id,
        tool_call_id=getattr(ctx, "tool_call_id", None),
        tool_input={
            "union_code": union_code,
            "society_code": society_code,
            "farmer_code": farmer_code,
            "species": species.value,
            "case_type": case_type.value,
            "remark": remark,
        },
    )
