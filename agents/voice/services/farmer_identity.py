"""Whether the caller's farmer identity is resolved for this turn, and what
follows from that.

Voice used to render "no farmer record exists" and "we have not established
whether one exists" identically — as an empty farmer block — while still
offering every identity-taking tool. The model, handed a required `farmer_code`
slot and nothing to fill it from, invented one: `MISSING`, `UNKNOWN`, `F12345`,
`UNION_CODE_FROM_CONTEXT`, farmer names in `farmerCode`. Measured 2026-09-01..09-14
on voice-production: create_health_call 20.2%, create_ai_call 10.4%,
get_farmer_milk_collection_details 10.2%. Every observed failure ran on a prompt
with no farmer block, and no booking has ever succeeded from one.

This module is the ONE place that decides the state, because two consumers must
never disagree about it:

  * `prepare_requires_farmer_identity` removes the identity-taking tools from
    the schema the model sees, so inventing a code is not something it can do.
  * `unavailable_capability_lines` tells the model which capabilities are off
    this turn and why, so it says something true instead of "that feature does
    not exist" or, worse, claiming a booking it never made.

If those were computed separately they would drift, and a prompt that promises a
tool which is not in the schema is worse than either failure alone.

See issue #282.
"""
import re
from typing import Literal, Optional, Sequence

from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition

from agents.deps import FarmerContext
from agents.voice.models.farmer import FarmerDataEnvelope
from helpers.utils import get_logger

logger = get_logger(__name__)

# found      — the caller has at least one farmer record we can act on.
# not_found  — upstream confirmed this mobile has no farmer record. Terminal for
#              the call: the caller must register with their milk society.
# unresolved — we do not know. The cold fetch timed out, a concurrent refresh
#              held the lock, or the read failed. Recoverable: a later turn may
#              well resolve it, so the caller is told to retry, never that the
#              service does not exist.
# anonymous  — there is no usable mobile number for this turn, so no lookup was
#              even attempted. Also terminal, but for a different reason than
#              not_found and with a different next action: call from the
#              registered number. Previously folded into "unresolved", which told
#              a caller with no number on file to "try again shortly" — advice
#              that cannot work, because nothing about the next turn differs.
FarmerIdentityState = Literal["found", "not_found", "unresolved", "anonymous"]

FOUND: FarmerIdentityState = "found"
NOT_FOUND: FarmerIdentityState = "not_found"
UNRESOLVED: FarmerIdentityState = "unresolved"
ANONYMOUS: FarmerIdentityState = "anonymous"

# The exact sentence to speak when a caller asks for a gated service and we have
# no profile. One per state, because the three cases give the caller three
# different next actions and getting that wrong is its own harm: telling a
# registered farmer whose fetch timed out that they are "not registered" is the
# bug this file exists to prevent, in reverse.
#
# Measured on voice-production 2026-09-26: of 173 turns that carried the
# not_found context lines, only 12 (7%) relayed anything about registration to
# the caller. Two of those callers were given textbook advice from
# search_documents instead and hung up believing the booking had been placed.
# Naming what is unavailable is not enough — the model needs a sentence it can
# say. Shape follows the micro-loan line already in the prompt ("I don't have X,
# so I can't Y; please Z") and stays short because it is spoken aloud through
# Gujarati output translation.
NO_PROFILE_SPOKEN_LINES: dict = {
    NOT_FOUND: (
        "Your mobile number is not registered with us, so I cannot book a visit "
        "or look up your records; please contact your milk society to get registered."
    ),
    UNRESOLVED: (
        "I cannot fetch your details right now, so I cannot book a visit or look "
        "up your records; please try again shortly."
    ),
    ANONYMOUS: (
        "I do not have your registered mobile number, so I cannot book a visit or "
        "look up your records; please call from the number registered with your "
        "milk society."
    ),
}


def no_profile_spoken_line(state: FarmerIdentityState) -> Optional[str]:
    """The sentence to say for this state, or None when a profile was resolved."""
    return NO_PROFILE_SPOKEN_LINES.get(state)


# Named in the context block so the model can tell the caller what it cannot do
# this turn. Kept in sync with IDENTITY_GATED_TOOLS below by
# test_farmer_identity_gate.py.
_GATED_CAPABILITY_NAMES = (
    "AI call booking",
    "health call booking",
    "milk collection lookup",
)

# The tools whose parameters are farmer identity codes. Any tool added here must
# also carry `prepare=prepare_requires_farmer_identity` in agents/tools/__init__.py.
IDENTITY_GATED_TOOLS = (
    "create_ai_call",
    "create_health_call",
    "get_farmer_milk_collection_details",
)


def identity_state_for_envelope(
    envelope: Optional[FarmerDataEnvelope],
) -> FarmerIdentityState:
    """Derive the turn's identity state from the farmer-cache read.

    A bare None is the signal that the lookup did not complete: both
    `refresh_farmer_data_bounded` (cold-fetch timeout) and `refresh_farmer_data`
    (refresh lock still held) return None rather than an envelope. That is
    "unresolved", never "not_found" — telling a registered farmer their number
    is unregistered because our fetch was slow is its own bug.
    """
    if envelope is None:
        return UNRESOLVED
    if envelope.farmers:
        return FOUND
    if envelope.lookupStatus == "not_found":
        return NOT_FOUND
    # An empty envelope with no explicit verdict: we cannot claim upstream said
    # "no such farmer", so treat it as not-yet-known.
    return UNRESOLVED


# This repo's FarmerContext has one identity field, chat's farmer_profile_status.
# The states are the same four; voice's "unresolved" is its "unavailable".
_STATE_BY_PROFILE_STATUS: dict = {
    "found": FOUND,
    "not_found": NOT_FOUND,
    "unavailable": UNRESOLVED,
    "anonymous": ANONYMOUS,
}


def identity_state_for_deps(deps: Optional[FarmerContext]) -> FarmerIdentityState:
    """Read the state off the per-turn deps, defaulting to unresolved.

    Tests and non-voice callers build a FarmerContext without the field, and an
    absent state must fail closed — the whole point is that we do not act on an
    identity we cannot see.
    """
    if deps is None:
        return UNRESOLVED
    state = _STATE_BY_PROFILE_STATUS.get(getattr(deps, "farmer_profile_status", None))
    if state is not None:
        return state
    return UNRESOLVED


def has_usable_farmer_identity(deps: Optional[FarmerContext]) -> bool:
    """True only when this turn can act on a real, known farmer account.

    Delegates to FarmerContext.has_usable_farmer_identity, which owns the
    predicate. The fallback path exists only for the SimpleNamespace deps stubs
    in the tool tests, and applies the same rule.
    """
    if deps is None:
        return False
    checker = getattr(deps, "has_usable_farmer_identity", None)
    if callable(checker):
        return bool(checker())
    if identity_state_for_deps(deps) != FOUND:
        return False
    return bool(getattr(deps, "farmer_accounts", None))


async def prepare_requires_farmer_identity(
    ctx: RunContext[FarmerContext], tool_def: ToolDefinition
) -> Optional[ToolDefinition]:
    """Withhold an identity-taking tool on a turn with no usable identity.

    Returning None drops the tool from the schema for this run, so the model
    cannot call it. This is deliberately not a prompt instruction: the prompt
    already tells the model to say the details are unavailable
    (assets/prompts/voice_system_translation_pipeline_en.md, and the
    create_ai_call docstring), and that instruction is what the measured 10-20%
    failure rate ignores. An instruction is a request; an absent tool is not.
    """
    if has_usable_farmer_identity(ctx.deps if ctx else None):
        return tool_def
    logger.info(
        "Hiding %s: farmer identity state=%s accounts=%s",
        tool_def.name,
        identity_state_for_deps(ctx.deps if ctx else None),
        len(getattr(ctx.deps, "farmer_accounts", []) or []) if ctx and ctx.deps else 0,
    )
    return None


def unavailable_capability_lines(state: FarmerIdentityState) -> list[str]:
    """Context lines naming what is off this turn and the exact line to speak.

    Phrased as availability status rather than prohibition. "Do not book" implies
    the model still could, which invites it to try or to apologise oddly for a
    tool it cannot see; "unavailable this turn" just hands it a fact to relay.

    The negative states give the caller DIFFERENT next actions — register, retry,
    or call from the registered number — which is why the state is four-valued
    rather than a boolean.

    The quoted sentence is the part that changes behaviour. Withholding the tools
    stopped the invented codes, but it left the model with a capability it could
    not name: it fell back on general advice from `search_documents` and the
    caller rang off with no idea why nothing had been booked. A status line is
    something to reason about; a quoted sentence is something to say.
    """
    if state == FOUND:
        return []
    capabilities = ", ".join(_GATED_CAPABILITY_NAMES)
    if state == NOT_FOUND:
        reason = "no farmer record is registered for this mobile number."
    elif state == ANONYMOUS:
        reason = "no registered mobile number is available for this call."
    else:
        reason = "could not be loaded for this turn."
    lines = [
        f"- Farmer identity: {reason}",
        f"- Unavailable for this turn: {capabilities}.",
        "- If the caller asks for any of those — an AI / beech daan visit, a vet "
        "visit, their milk records — say this and nothing more about it: "
        f'"{NO_PROFILE_SPOKEN_LINES[state]}"',
        "- Do not answer a booking request with general advice instead, and do "
        "not end the turn without saying that line.",
        "- Do not ask the caller for union, society, farmer or technician codes, "
        "and do not guess or construct them.",
    ]
    if state == UNRESOLVED:
        lines.append(
            "- Do not tell the caller they are unregistered. Do not say the "
            "service does not exist."
        )
    return lines


def identity_tool_groups(deps: Optional[FarmerContext]) -> list[str]:
    """Tool groups actually available this turn, for the runtime context line.

    Derived from the same predicate the gates use, so the line can never claim a
    tool group the model has not been given. It previously hardcoded "booking",
    advertising it on precisely the turns where booking was impossible.
    """
    groups = ["retrieval"]
    if has_usable_farmer_identity(deps):
        groups.append("booking")
    # Unchanged condition. This group's tools (get_union_scheme_data,
    # get_farmer_bonus_amount) carry their own prepare gates and resolve their
    # codes server-side from farmer_accounts, so they are not part of the
    # invented-identifier class and must not be withheld here — gating them on
    # identity too would make this line understate what the model was given.
    if getattr(deps, "signed_in", False) and getattr(deps, "mobile", None):
        groups.append("signed-in-farmer-data")
    return groups


# Shape of a real identity code. Codes are NOT always numeric — M001 and NA4192
# book fine — so the shape alone is permissive, and on its own it accepts
# MISSING, UNKNOWN, PLACEHOLDER, NA and None. Validated against 9,945 successful
# prod bookings (30d to 2026-09-08).
CODE_PATTERN = re.compile(r"^[A-Za-z0-9/-]{1,12}$")
# Every real code carries at least one digit. Verified against 28,089 successful
# AI bookings (22 unions, 2,032 societies, 2,548 farmer codes) and 110 health
# calls over 90 days: zero exceptions. This is what separates MISSING / UNKNOWN /
# NA / None from M001 and NA4192, and it is why the dbc2d23 guard only worked for
# create_ai_call — the technician-id check did the real work there, and health
# call, which has no technician id, kept leaking at 20%.
_HAS_DIGIT = re.compile(r"[0-9]")


def invalid_identity_code_field(
    union_code: str,
    society_code: str,
    farmer_code: str,
    accounts: "Optional[Sequence[object]]" = None,
) -> Optional[str]:
    """Name of the first supplied code that cannot be real, else None.

    A backstop, not the fix. By the time this runs the model has already decided
    to act on an identity it does not have; the gate above the tool layer is what
    prevents that. This exists so a tool reached by some other path — a fallback
    re-run, a future caller, a gate someone forgets to attach — still cannot
    forward an invented code to the partner API.

    When the caller's own accounts are known, the triple must be one of them:
    the model is told to copy these out of the farmer context, so a triple that
    is not in that context did not come from it. That catches the digit-bearing
    inventions (`F12345`, `U11223`) the shape and digit rules let through.
    """
    for field, value in (
        ("union_code", union_code),
        ("society_code", society_code),
        ("farmer_code", farmer_code),
    ):
        cleaned = (value or "").strip()
        if not CODE_PATTERN.match(cleaned) or not _HAS_DIGIT.search(cleaned):
            return field
    if accounts and not codes_match_known_account(
        accounts, union_code, society_code, farmer_code
    ):
        return "codes_not_in_context"
    return None


def codes_match_known_account(
    accounts: "Sequence[object]",
    union_code: str,
    society_code: str,
    farmer_code: str,
) -> bool:
    """True when the supplied triple is one of the caller's own accounts.

    The strongest available check on a model-supplied code, and structural rather
    than cosmetic: the model is told to copy these out of the farmer context, so
    a triple that is not in that context did not come from it.

    With no accounts in context there is nothing to check against, so this
    returns False — callers pair it with the state gate rather than using it
    alone.
    """
    if not accounts:
        return False
    supplied = (
        str(union_code or "").strip(),
        str(society_code or "").strip(),
        str(farmer_code or "").strip(),
    )
    known = {
        (
            str(getattr(a, "union_code", "") or "").strip(),
            str(getattr(a, "society_code", "") or "").strip(),
            str(getattr(a, "farmer_code", "") or "").strip(),
        )
        for a in accounts
    }
    return supplied in known
