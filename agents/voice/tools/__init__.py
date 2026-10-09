"""
Tools for the Sunbird VA API.
"""
import functools
import inspect

from pydantic_ai import Tool
from agents.voice.tools.milk_collection import get_farmer_milk_collection_details
from agents.voice.tools.terms import search_terms
from agents.voice.tools.search import search_documents
from agents.voice.tools.vet_offices import find_nearby_vet_offices
from agents.voice.tools.ai_call import create_ai_call
from agents.voice.tools.conversation_state import signal_conversation_state
from agents.voice.tools.farmer_cached import get_farmer_profile, get_herd_summary, list_animal_tags
from agents.voice.tools.common import fire_tool_call_nudge
from agents.voice.tools.union_schemes import get_union_scheme_data, prepare_get_union_scheme_data
from agents.voice.tools.loan import check_loan_eligibility, prepare_check_loan_eligibility
from agents.voice.tools.bonus import get_farmer_bonus_amount, prepare_get_farmer_bonus_amount
from agents.voice.services.farmer_identity import prepare_requires_farmer_identity


def _with_nudge_signal(func):
    """Wrap a tool function so it fires the nudge event on invocation."""
    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            fire_tool_call_nudge()
            return await func(*args, **kwargs)
        return wrapper
    else:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            fire_tool_call_nudge()
            return func(*args, **kwargs)
        return wrapper


BASE_TOOLS = [
    Tool(
        _with_nudge_signal(search_terms),
        takes_ctx=False,
    ),
    Tool(
        _with_nudge_signal(search_documents),
        takes_ctx=True,
    ),
    # The identity-taking tools. `prepare` withholds each one on a turn
    # whose farmer identity is unresolved: with no farmer block to copy codes
    # from, the model invented them (`MISSING`, `F12345`, farmer names in
    # `farmerCode`) rather than stopping. Telling it not to is already in the
    # prompt and is what the measured 10-20% failure rate ignores — the tool has
    # to be absent, not discouraged. See issue #282.
    Tool(
        _with_nudge_signal(create_ai_call),
        takes_ctx=True,
        docstring_format='auto',
        require_parameter_descriptions=True,
        prepare=prepare_requires_farmer_identity,
    ),
    Tool(
        _with_nudge_signal(get_farmer_milk_collection_details),
        takes_ctx=True,
        docstring_format='auto',
        require_parameter_descriptions=True,
        prepare=prepare_requires_farmer_identity,
    ),
    Tool(
        signal_conversation_state,
        takes_ctx=True,
        docstring_format='auto',
    ),
    # Not identity-gated: the lookup takes no farmer codes, and a caller whose
    # record never resolved can still be placed by the taluka they say out loud.
    Tool(
        _with_nudge_signal(find_nearby_vet_offices),
        takes_ctx=True,
        docstring_format='auto',
        require_parameter_descriptions=True,
    ),
    Tool(
        _with_nudge_signal(check_loan_eligibility),
        takes_ctx=True,
        docstring_format='auto',
        prepare=prepare_check_loan_eligibility,  # hidden unless feature on + caller phone resolved
    ),
]

SIGNED_IN_FARMER_TOOLS = [
    # # Get Farmer Profile (disabled — redundant with _build_compact_farmer_summary
    # # context; the brittle farmers[0]-only read returned "not available" when
    # # the cached record was missing fields, while context already had them).
    # Tool(
    #     _with_nudge_signal(get_farmer_profile),
    #     takes_ctx=True,
    #     docstring_format='auto',
    #     require_parameter_descriptions=False,
    # ),

    # # Get Herd Summary (disabled — same cache as context; root of the Turn-A
    # # "I don't have your herd info" failure. Counts now always surfaced in
    # # the runtime context with len(tags) fallback, matching chat's pattern).
    # Tool(
    #     _with_nudge_signal(get_herd_summary),
    #     takes_ctx=True,
    #     docstring_format='auto',
    #     require_parameter_descriptions=False,
    # ),

    # # List Animal Tags (disabled — runtime context now lists all tags inline
    # # (no truncation), so a tool round-trip is no longer needed).
    # Tool(
    #     _with_nudge_signal(list_animal_tags),
    #     takes_ctx=True,
    #     docstring_format='auto',
    #     require_parameter_descriptions=False,
    # ),

    Tool(
        _with_nudge_signal(get_union_scheme_data),
        takes_ctx=True,
        docstring_format='auto',
        require_parameter_descriptions=False,
        prepare=prepare_get_union_scheme_data,
    ),
    Tool(
        _with_nudge_signal(get_farmer_bonus_amount),
        takes_ctx=True,
        docstring_format='auto',
        require_parameter_descriptions=False,
        prepare=prepare_get_farmer_bonus_amount,
    ),
]

TOOLS = BASE_TOOLS
