"""Tools for the Sunbird VA API."""
import functools
import inspect

from pydantic_ai import Tool
from agents.tools.ai_call import create_ai_call
from agents.tools.health_call import create_health_call
from agents.tools.milk_collection import (
    get_farmer_milk_collection_details,
    get_farmer_milk_collection_details_voice,
    prepare_get_farmer_milk_collection_details,
)
from agents.tools.search import search_documents, search_videos
from agents.tools.terms import search_terms
from agents.tools.union_schemes import get_union_scheme_data, prepare_get_union_scheme_data
from agents.tools.conversation_state import signal_conversation_state
from agents.tools.common import fire_tool_call_nudge
from agents.tools.farmer_cached import get_farmer_profile, get_herd_summary, list_animal_tags
from agents.tools.loan import check_loan_eligibility, prepare_check_loan_eligibility
# from agents.tools.animal import get_animal_by_tag
# from agents.tools.cvcc import get_cvcc_health_details
# from agents.tools.farmer import get_farmer_by_mobile

TOOLS = [
    # # Search Terms
    # Tool(
    #     search_terms,
    #     takes_ctx=False,
    #     docstring_format='auto', 
    #     require_parameter_descriptions=True,

    # ),

    # Search Documents
    Tool(
        search_documents,
        takes_ctx=False, # No context is needed for this tool
        docstring_format='auto', 
        require_parameter_descriptions=True,
    ),

    Tool(
        create_ai_call,
        takes_ctx=True,  # needs ctx.deps.session_id for the booking idempotency guard
        docstring_format='auto',
        require_parameter_descriptions=True,
    ),

    Tool(
        create_health_call,
        takes_ctx=True,  # needs ctx.deps.session_id for the booking idempotency guard
        docstring_format='auto',
        require_parameter_descriptions=True,
    ),
    
    
    Tool(
        get_farmer_milk_collection_details,
        takes_ctx=False,
        docstring_format='auto',
        require_parameter_descriptions=True,
        prepare=prepare_get_farmer_milk_collection_details,  # hide unless a farmer is resolved
    ),

    Tool(
        get_union_scheme_data,
        takes_ctx=True,
        docstring_format='auto',
        require_parameter_descriptions=False,
        prepare=prepare_get_union_scheme_data,
    ),

    Tool(
        check_loan_eligibility,
        takes_ctx=True,
        docstring_format='auto',
        prepare=prepare_check_loan_eligibility,  # hidden unless feature on + caller phone resolved
    ),

    # # Get Animal by Tag (temporarily disabled)
    # Tool(
    #     get_animal_by_tag,
    #     takes_ctx=False,
    #     docstring_format='auto',
    #     require_parameter_descriptions=True,
    # ),

    # # Get CVCC Health Details (temporarily disabled)
    # Tool(
    #     get_cvcc_health_details,
    #     takes_ctx=False,
    #     docstring_format='auto',
    #     require_parameter_descriptions=True,
    # ),

    # # Get Farmer by Mobile Number (temporarily disabled)
    # Tool(
    #     get_farmer_by_mobile,
    #     takes_ctx=False,
    #     docstring_format='auto',
    #     require_parameter_descriptions=True,
    # ),

    # # Search Videos
    # Tool(
    #     search_videos,
    #     takes_ctx=False,
    #     docstring_format='auto', 
    #     require_parameter_descriptions=True,
    # ),

    # # Reverse Geocode - Do we need this?
    # Tool(
    #     reverse_geocode,
    #     takes_ctx=False,
    #     docstring_format='auto', 
    #     require_parameter_descriptions=True,
    # ),

    # # Weather Forecast
    # Tool(
    #     weather_forecast,
    #     takes_ctx=False,
    #     docstring_format='auto', 
    #     require_parameter_descriptions=True,
    # ),

    # # Weather Historical
    # Tool(
    #     weather_historical,
    #     takes_ctx=False,
    #     docstring_format='auto', 
    #     require_parameter_descriptions=True,
    # ),

    # # Mandi Prices
    # Tool(
    #     mandi_prices,
    #     takes_ctx=False,
    #     docstring_format='auto', 
    #     require_parameter_descriptions=True,
    # ),

    # # Agricultural Services (KVK, CHC, etc.)
    # Tool(
    #     agri_services,
    #     takes_ctx=False,
    #     docstring_format='auto', 
    #     require_parameter_descriptions=True,
    # ),
    
    # # Geocode
    # Tool(
    #     forward_geocode,
    #     takes_ctx=False,
    #     docstring_format='auto', 
    #     require_parameter_descriptions=True,
    # ),

    # # Agristack
    # Tool(
    #     fetch_agristack_data,
    #     takes_ctx=True,
    #     docstring_format='auto', 
    #     require_parameter_descriptions=False, # No params are needed for this tool
    # ),
    # # Scheme Codes
    # Tool(
    #     get_scheme_codes,
    #     takes_ctx=False,
    #     docstring_format='auto', 
    #     require_parameter_descriptions=False, # No params are needed for this tool
    # ),

    # # Scheme Info (single scheme)
    # Tool(
    #     get_scheme_info,
    #     takes_ctx=False,
    #     docstring_format='auto', 
    #     require_parameter_descriptions=True,
    #     ),

    # # Multiple Schemes Info (with automatic state-first prioritization)
    # Tool(
    #     get_multiple_schemes_info,
    #     takes_ctx=False,
    #     docstring_format='auto', 
    #     require_parameter_descriptions=True,
    #     ),

    # # MahaDBT
    # Tool(
    #     get_scheme_status,
    #     takes_ctx=True,
    #     docstring_format='auto', 
    #     require_parameter_descriptions=False,
    # ),

    # # Agricultural Staff Contact
    # Tool(
    #     contact_agricultural_staff,
    #     takes_ctx=False,
    #     docstring_format='auto',
    #     require_parameter_descriptions=True,
    # ),

]


# ── Voice agent tool registries (Inc 7.2) ────────────────────────────────────
# The voice agent uses its own tool set, kept SEPARATE from chat's TOOLS above
# (Option A). They legitimately differ per surface: a "working on it" nudge
# wrapper for telephony latency, different takes_ctx flags, and voice-only tools.
# Per the tool-by-tool reconciliation:
#   - search_documents uses the unified no-ctx fn (voice's ctx was unused)
#   - the milk-collection AND union-scheme prepare-guards are applied to voice too
#   - get_union_scheme_data stays signed-in-only
#   - the profile/herd/tags tools stay disabled (redundant with runtime context)


def _with_nudge_signal(func):
    """Wrap a tool so it fires the tool-call nudge event on invocation (voice
    telephony 'working on it' UX). Harmless on chat — fire_tool_call_nudge is a
    no-op when no nudge listener is attached to the request."""
    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            fire_tool_call_nudge()
            return await func(*args, **kwargs)
        return wrapper

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
        takes_ctx=False,  # unified no-ctx search_documents
    ),
    Tool(
        _with_nudge_signal(create_ai_call),
        takes_ctx=True,
        docstring_format='auto',
        require_parameter_descriptions=True,
    ),
    Tool(
        _with_nudge_signal(get_farmer_milk_collection_details_voice),
        takes_ctx=True,
        docstring_format='auto',
        require_parameter_descriptions=True,
    ),
    Tool(
        _with_nudge_signal(create_health_call),
        takes_ctx=True,
        docstring_format='auto',
        require_parameter_descriptions=True,
    ),
    Tool(
        signal_conversation_state,
        takes_ctx=True,
        docstring_format='auto',
    ),
]

SIGNED_IN_FARMER_TOOLS = [
    # get_farmer_profile / get_herd_summary / list_animal_tags are intentionally
    # disabled — redundant with the runtime farmer-context summary the voice
    # pipeline injects (two caused the "I don't have your herd info" failure).
    # Imported + kept here (Option A) so they can be re-enabled without surgery.
    # Tool(_with_nudge_signal(get_farmer_profile), takes_ctx=True, docstring_format='auto', require_parameter_descriptions=False),
    # Tool(_with_nudge_signal(get_herd_summary), takes_ctx=True, docstring_format='auto', require_parameter_descriptions=False),
    # Tool(_with_nudge_signal(list_animal_tags), takes_ctx=True, docstring_format='auto', require_parameter_descriptions=False),
    Tool(
        _with_nudge_signal(get_union_scheme_data),
        takes_ctx=True,
        docstring_format='auto',
        require_parameter_descriptions=False,
        prepare=prepare_get_union_scheme_data,  # added for voice (tool 5)
    ),
]
