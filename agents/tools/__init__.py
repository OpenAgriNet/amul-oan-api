"""Tools for the Sunbird VA API."""
import functools
import inspect

from pydantic_ai import Tool

from app.config import settings
from agents.tools.ai_call import create_ai_call
from agents.tools.health_call import create_health_call
from agents.tools.milk_collection import (
    get_farmer_milk_collection_details,
    prepare_get_farmer_milk_collection_details,
)
from agents.tools.search import search_documents
from agents.tools.union_schemes import get_union_scheme_data, prepare_get_union_scheme_data
from agents.tools.loan import check_loan_eligibility, prepare_check_loan_eligibility
from agents.tools.vistaar import (
    get_vistaar_weather,
    get_vistaar_mandi_prices,
    get_vistaar_scheme_info,
)

TOOLS = [
    # # Search Terms

    # Search Documents
    Tool(
        search_documents,
        takes_ctx=False, # No context is needed for this tool
        docstring_format='auto', 
        require_parameter_descriptions=True,
    ),

    Tool(
        create_ai_call,
        takes_ctx=True,  # needs ctx.deps.ensure_in_scope for moderation gating
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

    # # Get CVCC Health Details (temporarily disabled)

    # # Get Farmer by Mobile Number (temporarily disabled)

    # # Search Videos

    # # Reverse Geocode - Do we need this?

    # # Weather Forecast

    # # Weather Historical

    # # Mandi Prices

    # # Agricultural Services (KVK, CHC, etc.)
    
    # # Geocode

    # # Agristack
    # # Scheme Codes

    # # Scheme Info (single scheme)

    # # Multiple Schemes Info (with automatic state-first prioritization)

    # # MahaDBT

    # # Agricultural Staff Contact

]

# Bharat Vistaar discovery — weather, mandi prices, scheme info (Beckn shortcut).
# Gated on the same flag as the re-routed tools: with ENABLE_NETWORK off the agent
# must not see them, or an ungated Beckn call (35s timeout) can eat most of a turn.
if settings.enable_network:
    TOOLS.extend([
        Tool(
            get_vistaar_weather,
            takes_ctx=False,
            docstring_format='auto',
            require_parameter_descriptions=True,
        ),
        Tool(
            get_vistaar_mandi_prices,
            takes_ctx=False,
            docstring_format='auto',
            require_parameter_descriptions=True,
        ),
        Tool(
            get_vistaar_scheme_info,
            takes_ctx=False,
            docstring_format='auto',
            require_parameter_descriptions=True,
        ),
    ])

# ── Voice agent tool registries (Inc 7.2) ────────────────────────────────────
# The voice agent uses its own tool set, kept SEPARATE from chat's TOOLS above
# (Option A). They legitimately differ per surface: a "working on it" nudge
# wrapper for telephony latency, different takes_ctx flags, and voice-only tools.
# Per the tool-by-tool reconciliation:
#   - search_documents uses the unified no-ctx fn (voice's ctx was unused)
#   - the milk-collection AND union-scheme prepare-guards are applied to voice too
#   - get_union_scheme_data stays signed-in-only
#   - the profile/herd/tags tools stay disabled (redundant with runtime context)

