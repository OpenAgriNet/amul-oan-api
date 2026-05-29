"""Tools for the Sunbird VA API."""
from pydantic_ai import Tool, RunContext
from pydantic_ai.tools import ToolDefinition
from app.config import settings
from agents.deps import FarmerContext
from agents.tools.beckn_search import search_government_schemes, is_government_scheme_query
from agents.tools.ai_call import create_ai_call
from agents.tools.health_call import create_health_call
from agents.tools.milk_collection import get_farmer_milk_collection_details
from agents.tools.search import search_documents, search_videos
from agents.tools.terms import search_terms
from agents.tools.union_schemes import get_union_scheme_data
# from agents.tools.animal import get_animal_by_tag
# from agents.tools.cvcc import get_cvcc_health_details
# from agents.tools.farmer import get_farmer_by_mobile

async def _suppress_docs_for_scheme_queries(ctx: RunContext[FarmerContext], tool_def: ToolDefinition):
    """DEMO: force the Beckn path for government-scheme questions.

    When beckn is enabled and the (English) query is a government-scheme query, hide
    `search_documents` so the agent must use `search_government_schemes` — the agent
    otherwise often answers schemes from RAG and never hits the live Vistaar network.
    """
    if settings.beckn_enabled and is_government_scheme_query(ctx.deps.query):
        return None
    return tool_def


TOOLS = [
    # # Search Terms
    # Tool(
    #     search_terms,
    #     takes_ctx=False,
    #     docstring_format='auto',
    #     require_parameter_descriptions=True,

    # ),

    # Search Documents (suppressed for government-scheme queries when beckn is on)
    Tool(
        search_documents,
        takes_ctx=False, # No context is needed for this tool
        docstring_format='auto',
        require_parameter_descriptions=True,
        prepare=_suppress_docs_for_scheme_queries,
    ),

    Tool(
        create_ai_call,
        takes_ctx=False,
        docstring_format='auto',
        require_parameter_descriptions=True,
    ),

    Tool(
        create_health_call,
        takes_ctx=False,
        docstring_format='auto',
        require_parameter_descriptions=True,
    ),
    
    
    Tool(
        get_farmer_milk_collection_details,
        takes_ctx=False,
        docstring_format='auto',
        require_parameter_descriptions=True,
    ),

    Tool(
        get_union_scheme_data,
        takes_ctx=True,
        docstring_format='auto',
        require_parameter_descriptions=False,
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

# Beckn "network of networks" government-scheme discovery — registered ONLY in the
# demo image (BECKN_ENABLED=true). Keeps the tool out of normal dev/prod agents.
if settings.beckn_enabled:
    TOOLS.append(
        Tool(
            search_government_schemes,
            takes_ctx=False,
            docstring_format='auto',
            require_parameter_descriptions=True,
        )
    )
