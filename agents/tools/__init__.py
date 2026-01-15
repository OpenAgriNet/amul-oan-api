"""Tools for the Sunbird VA API."""
from pydantic_ai import Tool
from agents.tools.common import reasoning_tool, planning_tool
from agents.tools.search import search_documents, search_videos
from agents.tools.weather import weather_forecast, weather_historical
from agents.tools.mandi import mandi_prices
from agents.tools.agri_services import agri_services
from agents.tools.maps import reverse_geocode, forward_geocode  
from agents.tools.agristack import fetch_agristack_data
from agents.tools.mahadbt import get_scheme_status
from agents.tools.terms import search_terms
from agents.tools.scheme_info import get_scheme_codes, get_scheme_info, get_multiple_schemes_info
from agents.tools.staff_contact import contact_agricultural_staff

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
