"""Tools for the Sunbird VA API."""
from pydantic_ai import Tool
from agents.tools.search import search_documents
from agents.tools.terms import search_terms

TOOLS = [
    # Search Terms
    Tool(
        search_terms,
        takes_ctx=False,
        docstring_format='auto', 
        require_parameter_descriptions=True,
    ),
    # Search Documents
    Tool(
        search_documents,
        takes_ctx=True, # No context is needed for this tool
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

 
]
