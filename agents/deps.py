from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from langcodes import Language


class FarmerContext(BaseModel):
    """Context for the farmer agent.
    
    Args:
        query (str): The user's question.
        lang_code (str): The language code of the user's question.
        moderation_str (Optional[str]): The moderation result of the user's question.
        farmer_info (Optional[Dict[str, Any]]): Farmer's personal details and animals from JWT token.


    Example:
        **User:** "What is the weather in Mumbai?"
        **Selected Language:** Gujarati
        **Moderation Result:** "This is a valid agricultural question."
    """
    query: str = Field(description="The user's question.")
    lang_code: str = Field(description="The language code of the user's question.", default='gu')
    moderation_str: Optional[str] = Field(default=None, description="The moderation result of the user's question.")
    farmer_info: Optional[Dict[str, Any]] = Field(default=None, description="Farmer's personal details and animals from JWT token.")

    def update_moderation_str(self, moderation_str: str):
        """Update the moderation result of the user's question."""
        self.moderation_str = moderation_str

    # def update_farmer_id(self, farmer_id: str):
    #     """Update the farmer ID of the user."""
    #     self.farmer_id = farmer_id

    # def get_farmer_id(self) -> Optional[str]:
    #     """Get the farmer ID of the user."""
    #     return self.farmer_id
        
    def get_moderation_str(self) -> Optional[str]:
        """Get the moderation result of the user's question."""
        return self.moderation_str
    
    # def _language_string(self):
    #     """Get the language string for the agrinet agent."""
    #     if self.lang_code:
    #         return f"**Selected Language:** {Language.get(self.lang_code).display_name()}"
    #     else:
    #         return None
    
    def _query_string(self):
        """Get the query string for the agrinet agent."""
        return "**User:** " + '"' + self.query + '"'

    def _moderation_string(self):
        """Get the moderation string for the agrinet agent."""
        if self.moderation_str:
            return self.moderation_str
        else:
            return None
    
    # def _agristack_availability_string(self):
    #     """Get the farmer ID string for the agrinet agent."""
    #     if self.farmer_id:
    #         return "**Agristack Information Availability**: ✅"
    #     else:
    #         return "**Agristack Information Availability**: ❌"

    def get_farmer_context_string(self) -> Optional[str]:
        """Format farmer context information from JWT token for the system prompt."""
        if not self.farmer_info:
            return None
        
        def format_value(value: Any, indent: int = 0) -> str:
            """Recursively format values for display."""
            indent_str = "  " * indent
            if isinstance(value, dict):
                if not value:
                    return "{}"
                lines = []
                for k, v in value.items():
                    formatted_value = format_value(v, indent + 1)
                    if isinstance(v, (dict, list)) and v:
                        lines.append(f"{indent_str}- **{k}:**\n{formatted_value}")
                    else:
                        lines.append(f"{indent_str}- **{k}**: {formatted_value}")
                return "\n".join(lines)
            elif isinstance(value, list):
                if not value:
                    return "[]"
                lines = []
                for i, item in enumerate(value):
                    formatted_item = format_value(item, indent + 1)
                    if isinstance(item, (dict, list)) and item:
                        lines.append(f"{indent_str}- Item {i + 1}:\n{formatted_item}")
                    else:
                        lines.append(f"{indent_str}- {formatted_item}")
                return "\n".join(lines)
            else:
                return str(value)
        
        # Format the entire farmer_info dict
        formatted_context = format_value(self.farmer_info, indent=0)
        return formatted_context
    
    def get_user_message(self):
        """Get the user message for the agrinet agent."""
        strings = [
            self._query_string(), 
        #self._language_string(), 
        #self._moderation_string(), 
        ]
        return "\n".join([x for x in strings if x])