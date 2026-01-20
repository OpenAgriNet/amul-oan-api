from typing import Optional
from pydantic import BaseModel, Field, field_validator
from langcodes import Language


class FarmerContext(BaseModel):
    """Context for the farmer agent.
    
    Args:
        query (str): The user's question.
        lang_code (str): The language code of the user's question.
        moderation_str (Optional[str]): The moderation result of the user's question.


    Example:
        **User:** "What is the weather in Mumbai?"
        **Selected Language:** Gujarati
        **Moderation Result:** "This is a valid agricultural question."
    """
    query: str = Field(description="The user's question.")
    lang_code: str = Field(description="The language code of the user's question.", default='gu')
    moderation_str: Optional[str] = Field(default=None, description="The moderation result of the user's question.")
    farmer_id: Optional[str] = Field(default=None, description="The farmer ID of the user.")

    def update_moderation_str(self, moderation_str: str):
        """Update the moderation result of the user's question."""
        self.moderation_str = moderation_str

    def update_farmer_id(self, farmer_id: str):
        """Update the farmer ID of the user."""
        self.farmer_id = farmer_id

    def get_farmer_id(self) -> Optional[str]:
        """Get the farmer ID of the user."""
        return self.farmer_id
        
    def get_moderation_str(self) -> Optional[str]:
        """Get the moderation result of the user's question."""
        return self.moderation_str
    
    def _language_string(self):
        """Get the language string for the agrinet agent."""
        if self.lang_code:
            return f"**Selected Language:** {Language.get(self.lang_code).display_name()}"
        else:
            return None
    
    def _query_string(self):
        """Get the query string for the agrinet agent."""
        return "**User:** " + '"' + self.query + '"'

    def _moderation_string(self):
        """Get the moderation string for the agrinet agent."""
        if self.moderation_str:
            return self.moderation_str
        else:
            return None
    
    def _agristack_availability_string(self):
        """Get the farmer ID string for the agrinet agent."""
        if self.farmer_id:
            return "**Agristack Information Availability**: ✅"
        else:
            return "**Agristack Information Availability**: ❌"

    def get_user_message(self):
        """Get the user message for the agrinet agent."""
        strings = [self._query_string(), self._language_string(), self._moderation_string(), self._agristack_availability_string()]
        return "\n".join([x for x in strings if x])