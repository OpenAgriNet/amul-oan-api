from typing import Optional, Literal
from pydantic import BaseModel, Field


class FarmerContext(BaseModel):
    """Context for the voice agent.

    Args:
        query: The user's question.
        lang_code: The language code of the user's question.
        target_lang: The target language for the response.
        farmer_info: Pre-built markdown string with farmer profile, animals, vet visits.
        ai_technician_info: Pre-built internal AI technician context for booking.
        provider: The provider for the voice service.
        session_id: The session ID for the user.
        process_id: The process ID for tracking and hold messages.
    """
    query: str = Field(description="The user's question.")
    lang_code: str = Field(description="The language code of the user's question.", default='gu')
    target_lang: str = Field(description="The target language for the response (gu=Gujarati, en=English).", default='gu')
    provider: Optional[Literal['RAYA']] = Field(default=None, description="The provider for the voice service - can be RAYA or None.")
    session_id: Optional[str] = Field(default=None, description="The session ID for the user.")
    process_id: Optional[str] = Field(default=None, description="The process ID for tracking and hold messages.")
    farmer_info: str = Field(default="", description="Pre-built markdown farmer context string.")
    farmer_unions: list[str] = Field(default_factory=list, description="Normalized union names derived from the farmer context.")
    ai_technician_info: str = Field(default="", description="Pre-built internal AI technician context string.")
    signed_in: bool = Field(default=False, description="Whether the session is signed in/authenticated for farmer-specific tools.")
    mobile: Optional[str] = Field(default=None, description="Normalized mobile number when available.")

    def _query_string(self):
        """Get the query string for the agrinet agent."""
        return "**User:** " + '"' + self.query + '"'

    def get_farmer_context_string(self) -> Optional[str]:
        """Return the pre-built farmer context markdown string."""
        return self.farmer_info if self.farmer_info else None

    def get_preferred_union_name(self) -> Optional[str]:
        """Get the primary farmer union name when available."""
        return self.farmer_unions[0] if self.farmer_unions else None

    def get_runtime_context_message(self) -> str:
        """Compact runtime context that stays outside the static system prompt."""
        lines = [
            "Runtime context for this turn:",
            f"- Signed-in session: {'yes' if self.signed_in else 'no'}",
            f"- Normalized mobile available: {'yes' if self.mobile else 'no'}",
        ]
        if self.mobile:
            lines.append(f"- Normalized mobile: {self.mobile}")
        if self.farmer_unions:
            lines.append(f"- Farmer unions: {', '.join(self.farmer_unions)}")
        lines.append("- Core loop language: English")
        if self.signed_in:
            lines.append("- Farmer-data tools may be available for this turn.")
        else:
            lines.append("- Farmer-data tools should not be assumed available for this turn.")
        if self.farmer_info:
            lines.append("- Farmer context summary:")
            lines.append(self.farmer_info)
        if self.ai_technician_info:
            lines.append("- Internal AI technician context for booking:")
            lines.append("The caller does not know which AI technicians are available unless you tell them by name.")
            lines.append(self.ai_technician_info)
        return "\n".join(lines)

    def get_user_message(self):
        """Get the user message for the agrinet agent."""
        return self._query_string()
