from pydantic import BaseModel, Field
from typing import Optional, Literal

class ChatRequest(BaseModel):
    query: str = Field(..., description="The user's chat query")
    session_id: Optional[str] = Field(
        None,
        description="Session ID for conversation context and Langfuse Sessions (same ID = one session; omit for a new conversation)",
    )
    source_lang: Literal['gu', 'en'] = Field('gu', description="Source language code (gu=Gujarati, en=English)")
    target_lang: str = Field('gu', description="Target language code")
    user_id: str = Field('anonymous', description="User identifier (expected to be phone number for farmer context)")
    provider: Optional[Literal['RAYA']] = Field(None, description="Provider for the voice service - can be RAYA or None")
    process_id: Optional[str] = Field(None, description="Process ID for tracking and hold messages") 
