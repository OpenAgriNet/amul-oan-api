from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal

class VoiceChatRequest(BaseModel):
    """Query params for GET /voice (aligned with voice-oan-api ChatRequest)."""

    query: str = Field(..., description="The user's chat query")
    session_id: Optional[str] = Field(
        None,
        description="Session ID for conversation context and Langfuse (omit for a new conversation)",
    )
    source_lang: Literal["gu", "en"] = Field(
        "gu",
        description="Source language code (gu=Gujarati, en=English)",
    )
    target_lang: str = Field("gu", description="Target language code")
    user_id: str = Field(
        "anonymous",
        description="User identifier (expected to be phone number for farmer context)",
    )
    provider: Optional[Literal["RAYA"]] = Field(
        None,
        description="Provider for the voice service — RAYA or None",
    )
    process_id: Optional[str] = Field(
        None,
        description="Process ID for tracking and hold messages",
    )


class ChatRequest(BaseModel):
    query: str = Field(..., description="The user's chat query")
    session_id: Optional[str] = Field(None, description="Session ID for maintaining conversation context")
    source_lang: str = Field('gu', description="Source language code")
    target_lang: str = Field('gu', description="Target language code")
    user_id: str = Field('anonymous', description="User identifier")
    use_translation_pipeline: bool = Field(
        False,
        description="When True, use Gemma pre/post translation (query→en→agent→target_lang)",
    )
    stream: bool = Field(True, description="When True (default), return SSE stream. When False, return a single JSON response.")

    @field_validator("use_translation_pipeline", mode="before")
    @classmethod
    def _coerce_use_translation_pipeline(cls, v: object) -> bool:
        """Query params arrive as strings; accept common truthy/falsey forms."""
        if v is None or v == "":
            return False
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return v != 0
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes", "on")
        return bool(v)

    @field_validator("stream", mode="before")
    @classmethod
    def _coerce_stream(cls, v: object) -> bool:
        if v is None or v == "":
            return True
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return v != 0
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes", "on")
        return bool(v)

class TranscribeRequest(BaseModel):
    audio_content: str = Field(..., description="Base64 encoded audio content")
    source_lang: str = Field('gu', description="Source language code")
    service_type: Literal['bhashini', 'whisper'] = Field('bhashini', description="Transcription service to use")
    session_id: Optional[str] = Field(None, description="Session ID")

class SuggestionsRequest(BaseModel):
    session_id: str = Field(..., description="Session ID to get suggestions for")
    target_lang: str = Field('gu', description="Target language for suggestions")

class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech")
    target_lang: str = Field('gu', description="Target language code for TTS")
    session_id: Optional[str] = Field(None, description="Session ID")
    service_type: Literal['bhashini', 'raya'] = Field('bhashini', description="TTS service to use") 