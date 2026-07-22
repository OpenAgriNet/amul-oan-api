from pydantic import BaseModel, Field
from typing import Optional, Literal


class BaseChatRequest(BaseModel):
    """Fields common to the chat and voice request surfaces."""
    query: str = Field(..., description="The user's chat query")
    session_id: Optional[str] = Field(
        None,
        description="Session ID for conversation context and Langfuse Sessions (same ID = one session; omit for a new conversation)",
    )
    # Chat supports the full translation set (13 langs); voice narrows this to gu/en.
    source_lang: str = Field('gu', description="Source language code")
    target_lang: str = Field('gu', description="Target language code")
    user_id: str = Field('anonymous', description="User identifier (expected to be phone number for farmer context)")


class ChatRequest(BaseChatRequest):
    channel: Literal['web', 'whatsapp'] = Field('web', description="Calling channel")
    use_translation_pipeline: Optional[bool] = Field(True, description="When True, use Gemma pre/post translation (query→en→agent→target_lang)")
    stream: Optional[bool] = Field(True, description="When True (default), return SSE stream. When False, return a single JSON response.")


class VoiceRequest(BaseChatRequest):
    # Voice only supports Gujarati/English end to end.
    source_lang: Literal['gu', 'en'] = Field('gu', description="Source language code (gu=Gujarati, en=English)")
    provider: Optional[Literal['RAYA']] = Field(None, description="Provider for the voice service - can be RAYA or None")
    process_id: Optional[str] = Field(None, description="Process ID for tracking and hold messages")


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
