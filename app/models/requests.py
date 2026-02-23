from pydantic import BaseModel, Field
from typing import Optional, Literal

class ChatRequest(BaseModel):
    query: str = Field(..., description="The user's chat query")
    session_id: Optional[str] = Field(None, description="Session ID for maintaining conversation context")
    source_lang: str = Field('gu', description="Source language code")
    target_lang: str = Field('gu', description="Target language code")
    user_id: str = Field('anonymous', description="User identifier")
    use_translation_pipeline: Optional[bool] = Field(False, description="When True, use Gemma pre/post translation (query→en→agent→target_lang)")

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