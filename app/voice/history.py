"""What voice keeps in history in place of the caller's words, and the pair it writes.

From voice-oan-api (``app/services/voice.py`` at amul-dev ``3b19835``), shared by
the classifiers and the gate.
"""
from __future__ import annotations

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

HISTORY_MARKERS = {
    "greeting": "hello",
    "fragment": "[fragment]",
    "low_confidence": "[unclear-user-input]",
    "pretranslation_failed": "[pretranslation-failed]",
    "stt_no_audio": "[stt:no-audio]",
    "stt_unclear": "[stt:unclear-speech]",
    "moderation_reject": "[moderation-rejected]",
    "outbound_intro": "[outbound-call-started]",
}


def history_pair(user_text: str, assistant_text: str) -> tuple[ModelRequest, ModelResponse]:
    return (
        ModelRequest(parts=[UserPromptPart(content=user_text)]),
        ModelResponse(parts=[TextPart(content=assistant_text)]),
    )
