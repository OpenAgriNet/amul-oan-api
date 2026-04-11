"""Control layer for routing chat/voice conversation pipelines.

This module intentionally does not refactor business logic yet.
It provides a single entry point that forwards to existing services.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from app.services.chat import stream_chat_messages

CHAT_CONFIG = {
    "use_stt": False,
    "use_tts": False,
    "use_translation": False,  # set from request at router level
    # Chat pipeline must always run moderation (router also forces True).
    "use_moderation": True,
    "use_stt_signals": False,
    "use_concurrency_lock": False,
    "post_processing": "suggestions",
}

VOICE_CONFIG = {
    "use_stt": True,
    "use_tts": False,
    "use_translation": True,  # currently env-driven in voice stack
    "use_moderation": False,
    "use_stt_signals": True,
    "use_concurrency_lock": True,
    "post_processing": "feedback",
}


async def process_conversation(
    input_data: dict[str, Any],
    *,
    use_stt: bool = False,
    use_tts: bool = False,
    use_translation: bool = False,
    use_moderation: bool = False,
    use_stt_signals: bool = False,
    use_concurrency_lock: bool = False,
    post_processing: str | None = None,  # "suggestions" | "feedback"
) -> AsyncIterator[str]:
    """
    Unified conversation pipeline control layer.

    IMPORTANT:
    - This function only routes to existing implementations.
    - It does not alter internal chat/voice business logic.
    """
    _ = (
        use_tts,
        use_translation,
        use_moderation,
        use_stt_signals,
        use_concurrency_lock,
        post_processing,
    )

    if use_stt:
        # Voice pipeline not yet integrated into this repository.
        # Keep lazy import so chat behavior remains unaffected.
        try:
            from app.services.voice import stream_voice_messages  # type: ignore
        except ImportError:
            from app.services.voice import stream_voice_message as stream_voice_messages  # type: ignore
        return stream_voice_messages(**input_data)

    return stream_chat_messages(**input_data)

