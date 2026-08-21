from typing import Literal, cast

from app.config import settings


ChatPersona = Literal["farmer", "doctor"]


def resolve_chat_persona(
    user_info: dict | None,
    requested_persona: ChatPersona | None = None,
) -> ChatPersona:
    """Resolve the persona without allowing an unflagged client override.

    Signed JWTs remain authoritative in normal operation. The request override
    exists only for the feature-flagged chat UI used to test both paths.
    """
    jwt_persona = str((user_info or {}).get("user_type") or "").strip().lower()
    resolved: ChatPersona = "doctor" if jwt_persona == "doctor" else "farmer"

    if settings.chat_persona_override_enabled and requested_persona:
        return cast(ChatPersona, requested_persona)
    return resolved


def history_session_id_for_persona(session_id: str, persona: ChatPersona) -> str:
    """Keep Doctor context isolated from the farmer/Sarlaben conversation.

    Farmer history retains its legacy cache key to avoid disrupting existing
    sessions. Doctor history uses a namespaced key so changing the test selector
    cannot replay farmer tool calls, identity text, or audience framing into a
    clinical turn.
    """
    return f"{session_id}:persona:doctor" if persona == "doctor" else session_id
