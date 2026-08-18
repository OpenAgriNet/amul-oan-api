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
