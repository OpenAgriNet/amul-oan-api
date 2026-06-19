import asyncio
from typing import Optional, Literal
from pydantic import BaseModel, Field, PrivateAttr


class FarmerAccount(BaseModel):
    """One (union, society, farmer) account tied to the caller's mobile.

    A single mobile can map to several PashuGPT accounts (e.g. a separate
    cow account and buffalo account). Milk-collection lookups fan out over
    all of these so a farmer's data is never missed just because the agent
    happened to pick the wrong account's codes.
    """
    union_code: Optional[str] = None
    society_code: Optional[str] = None
    farmer_code: Optional[str] = None
    farmer_name: Optional[str] = None
    society_name: Optional[str] = None


class FarmerContext(BaseModel):
    """Unified context for the agent (chat + voice).

    Union of the chat and voice FarmerContext (bucket B/C reconciliation). Chat
    fields (moderation_str, use_translation_pipeline, response_max_chars) and voice
    fields (target_lang, provider, process_id, ai_technician_info, signed_in,
    mobile, the concurrent-moderation task) coexist; each channel sets/reads the
    subset it needs. farmer_info defaults to "" so a caller may omit it.

    Args:
        query: The user's question.
        lang_code: The language code of the user's question.
        target_lang: The target language for the response (voice).
        farmer_info: Pre-built markdown farmer context string.
        ai_technician_info: Pre-built internal AI technician context for booking (voice).
        provider: The provider for the voice service.
        session_id: Session id (booking-tool idempotency guards + voice session).
        process_id: The process ID for tracking and hold messages (voice).
    """
    query: str = Field(description="The user's question.")
    session_id: Optional[str] = Field(default=None, description="Session id, used for booking-tool idempotency guards (e.g. one AI/health call per session) and the voice session.")
    lang_code: str = Field(description="The language code of the user's question.", default='gu')
    target_lang: str = Field(description="The target language for the response (gu=Gujarati, en=English).", default='gu')
    provider: Optional[Literal['RAYA']] = Field(default=None, description="The provider for the voice service - can be RAYA or None.")
    process_id: Optional[str] = Field(default=None, description="The process ID for tracking and hold messages.")
    moderation_str: Optional[str] = Field(default=None, description="The moderation result of the user's question (chat).")
    farmer_info: str = Field(default="", description="Pre-built markdown string with farmer profile/animals/vet visits (from JWT or context bundle).")
    farmer_unions: list[str] = Field(default_factory=list, description="Normalized union names derived from the farmer context.")
    ai_technician_info: str = Field(default="", description="Pre-built internal AI technician context string (voice).")
    signed_in: bool = Field(default=False, description="Whether the session is signed in/authenticated for farmer-specific tools.")
    mobile: Optional[str] = Field(default=None, description="Normalized mobile number when available.")
    farmer_accounts: list[FarmerAccount] = Field(
        default_factory=list,
        description="All (union, society, farmer) accounts on the caller's mobile, for multi-account fan-out.",
    )
    use_translation_pipeline: bool = Field(default=False, description="When True, use English-only prompt; response is translated externally (chat).")
    response_max_chars: Optional[int] = Field(default=None, description="Optional channel-specific final response character guidance (chat).")

    # Handle to the per-turn content-moderation task, which runs concurrently with
    # the agent on the voice path (see app.services.voice). Side-effecting tools
    # await it via ensure_in_scope() so a rejected query can never produce a write,
    # even though the agent executes optimistically before the verdict is known.
    _moderation_task: Optional["asyncio.Task"] = PrivateAttr(default=None)

    def set_moderation_task(self, task: Optional["asyncio.Task"]) -> None:
        """Attach the concurrently-running moderation task for tool self-gating."""
        self._moderation_task = task

    async def ensure_in_scope(self) -> bool:
        """Block until the concurrent moderation verdict is known.

        Returns False ONLY when moderation explicitly rejected the query, so
        side-effecting tools (e.g. bookings) refuse instead of performing a write.
        Fail-open (returns True) when no task is attached or moderation errored —
        a flaky moderation check must never drop a real farmer booking.
        """
        task = self._moderation_task
        if task is None:
            return True
        try:
            verdict = await task
        except Exception:
            return True
        return not bool(verdict is not None and getattr(verdict, "rejected", False))

    def update_moderation_str(self, moderation_str: str):
        """Update the moderation result of the user's question (chat)."""
        self.moderation_str = moderation_str

    def get_moderation_str(self) -> Optional[str]:
        """Get the moderation result of the user's question (chat)."""
        return self.moderation_str

    def _query_string(self):
        """Get the query string for the agrinet agent."""
        return "**User:** " + '"' + self.query + '"'

    def _moderation_string(self):
        """Get the moderation string for the agrinet agent (chat)."""
        if self.moderation_str:
            return self.moderation_str
        else:
            return None

    def get_farmer_context_string(self) -> str:
        """Format farmer context information for the system prompt."""
        return self.farmer_info

    def get_preferred_union_name(self) -> Optional[str]:
        """Get the primary farmer union name when available."""
        return self.farmer_unions[0] if self.farmer_unions else None

    def get_response_max_chars(self) -> Optional[int]:
        """Get channel-specific final response character guidance (chat)."""
        return self.response_max_chars

    def get_runtime_context_message(self) -> str:
        """Compact runtime context that stays outside the static system prompt (voice)."""
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
