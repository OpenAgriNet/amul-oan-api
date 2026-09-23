"""Stable, provider-neutral contracts for historical telemetry analytics."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


FieldAvailability = Literal["recorded", "derived", "unavailable"]


class CanonicalChatTurn(BaseModel):
    """A stable chat-turn shape produced by version-specific Langfuse adapters.

    ``None`` means the source era did not provide a value to this adapter. The
    accompanying ``field_availability`` preserves whether that is a structural
    absence, a recorded value, or a value derived from a historical alias.
    """

    schema_version: Literal["chat-turn.v1"] = "chat-turn.v1"
    source_era: str
    source_schema_version: str
    source_era_extensions: list[str] = Field(default_factory=list)
    source_trace_id: str | None = None
    source_trace_name: str
    timestamp: datetime
    session_id: str | None = None
    user_id: str | None = None
    user_id_semantics: str | None = None
    channel: str | None = None
    pipeline: str | None = None
    pipeline_profile: str | None = None
    source_lang: str | None = None
    target_lang: str | None = None
    original_question: str | None = None
    answer: Any | None = None
    persona: str | None = None
    turn_outcome: str | None = None
    served_tier: str | None = None
    full_turn_latency_ms: float | None = None
    tool_calls: list[dict[str, Any]] | None = None
    root_input: dict[str, Any] | None = None
    root_output: Any | None = None
    observation_names: list[str] = Field(default_factory=list)
    score_names: list[str] = Field(default_factory=list)
    field_availability: dict[str, FieldAvailability] = Field(default_factory=dict)


class LangfuseScoreSchema(BaseModel):
    """The score fields that adapters need; provider extras remain ignorable."""

    model_config = ConfigDict(extra="ignore")

    name: str
    value: str | int | float | None = None


class ChatC3MetadataSchema(BaseModel):
    """Fields observed on the c3 root trace and its c3b extension."""

    model_config = ConfigDict(extra="ignore")

    pipeline: str | None = None
    channel: str | None = None
    source_lang: str | None = None
    target_lang: str | None = None
    user_id: str | None = None
    variant: str | None = None


class ChatC3TraceSchema(BaseModel):
    """Version-specific Langfuse root-trace contract for chat.c3/c3b."""

    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    name: Literal["Amul AI Agent"]
    timestamp: datetime
    session_id: str | None = Field(default=None, validation_alias="sessionId")
    input: dict[str, Any] | None = None
    output: Any | None = None
    metadata: ChatC3MetadataSchema = Field(default_factory=ChatC3MetadataSchema)


class ChatC5MetadataSchema(BaseModel):
    """c5 kept the c3 root shape but renamed the profile metadata key."""

    model_config = ConfigDict(extra="ignore")

    pipeline: str | None = None
    channel: str | None = None
    source_lang: str | None = None
    target_lang: str | None = None
    user_id: str | None = None
    pipeline_profile: str | None = None


class ChatC5TraceSchema(BaseModel):
    """Version-specific root-trace contract after c5's profile-key rename."""

    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    name: Literal["Amul AI Agent"]
    timestamp: datetime
    session_id: str | None = Field(default=None, validation_alias="sessionId")
    input: dict[str, Any] | None = None
    output: Any | None = None
    metadata: ChatC5MetadataSchema = Field(default_factory=ChatC5MetadataSchema)


class ChatC6MetadataSchema(BaseModel):
    """Fields recorded on c6+ root traces, including c8's translation-only path."""

    model_config = ConfigDict(extra="ignore")

    pipeline: str | None = None
    pipeline_profile: str | None = None
    channel: str | None = None
    source_lang: str | None = None
    target_lang: str | None = None
    user_id: str | None = None
    persona: str | None = None


class ChatC6TraceSchema(BaseModel):
    """Version-specific Langfuse root-trace contract for chat.c6 and later."""

    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    name: Literal["chat.default", "chat.translation"]
    timestamp: datetime
    session_id: str | None = Field(default=None, validation_alias="sessionId")
    input: dict[str, Any] | None = None
    output: Any | None = None
    metadata: ChatC6MetadataSchema = Field(default_factory=ChatC6MetadataSchema)


class ChatC8TraceSchema(ChatC6TraceSchema):
    """Translation-only continuation of the c6 root-turn contract."""

    name: Literal["chat.translation"]
