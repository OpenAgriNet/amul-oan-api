"""Contracts for historical voice telemetry. Voice text is only ever carried sanitized."""

from datetime import datetime
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from app.models.telemetry_analytics import FieldAvailability


class SanitizedVoiceText(BaseModel):
    """Output of voice_trace.sanitize_text; the full-text ``text`` key is dropped."""

    model_config = ConfigDict(extra="ignore")

    chars: int | None = None
    sha256: str | None = None
    preview: str | None = None


class CanonicalVoiceTurn(BaseModel):
    """One source trace becomes one turn.

    An utterance can emit several traces, so count delivered queries with
    ``outcome_class`` rather than raw turn totals.
    """

    # Version of this output shape. source_schema_version is the stamp that came in.
    schema_version: Literal["voice.canonical.v1"] = "voice.canonical.v1"
    source_era: str
    source_schema_version: str
    source_era_extensions: list[str] = Field(default_factory=list)
    source_trace_id: str | None = None
    source_trace_name: str
    timestamp: datetime
    session_id: str | None = None
    process_id: str | None = None
    user_id: str | None = None
    user_id_semantics: str | None = None
    user_id_hash: str | None = None
    signed_in: bool | None = None
    channel: Literal["voice"] = "voice"
    provider: str | None = None
    call_type: str | None = None
    route: str | None = None
    pipeline_profile: str | None = None
    source_lang: str | None = None
    target_lang: str | None = None
    question_sanitized: SanitizedVoiceText | None = None
    answer_sanitized: SanitizedVoiceText | None = None
    outcome: str | None = None
    outcome_class: str | None = None
    full_turn_latency_ms: float | None = None
    stage_totals_ms: dict[str, float] | None = None
    timings_ms: dict[str, float] | None = None
    observation_names: list[str] = Field(default_factory=list)
    score_names: list[str] = Field(default_factory=list)
    field_availability: dict[str, FieldAvailability] = Field(default_factory=dict)


class VoiceV0MetadataSchema(BaseModel):
    """Fields observed on the pydantic-ai voice agent roots (v0/v1/v2)."""

    model_config = ConfigDict(extra="ignore")

    process_id: str | int | None = None
    provider: str | None = None
    source_lang: str | None = None
    target_lang: str | None = None
    signed_in: bool | str | None = None
    pipeline_variant: str | None = None


class VoiceV0TraceSchema(BaseModel):
    """v0 root contract. Its output is a full message history, never read here."""

    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    name: Literal["Voice Agent run", "Voice Agent Signed In run"]
    timestamp: datetime
    session_id: str | None = Field(default=None, validation_alias=AliasChoices("sessionId", "session_id"))
    user_id: str | int | None = Field(default=None, validation_alias=AliasChoices("userId", "user_id"))
    metadata: VoiceV0MetadataSchema = Field(default_factory=VoiceV0MetadataSchema)


class VoiceTurnMetadataSchema(BaseModel):
    """VoiceTrace metadata from v3 on.

    Nested values stay ``Any``: they come as objects from the Langfuse API and
    as JSON strings from a ClickHouse export.
    """

    model_config = ConfigDict(extra="ignore")

    session_id: str | None = None
    process_id: str | int | None = None
    provider: str | None = None
    source_lang: str | None = None
    target_lang: str | None = None
    user_id_hash: str | None = None
    route: str | None = None
    call_type: str | None = None
    outcome: str | None = None
    pipeline_variant: str | None = None
    pipeline_profile: str | None = None
    agent: Any | None = None
    query: Any | None = None
    response: Any | None = None
    total_ms: Any | None = None
    timings_ms: Any | None = None
    stage_totals_ms: Any | None = None


class VoiceV3TraceSchema(BaseModel):
    """v3/v3b root contract: the voice_request trace."""

    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    name: Literal["voice_request"]
    timestamp: datetime
    session_id: str | None = Field(default=None, validation_alias=AliasChoices("sessionId", "session_id"))
    user_id: str | int | None = Field(default=None, validation_alias=AliasChoices("userId", "user_id"))
    input: Any | None = None
    output: Any | None = None
    metadata: VoiceTurnMetadataSchema = Field(default_factory=VoiceTurnMetadataSchema)


class VoiceV4TraceSchema(VoiceV3TraceSchema):
    """v4 renamed the root to agent_journey; v5/v5b only add keys to it."""

    name: Literal["agent_journey"]
