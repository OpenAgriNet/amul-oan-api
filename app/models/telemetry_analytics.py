"""Stable, provider-neutral contracts for historical telemetry analytics."""

from datetime import datetime
import hashlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


FieldAvailability = Literal["recorded", "derived", "unavailable"]


class SanitizedChatText(BaseModel):
    """Privacy-safe chat text retained by the importer."""

    chars: int
    sha256: str


class CanonicalToolCall(BaseModel):
    """Privacy-safe reference to a tool invocation, without tool arguments or results."""

    tool_name: str | None = None
    call_id: str | None = None


class CanonicalChatTurn(BaseModel):
    """A stable chat-turn shape produced by version-specific Langfuse adapters.

    ``None`` means the source era did not provide a value to this adapter. The
    accompanying ``field_availability`` preserves whether that is a structural
    absence, a recorded value, or a value derived from a historical alias.
    """

    # This is the normalized output contract. ``source_schema_version`` is the
    # incoming trace stamp, such as ``chat.turn.v1``.
    schema_version: Literal["chat.canonical.v1"] = "chat.canonical.v1"
    source_era: str
    source_schema_version: str
    source_era_extensions: list[str] = Field(default_factory=list)
    source_trace_id: str | None = None
    source_trace_name: str
    timestamp: datetime
    session_id: str | None = None
    user_id_hash: str | None = None
    user_id_semantics: str | None = None
    channel: str | None = None
    pipeline: str | None = None
    pipeline_profile: str | None = None
    source_lang: str | None = None
    target_lang: str | None = None
    question_sanitized: SanitizedChatText | None = None
    answer_sanitized: SanitizedChatText | None = None
    persona: str | None = None
    outcome: str | None = None
    outcome_class: Literal[
        "delivered", "non_question", "refused_or_blocked", "failed", "unclassified"
    ] | None = None
    served_tier: str | None = None
    full_turn_latency_ms: float | None = None
    tool_calls: list[CanonicalToolCall] | None = None
    observation_names: list[str] = Field(default_factory=list)
    score_names: list[str] = Field(default_factory=list)
    field_availability: dict[str, FieldAvailability] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _sanitize_import_values(cls, raw: Any) -> Any:
        """Accept legacy adapter inputs but never retain raw chat PII/text."""

        if not isinstance(raw, dict):
            return raw
        values = dict(raw)
        user_id = values.pop("user_id", None)
        question = values.pop("original_question", None)
        answer = values.pop("answer", None)
        turn_outcome = values.pop("turn_outcome", None)
        values.pop("root_input", None)
        values.pop("root_output", None)

        if values.get("user_id_hash") is None:
            values["user_id_hash"] = _hash_identifier(user_id)
        if values.get("question_sanitized") is None:
            values["question_sanitized"] = _sanitize_text(question)
        if values.get("answer_sanitized") is None:
            values["answer_sanitized"] = _sanitize_text(answer)
        if values.get("outcome") is None:
            values["outcome"] = turn_outcome if isinstance(turn_outcome, str) else None
        availability = dict(values.get("field_availability") or {})
        availability.pop("root_input", None)
        availability.pop("root_output", None)
        _move_availability(availability, "user_id", "user_id_hash", values["user_id_hash"])
        _move_availability(
            availability, "original_question", "question_sanitized", values["question_sanitized"]
        )
        _move_availability(availability, "answer", "answer_sanitized", values["answer_sanitized"])
        raw_outcome_availability = availability.pop("turn_outcome", None)
        if raw_outcome_availability is not None:
            availability["outcome"] = (
                raw_outcome_availability if values["outcome"] is not None else "unavailable"
            )
        elif "outcome" not in availability:
            availability["outcome"] = "unavailable"
        availability["outcome_class"] = (
            "derived" if values.get("outcome_class") is not None else "unavailable"
        )
        values["field_availability"] = availability
        return values

    @property
    def turn_outcome(self) -> str | None:
        """Backward-compatible read alias; new consumers use ``outcome``."""

        return self.outcome


def _hash_identifier(value: Any) -> str | None:
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        return None
    identifier = str(value)
    if not identifier or identifier.strip().lower() == "anonymous":
        return None
    return hashlib.sha256(f"amul-oan-api:{identifier}".encode()).hexdigest()


def _sanitize_text(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, str):
        return None
    return {"chars": len(value), "sha256": hashlib.sha256(value.encode()).hexdigest()}


def _move_availability(
    availability: dict[str, FieldAvailability], source: str, target: str, value: Any
) -> None:
    availability.pop(source, None)
    availability[target] = "derived" if value is not None else "unavailable"


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
    user_id: str | int | None = None
    variant: str | None = None


class ChatC2MetadataSchema(BaseModel):
    """Fields observed on c2 roots and their overlapping c2b/c2c extensions."""

    model_config = ConfigDict(extra="ignore")

    pipeline: str | None = None
    channel: str | None = None
    source_lang: str | None = None
    target_lang: str | None = None
    user_id: str | int | None = None


class ChatC2TraceSchema(BaseModel):
    """c2 root contract; final answer remains in its observations."""

    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    name: Literal["chat.default", "chat.translation"]
    timestamp: datetime
    session_id: str | None = Field(default=None, validation_alias="sessionId")
    input: dict[str, Any] | None = None
    output: Any | None = None
    metadata: ChatC2MetadataSchema = Field(default_factory=ChatC2MetadataSchema)


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
    user_id: str | int | None = None
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


class ChatC4MetadataSchema(BaseModel):
    """c4 overlaps the c3b-to-c5 profile-key transition."""

    model_config = ConfigDict(extra="ignore")

    pipeline: str | None = None
    channel: str | None = None
    source_lang: str | None = None
    target_lang: str | None = None
    user_id: str | None = None
    variant: str | None = None
    pipeline_profile: str | None = None


class ChatC4TraceSchema(BaseModel):
    """c4 root contract; tool calls are carried by child observations."""

    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    name: Literal["Amul AI Agent"]
    timestamp: datetime
    session_id: str | None = Field(default=None, validation_alias="sessionId")
    input: dict[str, Any] | None = None
    output: Any | None = None
    metadata: ChatC4MetadataSchema = Field(default_factory=ChatC4MetadataSchema)


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
