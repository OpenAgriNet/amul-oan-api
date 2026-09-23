"""Adapters from historical Langfuse telemetry eras to canonical chat turns."""

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from app.models.telemetry_analytics import (
    CanonicalChatTurn,
    ChatC3TraceSchema,
    ChatC5TraceSchema,
    ChatC6TraceSchema,
    ChatC8TraceSchema,
    LangfuseScoreSchema,
)
from app.services.telemetry_era_registry import (
    TelemetryEraRegistry,
    default_era_registry_path,
)


class UnsupportedTelemetryEra(ValueError):
    """Raised when a raw trace does not belong to an adapter's documented era."""


class ChatC3Adapter:
    """Adapt the 2026-05-13..2026-07-24 ``Amul AI Agent`` chat trace shape.

    The c3 root input is agent-internal (for example action/model), not the
    original farmer question.  The adapter deliberately leaves
    ``original_question`` unavailable rather than treating that payload as a
    question.  c3b's ``metadata.variant`` is normalized to ``pipeline_profile``.
    """

    era_id = "chat.c3"
    _trace_name = "Amul AI Agent"

    @classmethod
    def adapt(
        cls,
        trace: ChatC3TraceSchema,
        *,
        observations: Sequence[Mapping[str, Any]] = (),
        scores: Sequence[LangfuseScoreSchema] = (),
    ) -> CanonicalChatTurn:
        variant = trace.metadata.variant

        return CanonicalChatTurn(
            source_era=cls.era_id,
            source_schema_version="chat.c3.v1",
            source_era_extensions=["chat.c3b"] if variant is not None else [],
            source_trace_id=trace.id,
            source_trace_name=trace.name,
            timestamp=trace.timestamp,
            session_id=trace.session_id,
            user_id=trace.metadata.user_id,
            user_id_semantics="jwt_phone_then_query_param_then_anonymous",
            channel=trace.metadata.channel,
            pipeline=trace.metadata.pipeline,
            pipeline_profile=variant,
            source_lang=trace.metadata.source_lang,
            target_lang=trace.metadata.target_lang,
            # c3's agent-action input must not be mistaken for a user question.
            original_question=None,
            answer=trace.output,
            root_input=trace.input,
            root_output=trace.output,
            observation_names=_names(observations),
            score_names=[score.name for score in scores],
            field_availability={
                "session_id": "recorded",
                "user_id": "recorded",
                "channel": "recorded",
                "pipeline": "recorded",
                "pipeline_profile": "derived" if variant is not None else "unavailable",
                "source_lang": "recorded",
                "target_lang": "recorded",
                "original_question": "unavailable",
                "answer": "recorded",
                "root_input": "recorded",
                "root_output": "recorded",
                "persona": "unavailable",
                "turn_outcome": "unavailable",
                "served_tier": "unavailable",
                "full_turn_latency_ms": "unavailable",
                "tool_calls": "unavailable",
                "scores": "unavailable",
            },
        )


class ChatC5Adapter:
    """Adapt c5's c3-shaped root trace after the profile-key rename."""

    era_id = "chat.c5"
    @classmethod
    def adapt(
        cls,
        trace: ChatC5TraceSchema,
        *,
        observations: Sequence[Mapping[str, Any]] = (),
        scores: Sequence[LangfuseScoreSchema] = (),
    ) -> CanonicalChatTurn:
        return CanonicalChatTurn(
            source_era=cls.era_id,
            source_schema_version="chat.c5.v1",
            source_trace_id=trace.id,
            source_trace_name=trace.name,
            timestamp=trace.timestamp,
            session_id=trace.session_id,
            user_id=trace.metadata.user_id,
            user_id_semantics="jwt_phone_then_query_param_then_anonymous",
            channel=trace.metadata.channel,
            pipeline=trace.metadata.pipeline,
            pipeline_profile=trace.metadata.pipeline_profile,
            source_lang=trace.metadata.source_lang,
            target_lang=trace.metadata.target_lang,
            original_question=None,
            answer=trace.output,
            root_input=trace.input,
            root_output=trace.output,
            observation_names=_names(observations),
            score_names=[score.name for score in scores],
            field_availability={
                "session_id": "recorded",
                "user_id": "recorded",
                "channel": "recorded",
                "pipeline": "recorded",
                "pipeline_profile": _availability(trace.metadata.pipeline_profile),
                "source_lang": "recorded",
                "target_lang": "recorded",
                "original_question": "unavailable",
                "answer": "recorded",
                "root_input": "recorded",
                "root_output": "recorded",
                "persona": "unavailable",
                "turn_outcome": "unavailable",
                "served_tier": "unavailable",
                "full_turn_latency_ms": "unavailable",
                "tool_calls": "unavailable",
                "scores": "unavailable",
            },
        )


class ChatC6Adapter:
    """Adapt c6+ root spans, including c8's translation-only continuation."""

    era_id = "chat.c6"
    @classmethod
    def adapt(
        cls,
        trace: ChatC6TraceSchema,
        *,
        observations: Sequence[Mapping[str, Any]] = (),
        scores: Sequence[LangfuseScoreSchema] = (),
    ) -> CanonicalChatTurn:
        score_values = {score.name: _string_or_none(score.value) for score in scores}
        root_input = trace.input

        return CanonicalChatTurn(
            source_era=cls.era_id,
            source_schema_version="chat.c6.v1",
            # c8's deployment boundary is documented as unverified; do not
            # assign it from the merge date alone.
            source_era_extensions=[],
            source_trace_id=trace.id,
            source_trace_name=trace.name,
            timestamp=trace.timestamp,
            session_id=trace.session_id,
            user_id=trace.metadata.user_id,
            user_id_semantics="jwt_phone_then_query_param_then_anonymous",
            channel=_string_or_none((root_input or {}).get("channel")) or trace.metadata.channel,
            pipeline=trace.metadata.pipeline,
            pipeline_profile=trace.metadata.pipeline_profile,
            source_lang=_string_or_none((root_input or {}).get("source_lang")) or trace.metadata.source_lang,
            target_lang=_string_or_none((root_input or {}).get("target_lang")) or trace.metadata.target_lang,
            original_question=_string_or_none((root_input or {}).get("query")),
            answer=trace.output,
            persona=_string_or_none((root_input or {}).get("persona")) or trace.metadata.persona,
            turn_outcome=score_values.get("turn_outcome"),
            served_tier=score_values.get("served_tier"),
            root_input=root_input,
            root_output=trace.output,
            observation_names=_names(observations),
            score_names=[score.name for score in scores],
            field_availability={
                "session_id": "recorded",
                "user_id": "recorded",
                "channel": _availability((root_input or {}).get("channel") or trace.metadata.channel),
                "pipeline": _availability(trace.metadata.pipeline),
                "pipeline_profile": _availability(trace.metadata.pipeline_profile),
                "source_lang": _availability((root_input or {}).get("source_lang") or trace.metadata.source_lang),
                "target_lang": _availability((root_input or {}).get("target_lang") or trace.metadata.target_lang),
                "original_question": _availability((root_input or {}).get("query")),
                "answer": _availability(trace.output),
                "root_input": _availability(root_input),
                "root_output": _availability(trace.output),
                "persona": _availability((root_input or {}).get("persona") or trace.metadata.persona),
                "turn_outcome": _availability(score_values.get("turn_outcome")),
                "served_tier": _availability(score_values.get("served_tier")),
                "full_turn_latency_ms": "unavailable",
                "tool_calls": "unavailable",
            },
        )


class ChatC8Adapter:
    """Adapt the verified translation-only continuation of the c6 root shape."""

    era_id = "chat.c8"

    @classmethod
    def adapt(
        cls,
        trace: ChatC8TraceSchema,
        *,
        observations: Sequence[Mapping[str, Any]] = (),
        scores: Sequence[LangfuseScoreSchema] = (),
    ) -> CanonicalChatTurn:
        # c8 preserves c6's root I/O contract while restricting the root name.
        c6_turn = ChatC6Adapter.adapt(trace, observations=observations, scores=scores)
        return c6_turn.model_copy(
            update={
                "source_era": cls.era_id,
                "source_schema_version": "chat.c8.v1",
                "source_era_extensions": [],
            }
        )


def adapt_chat_trace(
    trace: Mapping[str, Any],
    *,
    observations: Sequence[Mapping[str, Any]] = (),
    scores: Sequence[Mapping[str, Any]] = (),
    era_registry: TelemetryEraRegistry | None = None,
) -> CanonicalChatTurn:
    """Resolve and adapt a supported chat trace using name *and* timestamp."""

    timestamp = _parse_timestamp(trace.get("timestamp") or trace.get("startTime"))
    name = trace.get("name")
    parsed_scores = [LangfuseScoreSchema.model_validate(score) for score in scores]
    registry = era_registry or TelemetryEraRegistry.from_yaml(default_era_registry_path())
    c3 = registry.require("chat.c3")
    c5 = registry.require("chat.c5")
    c6 = registry.require("chat.c6")
    c8 = registry.require("chat.c8")

    if name == ChatC3Adapter._trace_name and c3.valid_from <= timestamp < c5.valid_from:
        raw = dict(trace)
        raw["timestamp"] = timestamp
        return ChatC3Adapter.adapt(
            ChatC3TraceSchema.model_validate(raw), observations=observations, scores=parsed_scores
        )
    if name == ChatC3Adapter._trace_name and c5.valid_from <= timestamp < (c3.valid_to or c6.valid_from):
        raw = dict(trace)
        raw["timestamp"] = timestamp
        return ChatC5Adapter.adapt(
            ChatC5TraceSchema.model_validate(raw), observations=observations, scores=parsed_scores
        )
    if name in c6.root_trace_names and c6.valid_from <= timestamp < c8.valid_from:
        raw = dict(trace)
        raw["timestamp"] = timestamp
        return ChatC6Adapter.adapt(
            ChatC6TraceSchema.model_validate(raw), observations=observations, scores=parsed_scores
        )
    if name in c8.root_trace_names and timestamp >= c8.valid_from:
        if c8.valid_from_confidence != "high":
            raise UnsupportedTelemetryEra(
                "chat.c8 has a low-confidence production boundary and needs validation before dispatch"
            )
        raw = dict(trace)
        raw["timestamp"] = timestamp
        return ChatC8Adapter.adapt(
            ChatC8TraceSchema.model_validate(raw), observations=observations, scores=parsed_scores
        )
    raise UnsupportedTelemetryEra(f"No adapter registered for trace name={name!r} timestamp={timestamp.isoformat()}")


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise UnsupportedTelemetryEra("Trace timestamp is required")
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _availability(value: Any) -> str:
    return "recorded" if value is not None else "unavailable"


def _names(items: Sequence[Mapping[str, Any]]) -> list[str]:
    return [name for item in items if isinstance((name := item.get("name")), str)]
