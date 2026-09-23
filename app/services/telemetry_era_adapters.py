"""Adapters from historical Langfuse telemetry eras to canonical chat turns."""

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from app.models.telemetry_analytics import (
    CanonicalChatTurn,
    ChatC2TraceSchema,
    ChatC3TraceSchema,
    ChatC4TraceSchema,
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


class ChatC2Adapter:
    """Adapt c2 turns when the bundle contains its agent observation.

    c2 root payloads do not contain the final answer. The target-language answer
    is reconstructed from ``stream_translation`` when present, otherwise from
    the agent observation's recorded final result. A separate pretranslation
    trace cannot be joined safely from name or time alone, so the original
    question remains unavailable here.
    """

    era_id = "chat.c2"

    @classmethod
    def adapt(
        cls,
        trace: ChatC2TraceSchema,
        *,
        observations: Sequence[Mapping[str, Any]] = (),
        scores: Sequence[LangfuseScoreSchema] = (),
        source_era_extensions: list[str] | None = None,
        user_id_semantics: str = "query_param_then_anonymous",
        original_question: str | None = None,
    ) -> CanonicalChatTurn:
        agent_observation = _find_agent_observation(observations)
        if agent_observation is None:
            raise UnsupportedTelemetryEra(
                "chat.c2 requires an 'Amul AI Agent run' observation to distinguish an agent turn from c2 noise"
            )
        answer = _c2_answer(observations, agent_observation)

        return CanonicalChatTurn(
            source_era=cls.era_id,
            source_schema_version="chat.c2.v1",
            source_era_extensions=source_era_extensions or [],
            source_trace_id=trace.id,
            source_trace_name=trace.name,
            timestamp=trace.timestamp,
            session_id=trace.session_id,
            user_id=_identifier_or_none(trace.metadata.user_id),
            user_id_semantics=user_id_semantics,
            channel=trace.metadata.channel,
            pipeline=trace.metadata.pipeline,
            source_lang=trace.metadata.source_lang,
            target_lang=trace.metadata.target_lang,
            original_question=original_question,
            answer=answer,
            root_input=trace.input,
            root_output=trace.output,
            observation_names=_names(observations),
            score_names=[score.name for score in scores],
            field_availability={
                "session_id": _availability(trace.session_id),
                "user_id": _availability(trace.metadata.user_id),
                "channel": _availability(trace.metadata.channel),
                "pipeline": _availability(trace.metadata.pipeline),
                "pipeline_profile": "unavailable",
                "source_lang": _availability(trace.metadata.source_lang),
                "target_lang": _availability(trace.metadata.target_lang),
                "original_question": _availability(original_question),
                "answer": _availability(answer),
                "root_input": _availability(trace.input),
                "root_output": _availability(trace.output),
                "persona": "unavailable",
                "turn_outcome": "unavailable",
                "served_tier": "unavailable",
                "full_turn_latency_ms": "unavailable",
                "tool_calls": "unavailable",
                "scores": "unavailable",
            },
        )


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
            user_id=_identifier_or_none(trace.metadata.user_id),
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
            user_id=_identifier_or_none(trace.metadata.user_id),
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


class ChatC4Adapter:
    """Adapt c4 roots and derive tool-call history from TOOL observations."""

    era_id = "chat.c4"

    @classmethod
    def adapt(
        cls,
        trace: ChatC4TraceSchema,
        *,
        observations: Sequence[Mapping[str, Any]] = (),
        scores: Sequence[LangfuseScoreSchema] = (),
    ) -> CanonicalChatTurn:
        tool_calls = _tool_calls(observations)
        pipeline_profile = trace.metadata.pipeline_profile or trace.metadata.variant
        profile_availability = (
            "recorded"
            if trace.metadata.pipeline_profile is not None
            else "derived" if trace.metadata.variant is not None else "unavailable"
        )
        return CanonicalChatTurn(
            source_era=cls.era_id,
            source_schema_version="chat.c4.v1",
            source_era_extensions=["chat.c3b"] if trace.metadata.variant is not None else [],
            source_trace_id=trace.id,
            source_trace_name=trace.name,
            timestamp=trace.timestamp,
            session_id=trace.session_id,
            user_id=_identifier_or_none(trace.metadata.user_id),
            user_id_semantics="jwt_phone_then_query_param_then_anonymous",
            channel=trace.metadata.channel,
            pipeline=trace.metadata.pipeline,
            pipeline_profile=pipeline_profile,
            source_lang=trace.metadata.source_lang,
            target_lang=trace.metadata.target_lang,
            original_question=None,
            answer=trace.output,
            tool_calls=tool_calls,
            root_input=trace.input,
            root_output=trace.output,
            observation_names=_names(observations),
            score_names=[score.name for score in scores],
            field_availability={
                "session_id": _availability(trace.session_id),
                "user_id": _availability(trace.metadata.user_id),
                "channel": _availability(trace.metadata.channel),
                "pipeline": _availability(trace.metadata.pipeline),
                "pipeline_profile": profile_availability,
                "source_lang": _availability(trace.metadata.source_lang),
                "target_lang": _availability(trace.metadata.target_lang),
                "original_question": "unavailable",
                "answer": _availability(trace.output),
                "root_input": _availability(trace.input),
                "root_output": _availability(trace.output),
                "persona": "unavailable",
                "turn_outcome": "unavailable",
                "served_tier": "unavailable",
                "full_turn_latency_ms": "unavailable",
                "tool_calls": "derived" if tool_calls else "unavailable",
                "scores": "unavailable",
            },
        )


class ChatC6Adapter:
    """Adapt c6 root spans."""

    era_id = "chat.c6"
    @classmethod
    def adapt(
        cls,
        trace: ChatC6TraceSchema,
        *,
        observations: Sequence[Mapping[str, Any]] = (),
        scores: Sequence[LangfuseScoreSchema] = (),
        source_era_extensions: list[str] | None = None,
    ) -> CanonicalChatTurn:
        score_values = {score.name: _string_or_none(score.value) for score in scores}
        root_input = trace.input

        return CanonicalChatTurn(
            source_era=cls.era_id,
            source_schema_version="chat.c6.v1",
            source_era_extensions=source_era_extensions or [],
            source_trace_id=trace.id,
            source_trace_name=trace.name,
            timestamp=trace.timestamp,
            session_id=trace.session_id,
            user_id=_identifier_or_none(trace.metadata.user_id),
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
    related_traces: Sequence[Mapping[str, Any]] = (),
    era_registry: TelemetryEraRegistry | None = None,
) -> CanonicalChatTurn:
    """Resolve and adapt a supported chat trace using name *and* timestamp."""

    timestamp = _parse_timestamp(trace.get("timestamp") or trace.get("startTime"))
    name = trace.get("name")
    parsed_scores = [LangfuseScoreSchema.model_validate(score) for score in scores]
    registry = era_registry or TelemetryEraRegistry.from_yaml(default_era_registry_path())
    c2 = registry.require("chat.c2")
    c2b = registry.require("chat.c2b")
    c2c = registry.require("chat.c2c")
    c3 = registry.require("chat.c3")
    c4 = registry.require("chat.c4")
    c5 = registry.require("chat.c5")
    c6 = registry.require("chat.c6")
    c6b = registry.require("chat.c6b")
    c6c = registry.require("chat.c6c")
    c7 = registry.require("chat.c7")
    c8 = registry.require("chat.c8")

    if name in c2.root_trace_names and c2.valid_from <= timestamp < c3.valid_from:
        raw = dict(trace)
        raw["timestamp"] = timestamp
        extensions = []
        if timestamp >= c2b.valid_from:
            extensions.append("chat.c2b")
        if timestamp >= c2c.valid_from:
            extensions.append("chat.c2c")
        return ChatC2Adapter.adapt(
            ChatC2TraceSchema.model_validate(raw),
            observations=observations,
            scores=parsed_scores,
            source_era_extensions=extensions,
            user_id_semantics=(
                "jwt_phone_then_query_param_then_anonymous"
                if timestamp >= c2b.valid_from
                else "query_param_then_anonymous"
            ),
            original_question=_c2_original_question(raw, related_traces),
        )
    if name == ChatC3Adapter._trace_name and c3.valid_from <= timestamp < c4.valid_from:
        raw = dict(trace)
        raw["timestamp"] = timestamp
        return ChatC3Adapter.adapt(
            ChatC3TraceSchema.model_validate(raw), observations=observations, scores=parsed_scores
        )
    if name == ChatC3Adapter._trace_name and c4.valid_from <= timestamp < c5.valid_from:
        raw = dict(trace)
        raw["timestamp"] = timestamp
        return ChatC4Adapter.adapt(
            ChatC4TraceSchema.model_validate(raw), observations=observations, scores=parsed_scores
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
        extensions = _c6_extensions(timestamp, raw, parsed_scores, c6b=c6b, c6c=c6c, c7=c7)
        return ChatC6Adapter.adapt(
            ChatC6TraceSchema.model_validate(raw),
            observations=observations,
            scores=parsed_scores,
            source_era_extensions=extensions,
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


def _identifier_or_none(value: Any) -> str | None:
    if isinstance(value, (str, int)):
        return str(value) or None
    return None


def _availability(value: Any) -> str:
    return "recorded" if value is not None else "unavailable"


def _names(items: Sequence[Mapping[str, Any]]) -> list[str]:
    return [name for item in items if isinstance((name := item.get("name")), str)]


def _find_agent_observation(items: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for item in items:
        name = item.get("name")
        attributes = _attributes(item)
        if (
            isinstance(name, str)
            and name.startswith("Amul AI Agent run")
            or attributes.get("agent_name") == "Amul AI Agent"
            or attributes.get("logfire.msg") == "Amul AI Agent run"
        ):
            return item
    return None


def _c2_answer(
    observations: Sequence[Mapping[str, Any]], agent_observation: Mapping[str, Any]
) -> Any | None:
    for observation in observations:
        metadata = observation.get("metadata")
        stage = metadata.get("pipeline_stage") if isinstance(metadata, Mapping) else None
        if stage == "stream_translation" and observation.get("output") is not None:
            return observation["output"]
    attributes = _attributes(agent_observation)
    return attributes.get("final_result") or agent_observation.get("output")


def _tool_calls(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    calls = []
    for item in items:
        if item.get("type") != "TOOL":
            continue
        attributes = _attributes(item)
        calls.append(
            {
                "observation_id": item.get("id"),
                "name": attributes.get("gen_ai.tool.name") or item.get("name"),
                "call_id": attributes.get("gen_ai.tool.call.id"),
                "input": item.get("input"),
                "output": item.get("output"),
            }
        )
    return calls


def _attributes(item: Mapping[str, Any]) -> Mapping[str, Any]:
    direct = item.get("attributes")
    if isinstance(direct, Mapping):
        return direct
    metadata = item.get("metadata")
    nested = metadata.get("attributes") if isinstance(metadata, Mapping) else None
    return nested if isinstance(nested, Mapping) else {}


def _c2_original_question(
    trace: Mapping[str, Any], related_traces: Sequence[Mapping[str, Any]]
) -> str | None:
    """Return a question only from an explicitly supplied, same-session pretranslation trace.

    The adapter never performs a global time-based join. Callers that have
    already fetched related records may provide them as a bundle; a missing
    pretranslation trace remains an honest historical absence.
    """

    session_id = _session_id(trace)
    if session_id is None:
        return None
    for related in related_traces:
        if _session_id(related) != session_id or not _is_query_pretranslation(related):
            continue
        raw_input = related.get("input")
        if isinstance(raw_input, Mapping):
            question = _string_or_none(raw_input.get("text"))
            if question is not None:
                return question
    return None


def _session_id(trace: Mapping[str, Any]) -> str | None:
    for key in ("sessionId", "session_id"):
        session_id = _identifier_or_none(trace.get(key))
        if session_id is not None:
            return session_id
    metadata = trace.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    session_id = _identifier_or_none(metadata.get("session_id"))
    if session_id is not None:
        return session_id
    attributes = metadata.get("attributes")
    return _identifier_or_none(attributes.get("session.id")) if isinstance(attributes, Mapping) else None


def _is_query_pretranslation(trace: Mapping[str, Any]) -> bool:
    metadata = trace.get("metadata")
    return isinstance(metadata, Mapping) and metadata.get("pipeline_stage") == "query_pretranslation"


def _c6_extensions(
    timestamp: datetime,
    trace: Mapping[str, Any],
    scores: Sequence[LangfuseScoreSchema],
    *,
    c6b: Any,
    c6c: Any,
    c7: Any,
) -> list[str]:
    """Label overlapping c6 additions only when their data is actually present."""

    score_names = {score.name for score in scores}
    metadata = trace.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    root_input = trace.get("input")
    root_input = root_input if isinstance(root_input, Mapping) else {}
    extensions = []
    if timestamp >= c6b.valid_from and "turn_outcome" in score_names:
        extensions.append("chat.c6b")
    if timestamp >= c6c.valid_from and "served_tier" in score_names:
        extensions.append("chat.c6c")
    if timestamp >= c7.valid_from and (root_input.get("persona") is not None or metadata.get("persona") is not None):
        extensions.append("chat.c7")
    return extensions
