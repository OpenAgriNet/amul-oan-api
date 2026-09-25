"""Adapters from historical Langfuse voice eras to CanonicalVoiceTurn.

Only the two root renames (v3, v4) pick an adapter. Later eras just add keys and
become extensions when both the date and the key match. v1 has nothing to match.
A trace stamped with ``amul.schema_version`` is routed by the stamp, not the date,
and read through telemetry/mappings/voice.yaml. Its source_era is the stamp.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.models.telemetry_analytics import LangfuseScoreSchema
from app.models.telemetry_voice_analytics import (
    CanonicalVoiceTurn,
    SanitizedVoiceText,
    VoiceV0TraceSchema,
    VoiceV3TraceSchema,
    VoiceV4TraceSchema,
)
from app.services.telemetry_era_adapters import UnsupportedTelemetryEra
from app.services.telemetry_era_registry import (
    EraBoundary,
    TelemetryEraRegistry,
    TelemetryEraRegistryError,
    default_era_registry_path,
    load_registry_file,
)
from app.services.telemetry_mappings import ContractMapping, default_mappings_path, load_mappings, value_at


# Raw caller id on every voice trace since 8d82835, documented as the farmer's phone.
_USER_ID_SEMANTICS = "request_user_id_expected_phone_then_anonymous"

# A recorded outcome missing from voice_outcome_vocabulary. Kept visible so a new
# outcome shows up in counts instead of quietly falling out of every bucket.
UNCLASSIFIED_OUTCOME = "unclassified"

SCHEMA_VERSION_KEY = "amul.schema_version"

# Spans added by deeea7a, the voice.v2 commit.
_V2_EXTERNAL_API_OBSERVATIONS = frozenset(
    {
        "create_ai_call_api",
        "fetch_animal_amulpashudhan",
        "fetch_animal_herdman",
        "fetch_farmer_amulpashudhan",
        "fetch_farmer_herdman",
        "marqo_search",
        "send_nudge_message_raya",
    }
)


class VoiceOutcomeVocabulary:
    """Outcome buckets read from ``voice_outcome_vocabulary`` in telemetry/eras.yaml."""

    def __init__(self, bucket_by_outcome: Mapping[str, str]):
        self._bucket_by_outcome = dict(bucket_by_outcome)

    @classmethod
    def from_yaml(cls, path: Path) -> "VoiceOutcomeVocabulary":
        payload = load_registry_file(path)
        raw = payload.get("voice_outcome_vocabulary") if isinstance(payload, Mapping) else None
        if not isinstance(raw, Mapping):
            raise TelemetryEraRegistryError("telemetry/eras.yaml has no voice_outcome_vocabulary")
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "VoiceOutcomeVocabulary":
        bucket_by_outcome: dict[str, str] = {}
        for bucket, outcomes in raw.items():
            # e.g. pre_v3_convention, a note rather than a bucket
            if not isinstance(outcomes, list):
                continue
            for outcome in outcomes:
                if not isinstance(outcome, str):
                    continue
                if bucket == UNCLASSIFIED_OUTCOME:
                    raise TelemetryEraRegistryError(f"{UNCLASSIFIED_OUTCOME!r} is reserved for unknown outcomes")
                if outcome in bucket_by_outcome and bucket_by_outcome[outcome] != bucket:
                    raise TelemetryEraRegistryError(
                        f"voice outcome {outcome!r} is listed under both "
                        f"{bucket_by_outcome[outcome]!r} and {bucket!r}"
                    )
                bucket_by_outcome[outcome] = str(bucket)
        return cls(bucket_by_outcome)

    def classify(self, outcome: str | None) -> str | None:
        if not outcome:
            return None
        return self._bucket_by_outcome.get(outcome, UNCLASSIFIED_OUTCOME)


class VoiceV0Adapter:
    """Pre-VoiceTrace agent roots.

    The output is the whole message history and is never read, so question,
    answer and outcome stay unavailable.
    """

    era_id = "voice.v0"

    @classmethod
    def adapt(
        cls,
        trace: VoiceV0TraceSchema,
        *,
        observations: Sequence[Mapping[str, Any]] = (),
        scores: Sequence[LangfuseScoreSchema] = (),
        source_era_extensions: list[str] | None = None,
    ) -> CanonicalVoiceTurn:
        metadata = trace.metadata
        session_id = _string_or_none(trace.session_id)
        process_id = _identifier_or_none(metadata.process_id)
        user_id = _identifier_or_none(trace.user_id)
        provider = _string_or_none(metadata.provider)
        source_lang = _string_or_none(metadata.source_lang)
        target_lang = _string_or_none(metadata.target_lang)
        variant = _string_or_none(metadata.pipeline_variant)

        # "Voice Agent run" also served signed-in callers before the signed-in agent existed.
        recorded_signed_in = _bool_or_none(metadata.signed_in)
        if recorded_signed_in is not None:
            signed_in, signed_in_availability = recorded_signed_in, "recorded"
        elif trace.name == "Voice Agent Signed In run":
            signed_in, signed_in_availability = True, "derived"
        else:
            signed_in, signed_in_availability = None, "unavailable"

        return CanonicalVoiceTurn(
            source_era=cls.era_id,
            source_schema_version="voice.v0.v1",
            source_era_extensions=source_era_extensions or [],
            source_trace_id=trace.id,
            source_trace_name=trace.name,
            timestamp=trace.timestamp,
            session_id=session_id,
            process_id=process_id,
            user_id=user_id,
            user_id_semantics=_USER_ID_SEMANTICS,
            signed_in=signed_in,
            provider=provider,
            pipeline_profile=variant,
            source_lang=source_lang,
            target_lang=target_lang,
            observation_names=_names(observations),
            score_names=[score.name for score in scores],
            field_availability={
                "session_id": _availability(session_id),
                "process_id": _availability(process_id),
                "user_id": _availability(user_id),
                "user_id_hash": "unavailable",
                "signed_in": signed_in_availability,
                "channel": "derived",
                "provider": _availability(provider),
                "call_type": "unavailable",
                "route": "unavailable",
                "pipeline_profile": "derived" if variant else "unavailable",
                "source_lang": _availability(source_lang),
                "target_lang": _availability(target_lang),
                "question_sanitized": "unavailable",
                "answer_sanitized": "unavailable",
                "outcome": "unavailable",
                "outcome_class": "unavailable",
                "full_turn_latency_ms": "unavailable",
                "stage_totals_ms": "unavailable",
                "timings_ms": "unavailable",
            },
        )


class VoiceV3Adapter:
    """Adapt voice_request roots, the first VoiceTrace contract."""

    era_id = "voice.v3"

    @classmethod
    def adapt(
        cls,
        trace: VoiceV3TraceSchema,
        *,
        outcome_vocabulary: VoiceOutcomeVocabulary,
        observations: Sequence[Mapping[str, Any]] = (),
        scores: Sequence[LangfuseScoreSchema] = (),
        source_era_extensions: list[str] | None = None,
    ) -> CanonicalVoiceTurn:
        return _adapt_voice_turn(
            trace,
            era_id=cls.era_id,
            source_schema_version="voice.v3.v1",
            outcome_vocabulary=outcome_vocabulary,
            observations=observations,
            scores=scores,
            source_era_extensions=source_era_extensions,
        )


class VoiceV4Adapter:
    """Adapt agent_journey roots. v4 only renamed v3's root; v5/v5b add keys."""

    era_id = "voice.v4"

    @classmethod
    def adapt(
        cls,
        trace: VoiceV4TraceSchema,
        *,
        outcome_vocabulary: VoiceOutcomeVocabulary,
        observations: Sequence[Mapping[str, Any]] = (),
        scores: Sequence[LangfuseScoreSchema] = (),
        source_era_extensions: list[str] | None = None,
    ) -> CanonicalVoiceTurn:
        return _adapt_voice_turn(
            trace,
            era_id=cls.era_id,
            source_schema_version="voice.v4.v1",
            outcome_vocabulary=outcome_vocabulary,
            observations=observations,
            scores=scores,
            source_era_extensions=source_era_extensions,
        )


def adapt_voice_trace(
    trace: Mapping[str, Any],
    *,
    observations: Sequence[Mapping[str, Any]] = (),
    scores: Sequence[Mapping[str, Any]] = (),
    era_registry: TelemetryEraRegistry | None = None,
    outcome_vocabulary: VoiceOutcomeVocabulary | None = None,
    voice_mappings: Mapping[str, ContractMapping] | None = None,
) -> CanonicalVoiceTurn:
    """Resolve and adapt a voice trace by its stamp, or by name *and* timestamp when unstamped."""

    timestamp = _parse_timestamp(trace.get("timestamp") or trace.get("startTime"))
    name = trace.get("name")
    parsed_scores = [LangfuseScoreSchema.model_validate(score) for score in scores]
    vocabulary = outcome_vocabulary or VoiceOutcomeVocabulary.from_yaml(default_era_registry_path())

    raw = dict(trace)
    raw["timestamp"] = timestamp
    metadata = _mapping_or_none(trace.get("metadata")) or {}
    raw["metadata"] = metadata

    if SCHEMA_VERSION_KEY in metadata:
        return _adapt_stamped_voice_trace(
            raw,
            metadata[SCHEMA_VERSION_KEY],
            mappings=voice_mappings or load_voice_mappings(),
            outcome_vocabulary=vocabulary,
            observations=observations,
            scores=parsed_scores,
        )

    registry = era_registry or TelemetryEraRegistry.from_yaml(default_era_registry_path(), section="voice_eras")
    v0 = registry.require("voice.v0")
    v2 = registry.require("voice.v2")
    v3 = registry.require("voice.v3")
    v3b = registry.require("voice.v3b")
    v4 = registry.require("voice.v4")
    v5 = registry.require("voice.v5")
    v5b = registry.require("voice.v5b")

    if name in v0.root_trace_names and v0.valid_from <= timestamp < (v0.valid_to or v3.valid_from):
        _require_high_confidence(v3)
        extensions = []
        if timestamp >= v2.valid_from and _has_v2_signal(observations):
            extensions.append("voice.v2")
        return VoiceV0Adapter.adapt(
            VoiceV0TraceSchema.model_validate(raw),
            observations=observations,
            scores=parsed_scores,
            source_era_extensions=extensions,
        )
    if name in v3.root_trace_names and v3.valid_from <= timestamp < (v3.valid_to or v4.valid_from):
        _require_high_confidence(v3)
        _require_high_confidence(v4)
        return VoiceV3Adapter.adapt(
            VoiceV3TraceSchema.model_validate(raw),
            outcome_vocabulary=vocabulary,
            observations=observations,
            scores=parsed_scores,
            source_era_extensions=_voice_turn_extensions(timestamp, metadata, v3b=v3b),
        )
    # Open-ended until eras.yaml gives v4 a valid_to, e.g. once every trace is stamped.
    v4_open = v4.valid_to is None or timestamp < v4.valid_to
    if name in v4.root_trace_names and v4.valid_from <= timestamp and v4_open:
        _require_high_confidence(v4)
        return VoiceV4Adapter.adapt(
            VoiceV4TraceSchema.model_validate(raw),
            outcome_vocabulary=vocabulary,
            observations=observations,
            scores=parsed_scores,
            source_era_extensions=_voice_turn_extensions(timestamp, metadata, v3b=v3b, v5=v5, v5b=v5b),
        )
    raise UnsupportedTelemetryEra(
        f"No voice adapter registered for trace name={name!r} timestamp={timestamp.isoformat()}"
    )


def load_voice_mappings(path: Path | None = None) -> dict[str, ContractMapping]:
    return load_mappings(path or default_mappings_path("voice"), allowed_fields=_MAPPED_FIELDS)


def _adapt_stamped_voice_trace(
    raw: Mapping[str, Any],
    stamp: Any,
    *,
    mappings: Mapping[str, ContractMapping],
    outcome_vocabulary: VoiceOutcomeVocabulary,
    observations: Sequence[Mapping[str, Any]],
    scores: Sequence[LangfuseScoreSchema],
) -> CanonicalVoiceTurn:
    mapping = mappings.get(stamp) if isinstance(stamp, str) else None
    if mapping is None:
        raise UnsupportedTelemetryEra(f"Unknown voice schema version {stamp!r}")
    if raw.get("name") != mapping.root:
        raise UnsupportedTelemetryEra(f"{stamp} is emitted on {mapping.root!r} roots, not {raw.get('name')!r}")
    # The stamp is the era, so the turn doesn't wait on an eras.yaml entry.
    return _adapt_mapped_voice_turn(
        raw,
        mapping,
        era_id=mapping.schema_version,
        outcome_vocabulary=outcome_vocabulary,
        observations=observations,
        scores=scores,
    )


def _adapt_mapped_voice_turn(
    raw: Mapping[str, Any],
    mapping: ContractMapping,
    *,
    era_id: str,
    outcome_vocabulary: VoiceOutcomeVocabulary,
    observations: Sequence[Mapping[str, Any]],
    scores: Sequence[LangfuseScoreSchema],
) -> CanonicalVoiceTurn:
    values = {
        field: next(
            (value for path in mapping.fields.get(field, ()) if (value := parse(value_at(raw, path))) is not None),
            None,
        )
        for field, parse in _MAPPED_FIELDS.items()
    }
    outcome_class = outcome_vocabulary.classify(values["outcome"])
    availability = {field: _availability(value) for field, value in values.items()}
    availability["channel"] = "derived"
    availability["outcome_class"] = "derived" if outcome_class else "unavailable"

    # The stamp names the whole contract, so there are no dated extensions to guess.
    return CanonicalVoiceTurn(
        source_era=era_id,
        source_schema_version=mapping.schema_version,
        source_trace_id=raw.get("id"),
        source_trace_name=mapping.root,
        timestamp=raw["timestamp"],
        user_id_semantics=_USER_ID_SEMANTICS,
        outcome_class=outcome_class,
        observation_names=_names(observations),
        score_names=[score.name for score in scores],
        field_availability=availability,
        **values,
    )


def _adapt_voice_turn(
    trace: VoiceV3TraceSchema,
    *,
    era_id: str,
    source_schema_version: str,
    outcome_vocabulary: VoiceOutcomeVocabulary,
    observations: Sequence[Mapping[str, Any]],
    scores: Sequence[LangfuseScoreSchema],
    source_era_extensions: list[str] | None,
) -> CanonicalVoiceTurn:
    metadata = trace.metadata
    session_id = _string_or_none(trace.session_id) or _string_or_none(metadata.session_id)
    process_id = _identifier_or_none(metadata.process_id)
    user_id = _identifier_or_none(trace.user_id)
    user_id_hash = _string_or_none(metadata.user_id_hash)
    provider = _string_or_none(metadata.provider)
    call_type = _string_or_none(metadata.call_type)
    route = _string_or_none(metadata.route)
    source_lang = _string_or_none(metadata.source_lang)
    target_lang = _string_or_none(metadata.target_lang)
    recorded_profile = _string_or_none(metadata.pipeline_profile)
    variant = _string_or_none(metadata.pipeline_variant)
    question = _sanitized_or_none(metadata.query) or _sanitized_or_none(trace.input)
    answer = _sanitized_or_none(metadata.response) or _sanitized_or_none(trace.output)
    outcome = _string_or_none(metadata.outcome)
    outcome_class = outcome_vocabulary.classify(outcome)
    total_ms = _float_or_none(metadata.total_ms)
    stage_totals_ms = _float_mapping_or_none(metadata.stage_totals_ms)
    timings_ms = _float_mapping_or_none(metadata.timings_ms)

    if recorded_profile:
        pipeline_profile, profile_availability = recorded_profile, "recorded"
    elif variant:
        pipeline_profile, profile_availability = variant, "derived"
    else:
        pipeline_profile, profile_availability = None, "unavailable"

    return CanonicalVoiceTurn(
        source_era=era_id,
        source_schema_version=source_schema_version,
        source_era_extensions=source_era_extensions or [],
        source_trace_id=trace.id,
        source_trace_name=trace.name,
        timestamp=trace.timestamp,
        session_id=session_id,
        process_id=process_id,
        user_id=user_id,
        user_id_semantics=_USER_ID_SEMANTICS,
        user_id_hash=user_id_hash,
        provider=provider,
        call_type=call_type,
        route=route,
        pipeline_profile=pipeline_profile,
        source_lang=source_lang,
        target_lang=target_lang,
        question_sanitized=question,
        answer_sanitized=answer,
        outcome=outcome,
        outcome_class=outcome_class,
        full_turn_latency_ms=total_ms,
        stage_totals_ms=stage_totals_ms,
        timings_ms=timings_ms,
        observation_names=_names(observations),
        score_names=[score.name for score in scores],
        field_availability={
            "session_id": _availability(session_id),
            "process_id": _availability(process_id),
            "user_id": _availability(user_id),
            "user_id_hash": _availability(user_id_hash),
            "signed_in": "unavailable",
            "channel": "derived",
            "provider": _availability(provider),
            "call_type": _availability(call_type),
            "route": _availability(route),
            "pipeline_profile": profile_availability,
            "source_lang": _availability(source_lang),
            "target_lang": _availability(target_lang),
            "question_sanitized": _availability(question),
            "answer_sanitized": _availability(answer),
            "outcome": _availability(outcome),
            "outcome_class": "derived" if outcome_class else "unavailable",
            "full_turn_latency_ms": _availability(total_ms),
            "stage_totals_ms": _availability(stage_totals_ms),
            "timings_ms": _availability(timings_ms),
        },
    )


def _voice_turn_extensions(
    timestamp: datetime,
    metadata: Mapping[str, Any],
    *,
    v3b: EraBoundary,
    v5: EraBoundary | None = None,
    v5b: EraBoundary | None = None,
) -> list[str]:
    extensions = []
    if timestamp >= v3b.valid_from and _string_or_none(metadata.get("pipeline_variant")):
        extensions.append("voice.v3b")
    if (
        v5 is not None
        and timestamp >= v5.valid_from
        and (
            _string_or_none(metadata.get("pipeline_profile"))
            or any(isinstance(key, str) and key.startswith("pc_") for key in metadata)
        )
    ):
        extensions.append("voice.v5")
    if v5b is not None and timestamp >= v5b.valid_from and metadata.get("outcome") == "outbound_intro":
        extensions.append("voice.v5b")
    return extensions


def _has_v2_signal(observations: Sequence[Mapping[str, Any]]) -> bool:
    return any(name in _V2_EXTERNAL_API_OBSERVATIONS for name in _names(observations))


def _require_high_confidence(era: EraBoundary) -> None:
    if era.valid_from_confidence != "high":
        raise UnsupportedTelemetryEra(
            f"{era.era_id} has a {era.valid_from_confidence or 'unrated'}-confidence production "
            "boundary and needs validation before dispatch"
        )


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise UnsupportedTelemetryEra("Trace timestamp is required")
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _mapping_or_none(value: Any) -> Mapping[str, Any] | None:
    """Accept an object (Langfuse API) or a JSON-encoded object (ClickHouse export)."""
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str) and value.lstrip().startswith("{"):
        try:
            parsed = json.loads(value)
        except ValueError:
            return None
        return parsed if isinstance(parsed, Mapping) else None
    return None


def _sanitized_or_none(value: Any) -> SanitizedVoiceText | None:
    # A plain string would be raw caller text.
    mapping = _mapping_or_none(value)
    if mapping is None or ("sha256" not in mapping and "chars" not in mapping):
        return None
    return SanitizedVoiceText.model_validate(mapping)


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value:
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _float_mapping_or_none(value: Any) -> dict[str, float] | None:
    mapping = _mapping_or_none(value)
    if mapping is None:
        return None
    parsed = {
        str(key): number for key, raw in mapping.items() if (number := _float_or_none(raw)) is not None
    }
    return parsed or None


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return None


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _identifier_or_none(value: Any) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (str, int)):
        return str(value) or None
    return None


def _availability(value: Any) -> str:
    return "recorded" if value is not None else "unavailable"


def _names(items: Sequence[Mapping[str, Any]]) -> list[str]:
    return [name for item in items if isinstance((name := item.get("name")), str)]


# CanonicalVoiceTurn fields a mapping may fill, and how each raw value is read.
# A field name in voice.yaml that isn't here is rejected as a typo.
_MAPPED_FIELDS = {
    "session_id": _string_or_none,
    "process_id": _identifier_or_none,
    "user_id": _identifier_or_none,
    "user_id_hash": _string_or_none,
    "signed_in": _bool_or_none,
    "provider": _string_or_none,
    "call_type": _string_or_none,
    "route": _string_or_none,
    "pipeline_profile": _string_or_none,
    "source_lang": _string_or_none,
    "target_lang": _string_or_none,
    "question_sanitized": _sanitized_or_none,
    "answer_sanitized": _sanitized_or_none,
    "outcome": _string_or_none,
    "full_turn_latency_ms": _float_or_none,
    "stage_totals_ms": _float_mapping_or_none,
    "timings_ms": _float_mapping_or_none,
}
