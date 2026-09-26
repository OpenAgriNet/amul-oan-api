import hashlib

import pytest

from app.models.telemetry_analytics import (
    CanonicalChatTurn,
    ChatC3TraceSchema,
    ChatC5TraceSchema,
    ChatC6TraceSchema,
    ChatC8TraceSchema,
)
from app.services.telemetry_era_adapters import (
    ChatC3Adapter,
    ChatC5Adapter,
    ChatC6Adapter,
    ChatC8Adapter,
    UnsupportedTelemetryEra,
    adapt_chat_trace as _adapt_chat_trace,
    load_chat_mappings,
)
from app.services.telemetry_mappings import default_mappings_path
from app.services.telemetry_era_registry import OutcomeVocabulary, TelemetryEraRegistry


_CHAT_OUTCOME_VOCABULARY = OutcomeVocabulary.from_mapping(
    {"delivered": ["success"], "failed": ["error", "cancelled"]}
)


def adapt_chat_trace(*args, **kwargs):
    """Keep unit fixtures independent of the registry PR stacked below this branch."""

    kwargs.setdefault("outcome_vocabulary", _CHAT_OUTCOME_VOCABULARY)
    return _adapt_chat_trace(*args, **kwargs)


def _text(value: str) -> dict[str, object]:
    return {"chars": len(value), "sha256": hashlib.sha256(value.encode()).hexdigest()}


def _user_hash(value: str | int) -> str:
    return hashlib.sha256(f"amul-oan-api:{value}".encode()).hexdigest()


@pytest.fixture
def era_registry(tmp_path):
    return _era_registry(tmp_path, c8_confidence="low")


@pytest.fixture
def verified_c8_registry(tmp_path):
    return _era_registry(tmp_path, c8_confidence="high")


def _era_registry(tmp_path, *, c8_confidence):
    registry_path = tmp_path / "eras.yaml"
    registry_path.write_text(
        """
chat_eras:
  - era_id: chat.c2
    valid_from: 2026-03-10
    valid_to: 2026-05-13
    root_trace_names: [chat.default, chat.translation]
  - era_id: chat.c2b
    valid_from: 2026-03-25
    valid_to: null
    root_trace_names: []
  - era_id: chat.c2c
    valid_from: 2026-04-24
    valid_to: null
    root_trace_names: []
  - era_id: chat.c3
    valid_from: 2026-05-13
    valid_to: 2026-08-05
    root_trace_names: [Amul AI Agent]
  - era_id: chat.c4
    valid_from: 2026-07-22
    valid_to: null
    root_trace_names: []
  - era_id: chat.c5
    valid_from: 2026-07-24
    valid_to: null
    root_trace_names: []
  - era_id: chat.c6
    valid_from: 2026-08-05T06:30:00Z
    valid_to: null
    root_trace_names: [chat.default, chat.translation]
  - era_id: chat.c6b
    valid_from: 2026-08-03
    valid_to: null
    root_trace_names: []
  - era_id: chat.c6c
    valid_from: 2026-08-02
    valid_to: null
    root_trace_names: []
  - era_id: chat.c7
    valid_from: 2026-08-18
    valid_to: null
    root_trace_names: []
  - era_id: chat.c8
    valid_from: 2026-09-16
    valid_to: null
    valid_from_confidence: {c8_confidence}
    root_trace_names: [chat.translation]
""".strip().format(c8_confidence=c8_confidence),
        encoding="utf-8",
    )
    return TelemetryEraRegistry.from_yaml(registry_path)


def test_chat_c3_adapter_normalizes_variant_without_inventing_missing_fields():
    source = ChatC3TraceSchema.model_validate(
        {
            "id": "redacted-c3-trace",
            "name": "Amul AI Agent",
            "timestamp": "2026-05-30T23:59:08Z",
            "sessionId": "redacted-session",
            "input": {"action": "Proceed with the query.", "model_name": "gpt-5.1"},
            "output": "<redacted answer>",
            "metadata": {
                "pipeline": "translation",
                "channel": "web",
                "source_lang": "gu",
                "target_lang": "gu",
                "variant": "legacy",
            },
        }
    )
    turn = ChatC3Adapter.adapt(
        source,
        # The historical trace detail showed no associated observations or scores.
        # Do not infer parentage from a separately filtered observations table.
        observations=[],
        scores=[],
    )

    assert turn.schema_version == "chat.canonical.v1"
    assert turn.source_era == "chat.c3"
    assert turn.source_schema_version == "chat.c3.v1"
    assert turn.source_era_extensions == ["chat.c3b"]
    assert turn.pipeline == "translation"
    assert turn.pipeline_profile == "legacy"
    assert turn.field_availability["pipeline_profile"] == "derived"
    assert turn.answer_sanitized.model_dump() == _text("<redacted answer>")
    assert turn.question_sanitized is None
    assert "root_input" not in turn.model_dump()
    assert turn.score_names == []
    assert turn.observation_names == []
    assert turn.field_availability["question_sanitized"] == "unavailable"
    assert turn.outcome is None
    assert turn.outcome_class is None
    assert turn.field_availability["outcome"] == "unavailable"
    assert turn.field_availability["outcome_class"] == "unavailable"
    assert turn.field_availability["tool_calls"] == "unavailable"


def test_resolver_adapts_c2_agent_observation_without_guessing_pretranslation_link(era_registry):
    turn = adapt_chat_trace(
        {
            "id": "redacted-c2-trace",
            "name": "chat.translation",
            "timestamp": "2026-05-12T23:56:49Z",
            "sessionId": "redacted-session",
            "metadata": {
                "pipeline": "translation",
                "channel": "web",
                "source_lang": "gu",
                "target_lang": "gu",
                "user_id": 1234567890,
            },
        },
        observations=[
            {
                "type": "SPAN",
                "name": "Amul AI Agent run (redacted)",
                "metadata": {
                    "attributes": {
                        "agent_name": "Amul AI Agent",
                        "final_result": "<redacted English agent answer>",
                    }
                },
            },
            {
                "type": "GENERATION",
                "name": "stream_translation (redacted)",
                "output": "<redacted target-language answer>",
                "metadata": {"pipeline_stage": "stream_translation"},
            },
        ],
        era_registry=era_registry,
    )

    assert turn.source_era == "chat.c2"
    assert turn.source_era_extensions == ["chat.c2b", "chat.c2c"]
    assert turn.user_id_semantics == "jwt_phone_then_query_param_then_anonymous"
    assert turn.user_id_hash == _user_hash(1234567890)
    assert turn.answer_sanitized.model_dump() == _text("<redacted target-language answer>")
    assert turn.question_sanitized is None
    assert turn.field_availability["answer_sanitized"] == "derived"
    assert turn.field_availability["question_sanitized"] == "unavailable"


def test_resolver_rejects_c2_name_reuse_without_agent_observation(era_registry):
    with pytest.raises(UnsupportedTelemetryEra, match="requires an 'Amul AI Agent run' observation"):
        adapt_chat_trace(
            {"name": "chat.translation", "timestamp": "2026-05-12T23:56:49Z"},
            era_registry=era_registry,
        )


def test_c2_enriches_original_question_only_from_explicit_same_session_pretranslation(era_registry):
    turn = adapt_chat_trace(
        {
            "name": "chat.translation",
            "timestamp": "2026-05-12T23:56:49Z",
            "sessionId": "matching-session",
            "metadata": {"pipeline": "translation"},
        },
        observations=[
            {
                "name": "Amul AI Agent run (redacted)",
                "metadata": {"attributes": {"final_result": "<redacted answer>"}},
            }
        ],
        related_traces=[
            {
                "name": "chat.translation",
                "sessionId": "unrelated-session",
                "timestamp": "2026-05-12T23:56:48Z",
                "input": {"text": "<must not be used>"},
                "metadata": {"pipeline_stage": "query_pretranslation"},
            },
            {
                "name": "chat.translation",
                "sessionId": "matching-session",
                "timestamp": "2026-05-12T23:56:48Z",
                "input": {"text": "<redacted original question>"},
                "metadata": {"pipeline_stage": "query_pretranslation"},
            },
        ],
        era_registry=era_registry,
    )

    assert turn.question_sanitized.model_dump() == _text("<redacted original question>")
    assert turn.field_availability["question_sanitized"] == "derived"


def test_c2_matches_each_turn_to_its_uniquely_nearest_pretranslation(era_registry):
    observations = [
        {
            "name": "Amul AI Agent run (redacted)",
            "metadata": {"attributes": {"final_result": "<redacted answer>"}},
        }
    ]
    related_traces = [
        {
            "name": "chat.translation",
            "sessionId": "shared-session",
            "timestamp": "2026-05-12T10:00:01Z",
            "input": {"text": "<redacted first question>"},
            "metadata": {"pipeline_stage": "query_pretranslation"},
        },
        {
            "name": "chat.translation",
            "sessionId": "shared-session",
            "timestamp": "2026-05-12T10:03:01Z",
            "input": {"text": "<redacted second question>"},
            "metadata": {"pipeline_stage": "query_pretranslation"},
        },
    ]

    first_turn = adapt_chat_trace(
        {"name": "chat.translation", "timestamp": "2026-05-12T10:00:03Z", "sessionId": "shared-session"},
        observations=observations,
        related_traces=related_traces,
        era_registry=era_registry,
    )
    second_turn = adapt_chat_trace(
        {"name": "chat.translation", "timestamp": "2026-05-12T10:03:03Z", "sessionId": "shared-session"},
        observations=observations,
        related_traces=list(reversed(related_traces)),
        era_registry=era_registry,
    )

    assert first_turn.question_sanitized.model_dump() == _text("<redacted first question>")
    assert second_turn.question_sanitized.model_dump() == _text("<redacted second question>")


def test_c2_returns_no_question_for_ambiguous_or_stale_pretranslation(era_registry):
    observations = [
        {
            "name": "Amul AI Agent run (redacted)",
            "metadata": {"attributes": {"final_result": "<redacted answer>"}},
        }
    ]
    turn = adapt_chat_trace(
        {"name": "chat.translation", "timestamp": "2026-05-12T10:00:10Z", "sessionId": "shared-session"},
        observations=observations,
        related_traces=[
            {
                "name": "chat.translation",
                "sessionId": "shared-session",
                "timestamp": "2026-05-12T10:00:05Z",
                "input": {"text": "<redacted first question>"},
                "metadata": {"pipeline_stage": "query_pretranslation"},
            },
            {
                "name": "chat.translation",
                "sessionId": "shared-session",
                "timestamp": "2026-05-12T10:00:15Z",
                "input": {"text": "<redacted second question>"},
                "metadata": {"pipeline_stage": "query_pretranslation"},
            },
            {
                "name": "chat.translation",
                "sessionId": "shared-session",
                "timestamp": "2026-05-12T09:00:10Z",
                "input": {"text": "<stale question>"},
                "metadata": {"pipeline_stage": "query_pretranslation"},
            },
        ],
        era_registry=era_registry,
    )

    assert turn.question_sanitized is None
    assert turn.field_availability["question_sanitized"] == "unavailable"


def test_resolver_adapts_c4_tool_observations(era_registry):
    turn = adapt_chat_trace(
        {
            "id": "redacted-c4-trace",
            "name": "Amul AI Agent",
            "timestamp": "2026-07-23T23:56:41Z",
            "sessionId": "redacted-session",
            "input": {"action": "<redacted action>", "model_name": "<redacted model>"},
            "output": "<redacted answer>",
            "metadata": {
                "pipeline": "translation",
                "variant": "oss",
                "pipeline_profile": "oss",
                "channel": "web",
            },
        },
        observations=[
            {
                "id": "redacted-tool-observation",
                "type": "TOOL",
                "name": "get_farmer_milk_collection_details (redacted)",
                "input": {"farmer_code": "redacted"},
                "output": "<redacted tool response>",
                "metadata": {
                    "attributes": {
                        "gen_ai.tool.name": "get_farmer_milk_collection_details",
                        "gen_ai.tool.call.id": "redacted-call-id",
                    }
                },
            }
        ],
        era_registry=era_registry,
    )

    assert turn.source_era == "chat.c4"
    assert turn.pipeline_profile == "oss"
    assert turn.field_availability["pipeline_profile"] == "recorded"
    assert turn.field_availability["tool_calls"] == "derived"
    assert [call.model_dump() for call in turn.tool_calls] == [
        {
            "tool_name": "get_farmer_milk_collection_details",
            "call_id": "redacted-call-id",
        }
    ]
    assert "redacted-tool-observation" not in turn.model_dump_json()
    assert "redacted tool response" not in turn.model_dump_json()


def test_canonical_tool_calls_strip_arguments_and_results():
    turn = CanonicalChatTurn(
        source_era="test",
        source_schema_version="test.v1",
        source_trace_name="chat.translation",
        timestamp="2026-10-05T10:00:00Z",
        tool_calls=[
            {
                "tool_name": "fetch_farmer",
                "call_id": "stable-call-id",
                "input": {"farmer_data": "must not persist"},
                "output": "must not persist",
            }
        ],
    )

    assert [call.model_dump() for call in turn.tool_calls] == [
        {"tool_name": "fetch_farmer", "call_id": "stable-call-id"}
    ]
    assert "must not persist" not in turn.model_dump_json()


@pytest.mark.parametrize(
    ("adapter", "schema", "name", "timestamp", "root_input"),
    [
        (ChatC5Adapter, ChatC5TraceSchema, "Amul AI Agent", "2026-07-25T10:00:00Z", {"action": "redacted"}),
        (ChatC6Adapter, ChatC6TraceSchema, "chat.translation", "2026-08-10T10:00:00Z", {"query": "redacted"}),
        (ChatC8Adapter, ChatC8TraceSchema, "chat.translation", "2026-09-20T10:00:00Z", {"query": "redacted"}),
    ],
    ids=["c5", "c6", "c8"],
)
def test_later_chat_eras_keep_only_tool_references(adapter, schema, name, timestamp, root_input):
    turn = adapter.adapt(
        schema.model_validate(
            {
                "id": "redacted-trace",
                "name": name,
                "timestamp": timestamp,
                "sessionId": "redacted-session",
                "input": root_input,
                "output": "redacted answer",
                "metadata": {"pipeline": "translation", "channel": "web"},
            }
        ),
        observations=[
            {
                "type": "TOOL",
                "name": "farmer_lookup",
                "input": {"farmer": "must not persist"},
                "output": "must not persist",
                "metadata": {"attributes": {"gen_ai.tool.call.id": "stable-call-id"}},
            }
        ],
    )

    assert [call.model_dump() for call in turn.tool_calls] == [
        {"tool_name": "farmer_lookup", "call_id": "stable-call-id"}
    ]
    assert turn.field_availability["tool_calls"] == "derived"
    assert "must not persist" not in turn.model_dump_json()


def test_resolver_rejects_an_amul_agent_trace_outside_registered_c3_c5_dates(era_registry):
    with pytest.raises(UnsupportedTelemetryEra, match="No adapter registered"):
        adapt_chat_trace(
            {"name": "Amul AI Agent", "timestamp": "2026-08-05T00:00:00Z"},
            era_registry=era_registry,
        )


def test_resolver_uses_c5_schema_after_the_pipeline_profile_rename(era_registry):
    turn = adapt_chat_trace(
        {
            "name": "Amul AI Agent",
            "timestamp": "2026-07-24T00:00:00Z",
            "input": {"action": "<redacted action>"},
            "output": "<redacted answer>",
            "metadata": {
                "pipeline": "translation",
                "pipeline_profile": "oss",
                "channel": "web",
            },
        },
        era_registry=era_registry,
    )

    assert turn.source_era == "chat.c5"
    assert turn.source_schema_version == "chat.c5.v1"
    assert turn.pipeline_profile == "oss"
    assert turn.field_availability["pipeline_profile"] == "recorded"


def test_historical_chat_field_rename_needs_only_a_mapping_change(tmp_path, era_registry):
    mappings_path = tmp_path / "chat.yaml"
    mappings_path.write_text(
        """
chat.c5.v1:
  root: Amul AI Agent
  fields:
    pipeline_profile: [metadata.renamed_profile]
""".strip(),
        encoding="utf-8",
    )

    turn = adapt_chat_trace(
        {
            "name": "Amul AI Agent",
            "timestamp": "2026-07-24T00:00:00Z",
            "metadata": {"renamed_profile": "oss"},
        },
        era_registry=era_registry,
        chat_mappings=load_chat_mappings(mappings_path),
    )

    assert turn.source_schema_version == "chat.c5.v1"
    assert turn.pipeline_profile == "oss"


def test_resolver_adapts_c6_root_input_and_categorical_scores(era_registry):
    turn = adapt_chat_trace(
        {
            "id": "redacted-c6-trace",
            "name": "chat.translation",
            "timestamp": "2026-08-20T10:29:39.198Z",
            "input": {
                "query": "<redacted query>",
                "channel": "web",
                "source_lang": "hi",
                "target_lang": "hi",
                "persona": "farmer",
            },
            "output": "<redacted answer>",
            "metadata": {
                "pipeline": "translation",
                "pipeline_profile": "oss",
            },
        },
        scores=[
            {"name": "turn_outcome", "value": "success"},
            {"name": "served_tier", "value": "agent=vllm:gemma"},
            {"name": "pipeline_profile", "value": "oss"},
        ],
        era_registry=era_registry,
    )

    assert turn.source_era == "chat.c6"
    assert turn.source_schema_version == "chat.c6.v1"
    assert turn.source_era_extensions == ["chat.c6b", "chat.c6c", "chat.c7"]
    assert turn.question_sanitized.model_dump() == _text("<redacted query>")
    assert turn.answer_sanitized.model_dump() == _text("<redacted answer>")
    assert turn.pipeline_profile == "oss"
    assert turn.persona == "farmer"
    assert turn.turn_outcome == "success"
    assert turn.outcome == "success"
    assert turn.outcome_class == "delivered"
    assert turn.served_tier == "agent=vllm:gemma"
    assert turn.field_availability["outcome"] == "recorded"


@pytest.mark.parametrize(
    ("raw_outcome", "outcome_class"),
    [
        ("success", "delivered"),
        ("error", "failed"),
        ("cancelled", "failed"),
        ("future_value", "unclassified"),
        (None, None),
    ],
)
def test_chat_outcomes_use_the_registry_dashboard_taxonomy(raw_outcome, outcome_class):
    vocabulary = OutcomeVocabulary.from_mapping(
        {
            "delivered": ["success"],
            "failed": ["error", "cancelled"],
        }
    )

    assert vocabulary.classify(raw_outcome) == outcome_class


def test_resolver_refuses_low_confidence_c8_boundary(era_registry):
    with pytest.raises(UnsupportedTelemetryEra, match="low-confidence"):
        adapt_chat_trace(
            {"name": "chat.translation", "timestamp": "2026-09-21T10:29:39Z"},
            era_registry=era_registry,
        )


def test_resolver_adapts_c8_only_after_its_boundary_is_verified(verified_c8_registry):
    turn = adapt_chat_trace(
        {
            "name": "chat.translation",
            "timestamp": "2026-09-21T10:29:39Z",
            "input": {"query": "<redacted query>", "channel": "web"},
            "output": "<redacted answer>",
            "metadata": {"pipeline": "translation", "pipeline_profile": "oss"},
        },
        era_registry=verified_c8_registry,
    )

    assert turn.source_era == "chat.c8"
    assert turn.source_schema_version == "chat.c8.v1"


def _stamped_chat_trace(stamp="chat.turn.v1"):
    return {
        "id": "redacted-stamped-trace",
        "name": "chat.translation",
        "timestamp": "2026-10-05T10:00:00Z",
        "sessionId": "redacted-session",
        "input": {
            "query": "<redacted question>",
            "channel": "web",
            "source_lang": "gu",
            "target_lang": "gu",
            "persona": "farmer",
        },
        "output": "<redacted answer>",
        "metadata": {
            "amul.schema_version": stamp,
            "service": "amul-oan-api",
            "release": "test-release-sha",
            "user_id": "redacted-user",
            "pipeline": "translation",
            "pipeline_profile": "oss",
        },
    }


def test_stamped_chat_trace_uses_the_shared_mapping_engine_without_an_era_registry():
    turn = adapt_chat_trace(
        _stamped_chat_trace(),
        scores=[{"name": "turn_outcome", "value": "success"}],
    )

    assert turn.source_era == "chat.turn.v1"
    assert turn.source_schema_version == "chat.turn.v1"
    assert turn.question_sanitized.model_dump() == _text("<redacted question>")
    assert turn.answer_sanitized.model_dump() == _text("<redacted answer>")
    assert turn.pipeline_profile == "oss"
    assert turn.turn_outcome == "success"
    assert turn.field_availability["pipeline_profile"] == "recorded"

    imported = turn.model_dump_json()
    assert "redacted-user" not in imported
    assert "redacted question" not in imported
    assert "redacted answer" not in imported


def test_unknown_stamped_chat_schema_is_rejected():
    with pytest.raises(UnsupportedTelemetryEra, match="Unknown chat schema version"):
        adapt_chat_trace(_stamped_chat_trace("chat.turn.v2"))


def test_stamped_chat_field_rename_needs_only_a_mapping_change(tmp_path):
    mappings_path = tmp_path / "chat.yaml"
    mappings_path.write_text(
        default_mappings_path("chat").read_text(encoding="utf-8")
        + "\nchat.turn.v2:\n  extends: chat.turn.v1\n  fields:\n    pipeline_profile: [metadata.variant]\n",
        encoding="utf-8",
    )
    trace = _stamped_chat_trace("chat.turn.v2")
    trace["metadata"]["variant"] = trace["metadata"].pop("pipeline_profile")

    turn = adapt_chat_trace(trace, chat_mappings=load_chat_mappings(mappings_path))

    assert turn.source_schema_version == "chat.turn.v2"
    assert turn.pipeline_profile == "oss"


@pytest.mark.parametrize("anonymous_value", ["anonymous", "Anonymous", " anonymous "])
def test_anonymous_chat_user_id_is_not_hashed(anonymous_value):
    turn = CanonicalChatTurn(
        source_era="test",
        source_schema_version="test.v1",
        source_trace_name="chat.translation",
        timestamp="2026-10-05T10:00:00Z",
        user_id=anonymous_value,
    )

    assert turn.user_id_hash is None
    assert turn.field_availability["user_id_hash"] == "unavailable"
