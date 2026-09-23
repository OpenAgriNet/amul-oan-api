import pytest

from app.models.telemetry_analytics import ChatC3TraceSchema
from app.services.telemetry_era_adapters import (
    ChatC3Adapter,
    UnsupportedTelemetryEra,
    adapt_chat_trace,
)
from app.services.telemetry_era_registry import TelemetryEraRegistry


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

    assert turn.schema_version == "chat-turn.v1"
    assert turn.source_era == "chat.c3"
    assert turn.source_schema_version == "chat.c3.v1"
    assert turn.source_era_extensions == ["chat.c3b"]
    assert turn.pipeline == "translation"
    assert turn.pipeline_profile == "legacy"
    assert turn.answer == "<redacted answer>"
    assert turn.original_question is None
    assert turn.root_input == {"action": "Proceed with the query.", "model_name": "gpt-5.1"}
    assert turn.score_names == []
    assert turn.observation_names == []
    assert turn.field_availability["original_question"] == "unavailable"
    assert turn.field_availability["turn_outcome"] == "unavailable"
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
    assert turn.user_id == "1234567890"
    assert turn.answer == "<redacted target-language answer>"
    assert turn.original_question is None
    assert turn.field_availability["answer"] == "recorded"
    assert turn.field_availability["original_question"] == "unavailable"


def test_resolver_rejects_c2_name_reuse_without_agent_observation(era_registry):
    with pytest.raises(UnsupportedTelemetryEra, match="requires an 'Amul AI Agent run' observation"):
        adapt_chat_trace(
            {"name": "chat.translation", "timestamp": "2026-05-12T23:56:49Z"},
            era_registry=era_registry,
        )


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
    assert turn.tool_calls == [
        {
            "observation_id": "redacted-tool-observation",
            "name": "get_farmer_milk_collection_details",
            "call_id": "redacted-call-id",
            "input": {"farmer_code": "redacted"},
            "output": "<redacted tool response>",
        }
    ]


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
    assert turn.original_question == "<redacted query>"
    assert turn.answer == "<redacted answer>"
    assert turn.pipeline_profile == "oss"
    assert turn.persona == "farmer"
    assert turn.turn_outcome == "success"
    assert turn.served_tier == "agent=vllm:gemma"
    assert turn.field_availability["turn_outcome"] == "recorded"


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
