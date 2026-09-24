import json

import pytest

from app.services.telemetry_era_adapters import UnsupportedTelemetryEra
from app.services.telemetry_era_registry import TelemetryEraRegistry, TelemetryEraRegistryError
from app.services.telemetry_voice_era_adapters import VoiceOutcomeVocabulary, adapt_voice_trace


VOCABULARY = {
    "delivered": ["success"],
    "non_question": [
        "stale_request",
        "stt_signal",
        "hold_message",
        "greeting_fast_path",
        "identity_fast_path",
        "fragment_fast_path",
        "non_meaningful_hangup",
        "outbound_intro",
    ],
    "refused_or_blocked": ["moderation_rejected", "outbound_declined", "outbound_no_data"],
    "failed": ["error", "pretranslation_empty", "client_disconnected"],
}


def _write_registry(tmp_path, *, v3_confidence="high", v4_confidence="high"):
    registry_path = tmp_path / "eras.yaml"
    registry_path.write_text(
        """
chat_eras:
  - era_id: chat.c6
    valid_from: 2026-08-05T06:30:00Z
    valid_to: null
    root_trace_names: [chat.default, chat.translation]
voice_eras:
  - era_id: voice.v0
    valid_from: 2026-02-02
    valid_from_confidence: medium
    valid_to: 2026-05-20
    root_trace_names: ["Voice Agent run", "Voice Agent Signed In run"]
  - era_id: voice.v1
    valid_from: 2026-03-26
    valid_from_confidence: low
    valid_to: null
  - era_id: voice.v2
    valid_from: 2026-04-14
    valid_from_confidence: low
    valid_to: null
  - era_id: voice.v3
    valid_from: 2026-05-20
    valid_from_confidence: {v3_confidence}
    valid_to: 2026-07-22
    root_trace_names: [voice_request]
  - era_id: voice.v3b
    valid_from: 2026-05-25
    valid_from_confidence: medium
    valid_to: null
  - era_id: voice.v4
    valid_from: 2026-07-22
    valid_from_confidence: {v4_confidence}
    valid_to: null
    root_trace_names: [agent_journey]
  - era_id: voice.v5
    valid_from: 2026-07-24
    valid_from_confidence: low
    valid_to: null
  - era_id: voice.v5b
    valid_from: 2026-08-05
    valid_from_confidence: low
    valid_to: null
voice_outcome_vocabulary:
  delivered: [success]
  non_question: [stale_request, stt_signal, hold_message, greeting_fast_path, identity_fast_path, fragment_fast_path, non_meaningful_hangup, outbound_intro]
  refused_or_blocked: [moderation_rejected, outbound_declined, outbound_no_data]
  failed: [error, pretranslation_empty, client_disconnected]
  pre_v3_convention: success
""".strip().format(v3_confidence=v3_confidence, v4_confidence=v4_confidence),
        encoding="utf-8",
    )
    return registry_path


@pytest.fixture
def registry_path(tmp_path):
    return _write_registry(tmp_path)


@pytest.fixture
def adapt(registry_path):
    registry = TelemetryEraRegistry.from_yaml(registry_path, section="voice_eras")
    vocabulary = VoiceOutcomeVocabulary.from_yaml(registry_path)

    def _adapt(trace, **kwargs):
        return adapt_voice_trace(trace, era_registry=registry, outcome_vocabulary=vocabulary, **kwargs)

    return _adapt


def _sanitized(label, **extra):
    return {"chars": 42, "sha256": "0" * 64, "preview": f"<redacted {label} preview>", **extra}


def _voice_turn_trace(name, timestamp, **metadata):
    return {
        "id": "trace-redacted",
        "name": name,
        "timestamp": timestamp,
        "sessionId": "session-redacted",
        "userId": "<redacted-user-id>",
        "input": _sanitized("question"),
        "output": _sanitized("answer"),
        "metadata": {
            "process_id": "3",
            "provider": "<redacted-provider>",
            "source_lang": "gu",
            "target_lang": "gu",
            "user_id_hash": "1" * 64,
            "route": "agent",
            "query": _sanitized("question"),
            "response": _sanitized("answer"),
            "outcome": "success",
            "total_ms": 2400.5,
            "timings_ms": {"ttft_ms": 800.0, "ttfr_ms": 950.0},
            "stage_totals_ms": {"pretranslation": 120.0, "agent": 1900.0},
            **metadata,
        },
    }


def test_v0_root_never_reads_its_message_history_and_leaves_outcome_unavailable(adapt):
    dump = [{"role": "user", "content": "<redacted full conversation>"}] * 3
    turn = adapt(
        {
            "id": "trace-redacted",
            "name": "Voice Agent run",
            "timestamp": "2026-03-01T10:00:00Z",
            "sessionId": "session-redacted",
            "userId": "<redacted-user-id>",
            "output": dump,
            "metadata": {"process_id": "1", "source_lang": "gu", "target_lang": "gu"},
        }
    )

    assert turn.schema_version == "voice.turn.v1"
    assert turn.source_era == "voice.v0"
    assert turn.source_schema_version == "voice.v0.v1"
    assert turn.user_id == "<redacted-user-id>"
    assert turn.user_id_semantics == "request_user_id_expected_phone_then_anonymous"
    assert turn.field_availability["user_id_hash"] == "unavailable"
    assert turn.question_sanitized is None and turn.answer_sanitized is None
    assert turn.outcome is None and turn.outcome_class is None
    assert turn.field_availability["outcome"] == "unavailable"
    assert "<redacted full conversation>" not in turn.model_dump_json()


def test_only_the_signed_in_agent_name_proves_signed_in(adapt):
    base = {"timestamp": "2026-04-20T10:00:00Z", "sessionId": "session-redacted", "metadata": {}}

    anonymous_agent = adapt({**base, "name": "Voice Agent run"})
    signed_in_agent = adapt({**base, "name": "Voice Agent Signed In run"})
    recorded = adapt({**base, "name": "Voice Agent Signed In run", "metadata": {"signed_in": "false"}})

    assert anonymous_agent.signed_in is None
    assert anonymous_agent.field_availability["signed_in"] == "unavailable"
    assert signed_in_agent.signed_in is True
    assert signed_in_agent.field_availability["signed_in"] == "derived"
    assert recorded.signed_in is False
    assert recorded.field_availability["signed_in"] == "recorded"


def test_v2_label_needs_its_date_and_an_external_api_span(adapt):
    trace = {"name": "Voice Agent Signed In run", "metadata": {}}
    external_span = [{"name": "marqo_search"}]

    assert adapt({**trace, "timestamp": "2026-04-20T10:00:00Z"}, observations=external_span).source_era_extensions == [
        "voice.v2"
    ]
    assert adapt({**trace, "timestamp": "2026-04-01T10:00:00Z"}, observations=external_span).source_era_extensions == []
    assert adapt({**trace, "timestamp": "2026-04-20T10:00:00Z"}, observations=[{"name": "chat gpt-5.1"}]).source_era_extensions == []


def test_v1_is_never_labelled_because_nothing_observable_changed(adapt):
    turn = adapt({"name": "Voice Agent run", "timestamp": "2026-04-01T10:00:00Z", "metadata": {}})

    assert turn.source_era == "voice.v0"
    assert "voice.v1" not in turn.source_era_extensions


def test_v3_keeps_only_sanitized_text_and_normalizes_the_contract(adapt):
    turn = adapt(
        _voice_turn_trace(
            "voice_request",
            "2026-05-21T10:00:00Z",
            query=_sanitized("question", text="<full text must never be carried>"),
        )
    )

    assert turn.source_era == "voice.v3"
    assert turn.source_era_extensions == []
    assert turn.question_sanitized.preview == "<redacted question preview>"
    assert turn.answer_sanitized.sha256 == "0" * 64
    assert "<full text must never be carried>" not in turn.model_dump_json()
    assert turn.user_id_hash == "1" * 64
    assert turn.process_id == "3"
    assert turn.outcome == "success"
    assert turn.outcome_class == "delivered"
    assert turn.full_turn_latency_ms == 2400.5
    assert turn.stage_totals_ms == {"pretranslation": 120.0, "agent": 1900.0}
    assert turn.timings_ms == {"ttft_ms": 800.0, "ttfr_ms": 950.0}
    assert turn.signed_in is None
    assert turn.field_availability["question_sanitized"] == "recorded"
    assert turn.field_availability["outcome_class"] == "derived"


def test_plain_string_text_is_never_taken_as_a_question(adapt):
    trace = _voice_turn_trace("voice_request", "2026-05-21T10:00:00Z", query="<raw caller text>")
    trace["input"] = "<raw caller text>"

    turn = adapt(trace)

    assert turn.question_sanitized is None
    assert turn.field_availability["question_sanitized"] == "unavailable"
    assert "<raw caller text>" not in turn.model_dump_json()


def test_v3b_variant_becomes_a_derived_pipeline_profile_and_a_label_after_its_date(adapt):
    labelled = adapt(_voice_turn_trace("voice_request", "2026-06-01T10:00:00Z", pipeline_variant="oss"))
    before_v3b = adapt(_voice_turn_trace("voice_request", "2026-05-21T10:00:00Z", pipeline_variant="oss"))

    assert labelled.source_era_extensions == ["voice.v3b"]
    assert labelled.pipeline_profile == "oss"
    assert labelled.field_availability["pipeline_profile"] == "derived"
    assert before_v3b.source_era_extensions == []
    assert before_v3b.pipeline_profile == "oss"


def test_clickhouse_export_shape_adapts_the_same_as_the_langfuse_api_shape(adapt):
    api_trace = _voice_turn_trace("voice_request", "2026-06-01T10:00:00Z")
    export_trace = dict(api_trace)
    export_trace["timestamp"] = "2026-06-01 10:00:00.000"
    export_trace["metadata"] = {
        key: json.dumps(value) if isinstance(value, dict) else str(value)
        for key, value in api_trace["metadata"].items()
    }

    api_turn = adapt(api_trace)
    export_turn = adapt(export_trace)

    assert export_turn.model_dump(exclude={"field_availability"}) == api_turn.model_dump(
        exclude={"field_availability"}
    )
    assert export_turn.field_availability == api_turn.field_availability


def test_v4_is_the_v3_shape_under_a_new_root_name(adapt):
    turn = adapt(_voice_turn_trace("agent_journey", "2026-07-23T10:00:00Z"))

    assert turn.source_era == "voice.v4"
    assert turn.source_schema_version == "voice.v4.v1"
    assert turn.source_era_extensions == []
    assert turn.outcome_class == "delivered"


def test_v5_and_v5b_are_labels_on_the_v4_root(adapt):
    turn = adapt(
        _voice_turn_trace(
            "agent_journey",
            "2026-08-10T10:00:00Z",
            pipeline_profile="managed",
            pc_agent="<redacted-model-config>",
            call_type="outbound",
            outcome="outbound_intro",
        )
    )

    assert turn.source_era == "voice.v4"
    assert turn.source_era_extensions == ["voice.v5", "voice.v5b"]
    assert turn.pipeline_profile == "managed"
    assert turn.field_availability["pipeline_profile"] == "recorded"
    assert turn.call_type == "outbound"
    assert turn.outcome_class == "non_question"


def test_v5_label_waits_for_its_date(adapt):
    turn = adapt(_voice_turn_trace("agent_journey", "2026-07-23T10:00:00Z", pipeline_profile="managed"))

    assert turn.source_era_extensions == []


@pytest.mark.parametrize(
    ("outcome", "bucket"),
    [(outcome, bucket) for bucket, outcomes in VOCABULARY.items() for outcome in outcomes],
)
def test_every_vocabulary_outcome_lands_in_its_bucket(adapt, outcome, bucket):
    turn = adapt(_voice_turn_trace("agent_journey", "2026-08-10T10:00:00Z", outcome=outcome))

    assert turn.outcome == outcome
    assert turn.outcome_class == bucket


def test_unknown_or_missing_outcome_keeps_the_turn_without_a_class(adapt):
    unknown = adapt(_voice_turn_trace("agent_journey", "2026-08-10T10:00:00Z", outcome="stt_signal_auto_hangup"))
    missing = adapt(_voice_turn_trace("agent_journey", "2026-08-10T10:00:00Z", outcome=""))

    assert unknown.outcome == "stt_signal_auto_hangup"
    assert unknown.outcome_class is None
    assert unknown.field_availability["outcome"] == "recorded"
    assert unknown.field_availability["outcome_class"] == "unavailable"
    assert missing.outcome is None
    assert missing.field_availability["outcome"] == "unavailable"


@pytest.mark.parametrize("name", ["voice_agent_run", "Voice Service Agent run", "chat.translation"])
def test_resolver_rejects_names_the_registry_does_not_know(adapt, name):
    with pytest.raises(UnsupportedTelemetryEra):
        adapt(_voice_turn_trace(name, "2026-08-10T10:00:00Z"))


@pytest.mark.parametrize(
    ("name", "timestamp"),
    [
        ("Voice Agent run", "2026-02-01T10:00:00Z"),
        ("Voice Agent run", "2026-05-21T10:00:00Z"),
        ("voice_request", "2026-05-19T10:00:00Z"),
        ("voice_request", "2026-07-23T10:00:00Z"),
        ("agent_journey", "2026-07-21T10:00:00Z"),
    ],
)
def test_resolver_rejects_roots_outside_their_production_window(adapt, name, timestamp):
    with pytest.raises(UnsupportedTelemetryEra):
        adapt(_voice_turn_trace(name, timestamp))


@pytest.mark.parametrize(
    ("confidences", "name", "timestamp"),
    [
        ({"v4_confidence": "low"}, "agent_journey", "2026-08-10T10:00:00Z"),
        ({"v4_confidence": "low"}, "voice_request", "2026-06-01T10:00:00Z"),
        ({"v3_confidence": "medium"}, "voice_request", "2026-06-01T10:00:00Z"),
        ({"v3_confidence": "medium"}, "Voice Agent run", "2026-03-01T10:00:00Z"),
    ],
)
def test_resolver_refuses_dispatch_on_an_unverified_deciding_boundary(tmp_path, confidences, name, timestamp):
    path = _write_registry(tmp_path, **confidences)
    registry = TelemetryEraRegistry.from_yaml(path, section="voice_eras")
    vocabulary = VoiceOutcomeVocabulary.from_yaml(path)

    with pytest.raises(UnsupportedTelemetryEra, match="confidence"):
        adapt_voice_trace(
            _voice_turn_trace(name, timestamp), era_registry=registry, outcome_vocabulary=vocabulary
        )


def test_resolver_requires_a_timestamp(adapt):
    with pytest.raises(UnsupportedTelemetryEra):
        adapt({"name": "agent_journey", "metadata": {}})


def test_vocabulary_refuses_an_outcome_listed_in_two_buckets():
    with pytest.raises(TelemetryEraRegistryError, match="both"):
        VoiceOutcomeVocabulary.from_mapping({"delivered": ["success"], "failed": ["success"]})


def test_vocabulary_ignores_scalar_notes():
    vocabulary = VoiceOutcomeVocabulary.from_mapping({"delivered": ["success"], "pre_v3_convention": "success"})

    assert vocabulary.classify("success") == "delivered"
    assert vocabulary.classify(None) is None
