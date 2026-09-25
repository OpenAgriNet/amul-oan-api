import json

import pytest

from app.models.telemetry_voice_analytics import VoiceV4TraceSchema
from app.services.telemetry_era_adapters import UnsupportedTelemetryEra
from app.services.telemetry_era_registry import TelemetryEraRegistry, TelemetryEraRegistryError
from app.services.telemetry_mappings import default_mappings_path, mapping_or_none
from app.services.telemetry_voice_era_adapters import (
    VoiceOutcomeVocabulary,
    _adapt_mapped_voice_turn,
    _adapt_voice_turn,
    _parse_timestamp,
    adapt_voice_trace,
    load_voice_mappings,
)


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


# A future stamped era, written the way eras.yaml would record it once the stamp is live.
STAMPED_ERA = """
  - era_id: voice.v6
    valid_from: 2026-10-01
    valid_from_confidence: high
    valid_to: null
    root_trace_names: [agent_journey]
    schema_version: voice.turn.v1"""


def _write_registry(tmp_path, *, v3_confidence="high", v4_confidence="high", v4_valid_to="null", extra_eras=""):
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
    valid_to: {v4_valid_to}
    root_trace_names: [agent_journey]
  - era_id: voice.v5
    valid_from: 2026-07-24
    valid_from_confidence: low
    valid_to: null
  - era_id: voice.v5b
    valid_from: 2026-08-05
    valid_from_confidence: low
    valid_to: null{extra_eras}
voice_outcome_vocabulary:
  delivered: [success]
  non_question: [stale_request, stt_signal, hold_message, greeting_fast_path, identity_fast_path, fragment_fast_path, non_meaningful_hangup, outbound_intro]
  refused_or_blocked: [moderation_rejected, outbound_declined, outbound_no_data]
  failed: [error, pretranslation_empty, client_disconnected]
  pre_v3_convention: success
""".strip().format(
            v3_confidence=v3_confidence,
            v4_confidence=v4_confidence,
            v4_valid_to=v4_valid_to,
            extra_eras=extra_eras,
        ),
        encoding="utf-8",
    )
    return registry_path


def _adapter(registry_path):
    registry = TelemetryEraRegistry.from_yaml(registry_path, section="voice_eras")
    vocabulary = VoiceOutcomeVocabulary.from_yaml(registry_path)

    def _adapt(trace, **kwargs):
        return adapt_voice_trace(trace, era_registry=registry, outcome_vocabulary=vocabulary, **kwargs)

    return _adapt


@pytest.fixture
def registry_path(tmp_path):
    return _write_registry(tmp_path)


@pytest.fixture
def adapt(registry_path):
    return _adapter(registry_path)


@pytest.fixture
def adapt_with_stamped_era(tmp_path):
    return _adapter(_write_registry(tmp_path, v4_valid_to="2026-10-01", extra_eras=STAMPED_ERA))


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

    assert turn.schema_version == "voice.canonical.v1"
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


@pytest.mark.parametrize(
    ("name", "timestamp", "stamped"),
    [
        ("voice_request", "2026-06-01T10:00:00Z", False),
        ("agent_journey", "2026-08-10T10:00:00Z", False),
        ("agent_journey", "2026-10-05T10:00:00Z", True),
    ],
    ids=["v3", "v4", "stamped"],
)
def test_signed_in_is_read_from_the_agent_block(adapt, name, timestamp, stamped):
    trace = _voice_turn_trace(name, timestamp, agent={"signed_in": False, "tool_call_count": 1})
    trace = _stamped(trace) if stamped else trace
    exported = dict(trace)
    exported["metadata"] = {
        key: json.dumps(value) if isinstance(value, dict) else str(value) for key, value in trace["metadata"].items()
    }

    for shape in (trace, exported):
        turn = adapt(shape)
        assert turn.signed_in is False
        assert turn.field_availability["signed_in"] == "recorded"


def test_a_turn_that_never_reached_the_agent_has_no_signed_in(adapt):
    turn = adapt(_stamped(_voice_turn_trace("agent_journey", "2026-10-05T10:00:00Z", outcome="stt_signal")))

    assert turn.signed_in is None
    assert turn.field_availability["signed_in"] == "unavailable"


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


def test_unknown_outcome_is_counted_as_unclassified(adapt):
    unknown = adapt(_voice_turn_trace("agent_journey", "2026-08-10T10:00:00Z", outcome="stt_signal_auto_hangup"))

    assert unknown.outcome == "stt_signal_auto_hangup"
    assert unknown.outcome_class == "unclassified"
    assert unknown.field_availability["outcome"] == "recorded"
    assert unknown.field_availability["outcome_class"] == "derived"


def test_missing_outcome_has_no_class(adapt):
    missing = adapt(_voice_turn_trace("agent_journey", "2026-08-10T10:00:00Z", outcome=""))

    assert missing.outcome is None
    assert missing.outcome_class is None
    assert missing.field_availability["outcome"] == "unavailable"
    assert missing.field_availability["outcome_class"] == "unavailable"


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


def _stamped(trace, stamp="voice.turn.v1"):
    trace["metadata"].update({"amul.schema_version": stamp, "service": "voice-oan-api", "release": "test-release-sha"})
    return trace


def test_stamped_trace_is_routed_by_its_stamp_not_its_date(adapt_with_stamped_era):
    # Dated before voice.v6 starts, like a dev trace ahead of the prod rollout.
    turn = adapt_with_stamped_era(
        _stamped(_voice_turn_trace("agent_journey", "2026-09-01T10:00:00Z", pipeline_profile="managed"))
    )

    assert turn.source_era == "voice.turn.v1"
    assert turn.source_schema_version == "voice.turn.v1"
    assert turn.source_era_extensions == []
    assert turn.pipeline_profile == "managed"
    assert turn.outcome_class == "delivered"


def test_stamped_trace_does_not_need_the_dated_boundaries(tmp_path):
    path = _write_registry(tmp_path, v4_confidence="low", extra_eras=STAMPED_ERA)
    adapt = _adapter(path)

    assert adapt(_stamped(_voice_turn_trace("agent_journey", "2026-10-05T10:00:00Z"))).source_era == "voice.turn.v1"
    with pytest.raises(UnsupportedTelemetryEra, match="confidence"):
        adapt(_voice_turn_trace("agent_journey", "2026-10-05T10:00:00Z"))


@pytest.mark.parametrize("stamp", ["voice.turn.v2", "chat.turn.v1", "", None])
def test_unknown_stamp_is_rejected_even_on_a_known_root(adapt_with_stamped_era, stamp):
    with pytest.raises(UnsupportedTelemetryEra, match="Unknown voice schema version"):
        adapt_with_stamped_era(_stamped(_voice_turn_trace("agent_journey", "2026-10-05T10:00:00Z"), stamp))


def test_stamp_on_the_wrong_root_is_rejected(adapt_with_stamped_era):
    with pytest.raises(UnsupportedTelemetryEra, match="agent_journey"):
        adapt_with_stamped_era(_stamped(_voice_turn_trace("voice_request", "2026-10-05T10:00:00Z")))


def test_stamped_trace_is_read_before_eras_yaml_records_its_era(adapt):
    turn = adapt(_stamped(_voice_turn_trace("agent_journey", "2026-10-05T10:00:00Z")))

    assert turn.source_era == "voice.turn.v1"
    assert turn.outcome_class == "delivered"


def test_unstamped_agent_journey_stops_where_eras_yaml_closes_v4(adapt_with_stamped_era):
    assert adapt_with_stamped_era(_voice_turn_trace("agent_journey", "2026-09-30T10:00:00Z")).source_era == "voice.v4"
    with pytest.raises(UnsupportedTelemetryEra):
        adapt_with_stamped_era(_voice_turn_trace("agent_journey", "2026-10-02T10:00:00Z"))


def test_vocabulary_refuses_an_outcome_listed_in_two_buckets():
    with pytest.raises(TelemetryEraRegistryError, match="both"):
        VoiceOutcomeVocabulary.from_mapping({"delivered": ["success"], "failed": ["success"]})


def test_vocabulary_refuses_a_bucket_named_unclassified():
    with pytest.raises(TelemetryEraRegistryError, match="reserved"):
        VoiceOutcomeVocabulary.from_mapping({"unclassified": ["stt_signal"]})


def test_vocabulary_ignores_scalar_notes():
    vocabulary = VoiceOutcomeVocabulary.from_mapping({"delivered": ["success"], "pre_v3_convention": "success"})

    assert vocabulary.classify("success") == "delivered"
    assert vocabulary.classify(None) is None


def _prepared(trace):
    raw = dict(trace)
    raw["timestamp"] = _parse_timestamp(raw["timestamp"])
    raw["metadata"] = mapping_or_none(trace.get("metadata")) or {}
    return raw


def _clickhouse_shape(trace):
    exported = dict(trace, timestamp="2026-10-05 10:00:00.000")
    exported["metadata"] = {
        key: json.dumps(value) if isinstance(value, dict) else str(value) for key, value in trace["metadata"].items()
    }
    return exported


_FULL_STAMPED = _stamped(
    _voice_turn_trace(
        "agent_journey",
        "2026-10-05T10:00:00Z",
        pipeline_profile="managed",
        call_type="outbound",
        agent={"signed_in": True},
    )
)
_NO_SESSION_OR_QUERY = _stamped(_voice_turn_trace("agent_journey", "2026-10-05T10:00:00Z", session_id="s-meta"))
del _NO_SESSION_OR_QUERY["sessionId"]
del _NO_SESSION_OR_QUERY["metadata"]["query"], _NO_SESSION_OR_QUERY["metadata"]["response"]


@pytest.mark.parametrize(
    "trace",
    [
        _FULL_STAMPED,
        _clickhouse_shape(_FULL_STAMPED),
        _NO_SESSION_OR_QUERY,
        _stamped(_voice_turn_trace("agent_journey", "2026-10-05T10:00:00Z", process_id=7, outcome="new_outcome")),
        {"id": "t", "name": "agent_journey", "timestamp": "2026-10-05T10:00:00Z", "metadata": {"amul.schema_version": "voice.turn.v1"}},
    ],
    ids=["full", "clickhouse-export", "fallback-paths", "int-and-unknown-outcome", "empty"],
)
def test_voice_mapping_gives_the_same_turn_as_the_python_adapter(registry_path, trace):
    vocabulary = VoiceOutcomeVocabulary.from_yaml(registry_path)
    raw = _prepared(trace)
    common = dict(era_id="voice.turn.v1", outcome_vocabulary=vocabulary, observations=[{"name": "moderation"}], scores=[])

    mapped = _adapt_mapped_voice_turn(raw, load_voice_mappings()["voice.turn.v1"], **common)
    python = _adapt_voice_turn(
        VoiceV4TraceSchema.model_validate(raw), source_schema_version="voice.turn.v1", source_era_extensions=[], **common
    )

    mapped_dump, python_dump = mapped.model_dump(), python.model_dump()
    # Fields mapped after the switch have no Python counterpart; compare everything else.
    for field in set(mapped_dump["field_availability"]) - set(python_dump["field_availability"]):
        for dump in (mapped_dump, python_dump):
            dump.pop(field, None)
            dump["field_availability"].pop(field, None)
    assert mapped_dump == python_dump


def test_a_renamed_field_needs_only_a_mapping_change(tmp_path):
    mappings_path = tmp_path / "voice.yaml"
    mappings_path.write_text(
        default_mappings_path("voice").read_text(encoding="utf-8")
        + "\nvoice.turn.v2:\n  extends: voice.turn.v1\n  fields:\n    outcome: [metadata.turn_outcome]\n",
        encoding="utf-8",
    )
    adapt = _adapter(_write_registry(tmp_path))
    trace = _stamped(_voice_turn_trace("agent_journey", "2026-11-02T10:00:00Z"), "voice.turn.v2")
    trace["metadata"]["turn_outcome"] = trace["metadata"].pop("outcome")

    turn = adapt(trace, voice_mappings=load_voice_mappings(mappings_path))

    assert turn.source_era == "voice.turn.v2"
    assert turn.source_schema_version == "voice.turn.v2"
    assert turn.outcome == "success"
    assert turn.outcome_class == "delivered"
    assert turn.field_availability["outcome"] == "recorded"
    assert turn.question_sanitized.preview == "<redacted question preview>"


def test_the_voice_mappings_file_loads():
    assert load_voice_mappings()["voice.turn.v1"].root == "agent_journey"
