import re

import pytest

from app.services.telemetry_era_registry import TelemetryEraRegistryError
from app.services.telemetry_mappings import load_mappings, mapped_attributes, value_at

FIELDS = {"outcome", "route"}


def _mappings(tmp_path, text):
    path = tmp_path / "voice.yaml"
    path.write_text(text, encoding="utf-8")
    return load_mappings(path, allowed_fields=FIELDS)


def test_a_version_can_extend_another_and_override_one_field(tmp_path):
    mappings = _mappings(
        tmp_path,
        """
v1:
  root: agent_journey
  fields:
    outcome: [metadata.outcome]
    route: metadata.route
v2:
  extends: v1
  fields:
    outcome: [metadata.turn_outcome, metadata.outcome]
""",
    )

    assert mappings["v1"].fields == {"outcome": ("metadata.outcome",), "route": ("metadata.route",)}
    assert mappings["v2"].root == "agent_journey"
    assert mappings["v2"].fields == {
        "outcome": ("metadata.turn_outcome", "metadata.outcome"),
        "route": ("metadata.route",),
    }


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("v1:\n  root: r\n  fields:\n    outcom: [metadata.outcome]\n", "v1: 'outcom' is not a canonical field"),
        ("v1:\n  root: r\n  feilds:\n    outcome: [metadata.outcome]\n", "v1 has unknown keys ['feilds']"),
        ("v1:\n  fields:\n    outcome: [metadata.outcome]\n", "v1 needs a root trace name"),
        ("v1:\n  root: r\n  fields:\n    outcome: []\n", "v1.outcome needs one or more paths"),
        ("v1:\n  extends: v2\nv2:\n  extends: v1\n", "extends itself"),
        ("v1:\n  extends: v9\n", "v9 is not defined"),
        ("v1:\n  root: r\n  attributes:\n    route: [metadata.route]\n", "'route' is a canonical field"),
        ("v1:\n  root: r\n  attributes:\n    status: [metadata.status]\n", "'status' needs a specific lowercase snake_case name"),
        ("v1:\n  root: r\n  attributes:\n    score: [metadata.score]\n", "'score' needs a specific lowercase snake_case name"),
        ("v1:\n  root: r\n  attributes:\n    CallQuality: [metadata.q]\n", "'CallQuality' needs a specific lowercase snake_case name"),
        ("v1:\n  root: r\n  attributes:\n    asked: [metadata.query.preview]\n", "can hold farmer text or a phone number"),
        ("v1:\n  root: r\n  attributes:\n    caller: [userId]\n", "can hold farmer text or a phone number"),
        ("v1:\n  root: r\n  attributes:\n    said: [input.text]\n", "can hold farmer text or a phone number"),
        ("v1:\n  root: r\n  attributes:\n    call_quality: []\n", "v1.attributes.call_quality needs one or more paths"),
    ],
)
def test_mapping_problems_are_named(tmp_path, text, problem):
    with pytest.raises(TelemetryEraRegistryError, match=re.escape(problem)):
        _mappings(tmp_path, text)


def test_a_missing_mappings_file_is_named(tmp_path):
    with pytest.raises(TelemetryEraRegistryError, match="mappings not found"):
        load_mappings(tmp_path / "voice.yaml", allowed_fields=FIELDS)


def test_value_at_reads_trace_fields_and_metadata_keys_with_dots():
    trace = {"sessionId": "s", "metadata": {"outcome": "success", "amul.schema_version": "voice.turn.v1"}}

    assert value_at(trace, "sessionId") == "s"
    assert value_at(trace, "metadata.outcome") == "success"
    assert value_at(trace, "metadata.amul.schema_version") == "voice.turn.v1"
    assert value_at(trace, "metadata.missing") is None
    assert value_at(trace, "sessionId.anything") is None


def test_value_at_reads_nested_keys_from_objects_and_json_strings():
    api = {"metadata": {"agent": {"signed_in": True}}}
    export = {"metadata": {"agent": '{"signed_in": false}'}}

    assert value_at(api, "metadata.agent.signed_in") is True
    assert value_at(export, "metadata.agent.signed_in") is False
    assert value_at(api, "metadata.agent.missing") is None
    assert value_at({"metadata": {"agent": "not an object"}}, "metadata.agent.signed_in") is None


def test_a_key_with_dots_wins_over_a_nested_read():
    trace = {"metadata": {"amul.schema_version": "flat", "amul": {"schema_version": "nested"}}}

    assert value_at(trace, "metadata.amul.schema_version") == "flat"


def test_attributes_extend_like_fields_and_are_read_as_text(tmp_path):
    mappings = _mappings(
        tmp_path,
        """
v1:
  root: agent_journey
  fields:
    outcome: [metadata.outcome]
  attributes:
    call_quality: [metadata.call_quality]
v2:
  extends: v1
  attributes:
    retries: [metadata.retries, metadata.attempts]
    agent_signed_in: [metadata.agent.signed_in]
    farmer_context: [metadata.farmer_context]
""",
    )
    trace = {
        "metadata": {
            "call_quality": "good",
            "attempts": 2,
            "agent": '{"signed_in": true}',
            "farmer_context": {"source": "cache", "unions": ["<redacted>"]},
        }
    }

    assert set(mappings["v2"].attributes) == {"call_quality", "retries", "agent_signed_in", "farmer_context"}
    # A whole block is skipped rather than flattened into storage.
    assert mapped_attributes(mappings["v2"], trace) == {"call_quality": "good", "retries": "2", "agent_signed_in": "true"}
    assert mapped_attributes(mappings["v1"], {"metadata": {}}) == {}
