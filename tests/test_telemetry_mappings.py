import re

import pytest

from app.services.telemetry_era_registry import TelemetryEraRegistryError
from app.services.telemetry_mappings import load_mappings, value_at

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
