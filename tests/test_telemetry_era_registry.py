from datetime import datetime, timezone

import pytest
import yaml

from app.services.telemetry_era_registry import TelemetryEraRegistry, TelemetryEraRegistryError


@pytest.fixture
def registry_path(tmp_path):
    path = tmp_path / "eras.yaml"
    path.write_text(
        """
chat_eras:
  - era_id: chat.c6
    valid_from: 2026-08-05T06:30:00Z
    valid_to: null
    root_trace_names: [chat.default, chat.translation]
voice_eras:
  - era_id: voice.v3
    valid_from: 2026-05-20
    valid_from_confidence: high
    valid_to: 2026-07-22
    root_trace_names: [voice_request]
  - era_id: voice.v5
    valid_from: 2026-07-24
    valid_from_confidence: low
    valid_to: null
  - era_id: voice.v6
    valid_from: 2026-10-01
    valid_from_confidence: high
    valid_to: null
    root_trace_names: [agent_journey]
    schema_version: voice.turn.v1
""".strip(),
        encoding="utf-8",
    )
    return path


def test_from_yaml_reads_voice_eras(registry_path):
    registry = TelemetryEraRegistry.from_yaml(registry_path, section="voice_eras")

    v3 = registry.require("voice.v3")
    assert v3.valid_from == datetime(2026, 5, 20, tzinfo=timezone.utc)
    assert v3.valid_to == datetime(2026, 7, 22, tzinfo=timezone.utc)
    assert v3.valid_from_confidence == "high"
    assert v3.root_trace_names == frozenset({"voice_request"})

    v5 = registry.require("voice.v5")
    assert v5.valid_to is None
    assert v5.root_trace_names == frozenset()

    with pytest.raises(TelemetryEraRegistryError):
        registry.require("chat.c6")


def test_from_yaml_still_defaults_to_chat_eras(registry_path):
    registry = TelemetryEraRegistry.from_yaml(registry_path)

    assert registry.require("chat.c6").root_trace_names == frozenset({"chat.default", "chat.translation"})
    with pytest.raises(TelemetryEraRegistryError):
        registry.require("voice.v3")


def test_from_yaml_names_the_missing_section(registry_path):
    with pytest.raises(TelemetryEraRegistryError, match="no missing_eras list"):
        TelemetryEraRegistry.from_yaml(registry_path, section="missing_eras")


def test_an_era_can_be_found_by_its_schema_version_stamp(registry_path):
    registry = TelemetryEraRegistry.from_yaml(registry_path, section="voice_eras")

    assert registry.for_schema_version("voice.turn.v1").era_id == "voice.v6"
    assert registry.for_schema_version("voice.turn.v2") is None
    assert registry.require("voice.v3").schema_version is None


def test_two_eras_cannot_claim_one_schema_version(tmp_path):
    path = tmp_path / "eras.yaml"
    path.write_text(
        """
voice_eras:
  - era_id: voice.v6
    valid_from: 2026-10-01
    schema_version: voice.turn.v1
  - era_id: voice.v7
    valid_from: 2026-11-01
    schema_version: voice.turn.v1
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(TelemetryEraRegistryError, match="both declare schema_version voice.turn.v1"):
        TelemetryEraRegistry.from_yaml(path, section="voice_eras")


def test_registry_file_is_parsed_once_until_it_changes(registry_path, monkeypatch):
    parses = []
    real_safe_load = yaml.safe_load
    monkeypatch.setattr(yaml, "safe_load", lambda stream: parses.append(1) or real_safe_load(stream))

    TelemetryEraRegistry.from_yaml(registry_path)
    TelemetryEraRegistry.from_yaml(registry_path, section="voice_eras")
    assert len(parses) == 1

    registry_path.write_text(registry_path.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    TelemetryEraRegistry.from_yaml(registry_path)
    assert len(parses) == 2


def test_missing_registry_file_points_at_the_registry_pr(tmp_path):
    with pytest.raises(TelemetryEraRegistryError, match="#297"):
        TelemetryEraRegistry.from_yaml(tmp_path / "eras.yaml")
