from datetime import datetime, timezone

import pytest

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
