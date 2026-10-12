from datetime import datetime, timezone

import pytest
import yaml

from app.services.telemetry_era_registry import (
    TelemetryEraRegistry,
    TelemetryEraRegistryError,
    check_registry,
)


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


def _registry(chat=None, voice=None):
    return {
        "chat_eras": chat if chat is not None else [{"era_id": "chat.c1", "valid_from": "2026-02-02"}],
        "voice_eras": voice if voice is not None else [{"era_id": "voice.v1", "valid_from": "2026-02-02"}],
    }


def test_check_registry_passes_a_clean_file():
    assert check_registry(_registry()) == []


def test_check_registry_passes_a_root_handover_at_the_same_instant():
    chat = [
        {"era_id": "chat.c1", "valid_from": "2026-02-02", "valid_to": "2026-08-05T06:30:00Z", "root_trace_names": ["a"]},
        {"era_id": "chat.c2", "valid_from": "2026-08-05T06:30:00Z", "root_trace_names": ["b"]},
    ]

    assert check_registry(_registry(chat=chat)) == []


def test_check_registry_flags_a_gap_between_root_eras():
    chat = [
        {"era_id": "chat.c3", "valid_from": "2026-05-13", "valid_to": "2026-08-05", "root_trace_names": ["a"]},
        {"era_id": "chat.c6", "valid_from": "2026-08-05T06:30:00Z", "root_trace_names": ["b"]},
    ]

    assert check_registry(_registry(chat=chat)) == [
        "chat.c3 ends at 2026-08-05T00:00:00+00:00 but no chat_eras root era starts then"
    ]


def test_check_registry_ignores_an_extension_era_that_ends_without_a_successor():
    voice = [
        {"era_id": "voice.v1", "valid_from": "2026-02-02", "root_trace_names": ["a"]},
        {"era_id": "voice.v2", "valid_from": "2026-08-05", "valid_to": "2026-08-06"},
    ]

    assert check_registry(_registry(voice=voice)) == []


@pytest.mark.parametrize(
    ("payload", "problem"),
    [
        ({"chat_eras": []}, "voice_eras is missing or not a list"),
        (_registry(chat=[{"era_id": "voice.v9", "valid_from": "2026-02-02"}]), "chat_eras[0] needs an era_id starting with 'chat.'"),
        (_registry(chat=[{"era_id": "chat.c1", "valid_from": "2026-02-02"}] * 2), "chat.c1 is listed twice"),
        (_registry(chat=[{"era_id": "chat.c1", "valid_from": "someday"}]), "chat.c1: "),
        (_registry(chat=[{"era_id": "chat.c1", "valid_from": "2026-03-01", "valid_to": "2026-02-01"}]), "chat.c1 ends before it starts"),
        (_registry(chat=[{"era_id": "chat.c1", "valid_from": "2026-02-02", "valid_from_confidence": "sure"}]), "chat.c1.valid_from_confidence must be low, medium or high"),
        (_registry(chat=[{"era_id": "chat.c1", "valid_from": "2026-02-02", "root_trace_names": "chat.default"}]), "chat.c1.root_trace_names must be a list of trace names"),
        (_registry(voice=[{"era_id": "voice.v6", "valid_from": "2026-10-01", "schema_version": 1}]), "voice.v6.schema_version must be a stamp"),
        (
            _registry(
                voice=[
                    {"era_id": "voice.v6", "valid_from": "2026-10-01", "schema_version": "voice.turn.v1"},
                    {"era_id": "voice.v7", "valid_from": "2026-11-01", "schema_version": "voice.turn.v1"},
                ]
            ),
            "voice.v6 and voice.v7 both declare schema_version voice.turn.v1",
        ),
    ],
)
def test_check_registry_names_each_problem(payload, problem):
    assert any(line.startswith(problem) for line in check_registry(payload))
