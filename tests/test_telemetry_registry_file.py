"""Checks on the real telemetry/eras.yaml, so a bad edit fails CI instead of the import."""

import pytest

from app.services.telemetry_era_registry import (
    TelemetryEraRegistry,
    check_registry,
    default_era_registry_path,
    load_registry_file,
)
from app.services.telemetry_voice_era_adapters import VoiceOutcomeVocabulary

REGISTRY = default_era_registry_path()

# Eras the resolvers look up by id.
CHAT_ERAS_USED = [
    "chat.c2", "chat.c2b", "chat.c2c", "chat.c3", "chat.c4", "chat.c5",
    "chat.c6", "chat.c6b", "chat.c6c", "chat.c7", "chat.c8",
]
VOICE_ERAS_USED = ["voice.v0", "voice.v2", "voice.v3", "voice.v3b", "voice.v4", "voice.v5", "voice.v5b"]

pytestmark = pytest.mark.skipif(not REGISTRY.exists(), reason="telemetry/eras.yaml comes with #297")


def test_registry_has_no_problems():
    assert check_registry(load_registry_file(REGISTRY)) == []


@pytest.mark.parametrize(("section", "era_ids"), [("chat_eras", CHAT_ERAS_USED), ("voice_eras", VOICE_ERAS_USED)])
def test_registry_has_every_era_the_resolvers_use(section, era_ids):
    registry = TelemetryEraRegistry.from_yaml(REGISTRY, section=section)

    for era_id in era_ids:
        registry.require(era_id)


def test_voice_outcome_vocabulary_loads():
    VoiceOutcomeVocabulary.from_yaml(REGISTRY)
