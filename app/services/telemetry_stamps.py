"""Stable provenance stamps for telemetry emitted by this service."""


CHAT_TELEMETRY_SCHEMA_VERSION = "chat.turn.v1"
CHAT_TELEMETRY_SERVICE = "amul-oan-api"


def forward_chat_telemetry_metadata(release: str | None) -> dict[str, str]:
    return {
        "amul.schema_version": CHAT_TELEMETRY_SCHEMA_VERSION,
        "service": CHAT_TELEMETRY_SERVICE,
        "release": release or "unknown",
    }
