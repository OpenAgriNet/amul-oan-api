"""Read the append-only telemetry era registry used by historical adapters."""

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Mapping


class TelemetryEraRegistryError(ValueError):
    """The registry is unavailable or does not contain a required era."""


@dataclass(frozen=True)
class EraBoundary:
    era_id: str
    valid_from: datetime
    valid_to: datetime | None
    valid_from_confidence: str | None
    root_trace_names: frozenset[str]


class TelemetryEraRegistry:
    """A small typed view over ``telemetry/eras.yaml`` for adapter dispatch."""

    def __init__(self, eras: Mapping[str, EraBoundary]):
        self._eras = dict(eras)

    @classmethod
    def from_yaml(cls, path: Path) -> "TelemetryEraRegistry":
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - project already uses PyYAML
            raise TelemetryEraRegistryError("PyYAML is required to read telemetry/eras.yaml") from exc
        try:
            with path.open(encoding="utf-8") as registry_file:
                payload = yaml.safe_load(registry_file)
        except FileNotFoundError as exc:
            raise TelemetryEraRegistryError(
                f"Telemetry era registry not found at {path}. This adapter depends on PR #297."
            ) from exc
        if not isinstance(payload, Mapping) or not isinstance(payload.get("chat_eras"), list):
            raise TelemetryEraRegistryError("telemetry/eras.yaml has no chat_eras list")

        eras: dict[str, EraBoundary] = {}
        for raw_era in payload["chat_eras"]:
            if not isinstance(raw_era, Mapping) or not isinstance(raw_era.get("era_id"), str):
                continue
            era_id = raw_era["era_id"]
            valid_from = _as_utc_datetime(raw_era.get("valid_from"), era_id, "valid_from")
            valid_to_raw = raw_era.get("valid_to")
            eras[era_id] = EraBoundary(
                era_id=era_id,
                valid_from=valid_from,
                valid_to=(
                    _as_utc_datetime(valid_to_raw, era_id, "valid_to")
                    if valid_to_raw is not None
                    else None
                ),
                valid_from_confidence=(
                    raw_era.get("valid_from_confidence")
                    if isinstance(raw_era.get("valid_from_confidence"), str)
                    else None
                ),
                root_trace_names=frozenset(
                    value for value in raw_era.get("root_trace_names", []) if isinstance(value, str)
                ),
            )
        return cls(eras)

    def require(self, era_id: str) -> EraBoundary:
        try:
            return self._eras[era_id]
        except KeyError as exc:
            raise TelemetryEraRegistryError(f"telemetry/eras.yaml is missing {era_id}") from exc


def default_era_registry_path() -> Path:
    return Path(__file__).resolve().parents[2] / "telemetry" / "eras.yaml"


def _as_utc_datetime(value: Any, era_id: str, field_name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, time.min)
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise TelemetryEraRegistryError(f"{era_id}.{field_name} must be an ISO date or datetime")
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
