"""Read the append-only telemetry era registry used by historical adapters."""

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


class TelemetryEraRegistryError(ValueError):
    """The registry is unavailable or does not contain a required era."""


UNCLASSIFIED_OUTCOME = "unclassified"


class OutcomeVocabulary:
    """Maps recorded outcome values to shared dashboard buckets from eras.yaml."""

    def __init__(self, bucket_by_outcome: Mapping[str, str]):
        self._bucket_by_outcome = dict(bucket_by_outcome)

    @classmethod
    def from_yaml(cls, path: Path, *, section: str) -> "OutcomeVocabulary":
        payload = load_registry_file(path)
        raw = payload.get(section) if isinstance(payload, Mapping) else None
        if not isinstance(raw, Mapping):
            raise TelemetryEraRegistryError(f"telemetry/eras.yaml has no {section}")
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "OutcomeVocabulary":
        bucket_by_outcome: dict[str, str] = {}
        for bucket, outcomes in raw.items():
            # Scalar notes such as pre_v3_convention describe a missing field.
            if not isinstance(outcomes, list):
                continue
            for outcome in outcomes:
                if not isinstance(outcome, str):
                    continue
                if bucket == UNCLASSIFIED_OUTCOME:
                    raise TelemetryEraRegistryError(f"{UNCLASSIFIED_OUTCOME!r} is reserved for unknown outcomes")
                if outcome in bucket_by_outcome and bucket_by_outcome[outcome] != bucket:
                    raise TelemetryEraRegistryError(
                        f"outcome {outcome!r} is listed under both "
                        f"{bucket_by_outcome[outcome]!r} and {bucket!r}"
                    )
                bucket_by_outcome[outcome] = str(bucket)
        return cls(bucket_by_outcome)

    def classify(self, outcome: str | None) -> str | None:
        if not outcome:
            return None
        return self._bucket_by_outcome.get(outcome, UNCLASSIFIED_OUTCOME)


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
    def from_yaml(cls, path: Path, *, section: str = "chat_eras") -> "TelemetryEraRegistry":
        payload = load_registry_file(path)
        if not isinstance(payload, Mapping) or not isinstance(payload.get(section), list):
            raise TelemetryEraRegistryError(f"telemetry/eras.yaml has no {section} list")

        eras: dict[str, EraBoundary] = {}
        for raw_era in payload[section]:
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

    def root_trace_names(self) -> frozenset[str]:
        return frozenset(name for era in self._eras.values() for name in era.root_trace_names)


def default_era_registry_path() -> Path:
    return Path(__file__).resolve().parents[2] / "telemetry" / "eras.yaml"


def load_registry_file(path: Path) -> Any:
    """Parsed eras.yaml, read-only. Adapters load it per trace, so each file version is parsed once."""
    try:
        return load_yaml_file(path)
    except FileNotFoundError as exc:
        raise TelemetryEraRegistryError(
            f"Telemetry era registry not found at {path}. This adapter depends on PR #297."
        ) from exc


def load_yaml_file(path: Path) -> Any:
    """Parsed YAML, read-only and cached until the file changes."""
    stat = path.stat()
    return _parse_yaml_file(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=8)
def _parse_yaml_file(path: str, mtime_ns: int, size: int) -> Any:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - project already uses PyYAML
        raise TelemetryEraRegistryError("PyYAML is required to read telemetry/eras.yaml") from exc
    with open(path, encoding="utf-8") as registry_file:
        return yaml.safe_load(registry_file)


_SECTION_PREFIXES = {"chat_eras": "chat.", "voice_eras": "voice."}
_CONFIDENCE_LEVELS = {"low", "medium", "high"}


def check_registry(payload: Any) -> list[str]:
    """Problems in a parsed eras.yaml, one readable line each. Empty means it's usable."""
    if not isinstance(payload, Mapping):
        return ["eras.yaml must be a mapping"]
    problems: list[str] = []
    seen_ids: set[str] = set()
    schema_versions: dict[str, str] = {}
    for section, prefix in _SECTION_PREFIXES.items():
        eras = payload.get(section)
        if not isinstance(eras, list):
            problems.append(f"{section} is missing or not a list")
            continue
        root_starts: set[datetime] = set()
        root_ends: list[tuple[str, datetime]] = []
        for index, era in enumerate(eras):
            era_id = era.get("era_id") if isinstance(era, Mapping) else None
            if not isinstance(era_id, str) or not era_id.startswith(prefix):
                problems.append(f"{section}[{index}] needs an era_id starting with {prefix!r}")
                continue
            if era_id in seen_ids:
                problems.append(f"{era_id} is listed twice")
            seen_ids.add(era_id)
            schema_version = era.get("schema_version")
            if schema_version is not None:
                if not isinstance(schema_version, str) or not schema_version:
                    problems.append(f"{era_id}.schema_version must be a stamp like voice.turn.v1")
                elif schema_version in schema_versions:
                    problems.append(
                        f"{schema_versions[schema_version]} and {era_id} both declare schema_version {schema_version}"
                    )
                else:
                    schema_versions[schema_version] = era_id
            try:
                valid_from = _as_utc_datetime(era.get("valid_from"), era_id, "valid_from")
                valid_to = (
                    _as_utc_datetime(era["valid_to"], era_id, "valid_to") if era.get("valid_to") is not None else None
                )
            except ValueError as exc:
                problems.append(f"{era_id}: {exc}")
                continue
            if valid_to is not None and valid_to <= valid_from:
                problems.append(f"{era_id} ends before it starts")
            confidence = era.get("valid_from_confidence")
            if confidence is not None and confidence not in _CONFIDENCE_LEVELS:
                problems.append(f"{era_id}.valid_from_confidence must be low, medium or high")
            names = era.get("root_trace_names", [])
            if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
                problems.append(f"{era_id}.root_trace_names must be a list of trace names")
            elif names:
                root_starts.add(valid_from)
                if valid_to is not None:
                    root_ends.append((era_id, valid_to))
        # A root era that ends must hand over to another one at the same instant,
        # otherwise traces in between match no adapter.
        for era_id, valid_to in root_ends:
            if valid_to not in root_starts:
                problems.append(
                    f"{era_id} ends at {valid_to.isoformat()} but no {section} root era starts then"
                )
    return problems


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
