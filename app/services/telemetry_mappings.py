"""Field mappings for stamped telemetry, read from telemetry/mappings/<channel>.yaml.

Each schema version names the root trace it is emitted on and where every
canonical field lives in the raw trace, so a renamed field is a mapping change
rather than a code change.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Collection, Mapping

from app.services.telemetry_era_registry import TelemetryEraRegistryError, load_yaml_file

_CONTRACT_KEYS = {"root", "fields", "extends"}


@dataclass(frozen=True)
class ContractMapping:
    schema_version: str
    root: str
    fields: Mapping[str, tuple[str, ...]]


def default_mappings_path(channel: str) -> Path:
    return Path(__file__).resolve().parents[2] / "telemetry" / "mappings" / f"{channel}.yaml"


def load_mappings(path: Path, *, allowed_fields: Collection[str]) -> dict[str, ContractMapping]:
    try:
        payload = load_yaml_file(path)
    except FileNotFoundError as exc:
        raise TelemetryEraRegistryError(f"Telemetry mappings not found at {path}") from exc
    if not isinstance(payload, Mapping):
        raise TelemetryEraRegistryError(f"{path.name} must map schema versions to contracts")
    return {version: _resolve(version, payload, allowed_fields, seen=()) for version in payload}


def value_at(trace: Mapping[str, Any], path: str) -> Any:
    """Read `field`, `field.key` or `field.key.nested` from a trace.

    A key that itself has dots (amul.schema_version) wins over a nested read.
    """
    head, _, rest = path.partition(".")
    return _lookup(trace.get(head), rest) if rest else trace.get(head)


def mapped_values(
    mapping: ContractMapping,
    trace: Mapping[str, Any],
    parsers: Mapping[str, Callable[[Any], Any]],
) -> dict[str, Any]:
    """Extract a contract's canonical fields using ordered paths.

    Both historical-era adapters and stamped-contract adapters use this helper.
    Structural reconstruction remains in the channel adapter; ordinary field
    location and alias changes stay in the mapping file.
    """

    return {
        field: mapped_value_and_source(mapping, trace, field, parse)[0]
        for field, parse in parsers.items()
    }


def mapped_value_and_source(
    mapping: ContractMapping,
    trace: Mapping[str, Any],
    field: str,
    parse: Callable[[Any], Any],
) -> tuple[Any, str | None]:
    """Return the first mapped value and the raw path that supplied it."""

    for path in mapping.fields.get(field, ()):
        value = parse(value_at(trace, path))
        if value is not None:
            return value, path
    return None, None


def mapping_or_none(value: Any) -> Mapping[str, Any] | None:
    """Accept an object (Langfuse API) or a JSON-encoded object (ClickHouse export)."""
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str) and value.lstrip().startswith("{"):
        try:
            parsed = json.loads(value)
        except ValueError:
            return None
        return parsed if isinstance(parsed, Mapping) else None
    return None


def _lookup(value: Any, path: str) -> Any:
    mapping = mapping_or_none(value)
    if mapping is None:
        return None
    if path in mapping:
        return mapping[path]
    head, _, rest = path.partition(".")
    return _lookup(mapping.get(head), rest) if rest else None


def _resolve(
    version: str, payload: Mapping[str, Any], allowed_fields: Collection[str], seen: tuple[str, ...]
) -> ContractMapping:
    if version in seen:
        raise TelemetryEraRegistryError(f"{version} extends itself through {' -> '.join(seen)}")
    contract = payload.get(version)
    if not isinstance(contract, Mapping):
        raise TelemetryEraRegistryError(f"{version} is not defined")
    unknown = set(contract) - _CONTRACT_KEYS
    if unknown:
        raise TelemetryEraRegistryError(f"{version} has unknown keys {sorted(unknown)}; use root, fields, extends")

    parent = _resolve(contract["extends"], payload, allowed_fields, seen + (version,)) if contract.get("extends") else None
    root = contract.get("root") or (parent.root if parent else None)
    if not isinstance(root, str):
        raise TelemetryEraRegistryError(f"{version} needs a root trace name")

    fields = dict(parent.fields) if parent else {}
    for field, paths in (contract.get("fields") or {}).items():
        if field not in allowed_fields:
            raise TelemetryEraRegistryError(f"{version}: {field!r} is not a canonical field")
        paths = [paths] if isinstance(paths, str) else paths
        if not isinstance(paths, list) or not paths or not all(isinstance(p, str) and p for p in paths):
            raise TelemetryEraRegistryError(f"{version}.{field} needs one or more paths")
        fields[field] = tuple(paths)
    return ContractMapping(schema_version=version, root=root, fields=fields)
