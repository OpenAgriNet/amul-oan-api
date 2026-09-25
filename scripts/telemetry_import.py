#!/usr/bin/env python3
"""Import voice turns from Langfuse's ClickHouse into the telemetry database.

    python scripts/telemetry_import.py --env voice-development --from 2026-09-20 --to 2026-09-25 --dry-run
    python scripts/telemetry_import.py --env voice-production     # yesterday (UTC), for a daily cron

Connects to TELEMETRY_CLICKHOUSE_HOST:TELEMETRY_CLICKHOUSE_PORT (default
localhost:8123) as telemetry_reader, and as telemetry_writer unless --dry-run.
Each password comes from TELEMETRY_READER_PASSWORD / TELEMETRY_WRITER_PASSWORD,
or else ~/.telemetry_reader_password / ~/.telemetry_writer_password.
"""

import argparse
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.services.telemetry_era_registry import TelemetryEraRegistry, default_era_registry_path  # noqa: E402
from app.services.telemetry_import import import_voice_days  # noqa: E402
from app.services.telemetry_voice_era_adapters import VoiceOutcomeVocabulary, load_voice_mappings  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    registry_path = default_era_registry_path()
    report = import_voice_days(
        _client("reader", database="default"),
        None if args.dry_run else _client("writer", database="telemetry"),
        environment=args.env,
        first_day=args.first_day,
        last_day=args.last_day,
        registry=TelemetryEraRegistry.from_yaml(registry_path, section="voice_eras"),
        vocabulary=VoiceOutcomeVocabulary.from_yaml(registry_path),
        mappings=load_voice_mappings(),
    )
    print("\n".join(report.lines()))
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    parser = argparse.ArgumentParser(description="Import voice turns into the telemetry database.")
    parser.add_argument("--env", required=True, help="Langfuse environment, e.g. voice-production")
    parser.add_argument("--from", dest="first_day", type=date.fromisoformat, default=yesterday, help="first UTC day")
    parser.add_argument("--to", dest="last_day", type=date.fromisoformat, default=None, help="last UTC day, inclusive")
    parser.add_argument("--dry-run", action="store_true", help="read and report, write nothing")
    args = parser.parse_args(argv)
    args.last_day = args.last_day or args.first_day
    if args.last_day < args.first_day:
        parser.error("--to is before --from")
    return args


def _client(role: str, *, database: str):
    import clickhouse_connect

    return clickhouse_connect.get_client(
        host=os.getenv("TELEMETRY_CLICKHOUSE_HOST", "localhost"),
        port=int(os.getenv("TELEMETRY_CLICKHOUSE_PORT", "8123")),
        username=f"telemetry_{role}",
        password=_password(role),
        database=database,
    )


def _password(role: str) -> str:
    password = os.getenv(f"TELEMETRY_{role.upper()}_PASSWORD")
    if password:
        return password
    path = Path.home() / f".telemetry_{role}_password"
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        sys.exit(f"No password for telemetry_{role}: set TELEMETRY_{role.upper()}_PASSWORD or write it to {path}")


if __name__ == "__main__":
    sys.exit(main())
