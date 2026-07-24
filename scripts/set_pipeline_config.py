#!/usr/bin/env python3
"""Ops helper: PUT / GET / CLEAR the LIVE pipeline config in Redis (M2).

The app's ``app/llm_core/config_source.py`` reads a ``PipelineConfig`` JSON from
the Redis key ``llm_pipeline_config:{channel}`` and applies it within the refresh
TTL (``PIPELINE_CONFIG_REFRESH_S``, default 10s) — WITHOUT a redeploy — provided
the deployment has ``PIPELINE_CONFIG_REDIS_ENABLED=true``. Because the split
re-buckets every request against the current weights, continuing sessions follow
the new % automatically (e.g. an OSS profile 0 -> 50% moves ~half of them).

This script uses the SAME redis connection env the app uses (via
``config_source.build_redis_client()`` -> ``app.config.settings``) and validates
with the SAME ``PipelineConfig`` model before writing, so an invalid config can
never reach the key. Secrets are never in the config (tiers name ``api_key_env``
only) — do not put key VALUES in the JSON.

Usage
-----
    python scripts/set_pipeline_config.py set   <channel> <path-to-config.json>
    python scripts/set_pipeline_config.py get   <channel>
    python scripts/set_pipeline_config.py clear <channel>

``<channel>`` is normally ``chat`` or ``voice`` (each app reads its own key). Omit
nothing — the channel is explicit so you can target either service from one host.

Exact ops flow (hot % change, no redeploy)
------------------------------------------
  1. Confirm the deployment is armed: ``PIPELINE_CONFIG_REDIS_ENABLED=true`` in
     the app env (default OFF -> the app ignores the key entirely). One-time.
  2. Capture the CURRENT boot config to edit from. Easiest source is the app's
     boot log line ``llm_core.full_config`` / ``pipeline config: loaded LIVE ...``,
     or hand-author a PipelineConfig JSON (profiles + weights + per-step tiers).
  3. Edit the weights (they MUST sum to 100; names unique) and save to a .json.
  4. ``python scripts/set_pipeline_config.py set <channel> ./new.json``
     -> validates via PipelineConfig(**data); refuses to write if invalid.
  5. Within ``PIPELINE_CONFIG_REFRESH_S`` (~10s) the app's get_pipeline() reloads
     it; the split re-buckets continuing sessions onto the new %. Verify with
     ``get <channel>`` and the app's ``pipeline config: loaded LIVE ...`` log.
  6. To REVERT to the boot (env/YAML) config: ``clear <channel>`` deletes the key;
     within the TTL the app falls back to its boot config (last-good until then).

No real network is required to import this module; the redis client is built only
when a subcommand runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make the repo root importable when run as ``python scripts/set_pipeline_config.py``.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.llm_core import config_source
from app.llm_core.config_model import PipelineConfig


def _client():
    client = config_source.build_redis_client()
    if client is None:
        print("ERROR: could not build a redis client (check app.config redis_* env)", file=sys.stderr)
        raise SystemExit(2)
    return client


def cmd_set(channel: str, path: str) -> int:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        cfg = PipelineConfig(**data)  # validates weights==100 + unique profile names
    except Exception as e:
        print(f"REFUSED: invalid PipelineConfig in {path}: {e}", file=sys.stderr)
        return 1
    key = config_source.key(channel)
    payload = json.dumps(cfg.model_dump(mode="json"))
    _client().set(key, payload)
    print(f"OK: set {key} ({len(payload)} bytes) profiles={[f'{p.name}:{p.weight}' for p in cfg.profiles]}")
    print("   live within PIPELINE_CONFIG_REFRESH_S on any deployment with PIPELINE_CONFIG_REDIS_ENABLED=true")
    return 0


def cmd_get(channel: str) -> int:
    key = config_source.key(channel)
    raw = _client().get(key)
    if raw is None:
        print(f"(no live config at {key} — app is serving its boot config)")
        return 0
    try:
        parsed = json.loads(raw)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except Exception:
        print(raw)  # print whatever is there even if unparseable (aids debugging)
    return 0


def cmd_clear(channel: str) -> int:
    key = config_source.key(channel)
    deleted = _client().delete(key)
    if deleted:
        print(f"OK: cleared {key} — app reverts to its boot config within the TTL")
    else:
        print(f"(nothing to clear at {key})")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) >= 3 and argv[1] == "set" and len(argv) == 4:
        return cmd_set(argv[2], argv[3])
    if len(argv) == 3 and argv[1] == "get":
        return cmd_get(argv[2])
    if len(argv) == 3 and argv[1] == "clear":
        return cmd_clear(argv[2])
    print(__doc__)
    print("\nERROR: bad arguments.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
