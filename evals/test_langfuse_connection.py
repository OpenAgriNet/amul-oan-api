"""
Proof-of-concept: verify Langfuse connectivity and list recent traces.

Langfuse SDK / API methods used:
  - langfuse.Langfuse(...)                         -> construct client from env vars
  - client.api.health.health()                     -> GET /api/public/health
  - client.api.trace.list(limit=..., order_by=...) -> GET /api/public/traces

Usage (from repo root):
    python evals/test_langfuse_connection.py
    python evals/test_langfuse_connection.py --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evals.langfuse_eval_utils import (
    LangfuseEvalError,
    get_langfuse_api,
    get_langfuse_settings,
    json_default,
    list_latest_traces,
    summarize_trace,
    verify_langfuse_connection,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify Langfuse connectivity and list recent traces.")
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of latest traces to fetch (default: 10)",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    settings = get_langfuse_settings()
    print("Langfuse configuration (keys redacted):")
    print(
        json.dumps(
            {
                "public_key_set": bool(settings["public_key"]),
                "secret_key_set": bool(settings["secret_key"]),
                "base_url": settings["base_url"],
                "environment": settings["environment"],
                "release": settings["release"],
            },
            indent=2,
        )
    )
    print("")
    print("SDK/API methods:")
    print("  - Langfuse(public_key, secret_key, base_url, environment)")
    print("  - client.api.health.health()")
    print(f"  - client.api.trace.list(limit={args.limit}, order_by='timestamp.desc')")
    print("")

    try:
        api = get_langfuse_api()
        health = verify_langfuse_connection(api)
        print("Health check OK:")
        print(json.dumps(health, indent=2, default=json_default))
        print("")

        traces = list_latest_traces(api, limit=args.limit)
        print(f"Latest {len(traces)} trace(s):")
        for index, trace in enumerate(traces, start=1):
            summary = summarize_trace(trace)
            summary["timestamp"] = json_default(summary["timestamp"])
            print(f"[{index}] {json.dumps(summary, ensure_ascii=False)}")
        return 0
    except LangfuseEvalError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Langfuse connectivity failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
