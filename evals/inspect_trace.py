"""
Inspect a single Langfuse trace with all observations/spans.

Langfuse SDK / API methods used:
  - langfuse.Langfuse(...)                              -> construct client from env vars
  - client.api.trace.get(trace_id, fields='core,io,observations,scores,metrics')

Usage (from repo root):
    python evals/inspect_trace.py <trace_id>
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
    TRACE_FIELDS_FULL,
    fetch_trace,
    get_langfuse_api,
    json_default,
    to_jsonable,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a Langfuse trace and print JSON details.")
    parser.add_argument("trace_id", help="Langfuse trace ID")
    return parser


def _build_inspection_payload(trace) -> dict:
    payload = to_jsonable(trace)
    observations = getattr(trace, "observations", None) or []
    payload["observation_summaries"] = [
        {
            "id": getattr(obs, "id", None),
            "name": getattr(obs, "name", None),
            "type": getattr(obs, "type", None),
            "input": to_jsonable(getattr(obs, "input", None)),
            "output": to_jsonable(getattr(obs, "output", None)),
            "metadata": to_jsonable(getattr(obs, "metadata", None)),
            "latency_ms": round((getattr(obs, "latency", 0) or 0) * 1000, 2),
            "parent_observation_id": getattr(obs, "parent_observation_id", None),
        }
        for obs in observations
    ]
    payload["span_names"] = [getattr(obs, "name", None) for obs in observations]
    return payload


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    print("SDK/API methods:")
    print("  - Langfuse(public_key, secret_key, base_url, environment)")
    print(f"  - client.api.trace.get({args.trace_id!r}, fields={TRACE_FIELDS_FULL!r})")
    print("")

    try:
        api = get_langfuse_api()
        trace = fetch_trace(api, args.trace_id)
        payload = _build_inspection_payload(trace)
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default))
        return 0
    except LangfuseEvalError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Trace inspection failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
