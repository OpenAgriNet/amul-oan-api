"""
Reconstruct one chat session from Langfuse traces into an eval artifact JSON file.

Langfuse SDK / API methods used:
  - client.api.trace.list(session_id=..., limit=..., order_by='timestamp.asc')
  - client.api.trace.get(trace_id, fields='core,io,observations,scores,metrics')

Usage (from repo root):
    python evals/reconstruct_session.py <session_id>
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

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from evals.langfuse_eval_utils import (
    DEFAULT_OUTPUT_DIR,
    LangfuseEvalError,
    get_langfuse_api,
    json_default,
    reconstruct_session,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconstruct a Langfuse session into eval JSON.")
    parser.add_argument("session_id", help="Langfuse session ID")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR / "reconstructed_session.json"),
        help="Output JSON path (default: eval_outputs/reconstructed_session.json)",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parents[1] / output_path

    print("SDK/API methods:")
    print(f"  - client.api.trace.list(session_id={args.session_id!r}, order_by='timestamp.asc')")
    print("  - client.api.trace.get(trace_id, fields='core,io,observations,scores,metrics')")
    print("")

    try:
        api = get_langfuse_api()
        artifact = reconstruct_session(api, args.session_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2, default=json_default),
            encoding="utf-8",
        )
        print(json.dumps(artifact, ensure_ascii=False, indent=2, default=json_default))
        print("")
        print(f"Saved reconstructed session to: {output_path.resolve()}")
        return 0
    except LangfuseEvalError as exc:
        print(f"Session reconstruction error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Session reconstruction failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
