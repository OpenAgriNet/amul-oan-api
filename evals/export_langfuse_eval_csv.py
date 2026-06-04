"""
Export a CSV of reconstructed Langfuse sessions for offline evaluation.

Langfuse SDK / API methods used:
  - client.api.sessions.list(limit=...)
  - client.api.trace.list(session_id=..., order_by='timestamp.asc')
  - client.api.trace.get(trace_id, fields='core,io,observations,scores,metrics')

Usage (from repo root):
    python evals/export_langfuse_eval_csv.py
    python evals/export_langfuse_eval_csv.py --sessions 20
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evals.langfuse_eval_utils import (
    DEFAULT_OUTPUT_DIR,
    LangfuseEvalError,
    flatten_session_for_csv,
    get_langfuse_api,
    list_latest_sessions,
    reconstruct_session,
)

CSV_COLUMNS = [
    "question_gu",
    "question_en",
    "response_en",
    "moderation_category",
    "moderation_action",
    "final_answer",
    "translated_answer",
    "tool_names",
    "tool_count",
    "session_id",
    "trace_ids",
    "latency",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export reconstructed Langfuse sessions to CSV.")
    parser.add_argument(
        "--sessions",
        type=int,
        default=10,
        help="Number of latest sessions to export (default: 10)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR / "langfuse_eval.csv"),
        help="Output CSV path (default: eval_outputs/langfuse_eval.csv)",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parents[1] / output_path

    print("SDK/API methods:")
    print(f"  - client.api.sessions.list(limit={args.sessions})")
    print("  - client.api.trace.list(session_id=..., order_by='timestamp.asc')")
    print("  - client.api.trace.get(trace_id, fields='core,io,observations,scores,metrics')")
    print("")

    succeeded = 0
    failed = 0
    rows: list[dict[str, str | int | float]] = []

    try:
        api = get_langfuse_api()
        sessions = list_latest_sessions(api, limit=args.sessions)
        if not sessions:
            print("No sessions returned by Langfuse.")
            return 1

        for index, session in enumerate(sessions, start=1):
            session_id = session.id
            try:
                reconstructed = reconstruct_session(api, session_id)
                rows.append(flatten_session_for_csv(reconstructed))
                succeeded += 1
                print(f"[{index}/{len(sessions)}] reconstructed session_id={session_id}")
            except Exception as exc:
                failed += 1
                print(f"[{index}/{len(sessions)}] failed session_id={session_id}: {exc}", file=sys.stderr)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        print("")
        print(f"Sessions requested: {len(sessions)}")
        print(f"Succeeded: {succeeded}")
        print(f"Failed: {failed}")
        print(f"Output CSV: {output_path.resolve()}")
        return 0 if failed == 0 else 1
    except LangfuseEvalError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
