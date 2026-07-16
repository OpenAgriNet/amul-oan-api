"""
Re-export Langfuse sessions into query_XXXX.json and rebuild eval CSVs.

Use after traces already exist (e.g. fix partial rows without re-calling chat).

Usage (from repo root):
    python evals/reexport_sessions.py --from-team-csv eval_outputs/langfuse_eval_all_queries_vm5.csv --status partial
    python evals/reexport_sessions.py --session-id <uuid> --query-index 67
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(_REPO_ROOT / ".env")

from evals.langfuse_eval_utils import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    RAW_JSON_DIR,
    get_langfuse_api,
    reconstruct_session,
    rebuild_detail_csv_from_raw,
    write_team_shareable_csv,
)


def _save_raw_json(index: int, payload: dict) -> None:
    import json

    RAW_JSON_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_JSON_DIR / f"query_{index:04d}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Re-export Langfuse sessions by query index.")
    parser.add_argument(
        "--from-team-csv",
        type=Path,
        help="Team CSV path; re-export rows matching --status",
    )
    parser.add_argument(
        "--status",
        default="partial",
        help="Comma-separated status values to re-export from team CSV (default: partial)",
    )
    parser.add_argument("--session-id", help="Single session UUID")
    parser.add_argument("--query-index", type=int, help="1-based query index for --session-id")
    parser.add_argument(
        "--shareable-csv",
        default=str(DEFAULT_OUTPUT_DIR / "langfuse_eval_all_queries.csv"),
    )
    parser.add_argument(
        "--detail-csv",
        default=str(DEFAULT_OUTPUT_DIR / "langfuse_queries_eval.csv"),
    )
    parser.add_argument("--max-index", type=int, default=200)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    api = get_langfuse_api()

    items: list[tuple[int, str, str]] = []
    if args.from_team_csv:
        csv_path = args.from_team_csv
        if not csv_path.is_absolute():
            csv_path = _REPO_ROOT / csv_path
        statuses = {s.strip().lower() for s in args.status.split(",") if s.strip()}
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if (row.get("status") or "").strip().lower() not in statuses:
                    continue
                items.append(
                    (
                        int(row["query_index"]),
                        row["session_id"],
                        row.get("question_gu") or "",
                    )
                )
    elif args.session_id and args.query_index:
        items.append((args.query_index, args.session_id, ""))
    else:
        raise SystemExit("Provide --from-team-csv or both --session-id and --query-index")

    succeeded = 0
    failed = 0
    for query_index, session_id, question_gu in items:
        try:
            reconstructed = reconstruct_session(api, session_id)
            if question_gu:
                reconstructed["question_gu"] = question_gu
            _save_raw_json(query_index, reconstructed)
            succeeded += 1
            print(f"Q{query_index} OK session_id={session_id}")
        except Exception as exc:
            failed += 1
            print(f"Q{query_index} FAIL session_id={session_id}: {exc}")

    shareable = Path(args.shareable_csv)
    detail = Path(args.detail_csv)
    if not shareable.is_absolute():
        shareable = _REPO_ROOT / shareable
    if not detail.is_absolute():
        detail = _REPO_ROOT / detail

    detail_count = rebuild_detail_csv_from_raw(RAW_JSON_DIR, detail, max_index=args.max_index)
    team_count = write_team_shareable_csv(RAW_JSON_DIR, shareable, max_queries=args.max_index)

    print("")
    print(f"Re-exported: {succeeded} ok, {failed} failed")
    print(f"Detail CSV: {detail} ({detail_count} rows)")
    print(f"Team CSV: {shareable} ({team_count} rows)")


if __name__ == "__main__":
    main()
