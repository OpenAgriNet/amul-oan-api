"""Rebuild golden-set eval CSV from saved Langfuse JSON artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evals.langfuse_eval_utils import (  # noqa: E402
    GOLDEN_FULL_CSV_COLUMNS,
    GOLDEN_SET_EVAL_CSV_COLUMNS,
    TEAM_SHARE_CSV_COLUMNS,
    write_team_shareable_csv,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild golden-set CSV from raw JSON")
    parser.add_argument(
        "--raw-json-dir",
        type=Path,
        default=_REPO_ROOT / "eval_outputs" / "golden_langfuse_raw_json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "eval_outputs" / "golden_set_eval_full.csv",
    )
    parser.add_argument(
        "--format",
        choices=("full", "golden", "team"),
        default="full",
        help="full=row_id+category+all team cols; golden=query-only; team=200-query style",
    )
    parser.add_argument("--max-index", type=int, default=679)
    args = parser.parse_args()

    if args.format == "full":
        columns = GOLDEN_FULL_CSV_COLUMNS
    elif args.format == "golden":
        columns = GOLDEN_SET_EVAL_CSV_COLUMNS
    else:
        columns = TEAM_SHARE_CSV_COLUMNS

    count = write_team_shareable_csv(
        args.raw_json_dir,
        args.output,
        max_queries=args.max_index,
        fieldnames=columns,
    )
    print(f"Wrote {count} rows -> {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
