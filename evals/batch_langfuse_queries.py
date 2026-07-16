"""
Run queries from a CSV against dev chat, then export collection rows from Langfuse traces.

This is an answer-collection harness for human review — not an automated evaluator.
Exported status=success means the chat/trace path completed without a transport error;
it does not judge answer quality.

You do NOT need session_id in queries.csv — a fresh UUID is generated per row and sent
to /api/chat so Langfuse groups traces under that session.

Langfuse SDK / API:
  - client.api.trace.list(session_id=...)
  - client.api.trace.get(trace_id, fields=...)

Usage (from repo root):
    python evals/batch_langfuse_queries.py evals/queries.csv --limit 3 \\
        --chat-base-url https://<dev-api-host> \\
        --jwt-token "<token>"

    # Or API-key auth (WhatsApp-style):
    python evals/batch_langfuse_queries.py evals/queries.csv --limit 3 \\
        --chat-base-url https://<dev-api-host> \\
        --chat-api-key "<key>" --user-phone 9876543210

    # Re-export only (CSV must have session_id column — no chat calls):
    python evals/batch_langfuse_queries.py evals/sessions.csv --langfuse-only

    # Re-run failed/partial rows and rebuild both CSVs from raw JSON:
    python evals/batch_langfuse_queries.py evals/queries.csv --retry-failed --langfuse-wait 180
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(_REPO_ROOT / ".env")

from evals.langfuse_eval_utils import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    GOLDEN_FULL_CSV_COLUMNS,
    GOLDEN_SET_EVAL_CSV_COLUMNS,
    LangfuseEvalError,
    RAW_JSON_DIR,
    collect_retry_indices,
    flatten_session_for_csv,
    get_langfuse_api,
    reconstruct_session,
    rebuild_detail_csv_from_raw,
    wait_for_session_traces,
    write_team_shareable_csv,
)

OUTPUT_COLUMNS = [
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
    "chat_response_preview",
    "error",
]



def _parse_indices_arg(value: str) -> set[int]:
    indices: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start_i, end_i = int(start_s.strip()), int(end_s.strip())
            indices.update(range(start_i, end_i + 1))
        else:
            indices.add(int(part))
    return indices


def _read_questions_indexed(
    csv_path: Path,
    *,
    limit: Optional[int],
    indices_filter: Optional[set[int]] = None,
    start_from: Optional[int] = None,
) -> list[tuple[int, str, dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "question_gu" not in reader.fieldnames:
            raise ValueError(f"CSV must contain a 'question_gu' column: {csv_path}")
        items: list[tuple[int, str, dict[str, str]]] = []
        for index, row in enumerate(reader, start=1):
            if start_from is not None and index < start_from:
                continue
            if indices_filter is not None and index not in indices_filter:
                continue
            question = (row.get("question_gu") or "").strip()
            if question:
                metadata = {
                    key: (row.get(key) or "").strip()
                    for key in ("row_id", "category")
                    if (row.get(key) or "").strip()
                }
                items.append((index, question, metadata))
            if limit is not None and len(items) >= limit:
                break
    return items


def _read_session_rows(csv_path: Path, limit: Optional[int]) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "session_id" not in reader.fieldnames:
            raise ValueError(
                f"--langfuse-only requires a 'session_id' column in {csv_path}"
            )
        rows: list[dict[str, str]] = []
        for row in reader:
            session_id = (row.get("session_id") or "").strip()
            if session_id:
                rows.append(
                    {
                        "session_id": session_id,
                        "question_gu": (row.get("question_gu") or "").strip(),
                    }
                )
            if limit is not None and len(rows) >= limit:
                break
    return rows


async def _call_dev_chat(
    *,
    base_url: str,
    query: str,
    session_id: str,
    source_lang: str,
    target_lang: str,
    use_translation_pipeline: bool,
    jwt_token: Optional[str],
    chat_api_key: Optional[str],
    user_phone: Optional[str],
    timeout_s: float,
) -> dict[str, Any]:
    import httpx

    base = base_url.rstrip("/")
    params = {
        "query": query,
        "session_id": session_id,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "channel": "web",
        "user_id": user_phone or "eval-runner",
        "use_translation_pipeline": str(use_translation_pipeline).lower(),
        "stream": "false",
    }
    url = f"{base}/api/chat/?{urlencode(params)}"
    headers: dict[str, str] = {}
    if jwt_token:
        headers["Authorization"] = f"Bearer {jwt_token}"
    if chat_api_key:
        headers["X-API-Key"] = chat_api_key
        if not user_phone:
            raise LangfuseEvalError("X-User-Phone is required when using --chat-api-key")
        headers["X-User-Phone"] = user_phone

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()


def _truncate(text: str, limit: int = 300) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [{len(text) - limit} more chars]"


def _append_csv_row(csv_path: Path, row: dict[str, Any], write_header: bool) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _save_raw_json(raw_json_dir: Path, index: int, payload: dict[str, Any]) -> Path:
    raw_json_dir.mkdir(parents=True, exist_ok=True)
    path = raw_json_dir / f"query_{index:04d}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CSV queries on dev chat and export Langfuse-reconstructed eval CSV.",
    )
    parser.add_argument("csv_path", help="Input CSV with question_gu (or session_id for --langfuse-only)")
    parser.add_argument("--limit", type=int, default=None, help="Process first N rows only")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR / "langfuse_queries_eval.csv"),
        help="Output CSV path (default: eval_outputs/langfuse_queries_eval.csv)",
    )
    parser.add_argument(
        "--langfuse-only",
        action="store_true",
        help="Skip chat calls; CSV must include session_id (reconstruct from Langfuse only)",
    )
    parser.add_argument(
        "--chat-base-url",
        default=os.getenv("EVAL_CHAT_BASE_URL", "http://localhost:8000"),
        help="Chat API base URL (default: EVAL_CHAT_BASE_URL or http://localhost:8000)",
    )
    parser.add_argument("--jwt-token", default=os.getenv("EVAL_CHAT_JWT"), help="Bearer JWT for /api/chat")
    parser.add_argument(
        "--chat-api-key",
        default=os.getenv("EVAL_CHAT_API_KEY") or os.getenv("CHAT_API_KEY") or None,
        help="X-API-Key (EVAL_CHAT_API_KEY or CHAT_API_KEY). No hardcoded default.",
    )
    parser.add_argument(
        "--user-phone",
        default=os.getenv("EVAL_CHAT_USER_PHONE"),
        help="X-User-Phone for API-key auth (EVAL_CHAT_USER_PHONE). Use a synthetic test phone only.",
    )
    parser.add_argument("--source-lang", default="gu")
    parser.add_argument("--target-lang", default="gu")
    parser.add_argument(
        "--no-translation-pipeline",
        action="store_true",
        help="Disable gu->en->gu translation pipeline (default: enabled)",
    )
    parser.add_argument(
        "--chat-timeout",
        type=float,
        default=300.0,
        help="HTTP timeout per chat request in seconds (default: 300)",
    )
    parser.add_argument(
        "--langfuse-wait",
        type=float,
        default=120.0,
        help="Seconds to wait for Langfuse traces after each chat (default: 120)",
    )
    parser.add_argument(
        "--shareable-csv",
        default=str(DEFAULT_OUTPUT_DIR / "langfuse_eval_all_queries.csv"),
        help="Team-shareable summary CSV path (default: eval_outputs/langfuse_eval_all_queries.csv)",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=None,
        metavar="N",
        help="1-based query index to start from (skips earlier rows in the CSV)",
    )
    parser.add_argument(
        "--indices",
        default=None,
        help="Comma-separated 1-based indices to run (e.g. 2,43,45-48)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-run failed/partial team-CSV rows or incomplete existing raw JSON (never invents never-run indices)",
    )
    parser.add_argument(
        "--max-index",
        type=int,
        default=None,
        help=(
            "Highest query index when scanning raw JSON / --retry-failed "
            "(default: infer from existing query_*.json files)"
        ),
    )
    parser.add_argument(
        "--raw-json-dir",
        default=str(RAW_JSON_DIR),
        help="Directory for query_XXXX.json artifacts (default: eval_outputs/langfuse_raw_json)",
    )
    parser.add_argument(
        "--csv-columns",
        default=None,
        help=(
            "Comma-separated output CSV columns for --shareable-csv "
            "(default: full team columns; use 'golden' for query-only, 'full' for golden+team cols)"
        ),
    )
    return parser


async def _run_batch(args: argparse.Namespace) -> int:
    csv_path = Path(args.csv_path)
    if not csv_path.is_absolute():
        csv_path = _REPO_ROOT / csv_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = _REPO_ROOT / output_path

    shareable_path = Path(args.shareable_csv)
    if not shareable_path.is_absolute():
        shareable_path = _REPO_ROOT / shareable_path

    raw_json_dir = Path(args.raw_json_dir)
    if not raw_json_dir.is_absolute():
        raw_json_dir = _REPO_ROOT / raw_json_dir

    csv_columns: Optional[list[str]] = None
    if args.csv_columns:
        if args.csv_columns.strip().lower() == "golden":
            csv_columns = GOLDEN_SET_EVAL_CSV_COLUMNS
        elif args.csv_columns.strip().lower() == "full":
            csv_columns = GOLDEN_FULL_CSV_COLUMNS
        else:
            csv_columns = [part.strip() for part in args.csv_columns.split(",") if part.strip()]

    indices_filter: Optional[set[int]] = None
    if args.indices:
        indices_filter = _parse_indices_arg(args.indices)
    if args.retry_failed:
        retry_from_raw = collect_retry_indices(
            raw_json_dir,
            team_csv_path=shareable_path if shareable_path.exists() else None,
            max_index=args.max_index,
        )
        indices_filter = (
            sorted(indices_filter | set(retry_from_raw))
            if indices_filter
            else set(retry_from_raw)
        )
        print(f"Retry mode: {len(indices_filter)} indices — {indices_filter}")

    merge_outputs = bool(indices_filter or args.start_from)
    if not merge_outputs and output_path.exists():
        output_path.unlink()

    use_translation = not args.no_translation_pipeline
    api = get_langfuse_api()

    succeeded = 0
    failed = 0

    if args.langfuse_only:
        rows = _read_session_rows(csv_path, args.limit)
        total = len(rows)
        for index, row in enumerate(rows, start=1):
            session_id = row["session_id"]
            question_gu = row.get("question_gu") or ""
            try:
                reconstructed = reconstruct_session(api, session_id)
                if question_gu:
                    reconstructed["question_gu"] = question_gu
                flat = flatten_session_for_csv(reconstructed)
                flat["chat_response_preview"] = ""
                flat["error"] = ""
                _save_raw_json(raw_json_dir, index, reconstructed)
                _append_csv_row(output_path, flat, write_header=index == 1)
                succeeded += 1
            except Exception as exc:
                failed += 1
                _append_csv_row(
                    output_path,
                    {
                        "question_gu": question_gu,
                        "session_id": session_id,
                        "error": str(exc),
                    },
                    write_header=index == 1 and succeeded == 0,
                )
            print(f"[{index}/{total}] completed")
    else:
        has_jwt = bool(args.jwt_token)
        has_key = bool(args.chat_api_key)
        if has_jwt == has_key:
            raise LangfuseEvalError(
                "Provide exactly one of --jwt-token or --chat-api-key for chat authentication."
            )
        if has_key and not args.user_phone:
            raise LangfuseEvalError(
                "Provide --user-phone (synthetic test phone) when using --chat-api-key."
            )

        items = _read_questions_indexed(
            csv_path,
            limit=args.limit,
            indices_filter=indices_filter,
            start_from=args.start_from,
        )
        total = len(items)
        if total == 0:
            print("No questions found.")
            return 1

        for run_no, (query_index, question_gu, metadata) in enumerate(items, start=1):
            session_id = str(uuid.uuid4())
            row: dict[str, Any] = {"question_gu": question_gu, "session_id": session_id}
            try:
                print(f"[{run_no}/{total}] Q{query_index} chat session_id={session_id}")
                chat_payload = await _call_dev_chat(
                    base_url=args.chat_base_url,
                    query=question_gu,
                    session_id=session_id,
                    source_lang=args.source_lang,
                    target_lang=args.target_lang,
                    use_translation_pipeline=use_translation,
                    jwt_token=args.jwt_token,
                    chat_api_key=(None if args.jwt_token else args.chat_api_key),
                    user_phone=(None if args.jwt_token else args.user_phone),
                    timeout_s=args.chat_timeout,
                )
                chat_response = str(chat_payload.get("response") or "")
                row["chat_response_preview"] = _truncate(chat_response)

                print(f"[{run_no}/{total}] Q{query_index} waiting for Langfuse traces...")
                langfuse_error = ""
                try:
                    wait_for_session_traces(
                        api,
                        session_id,
                        timeout_s=args.langfuse_wait,
                    )
                    reconstructed = reconstruct_session(api, session_id)
                    reconstructed["question_gu"] = question_gu
                    reconstructed.update(metadata)
                    flat = flatten_session_for_csv(reconstructed)
                    flat["chat_response_preview"] = row["chat_response_preview"]
                    flat["error"] = ""
                    _save_raw_json(raw_json_dir, query_index, reconstructed)
                    if not merge_outputs:
                        _append_csv_row(output_path, flat, write_header=run_no == 1)
                    succeeded += 1
                except LangfuseEvalError as langfuse_exc:
                    langfuse_error = str(langfuse_exc)
                    flat = {
                        "question_gu": question_gu,
                        "session_id": session_id,
                        "final_answer": chat_response,
                        "translated_answer": chat_response,
                        "chat_response_preview": row["chat_response_preview"],
                        "error": (
                            f"{langfuse_error} "
                            "(Chat succeeded; Langfuse traces missing — "
                            "restart API with LANGFUSE_* in .env or use dev API host.)"
                        ),
                    }
                    _save_raw_json(
                        raw_json_dir,
                        query_index,
                        {
                            "question_gu": question_gu,
                            "session_id": session_id,
                            "chat_response": chat_response,
                            "langfuse_error": langfuse_error,
                            **metadata,
                        },
                    )
                    if not merge_outputs:
                        _append_csv_row(output_path, flat, write_header=run_no == 1)
                    failed += 1
            except Exception as exc:
                failed += 1
                row["error"] = f"{exc}\n{traceback.format_exc()}".strip()
                _save_raw_json(
                    raw_json_dir,
                    query_index,
                    {
                        "question_gu": question_gu,
                        "session_id": session_id,
                        "error": row["error"],
                        **metadata,
                    },
                )
                if not merge_outputs:
                    _append_csv_row(
                        output_path,
                        row,
                        write_header=run_no == 1 and succeeded == 0,
                    )
            print(f"[{run_no}/{total}] Q{query_index} completed")

    print("")
    if merge_outputs:
        detail_count = rebuild_detail_csv_from_raw(
            raw_json_dir,
            output_path,
            max_index=args.max_index,
        )
        print(f"Rebuilt detail CSV: {output_path.resolve()} ({detail_count} rows)")
    shareable_count = write_team_shareable_csv(
        raw_json_dir,
        shareable_path,
        max_queries=args.max_index,
        fieldnames=csv_columns,
    )

    print(f"Succeeded: {succeeded}")
    print(f"Failed: {failed}")
    print(f"Output CSV: {output_path.resolve()}")
    print(f"Team shareable CSV: {shareable_path.resolve()} ({shareable_count} rows)")
    print(f"Raw JSON dir: {raw_json_dir.resolve()}")
    return 0 if failed == 0 else 1


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run_batch(args)))


if __name__ == "__main__":
    main()
