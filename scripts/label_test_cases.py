"""Dry run of the test case labeller (#488): label cases and write a table, nothing else.

Reads cases from files, one case per session, labels each with Gemma 4 through
LiteLLM (app/services/test_case_labeller.py) and writes one CSV row per case, the
set's own hints beside the labels, for Kanav to review as a sheet. It writes to
Langfuse nothing and needs no Langfuse access.

    SCENARIO_GEMMA_BASE_URL=... LITELLM_API_KEY=... \\
    python scripts/label_test_cases.py --seed amul-seed-questions.json \\
        --v0 amul_eval_set_v0.csv --v0-meta amul_eval_set_v0_metadata.json --out labels.csv

--seed and --submissions take a dataset export from Langfuse's public API
(GET /api/public/dataset-items, the {"data": [...]} body); archived items are left
out. --v0 is oan-evaluation's inference/amul_eval_set_v0.csv (branch
amul-eval-set-v0-metadata). Its expected_action goes in the table, not the prompt,
so the label's answer_type can be checked against it. --limit labels only the first
N cases, for a quick look before a full run.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.personal_data import mask_personal_data  # noqa: E402
from app.services.test_case_labeller import Case, LabelError, LiteLLM, Turn, label_case  # noqa: E402

COLUMNS = [
    "set", "case_id", "turns", "set_hints", "set_expected_action",
    "feature", "sub_tags", "use_case", "edge_case", "answer_type", "error",
]


def _question(gujarati: str, english: str) -> str:
    gujarati, english = (gujarati or "").strip(), (english or "").strip()
    if gujarati and english and gujarati != english:
        return f"{gujarati} (English: {english})"
    return gujarati or english


def _turn_number(question_id: str) -> tuple[int, str]:
    tail = question_id.rsplit("-", 1)[-1]
    return (int(tail), question_id) if tail.isdigit() else (0, question_id)


def _active(path: Path) -> list[dict]:
    items = json.loads(path.read_text(encoding="utf-8"))["data"]
    return [item for item in items if item.get("status") != "ARCHIVED" and isinstance(item.get("input"), dict)]


def load_seed(path: Path) -> list[Case]:
    """amul-seed-questions: one item per farmer question, grouped by session_id."""
    sessions: dict[str, list[dict]] = defaultdict(list)
    for item in _active(path):
        sessions[item["input"].get("session_id") or item["id"]].append(item)
    cases = []
    for session_id, items in sessions.items():
        items.sort(key=lambda item: _turn_number(item["input"].get("question_id", "")))
        metadata = [item.get("metadata") or {} for item in items]
        hints = {
            "category": ", ".join(sorted({m.get("category", "") for m in metadata} - {""})),
            "tool_hint": ", ".join(sorted({m.get("tool_hint", "") for m in metadata} - {""})),
        }
        turns = tuple(
            Turn("user", _question(item["input"].get("question_gujarati", ""), item["input"].get("question_english", "")))
            for item in items
        )
        cases.append(Case("amul-seed-questions", session_id, turns, hints=hints))
    return cases


def load_v0(path: Path, meta_path: Path | None = None) -> tuple[list[Case], dict[str, str]]:
    """Eval set v0: one row per farmer question. Returns the cases and each case's
    expected_action, turn by turn, for the table."""
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path else {}
    sessions: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sessions[row["session_id"]].append(row)
    cases, actions = [], {}
    for session_id, rows in sessions.items():
        rows.sort(key=lambda row: int(row["turn"]) if row["turn"].isdigit() else 0)
        hints = {
            "category": ", ".join(sorted({row["category"] for row in rows} - {""})),
            "expected_tools": ", ".join(sorted({row["expected_tools"] for row in rows} - {""})),
            "must_not_call": ", ".join(sorted({row["must_not_call"] for row in rows} - {""})),
        }
        turns = tuple(Turn("user", _question(row["question_gujarati"], row["question_english"])) for row in rows)
        cases.append(Case("eval-set-v0", session_id, turns, hints=hints))
        actions[session_id] = " | ".join(
            str((meta.get(row["question_id"]) or {}).get("expected_action") or "") for row in rows
        )
    return cases, actions


def load_submissions(path: Path) -> list[Case]:
    """amul-test-case-submissions: one v1 body per item (POST /api/test-cases/)."""
    cases = []
    for item in _active(path):
        body = item["input"]
        turns = []
        for turn in body.get("turns") or []:
            if not isinstance(turn, dict) or turn.get("role") not in ("user", "assistant"):
                continue
            text = str(turn.get("content") or "")
            if turn.get("original_content"):
                text += f" [the tester corrected this; the bot first said: {turn['original_content']}]"
            turns.append(Turn(turn["role"], text))
        hints = {
            "tester_tags": ", ".join(body.get("tags") or []),
            "tester_case_types": ", ".join(body.get("case_types") or []),
            "other_topic": str(body.get("other_topic") or ""),
            "persona": str(body.get("persona") or ""),
            "language": str(body.get("language") or ""),
        }
        cases.append(Case("amul-test-case-submissions", item["id"], tuple(turns), str(body.get("description") or ""), hints))
    return cases


def label_rows(cases: list[Case], complete: Callable[[str], str], actions: dict[str, str]) -> list[dict]:
    rows = []
    for number, case in enumerate(cases, start=1):
        print(f"\rlabelling {number}/{len(cases)}", end="", file=sys.stderr, flush=True)
        row = {
            "set": case.set_name,
            "case_id": case.case_id,
            "turns": " | ".join(
                f"{'Farmer' if turn.role == 'user' else 'Bot'}: {mask_personal_data(turn.text)}" for turn in case.turns
            ),
            "set_hints": "; ".join(f"{key}: {value}" for key, value in case.hints.items() if value),
            "set_expected_action": actions.get(case.case_id, "") if case.set_name == "eval-set-v0" else "",
        }
        try:
            label = label_case(case, complete)
        except Exception as exc:  # noqa: BLE001 - one bad case is an error row, not a stopped run
            row["error"] = str(exc) if isinstance(exc, LabelError) else f"{type(exc).__name__}: {exc}"[:200]
        else:
            row.update(label.model_dump())
            row["sub_tags"] = ", ".join(label.sub_tags)
        rows.append(row)
    print(file=sys.stderr)
    return rows


def write_table(rows: list[dict], out: Path) -> None:
    # utf-8-sig so Excel shows Gujarati too; Google Sheets reads it either way.
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=Path, help="amul-seed-questions export (JSON)")
    parser.add_argument("--v0", type=Path, help="inference/amul_eval_set_v0.csv")
    parser.add_argument("--v0-meta", type=Path, help="inference/amul_eval_set_v0_metadata.json")
    parser.add_argument("--submissions", type=Path, help="amul-test-case-submissions export (JSON)")
    parser.add_argument("--out", type=Path, required=True, help="CSV to write")
    parser.add_argument("--limit", type=int, help="label only the first N cases")
    args = parser.parse_args(argv)

    cases: list[Case] = []
    actions: dict[str, str] = {}
    if args.seed:
        cases += load_seed(args.seed)
    if args.v0:
        v0_cases, actions = load_v0(args.v0, args.v0_meta)
        cases += v0_cases
    if args.submissions:
        cases += load_submissions(args.submissions)
    if not cases:
        parser.error("give at least one of --seed, --v0 or --submissions")
    if args.limit:
        cases = cases[: args.limit]

    try:
        model = LiteLLM()
    except LabelError as exc:
        parser.error(str(exc))
    rows = label_rows(cases, model.complete, actions)
    write_table(rows, args.out)
    failed = sum(1 for row in rows if row.get("error"))
    print(f"labelled {len(rows) - failed} of {len(rows)} cases, {failed} errors, table in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
