"""
Deduplicate GoldenSet.csv using exact match, string similarity, vector similarity,
and optional Gemma LLM judge for borderline pairs.

Usage (from repo root):
    python evals/deduplicate_golden_set.py evals/GoldenSet.csv
    python evals/deduplicate_golden_set.py evals/GoldenSet.csv --skip-gemma
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(_REPO_ROOT / ".env")

DEFAULT_OUTPUT_DIR = _REPO_ROOT / "eval_outputs" / "golden_dedup"
EMBED_MODEL = "intfloat/multilingual-e5-base"


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


@dataclass
class Row:
    index: int
    row_id: int
    category: str
    question_gu: str
    normalized: str = ""


@dataclass
class MergeRecord:
    dropped_row_id: int
    kept_row_id: int
    stage: str
    string_score: Optional[float] = None
    vector_score: Optional[float] = None
    gemma_duplicate: Optional[bool] = None
    gemma_confidence: Optional[float] = None
    gemma_reason: str = ""


@dataclass
class DedupState:
    merges: list[MergeRecord] = field(default_factory=list)
    gemma_review: list[dict[str, Any]] = field(default_factory=list)
    audit_rows: list[dict[str, Any]] = field(default_factory=list)



def parse_boolish(value: Any) -> bool:
    """Parse LLM/JSON bools without treating string false as truthy."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def normalize_question(text: str) -> str:
    text = unicodedata.normalize("NFC", (text or "").strip())
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip("?").rstrip("؟").strip()

    def lower_paren(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        return f"({inner.lower()})"

    text = re.sub(r"\(([^)]+)\)", lower_paren, text)
    return text.lower()


def load_rows(csv_path: Path) -> list[Row]:
    rows: list[Row] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        category_col = None
        question_col = None
        for name in reader.fieldnames or []:
            lower = name.strip().lower()
            if lower in {"category", "cat"}:
                category_col = name
            if "gu" in lower or lower in {"question_gu", "q (gu)", "question"}:
                question_col = name
        if not question_col:
            raise ValueError(f"Could not find Gujarati question column in {csv_path}")
        for idx, raw in enumerate(reader):
            question = (raw.get(question_col) or "").strip()
            if not question:
                continue
            category = (raw.get(category_col) or "").strip() if category_col else ""
            rows.append(
                Row(
                    index=len(rows),
                    row_id=len(rows) + 1,
                    category=category,
                    question_gu=question,
                    normalized=normalize_question(question),
                )
            )
    return rows


def string_score(a: str, b: str) -> float:
    from rapidfuzz import fuzz

    return float(fuzz.token_set_ratio(a, b))


def pick_canonical(indices: list[int], rows: list[Row]) -> int:
    return max(indices, key=lambda i: (len(rows[i].question_gu), -rows[i].row_id))


def stage_exact(rows: list[Row], uf: UnionFind, state: DedupState) -> int:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        groups[(row.category, row.normalized)].append(row.index)
    merged = 0
    for indices in groups.values():
        if len(indices) < 2:
            continue
        keep = pick_canonical(indices, rows)
        for idx in indices:
            if idx == keep:
                continue
            uf.union(keep, idx)
            state.merges.append(
                MergeRecord(
                    dropped_row_id=rows[idx].row_id,
                    kept_row_id=rows[keep].row_id,
                    stage="exact",
                    string_score=100.0,
                )
            )
            merged += 1
    return merged


def stage_string(
    rows: list[Row],
    uf: UnionFind,
    state: DedupState,
    *,
    auto_threshold: float,
    borderline_low: float,
    borderline_pairs: set[tuple[int, int]],
) -> int:
    by_cat: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_cat[row.category].append(row.index)

    merged = 0
    for indices in by_cat.values():
        for i_pos, i in enumerate(indices):
            for j in indices[i_pos + 1 :]:
                if uf.find(i) == uf.find(j):
                    continue
                score = string_score(rows[i].normalized, rows[j].normalized)
                if score >= auto_threshold:
                    keep = pick_canonical([i, j], rows)
                    drop = j if keep == i else i
                    uf.union(keep, drop)
                    state.merges.append(
                        MergeRecord(
                            dropped_row_id=rows[drop].row_id,
                            kept_row_id=rows[keep].row_id,
                            stage="string",
                            string_score=score,
                        )
                    )
                    merged += 1
                elif score >= borderline_low:
                    borderline_pairs.add(tuple(sorted((i, j))))
    return merged


def _active_borderline_pairs(
    borderline_pairs: set[tuple[int, int]],
    uf: UnionFind,
) -> set[tuple[int, int]]:
    return {
        pair
        for pair in borderline_pairs
        if uf.find(pair[0]) != uf.find(pair[1])
    }


def embed_questions(rows: list[Row], model_name: str) -> Any:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    texts = [f"query: {row.question_gu}" for row in rows]
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=True)


def stage_vector(
    rows: list[Row],
    uf: UnionFind,
    state: DedupState,
    embeddings: Any,
    *,
    auto_threshold: float,
    auto_string_min: float,
    borderline_low: float,
    borderline_pairs: set[tuple[int, int]],
) -> int:
    import numpy as np

    merged = 0
    by_cat: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_cat[row.category].append(row.index)

    for indices in by_cat.values():
        if len(indices) < 2:
            continue
        idx_arr = np.array(indices)
        vecs = embeddings[idx_arr]
        sim = vecs @ vecs.T
        for a_pos, i in enumerate(indices):
            for b_pos in range(a_pos + 1, len(indices)):
                j = indices[b_pos]
                if uf.find(i) == uf.find(j):
                    continue
                cos = float(sim[a_pos, b_pos])
                str_sc = string_score(rows[i].normalized, rows[j].normalized)
                if cos >= auto_threshold and str_sc >= auto_string_min:
                    keep = pick_canonical([i, j], rows)
                    drop = j if keep == i else i
                    uf.union(keep, drop)
                    state.merges.append(
                        MergeRecord(
                            dropped_row_id=rows[drop].row_id,
                            kept_row_id=rows[keep].row_id,
                            stage="vector",
                            string_score=str_sc,
                            vector_score=cos,
                        )
                    )
                    merged += 1
                elif borderline_low <= cos < auto_threshold and str_sc >= 70.0:
                    borderline_pairs.add(tuple(sorted((i, j))))
    return merged


def _gemma_config(args: argparse.Namespace) -> tuple[Optional[str], str, str]:
    base = (
        args.gemma_base_url
        or os.getenv("OSS_INFERENCE_ENDPOINT_URL")
        or os.getenv("INFERENCE_ENDPOINT_URL")
        or os.getenv("VLLM_BASE_URL")
    )
    if base and not base.rstrip("/").endswith("/v1"):
        base = base.rstrip("/") + "/v1"
    model = (
        args.gemma_model
        or os.getenv("OSS_LLM_MODEL_NAME")
        or os.getenv("LLM_MODEL_NAME")
        or "gemma-4-31b-it"
    )
    api_key = (
        args.gemma_api_key
        or os.getenv("OSS_INFERENCE_API_KEY")
        or os.getenv("INFERENCE_API_KEY")
        or "dummy"
    )
    return base, model, api_key


def gemma_judge_pair(
    q1: str,
    q2: str,
    *,
    base_url: str,
    model: str,
    api_key: str,
    timeout: float,
) -> dict[str, Any]:
    import httpx

    prompt = f"""You deduplicate farmer evaluation questions in Gujarati.

Q1: {q1}
Q2: {q2}

Are these the SAME farmer intent for evaluation (same expected answer/tools)?
Different symptoms, treatments, or actions = NOT duplicate.

Reply with JSON only:
{{"duplicate": true or false, "confidence": 0.0 to 1.0, "reason": "brief"}}"""

    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 256,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?|```$", "", content, flags=re.MULTILINE).strip()
    return json.loads(content)


def stage_gemma(
    rows: list[Row],
    uf: UnionFind,
    state: DedupState,
    borderline_pairs: set[tuple[int, int]],
    *,
    base_url: str,
    model: str,
    api_key: str,
    confidence_threshold: float,
    timeout: float,
) -> int:
    merged = 0
    for i, j in sorted(borderline_pairs):
        if uf.find(i) == uf.find(j):
            continue
        str_sc = string_score(rows[i].normalized, rows[j].normalized)
        try:
            verdict = gemma_judge_pair(
                rows[i].question_gu,
                rows[j].question_gu,
                base_url=base_url,
                model=model,
                api_key=api_key,
                timeout=timeout,
            )
            duplicate = parse_boolish(verdict.get("duplicate"))
            confidence = float(verdict.get("confidence") or 0.0)
            reason = str(verdict.get("reason") or "")
        except Exception as exc:
            err_rec = {
                    "row_id_a": rows[i].row_id,
                    "row_id_b": rows[j].row_id,
                    "question_a": rows[i].question_gu,
                    "question_b": rows[j].question_gu,
                    "string_score": str_sc,
                    "error": str(exc),
                }
            state.gemma_review.append(err_rec)
            state.audit_rows.append(
                {
                    **err_rec,
                    "stage": "gemma",
                    "action": "error",
                    "kept_row_id": "",
                    "dropped_row_id": "",
                    "gemma_duplicate": "",
                    "gemma_confidence": "",
                    "gemma_reason": "",
                }
            )
            continue

        record = {
            "row_id_a": rows[i].row_id,
            "row_id_b": rows[j].row_id,
            "question_a": rows[i].question_gu,
            "question_b": rows[j].question_gu,
            "string_score": str_sc,
            "gemma_duplicate": duplicate,
            "gemma_confidence": confidence,
            "gemma_reason": reason,
        }
        if duplicate and confidence >= confidence_threshold:
            keep = pick_canonical([i, j], rows)
            drop = j if keep == i else i
            uf.union(keep, drop)
            state.merges.append(
                MergeRecord(
                    dropped_row_id=rows[drop].row_id,
                    kept_row_id=rows[keep].row_id,
                    stage="gemma",
                    string_score=str_sc,
                    gemma_duplicate=duplicate,
                    gemma_confidence=confidence,
                    gemma_reason=reason,
                )
            )
            state.audit_rows.append(
                {
                    **record,
                    "stage": "gemma",
                    "action": "merged",
                    "kept_row_id": rows[keep].row_id,
                    "dropped_row_id": rows[drop].row_id,
                }
            )
            merged += 1
        else:
            state.gemma_review.append(record)
            state.audit_rows.append(
                {
                    **record,
                    "stage": "gemma",
                    "action": "kept_distinct",
                    "kept_row_id": "",
                    "dropped_row_id": "",
                }
            )
    return merged


def write_outputs(
    rows: list[Row],
    uf: UnionFind,
    state: DedupState,
    output_dir: Path,
    source_name: str,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    clusters: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        clusters[uf.find(row.index)].append(row.index)

    deduped_rows: list[dict[str, Any]] = []
    cluster_json: list[dict[str, Any]] = []
    for root, indices in sorted(clusters.items(), key=lambda item: min(item[1])):
        keep_idx = pick_canonical(indices, rows)
        keep = rows[keep_idx]
        members = [rows[i] for i in sorted(indices, key=lambda x: rows[x].row_id)]
        deduped_rows.append(
            {
                "row_id": keep.row_id,
                "category": keep.category,
                "question_gu": keep.question_gu,
                "cluster_size": len(members),
            }
        )
        cluster_json.append(
            {
                "kept_row_id": keep.row_id,
                "cluster_size": len(members),
                "members": [
                    {"row_id": m.row_id, "question_gu": m.question_gu}
                    for m in members
                ],
            }
        )

    deduped_path = output_dir / "GoldenSet_deduped.csv"
    with deduped_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_id", "category", "question_gu", "cluster_size"],
        )
        writer.writeheader()
        writer.writerows(deduped_rows)

    report_path = output_dir / "GoldenSet_duplicates_report.csv"
    with report_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dropped_row_id",
                "kept_row_id",
                "stage",
                "string_score",
                "vector_score",
                "gemma_duplicate",
                "gemma_confidence",
                "gemma_reason",
            ],
        )
        writer.writeheader()
        for merge in state.merges:
            writer.writerow(
                {
                    "dropped_row_id": merge.dropped_row_id,
                    "kept_row_id": merge.kept_row_id,
                    "stage": merge.stage,
                    "string_score": merge.string_score,
                    "vector_score": merge.vector_score,
                    "gemma_duplicate": merge.gemma_duplicate,
                    "gemma_confidence": merge.gemma_confidence,
                    "gemma_reason": merge.gemma_reason,
                }
            )

    review_path = output_dir / "GoldenSet_review.csv"
    if state.gemma_review:
        review_fields: list[str] = []
        for row in state.gemma_review:
            for key in row:
                if key not in review_fields:
                    review_fields.append(key)
        with review_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=review_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(state.gemma_review)

    audit_rows = list(state.audit_rows)
    for merge in state.merges:
        if merge.stage == "gemma":
            continue
        audit_rows.append(
            {
                "row_id_a": merge.dropped_row_id,
                "row_id_b": merge.kept_row_id,
                "question_a": "",
                "question_b": "",
                "string_score": merge.string_score,
                "vector_score": merge.vector_score,
                "stage": merge.stage,
                "action": "merged",
                "kept_row_id": merge.kept_row_id,
                "dropped_row_id": merge.dropped_row_id,
                "gemma_duplicate": merge.gemma_duplicate,
                "gemma_confidence": merge.gemma_confidence,
                "gemma_reason": merge.gemma_reason,
                "error": "",
            }
        )
    audit_path = output_dir / "GoldenSet_merge_audit.csv"
    audit_fields = [
        "row_id_a",
        "row_id_b",
        "question_a",
        "question_b",
        "stage",
        "action",
        "kept_row_id",
        "dropped_row_id",
        "string_score",
        "vector_score",
        "gemma_duplicate",
        "gemma_confidence",
        "gemma_reason",
        "error",
    ]
    with audit_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(audit_rows)

    summary_path = output_dir / "GoldenSet_dedup_summary.json"
    summary = {
        "source": source_name,
        "input_rows": len(rows),
        "deduped_rows": len(deduped_rows),
        "duplicates_removed": len(rows) - len(deduped_rows),
        "merge_events": len(state.merges),
        "review_pairs": len(state.gemma_review),
        "audit_rows": len(audit_rows),
        "stages": {
            stage: sum(1 for m in state.merges if m.stage == stage)
            for stage in ["exact", "string", "vector", "gemma"]
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "GoldenSet_clusters.json").write_text(
        json.dumps(cluster_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deduplicate GoldenSet.csv")
    parser.add_argument("csv_path", type=Path, help="Input GoldenSet.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: eval_outputs/golden_dedup)",
    )
    parser.add_argument("--string-auto", type=float, default=95.0)
    parser.add_argument("--string-borderline", type=float, default=85.0)
    parser.add_argument("--vector-auto", type=float, default=0.92)
    parser.add_argument("--vector-borderline", type=float, default=0.85)
    parser.add_argument("--vector-string-min", type=float, default=80.0)
    parser.add_argument("--embed-model", default=EMBED_MODEL)
    parser.add_argument("--skip-gemma", action="store_true")
    parser.add_argument("--gemma-base-url", default=None)
    parser.add_argument("--gemma-model", default=None)
    parser.add_argument("--gemma-api-key", default=None)
    parser.add_argument("--gemma-confidence", type=float, default=0.8)
    parser.add_argument("--gemma-timeout", type=float, default=120.0)
    parser.add_argument("--max-gemma-pairs", type=int, default=200)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    csv_path = args.csv_path if args.csv_path.is_absolute() else _REPO_ROOT / args.csv_path
    output_dir = args.output_dir if args.output_dir.is_absolute() else _REPO_ROOT / args.output_dir

    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit(f"No rows loaded from {csv_path}")

    uf = UnionFind(len(rows))
    state = DedupState()
    borderline_pairs: set[tuple[int, int]] = set()

    print(f"Loaded {len(rows)} rows from {csv_path}")
    exact_n = stage_exact(rows, uf, state)
    print(f"Exact merges: {exact_n}")

    string_n = stage_string(
        rows,
        uf,
        state,
        auto_threshold=args.string_auto,
        borderline_low=args.string_borderline,
        borderline_pairs=borderline_pairs,
    )
    print(f"String auto merges: {string_n}")

    print(f"Embedding with {args.embed_model}...")
    embeddings = embed_questions(rows, args.embed_model)
    vector_n = stage_vector(
        rows,
        uf,
        state,
        embeddings,
        auto_threshold=args.vector_auto,
        auto_string_min=args.vector_string_min,
        borderline_low=args.vector_borderline,
        borderline_pairs=borderline_pairs,
    )
    print(f"Vector auto merges: {vector_n}")
    active_borderline = _active_borderline_pairs(borderline_pairs, uf)
    print(f"Borderline pairs for judge: {len(active_borderline)}")

    gemma_n = 0
    if args.skip_gemma:
        print("Skipping Gemma judge (--skip-gemma)")
    else:
        base_url, model, api_key = _gemma_config(args)
        if not base_url:
            print("No Gemma/vLLM URL configured — skipping judge (use --skip-gemma to silence)")
        else:
            ranked = sorted(
                active_borderline,
                key=lambda pair: string_score(
                    rows[pair[0]].normalized,
                    rows[pair[1]].normalized,
                ),
                reverse=True,
            )
            limited = ranked[: args.max_gemma_pairs]
            skipped = ranked[args.max_gemma_pairs :]
            for i, j in skipped:
                skip_rec = {
                    "row_id_a": rows[i].row_id,
                    "row_id_b": rows[j].row_id,
                    "question_a": rows[i].question_gu,
                    "question_b": rows[j].question_gu,
                    "string_score": string_score(rows[i].normalized, rows[j].normalized),
                    "stage": "gemma_cap",
                    "action": "skipped_cap",
                    "kept_row_id": "",
                    "dropped_row_id": "",
                    "gemma_duplicate": "",
                    "gemma_confidence": "",
                    "gemma_reason": f"beyond --max-gemma-pairs={args.max_gemma_pairs}",
                    "error": "",
                }
                state.gemma_review.append(skip_rec)
                state.audit_rows.append(skip_rec)
            print(
                f"Gemma judge on {len(limited)} pairs via {base_url} model={model} "
                f"(skipped_cap={len(skipped)})"
            )
            gemma_n = stage_gemma(
                rows,
                uf,
                state,
                set(limited),
                base_url=base_url,
                model=model,
                api_key=api_key,
                confidence_threshold=args.gemma_confidence,
                timeout=args.gemma_timeout,
            )
            print(f"Gemma merges: {gemma_n}")

    summary = write_outputs(rows, uf, state, output_dir, csv_path.name)
    print("")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nOutputs written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
