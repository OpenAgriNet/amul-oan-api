#!/usr/bin/env python3
"""
Blind pairwise + faithfulness LLM-as-judge for the guided-decoding bench.

Judge: gemma-4-31b-it @ http://localhost:8020/v1  (same model that produced the
answers -> a plain single-answer quality score would be self-preference biased,
so the head-to-head metric is a BLIND PAIRWISE comparison with per-row
randomised A/B assignment: bias is symmetric across arms and cancels.)

Prompt shapes reuse ~/eval/04_score.py (RAGAS-style) with Gujarati framing.

Usage:
  python3 judge.py --pair voice
  python3 judge.py --pair voice --pair voiceprod --pair chat
  python3 judge.py --all
"""

import argparse
import csv
import json
import math
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

import httpx

csv.field_size_limit(10 ** 9)

OUT_DIR = Path.home() / "eval-runs" / "out"
JUDGE_URL = "http://localhost:8020/v1/chat/completions"
JUDGE_MODEL = "gemma-4-31b-it"
CONCURRENCY = 6          # HARD CAP - shared GPU pool with production traffic
SEED = 20260801

PAIRS = {
    "voice":     ("voice_control.csv", "voice_guided.csv"),
    "voiceprod": ("voiceprod_control.csv", "voiceprod_guided.csv"),
    "chat":      ("chat_control.csv", "chat_guided.csv"),
}

MAX_ANS = 3000
MAX_CTX = 6000

JUDGE_PROMPT_PAIRWISE = """You are evaluating an AI assistant that answers questions from Indian dairy farmers.

The farmer's question and both candidate answers are written in GUJARATI. Evaluate them as Gujarati text: judge the agricultural/veterinary correctness, the completeness and usefulness of the advice for a farmer, and whether the Gujarati is natural, fluent and consistently in Gujarati script (mixed or drifting script is a defect).

Farmer's question:
{question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Which answer better answers the farmer's question? If they are equally good (or equally bad), say TIE.

Respond with ONLY one word: "A", "B", or "TIE". Nothing else."""

JUDGE_PROMPT_FAITHFULNESS = """You are evaluating whether an AI answer is grounded in the provided context.

The answer is written in GUJARATI; the retrieved context may be in English or Gujarati. Judge whether the claims made in the Gujarati answer are supported by the context, across languages.

Retrieved context (from document search):
{context}

Model's answer (Gujarati):
{answer}

Score how well the answer is supported by the retrieved context.
- 1.0: Every claim is directly supported by the context
- 0.7: Most claims supported, minor extrapolations
- 0.5: Some claims supported but some appear fabricated
- 0.3: Answer mostly ignores context, relies on general knowledge
- 0.0: Answer contradicts or is completely ungrounded

Respond with ONLY a number between 0 and 1. Nothing else."""


def call_judge(prompt: str, max_tokens: int) -> str | None:
    """Return raw judge text, or None on transport failure (after retries)."""
    for attempt in range(3):
        try:
            r = httpx.post(
                JUDGE_URL,
                json={
                    "model": JUDGE_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                },
                timeout=180,
                headers={"Authorization": "Bearer dummy"},
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception:
            if attempt == 2:
                return None
    return None


def parse_verdict(text: str | None) -> str | None:
    """-> 'A' | 'B' | 'TIE' | None (unparseable)."""
    if text is None:
        return None
    t = text.strip().upper()
    t = t.strip("*_`\"' .\n")
    if t in ("A", "B", "TIE"):
        return t
    m = re.match(r"^\W*(TIE|A|B)\b", t)
    if m:
        return m.group(1)
    # last resort: a single unambiguous mention
    hits = set(re.findall(r"\b(TIE|A|B)\b", t))
    if len(hits) == 1:
        return hits.pop()
    return None


def parse_score(text: str | None) -> float | None:
    if text is None:
        return None
    m = re.search(r"\d+\.?\d*", text.strip())
    if not m:
        return None
    try:
        return min(max(float(m.group()), 0.0), 1.0)
    except ValueError:
        return None


def load(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def context_of(row: dict) -> str:
    try:
        tl = json.loads(row.get("tool_log") or "[]")
    except Exception:
        return ""
    parts = [str(e.get("content") or "") for e in tl if e.get("t") == "ret"]
    parts = [p for p in parts if p.strip()]
    return "\n\n---\n\n".join(parts)


def binom_two_sided_p(k: int, n: int) -> float | None:
    """Exact two-sided sign test p-value for k successes in n trials, p=0.5."""
    if n == 0:
        return None
    def pmf(i):
        return math.comb(n, i) * 0.5 ** n
    obs = pmf(k)
    tol = obs * (1 + 1e-9)
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= tol))


def run_pair(name: str) -> dict | None:
    cfile, gfile = PAIRS[name]
    cpath, gpath = OUT_DIR / cfile, OUT_DIR / gfile
    if not cpath.exists() or not gpath.exists():
        print(f"[skip] {name}: missing CSV ({cpath.name} / {gpath.name})")
        return None
    for log in (OUT_DIR / (cfile[:-4] + ".log"), OUT_DIR / (gfile[:-4] + ".log")):
        if log.exists() and "[done]" not in log.read_text(errors="ignore"):
            print(f"[skip] {name}: {log.name} has no [done] line -> run incomplete")
            return None

    ctrl, guid = load(cpath), load(gpath)
    by_qid_g = {r["qid"]: r for r in guid}
    rows = [(c, by_qid_g[c["qid"]]) for c in ctrl if c["qid"] in by_qid_g]
    print(f"[{name}] {len(ctrl)} control / {len(guid)} guided -> {len(rows)} aligned qids")

    rng = random.Random(SEED)
    # ---- build pairwise tasks (blind, per-row randomised A/B) ----
    pw_tasks = []
    for c, g in rows:
        control_is_a = rng.random() < 0.5
        a, b = (c, g) if control_is_a else (g, c)
        prompt = JUDGE_PROMPT_PAIRWISE.format(
            question=(c["question"] or "")[:1500],
            answer_a=(a["answer"] or "")[:MAX_ANS],
            answer_b=(b["answer"] or "")[:MAX_ANS],
        )
        pw_tasks.append({"qid": c["qid"], "control_is_a": control_is_a, "prompt": prompt})

    # ---- build faithfulness tasks (per answer, own context) ----
    f_tasks = []
    skipped_ctx = {"control": 0, "guided": 0}
    for c, g in rows:
        for arm, row in (("control", c), ("guided", g)):
            ctx = context_of(row)
            if not ctx.strip():
                skipped_ctx[arm] += 1
                continue
            f_tasks.append({
                "qid": c["qid"], "arm": arm,
                "prompt": JUDGE_PROMPT_FAITHFULNESS.format(
                    context=ctx[:MAX_CTX], answer=(row["answer"] or "")[:MAX_ANS]),
            })

    total = len(pw_tasks) + len(f_tasks)
    print(f"[{name}] judge calls: {len(pw_tasks)} pairwise + {len(f_tasks)} faithfulness = {total}")

    done = [0]

    def work(task):
        if "arm" in task:
            raw = call_judge(task["prompt"], 10)
            task["raw"] = raw
            task["score"] = parse_score(raw)
        else:
            raw = call_judge(task["prompt"], 8)
            task["raw"] = raw
            v = parse_verdict(raw)
            task["verdict_ab"] = v
            if v is None:
                task["winner"] = None
            elif v == "TIE":
                task["winner"] = "tie"
            elif v == "A":
                task["winner"] = "control" if task["control_is_a"] else "guided"
            else:
                task["winner"] = "guided" if task["control_is_a"] else "control"
        done[0] += 1
        if done[0] % 50 == 0:
            print(f"[{name}] {done[0]}/{total}", flush=True)
        return task

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        list(ex.map(work, pw_tasks + f_tasks))

    # ---- aggregate pairwise ----
    wins_c = sum(1 for t in pw_tasks if t["winner"] == "control")
    wins_g = sum(1 for t in pw_tasks if t["winner"] == "guided")
    ties = sum(1 for t in pw_tasks if t["winner"] == "tie")
    pw_fail = sum(1 for t in pw_tasks if t["winner"] is None)
    decisive = wins_c + wins_g
    p_val = binom_two_sided_p(wins_g, decisive)

    # position-bias diagnostic: how often did the judge pick slot A?
    picked_a = sum(1 for t in pw_tasks if t.get("verdict_ab") == "A")
    picked_b = sum(1 for t in pw_tasks if t.get("verdict_ab") == "B")

    # ---- aggregate faithfulness ----
    faith = {}
    for arm in ("control", "guided"):
        vals = [t["score"] for t in f_tasks if t["arm"] == arm and t["score"] is not None]
        fails = sum(1 for t in f_tasks if t["arm"] == arm and t["score"] is None)
        faith[arm] = {
            "mean": round(mean(vals), 4) if vals else None,
            "n_scored": len(vals),
            "n_skipped_no_context": skipped_ctx[arm],
            "n_judge_failed": fails,
        }

    # ---- divergence candidates: decisive verdicts, biggest answer difference ----
    verdict_by_qid = {t["qid"]: t for t in pw_tasks}
    faith_by = {(t["qid"], t["arm"]): t["score"] for t in f_tasks}
    diverged = []
    for c, g in rows:
        t = verdict_by_qid.get(c["qid"])
        if not t or t["winner"] in (None, "tie"):
            continue
        fc, fg = faith_by.get((c["qid"], "control")), faith_by.get((c["qid"], "guided"))
        fdelta = abs(fc - fg) if (fc is not None and fg is not None) else 0.0
        la, lb = len(c["answer"] or ""), len(g["answer"] or "")
        lendelta = abs(la - lb) / max(la, lb, 1)
        diverged.append({
            "qid": c["qid"], "winner": t["winner"], "question": c["question"],
            "control_answer": c["answer"], "guided_answer": g["answer"],
            "faith_control": fc, "faith_guided": fg,
            "faith_delta": round(fdelta, 3), "len_delta_frac": round(lendelta, 3),
            "rank_key": round(fdelta + lendelta, 4),
        })
    diverged.sort(key=lambda d: d["rank_key"], reverse=True)

    result = {
        "pair": name,
        "judge_model": JUDGE_MODEL,
        "judge_url": JUDGE_URL,
        "seed": SEED,
        "concurrency": CONCURRENCY,
        "n_rows": len(rows),
        "pairwise": {
            "wins_control": wins_c,
            "wins_guided": wins_g,
            "ties": ties,
            "unparseable": pw_fail,
            "n_decisive": decisive,
            "guided_win_rate_decisive": round(wins_g / decisive, 4) if decisive else None,
            "guided_win_rate_all": round(wins_g / len(pw_tasks), 4) if pw_tasks else None,
            "sign_test_two_sided_p": round(p_val, 5) if p_val is not None else None,
            "position_picked_A": picked_a,
            "position_picked_B": picked_b,
        },
        "faithfulness": faith,
        "judge_failure_rate_overall": round(
            (pw_fail + sum(f["n_judge_failed"] for f in faith.values())) / total, 4) if total else None,
        "top_divergences": diverged[:10],
        "per_row": [
            {"qid": t["qid"], "control_is_a": t["control_is_a"],
             "verdict_ab": t.get("verdict_ab"), "winner": t["winner"],
             "raw": (t.get("raw") or "")[:80]}
            for t in pw_tasks
        ],
        "per_row_faithfulness": [
            {"qid": t["qid"], "arm": t["arm"], "score": t["score"],
             "raw": (t.get("raw") or "")[:80]}
            for t in f_tasks
        ],
    }

    out = OUT_DIR / f"judge_{name}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[{name}] wrote {out}")
    print(f"[{name}] pairwise: control={wins_c} guided={wins_g} tie={ties} unparseable={pw_fail} p={result['pairwise']['sign_test_two_sided_p']}")
    print(f"[{name}] faithfulness: {faith}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", choices=list(PAIRS))
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    names = list(PAIRS) if args.all else (args.pair or [])
    if not names:
        ap.error("give --pair NAME (repeatable) or --all")
    for n in names:
        run_pair(n)


if __name__ == "__main__":
    main()
