#!/usr/bin/env python3
"""Bare-gemma4 bench runner with optional vLLM guided decoding.

Derived from scripts/run_goldenset_bare.py (exp1, 2026-05-26), which is left
untouched so the baseline stays reproducible. Additions:
  --questions  arbitrary CSV (id,question) instead of the goldenset schema
  --guided     apply the #142 script constraint via structured_outputs regex
  --strict-142 use #142's char set verbatim (no degree sign) for the A/B on it
  --out        explicit output path
  --arm        label written into every row

Must run with cwd = repo root; assets load via relative paths.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path

csv.field_size_limit(sys.maxsize)

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env.experiment"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
else:
    print(f"[warn] .env.experiment not found at {ENV_PATH}")

os.environ.setdefault("LOGFIRE_SEND_TO_LOGFIRE", "false")
os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")
os.environ.setdefault("LOGFIRE_TOKEN", "")

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from guided_patch import guided_settings  # noqa: E402

FIELDS = [
    "row_num", "qid", "arm", "guided", "question", "answer",
    "latency_s", "error", "num_tool_calls", "tool_names", "tool_log",
]


def _import_agent():
    from agents.agrinet import agrinet_agent
    from agents.deps import FarmerContext
    return agrinet_agent, FarmerContext


async def run_one(agent, FarmerContext, row_num, qid, question, lang, settings, arm, sem):
    async with sem:
        t0 = time.perf_counter()
        row = {
            "row_num": row_num, "qid": qid, "arm": arm,
            "guided": bool(settings), "question": question, "answer": "",
            "latency_s": 0.0, "error": "", "num_tool_calls": 0,
            "tool_names": "", "tool_log": "",
        }
        try:
            deps = FarmerContext(
                query=question, lang_code=lang, session_id=f"bench-{arm}-{row_num}",
            )
            kwargs = {"user_prompt": deps.get_user_message(), "deps": deps}
            if settings is not None:
                kwargs["model_settings"] = settings
            result = await agent.run(**kwargs)
            row["answer"] = (result.output if hasattr(result, "output") else str(result)) or ""

            names, log, n = [], [], 0
            try:
                for m in (result.all_messages() if hasattr(result, "all_messages") else []):
                    for p in getattr(m, "parts", []) or []:
                        ks = str(getattr(p, "part_kind", "") or type(p).__name__).lower()
                        if "tool-call" in ks:
                            n += 1
                            tn = getattr(p, "tool_name", "?")
                            names.append(tn)
                            log.append({"t": "call", "tool": tn,
                                        "args": str(getattr(p, "args", ""))[:600]})
                        elif "tool-return" in ks:
                            log.append({"t": "ret", "tool": getattr(p, "tool_name", "?"),
                                        "content": str(getattr(p, "content", ""))[:2000]})
            except Exception:
                pass
            row["num_tool_calls"] = n
            row["tool_names"] = "|".join(names)
            row["tool_log"] = json.dumps(log, ensure_ascii=False)
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
        row["latency_s"] = round(time.perf_counter() - t0, 3)
        return row


def load_questions(path, limit):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            q = next((r[c] for c in ("question", "question_gu", "query", "input")
                      if r.get(c)), "")
            if not q:
                continue
            rows.append((i, r.get("id", str(i)), q, (r.get("lang") or "").strip().lower()))
            if limit and len(rows) >= limit:
                break
    return rows


async def main_async(a):
    agent, FarmerContext = _import_agent()
    print(f"[ok] agent={getattr(agent, 'name', '?')}")
    if a.dry_run:
        s = guided_settings(a.lang, extended=not a.strict_142)
        print(f"[dry-run] guided={'ON' if (a.guided and s) else 'OFF'}")
        if a.guided and s:
            print(f"[dry-run] pattern={s.get('extra_body')['structured_outputs']['regex'][:120]}")
        return 0

    rows = load_questions(a.questions, a.limit)

    # Per-row language: the bench set mixes gu/en/hi. Constrain only languages that
    # have a script range, exactly as #142 does -- English stays unconstrained in
    # BOTH arms, so it contributes no A/B signal but keeps the tool-flow comparison
    # on the real production mix.
    def _settings_for(lang):
        if not a.guided:
            return None
        return guided_settings(lang or a.lang, extended=not a.strict_142)

    n_con = sum(1 for _, _, _, lg in rows if _settings_for(lg) is not None)
    print(f"[info] {len(rows)} rows | arm={a.arm} | guided_flag={a.guided} "
          f"| rows_constrained={n_con} | strict142={a.strict_142} "
          f"| concurrency={a.concurrency}")

    sem = asyncio.Semaphore(a.concurrency)
    tasks = [run_one(agent, FarmerContext, rn, qid, q, (lg or a.lang), _settings_for(lg), a.arm, sem)
             for rn, qid, q, lg in rows]

    out, done = [], 0
    for coro in asyncio.as_completed(tasks):
        out.append(await coro)
        done += 1
        if done % 25 == 0 or done == len(tasks):
            errs = sum(1 for x in out if x["error"])
            print(f"[progress] {done}/{len(tasks)} errors={errs}", flush=True)

    out.sort(key=lambda r: r["row_num"])
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)
    errs = sum(1 for x in out if x["error"])
    print(f"[done] {len(out)} rows -> {a.out} ({errs} errors)")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", required=False, default="")
    p.add_argument("--out", default="bench_out.csv")
    p.add_argument("--arm", default="bare")
    p.add_argument("--lang", default="gu")
    p.add_argument("--guided", action="store_true")
    p.add_argument("--strict-142", action="store_true",
                   help="use #142's char set verbatim (omits the degree sign)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if not a.dry_run and not a.questions:
        p.error("--questions required unless --dry-run")
    sys.exit(asyncio.run(main_async(a)))


if __name__ == "__main__":
    main()
