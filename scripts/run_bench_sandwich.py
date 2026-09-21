#!/usr/bin/env python3
"""PROD-EQUIVALENT "sandwich" bench runner.

Drives the REAL production chat path — ``app.services.chat.stream_chat_messages``
— end to end, one call per bench question, and writes the SAME CSV schema as
``scripts/run_bench.py`` so the two arms are directly comparable.

What runs for real (nothing is reimplemented here):
  gu/hi -> en pre-translation  (translation.translate_to_english_pretranslation)
  moderation agent             (agents.moderation)
  agrinet agent in ENGLISH     (agents.agrinet, streamed via agent.iter)
  sentence-batched en -> gu post-translation through TranslateGemma with
  GU_PREFERRED_TRANSLATION_RULES + mini-glossary Terminology Rules in the
  instruction and GU_POST_REPLACEMENTS / gu_term_policy.json applied per chunk.

Only the infrastructure the box cannot provide is stubbed (Redis). See PATCHES.

Must run with cwd = repo root; assets load via relative paths.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import threading
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

FIELDS = [
    "row_num", "qid", "arm", "guided", "question", "answer",
    "latency_s", "error", "num_tool_calls", "tool_names", "tool_log",
]

# ── evidence collectors (proof that the policy layers actually ran) ───────────
_EV_LOCK = threading.Lock()
POLICY_HITS: list[dict] = []          # chunks where GU_POST_REPLACEMENTS changed text
INSTRUCTION_SAMPLE: list[str] = []    # first built translation instruction
TERM_HIT_COUNTS: list[str] = []       # every forbidden key actually replaced
COSMETIC = [0]                        # diffs that were whitespace-only
MAX_POLICY_HITS = 400


def _patch_everything(capture):
    """Install the Redis stubs + evidence hooks. Returns a dict of notes."""
    import app.services.chat as chat_mod
    import app.services.translation as tr_mod

    notes = {}

    # ── PATCH 1: Redis is not running on this box. -----------------------------
    async def _noop_set_cache(*a, **k):
        return True

    class _StubCache:
        async def delete(self, *a, **k):
            return True

        async def set(self, *a, **k):
            return True

        async def get(self, *a, **k):
            return None

    chat_mod.set_cache = _noop_set_cache
    chat_mod.cache = _StubCache()

    # ── PATCH 2: capture the agent message list (tool calls) -------------------
    # stream_chat_messages hands the completed message list to
    # update_message_history() as its last act; we intercept it instead of
    # writing to Redis. Same call site, same object prod persists.
    async def _capture_history(session_id, all_messages):
        capture[session_id] = all_messages
        return True

    chat_mod.update_message_history = _capture_history

    # ── PATCH 3: no Langfuse (no keys, no collector on this box) ---------------
    chat_mod.get_langfuse_client = None
    chat_mod.propagate_attributes = None
    tr_mod.get_langfuse_client = None

    # ── EVIDENCE A: did GU_POST_REPLACEMENTS (83-pair policy) actually fire? ---
    _orig_norm = tr_mod._post_normalize_gu_translation
    _forbidden = list((tr_mod.GU_TERM_POLICY.get("forbidden") or {}).items())

    def _norm_probe(text, target_lang, **kw):
        out = _orig_norm(text, target_lang, **kw)
        if out != text:
            keys = [k for k, _ in _forbidden if k and k in text and k not in out]
            with _EV_LOCK:
                COSMETIC[0] += 0 if keys else 1
                if keys:
                    TERM_HIT_COUNTS.extend(keys)
                if keys and len(POLICY_HITS) < MAX_POLICY_HITS:
                    POLICY_HITS.append({"before": text, "after": out,
                                        "forbidden_keys_replaced": keys,
                                        "target_lang": target_lang})
        return out

    tr_mod._post_normalize_gu_translation = _norm_probe

    # ── EVIDENCE B: what instruction is handed to TranslateGemma? -------------
    _orig_instr = tr_mod._build_translation_instruction

    def _instr_probe(*a, **k):
        out = _orig_instr(*a, **k)
        with _EV_LOCK:
            if not INSTRUCTION_SAMPLE:
                INSTRUCTION_SAMPLE.append(out)
        return out

    tr_mod._build_translation_instruction = _instr_probe

    notes["n_gu_rules"] = len(tr_mod.GU_PREFERRED_TRANSLATION_RULES)
    notes["n_voice_gu_rules"] = len(tr_mod.VOICE_GU_PREFERRED_TRANSLATION_RULES)
    notes["n_policy_replacements"] = len(tr_mod.GU_POLICY_REPLACEMENTS)
    notes["n_post_replacements_total"] = len(tr_mod.GU_POST_REPLACEMENTS)
    notes["post_translation_chain"] = [
        f"{t.provider}:{t.model_name}@{t.endpoint}" for t in tr_mod._post_translation_chain()
    ]
    return notes


def _tool_rows(messages):
    names, log, n = [], [], 0
    try:
        for m in messages or []:
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
    return n, "|".join(names), json.dumps(log, ensure_ascii=False)


async def run_one(row_num, qid, question, lang, arm, profile, capture, sem):
    from fastapi import BackgroundTasks
    from app.services.chat import stream_chat_messages

    async with sem:
        t0 = time.perf_counter()
        row = {
            "row_num": row_num, "qid": qid, "arm": arm,
            "guided": False, "question": question, "answer": "",
            "latency_s": 0.0, "error": "", "num_tool_calls": 0,
            "tool_names": "", "tool_log": "",
        }
        session_id = f"bench-{arm}-{row_num}"
        try:
            chunks = []
            gen = stream_chat_messages(
                query=question,
                session_id=session_id,
                source_lang=lang,
                target_lang=lang,
                channel="web",
                user_id="bench",
                history=[],
                user_info={},
                background_tasks=BackgroundTasks(),
                use_translation_pipeline=True,
                pipeline_profile=profile,
            )
            async for chunk in gen:
                chunks.append(chunk)
            row["answer"] = "".join(chunks)
            n, names, log = _tool_rows(capture.pop(session_id, []))
            row["num_tool_calls"] = n
            row["tool_names"] = names
            row["tool_log"] = log
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            capture.pop(session_id, None)
        row["latency_s"] = round(time.perf_counter() - t0, 3)
        return row


def load_questions(path, limit, only_lang=""):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            q = next((r[c] for c in ("question", "question_gu", "query", "input")
                      if r.get(c)), "")
            if not q:
                continue
            lg = (r.get("lang") or "").strip().lower()
            if only_lang and lg != only_lang:
                continue
            rows.append((i, r.get("id", str(i)), q, lg))
            if limit and len(rows) >= limit:
                break
    return rows


async def main_async(a):
    capture: dict = {}
    notes = _patch_everything(capture)
    print(f"[cfg] chat GU style rules injected  : {notes['n_gu_rules']} "
          f"(voice-only list has {notes['n_voice_gu_rules']})")
    print(f"[cfg] gu_term_policy forbidden pairs : {notes['n_policy_replacements']}")
    print(f"[cfg] GU_POST_REPLACEMENTS total     : {notes['n_post_replacements_total']}")
    print(f"[cfg] post-translation chain         : {notes['post_translation_chain']}")

    from app.llm_core import resolver as _r
    from app.llm_core.config_model import Step as _S
    for step in (_S.PRE_TRANSLATION, _S.MODERATION, _S.AGENT, _S.POST_TRANSLATION):
        t = _r.primary_tier(step, a.profile)
        print(f"[cfg] {step.value:16} primary -> kind={t.kind} provider={t.provider} "
              f"model={t.model_name} endpoint={t.endpoint}")

    rows = load_questions(a.questions, a.limit, a.only_lang)
    print(f"[info] {len(rows)} rows | arm={a.arm} | profile={a.profile} "
          f"| concurrency={a.concurrency}", flush=True)
    if a.dry_run:
        return 0

    sem = asyncio.Semaphore(a.concurrency)
    tasks = [run_one(rn, qid, q, lg or a.lang, a.arm, a.profile, capture, sem)
             for rn, qid, q, lg in rows]

    out, done = [], 0
    for coro in asyncio.as_completed(tasks):
        out.append(await coro)
        done += 1
        if done % 25 == 0 or done == len(tasks):
            errs = sum(1 for x in out if x["error"])
            print(f"[progress] {done}/{len(tasks)} errors={errs} "
                  f"forbidden_replacements={len(TERM_HIT_COUNTS)}", flush=True)

    out.sort(key=lambda r: r["row_num"])
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)
    errs = sum(1 for x in out if x["error"])
    print(f"[done] {len(out)} rows -> {a.out} ({errs} errors)")

    ev = Path(a.out).with_name(Path(a.out).stem + "_evidence.json")
    with open(ev, "w", encoding="utf-8") as f:
        json.dump({
            "config": notes,
            "instruction_sample": INSTRUCTION_SAMPLE[:1],
            "forbidden_term_replacements_total": len(TERM_HIT_COUNTS),
            "forbidden_term_replacement_counts": {
                k: TERM_HIT_COUNTS.count(k) for k in sorted(set(TERM_HIT_COUNTS))
            },
            "whitespace_only_normalizations": COSMETIC[0],
            "policy_hits_recorded": len(POLICY_HITS),
            "policy_hits": POLICY_HITS,
        }, f, ensure_ascii=False, indent=2)
    print(f"[done] evidence -> {ev} "
          f"({len(TERM_HIT_COUNTS)} forbidden-term replacements, "
          f"{COSMETIC[0]} whitespace-only)")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", required=False, default="")
    p.add_argument("--out", default="bench_sandwich_out.csv")
    p.add_argument("--arm", default="chat_sandwich")
    p.add_argument("--lang", default="gu")
    p.add_argument("--only-lang", default="")
    p.add_argument("--profile", default="oss",
                   help="llm_core named profile (prod chat runs 100%% 'oss')")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if not a.dry_run and not a.questions:
        p.error("--questions required unless --dry-run")
    sys.exit(asyncio.run(main_async(a)))


if __name__ == "__main__":
    main()
