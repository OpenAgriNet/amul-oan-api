#!/usr/bin/env python3
"""Bare-gemma4 bench runner + TERMINOLOGY PARITY arms.

Copy of scripts/run_bench.py (left untouched so the exp2 baseline stays
reproducible) plus the two gates that production applies in its TRANSLATION
layer and that the bare gu-native arm never received:

  --inject-rules  append the 33 GU_PREFERRED_TRANSLATION_RULES to the agent's
                  system prompt at RUNTIME (no prompt file is edited on disk).
                  The rules are re-phrased from "instructions to a translator"
                  to "instructions to the answering agent"; every Gujarati term
                  is kept byte-identical to the source list.
  --apply-map     apply the 83 forbidden->preferred pairs from
                  gu_term_policy.json to the model's answer as a post-step,
                  longest-key-first, exactly as _build_gu_policy_replacements()
                  does in app/services/translation.py.

Output schema is run_bench.py's EXACT 11 columns, then three appended columns:
  inject_rules, apply_map, answer_raw   (answer_raw = pre-map answer)

Must run with cwd = repo root; assets load via relative paths.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
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
    # appended (downstream scorers read by name via DictReader)
    "inject_rules", "apply_map", "answer_raw",
]

DEFAULT_POLICY = Path.home() / "eval-runs" / "chat_gu_term_policy.json"

# ---------------------------------------------------------------------------
# The 33 rules. Source of truth: GU_PREFERRED_TRANSLATION_RULES in
# ~/eval-runs/voice-oan-api/app/services/translation.py (line 111).
# 27 are verbatim. 6 are re-pointed from translator-voice to agent-voice
# (marked ADAPTED); no Gujarati term is altered in any of them.
# ---------------------------------------------------------------------------
GU_AGENT_TERM_RULES = [
    "Use farmer-preferred Gujarati livestock terms.",
    # ADAPTED: "caller" -> "farmer" (the agent is answering, not dubbing a call)
    "Address the farmer respectfully with gender-neutral 'આપ' forms; never infer the farmer's gender.",
    # ADAPTED: third-person instruction about Sarlaben -> second-person identity
    "You are Sarlaben: always use feminine self-reference in Gujarati (e.g. શકતી છું, કરૂં, આપી શકતી છું — never શકું, કરું, આવું).",
    # ADAPTED: "Keep the tone" -> "Keep your tone"
    "Keep your tone professional, cordial, and detached; do not become overly familiar or chatty.",
    # ADAPTED: "Do not translate English address markers ... into caller labels"
    "Do not address the farmer with labels like બહેન, ભાઈ, મેડમ, or સાહેબ, even if the question uses English address markers such as sister, brother, bhai, ben, madam, or sir. Use respectful gender-neutral 'આપ' wording instead.",
    # ADAPTED: "If the English source mentions 'sister'" -> "If the farmer addresses you as 'sister'"
    "If the farmer addresses you as 'sister' or બહેન, do not call the farmer બહેન. Omit the address marker or render it as a neutral reference to સરલાબેન only when necessary.",
    "Never use slang body terms like 'બૈડા/બૈડું/બરડા/બરડું'. Prefer 'પીઠ' for back/flank context and 'શરીર' for general body context.",
    "Prefer 'બાવલું' over 'પાહો' for udder context.",
    "Prefer 'ધાર' over 'ટીપાં' for milk streams.",
    "Use 'ગાભણ' for pregnant livestock context.",
    "Use 'ફેટ' for fat/milk-fat (not 'ચરબી').",
    "Use 'એસ.એન.એફ.' for SNF (not 'ઘન પદાર્થો').",
    "Use 'બેક્ટેરિયા' for bacteria (not 'જંતુઓ').",
    "Use 'ધણ' for herd (not 'ટોળું').",
    "Use one mastitis term consistently: 'આંચળનો સોજો'. Do not combine 'આઉ નો સોજો' and 'બાવલાનો સોજો'.",
    "NEVER use 'સ્તન' for animal udder/teat. Use 'આંચળ' for teat and 'બાવલું' or 'આઉ' for udder.",
    "Use 'બુલ' for bull (not 'બળદ' which means bullock/ox).",
    "For bloat (આફરો), use 'ફુલેલા' (distended/puffed) not 'સોજેલા' (swollen) when describing the flank.",
    "Avoid brackets, markdown, list scaffolding, and repeated parenthetical restatements.",
    "Use 'માખણ' for butter, 'મલાઈ' for cream, 'વલોણું/વલોણાથી' for churning, and 'ઘી બનાવવું' for making ghee.",
    "Use 'ચીરો' for incision/cut (not 'ચૂભો' which is not a real word).",
    "Use 'માનસિક આઘાત' for mental trauma/stress in animals (not 'તણાવ').",
    "Use 'ફીણ' for foam (not 'ફી').",
    "Use 'દવા' for medicine (Gujarati does not pluralise as 'દવાઓ').",
    "For feed meant for a pregnant animal, say 'ગાભણ પશુ માટેનું દાણ' or 'ગાભણ દાણ'. Never invent 'ગર્ભચારો' and never say 'ગર્ભ માટેનો ચારો'.",
    "Never use the phrase 'સામાન્ય જાળવણી ચારો'. Always use natural farmer wording such as 'રોજિંદો ઘાસચારો' or 'નિયમિત સૂકો અને લીલો ચારો'.",
    # ADAPTED: "if ASR/transcription suggests" -> "if the farmer's question suggests"
    "In dairy feed context, if the farmer's question suggests 'સમુદ્રી' but livestock feed is the likely meaning, prefer asking or keeping the term conservative over drifting into marine feed or seaweed advice.",
    "Use 'તેને' (not archaic 'તેણીને') for 'to her/it'.",
    "Use 'ભૌતિક' for physical (examination/condition), not 'શારીરિક'.",
    "Never use the hallucinated fodder word 'બરબા'. Use 'બરસીમ' (or 'રજકો' where contextually better).",
    "Never output placeholder quantities like '-', '--', or '–' for feed or dose lines. If exact values are missing, keep the wording non-numeric rather than inventing a quantity.",
    "'Amul AI', 'Amul A I', 'AMUL AI', 'AI helpline', 'amul helpline', 'AI helpline advisor', and 'AI-powered helpline' refer to the Amul Artificial Intelligence digital advisory helpline, not artificial insemination. Write them as 'અમૂલ એ.આઈ.' / 'એ.આઈ. હેલ્પલાઇન'; never as 'કૃત્રિમ બીજદાન' or other insemination wording in helpline or assistant identity context.",
    "When 'AI' appears in product or helpline naming (Amul AI, AI helpline, AI assistant, AI-powered helpline), treat it as Artificial Intelligence, not breeding artificial insemination, unless the sentence is clearly about beejdan, semen, technician booking, or insemination procedure.",
]
assert len(GU_AGENT_TERM_RULES) == 33, len(GU_AGENT_TERM_RULES)

RULES_HEADER = (
    "## GUJARATI TERMINOLOGY AND STYLE RULES (PARITY-INJECTED)\n"
    "When you write your Gujarati answer you MUST follow every rule below "
    "exactly. Use the preferred term and never the rejected one."
)


def rules_block() -> str:
    return RULES_HEADER + "\n- " + "\n- ".join(GU_AGENT_TERM_RULES)


# ---------------------------------------------------------------------------
# Deterministic post-map (83 forbidden pairs), same construction as
# _build_gu_policy_replacements() in app/services/translation.py.
# ---------------------------------------------------------------------------
def build_policy_replacements(policy_path):
    with open(policy_path, encoding="utf-8") as f:
        pol = json.load(f)
    forbidden = pol.get("forbidden", {}) if isinstance(pol, dict) else {}
    items = sorted(
        [(str(k).strip(), str(v).strip()) for k, v in forbidden.items()
         if str(k).strip() and str(v).strip()],
        key=lambda kv: len(kv[0]),
        reverse=True,
    )
    return [(re.escape(src), dst) for src, dst in items]


def apply_policy(text, reps):
    for pattern, dst in reps:
        text = re.sub(pattern, dst, text)
    return text


def _import_agent():
    from agents.agrinet import agrinet_agent
    from agents.deps import FarmerContext
    return agrinet_agent, FarmerContext


CAPTURED_PROMPT = {}


def _capture_system_prompt(result):
    """Pull EVERY SystemPromptPart out of the messages actually sent.

    pydantic-ai emits one SystemPromptPart per registered system_prompt
    function, so capturing only the first would hide the injected block.
    """
    try:
        parts = []
        for m in result.all_messages():
            for p in getattr(m, "parts", []) or []:
                kind = str(getattr(p, "part_kind", "") or type(p).__name__).lower()
                if "system" in kind:
                    parts.append(str(getattr(p, "content", "")))
        if parts:
            CAPTURED_PROMPT.setdefault("n_parts", len(parts))
            CAPTURED_PROMPT.setdefault("text", "\n\n".join(parts))
    except Exception:
        pass


async def run_one(agent, FarmerContext, row_num, qid, question, lang, settings,
                  arm, sem, reps, inject):
    async with sem:
        t0 = time.perf_counter()
        row = {
            "row_num": row_num, "qid": qid, "arm": arm,
            "guided": bool(settings), "question": question, "answer": "",
            "latency_s": 0.0, "error": "", "num_tool_calls": 0,
            "tool_names": "", "tool_log": "",
            "inject_rules": bool(inject), "apply_map": bool(reps),
            "answer_raw": "",
        }
        try:
            deps = FarmerContext(
                query=question, lang_code=lang, session_id=f"bench-{arm}-{row_num}",
            )
            kwargs = {"user_prompt": deps.get_user_message(), "deps": deps}
            if settings is not None:
                kwargs["model_settings"] = settings
            result = await agent.run(**kwargs)
            raw = (result.output if hasattr(result, "output") else str(result)) or ""
            row["answer_raw"] = raw
            row["answer"] = apply_policy(raw, reps) if reps else raw
            _capture_system_prompt(result)

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

    n_before = len(getattr(agent, "_system_prompt_functions", []) or [])
    if a.inject_rules:
        @agent.system_prompt(dynamic=True)
        def _parity_terminology_rules(ctx) -> str:  # noqa: ANN001
            return rules_block()
        n_after = len(getattr(agent, "_system_prompt_functions", []) or [])
        print(f"[inject] registered parity rules system_prompt "
              f"({n_before} -> {n_after} system_prompt funcs); "
              f"{len(GU_AGENT_TERM_RULES)} rules, {len(rules_block())} chars")

    reps = build_policy_replacements(a.policy) if a.apply_map else []
    if a.apply_map:
        print(f"[map] {len(reps)} forbidden pairs loaded from {a.policy} "
              f"(longest-key-first; longest key len={len(reps[0][0]) if reps else 0})")

    rows = load_questions(a.questions, a.limit)

    def _settings_for(lang):
        if not a.guided:
            return None
        return guided_settings(lang or a.lang, extended=not a.strict_142)

    n_con = sum(1 for _, _, _, lg in rows if _settings_for(lg) is not None)
    print(f"[info] {len(rows)} rows | arm={a.arm} | guided_flag={a.guided} "
          f"| rows_constrained={n_con} | strict142={a.strict_142} "
          f"| inject_rules={a.inject_rules} | apply_map={a.apply_map} "
          f"| concurrency={a.concurrency}")

    sem = asyncio.Semaphore(a.concurrency)
    tasks = [run_one(agent, FarmerContext, rn, qid, q, (lg or a.lang),
                     _settings_for(lg), a.arm, sem, reps, a.inject_rules)
             for rn, qid, q, lg in rows]

    out, done = [], 0
    for coro in asyncio.as_completed(tasks):
        out.append(await coro)
        done += 1
        if done % 25 == 0 or done == len(tasks):
            errs = sum(1 for x in out if x["error"])
            print(f"[progress] {done}/{len(tasks)} errors={errs}", flush=True)

    out.sort(key=lambda r: r["row_num"])

    sp = CAPTURED_PROMPT.get("text", "")
    if sp:
        present = RULES_HEADER.split("\n")[0] in sp
        print(f"[proof] system prompt sent to model: {len(sp)} chars in "
              f"{CAPTURED_PROMPT.get('n_parts')} SystemPromptPart(s) | "
              f"parity header present={present}")
        for probe in ("Use 'ધણ' for herd (not 'ટોળું').",
                      "NEVER use 'સ્તન' for animal udder/teat.",
                      "Use 'ફેટ' for fat/milk-fat (not 'ચરબી')."):
            print(f"[proof] fragment present={probe in sp!s:5} :: {probe}")
        if a.dry_run or a.show_prompt:
            print("[proof] ---- tail of system prompt actually sent ----")
            print(sp[-len(rules_block()) - 200:] if present else sp[-800:])
            print("[proof] ---- end ----")
    else:
        print("[proof] WARNING: could not capture a system prompt from messages")

    changed = sum(1 for r in out if r["answer"] != r["answer_raw"])
    print(f"[map] rows changed by deterministic map: {changed}/{len(out)}")

    if a.dry_run:
        print("[dry-run] not writing CSV")
        return 0

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)
    errs = sum(1 for x in out if x["error"])
    print(f"[done] {len(out)} rows -> {a.out} ({errs} errors)")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", required=True)
    p.add_argument("--out", default="bench_out.csv")
    p.add_argument("--arm", default="bare")
    p.add_argument("--lang", default="gu")
    p.add_argument("--guided", action="store_true")
    p.add_argument("--strict-142", action="store_true")
    p.add_argument("--inject-rules", action="store_true",
                   help="append the 33 GU_PREFERRED_TRANSLATION_RULES to the system prompt")
    p.add_argument("--apply-map", action="store_true",
                   help="apply the 83 forbidden pairs to the answer, longest-key-first")
    p.add_argument("--policy", default=str(DEFAULT_POLICY))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--dry-run", action="store_true",
                   help="run --limit rows, print the real system prompt, write nothing")
    p.add_argument("--show-prompt", action="store_true")
    a = p.parse_args()
    sys.exit(asyncio.run(main_async(a)))


if __name__ == "__main__":
    main()
