#!/usr/bin/env python3
"""Side-by-side rubric report for a control/guided arm pair."""
import json
import subprocess
import sys

KEYS = [
    ("rows", "rows"), ("errors", "errors"), ("empty", "empty answers"),
    ("drift_pct", "R-drift  cross-script %"),
    ("drift_rows", "R-drift  rows"),
    ("drift_scored_rows", "R-drift  rows scored (gu+hi)"),
    ("forbidden_rows", "R1 forbidden-term rows"),
    ("forbidden_hits_total", "R1 forbidden hits"),
    ("masc_self_ref_rows", "R2 masc self-ref rows"),
    ("fem_self_ref_rows", "R2 fem self-ref rows"),
    ("insemination_wording_rows", "R3 insemination wording"),
    ("ai_helpline_wording_rows", "R3 AI-helpline wording"),
    ("placeholder_rows", "R7 placeholder qty rows"),
    ("markdown_rows", "R7 markdown rows"),
    ("degree_rows", "gap: rows using °"),
    ("gap_char_rows", "gap: rows using °…×≥→±"),
    ("latency_mean", "latency mean s"),
    ("latency_p90", "latency p90 s"),
    ("tools_per_turn", "tools/turn"),
    ("zero_tool_rows", "zero-tool rows"),
    ("answer_chars_mean", "answer chars mean"),
]


def main():
    policy, langmap, files = sys.argv[1], sys.argv[2], sys.argv[3:]
    raw = subprocess.run(
        [sys.executable, "rubric_score.py", *files, "--policy", policy,
         "--langmap", langmap],
        capture_output=True, text=True, cwd=".")
    if raw.returncode != 0:
        print(raw.stderr[-2000:]); sys.exit(1)
    d = json.loads(raw.stdout[raw.stdout.find("["):])
    labels = [f.split("/")[-1].replace(".csv", "") for f in files]

    w = max(len(x) for x in labels) + 2
    print(f"{'metric':30}" + "".join(f"{x:>{w}}" for x in labels))
    print("-" * (30 + w * len(labels)))
    for k, label in KEYS:
        print(f"{label:30}" + "".join(f"{str(x.get(k, '')):>{w}}" for x in d))
    print()
    for lab, x in zip(labels, d):
        print(f"drift scripts [{lab}]: {x.get('drift_scripts')}")
    print()
    for lab, x in zip(labels, d):
        print(f"tool mix [{lab}]: {x.get('tool_mix')}")
    print()
    for lab, x in zip(labels, d):
        top = x.get("forbidden_top") or {}
        if top:
            print(f"top forbidden [{lab}]: {top}")


if __name__ == "__main__":
    main()
