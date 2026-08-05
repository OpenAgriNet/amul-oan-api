#!/usr/bin/env python3
"""Mechanical scoring for the bare-gemma4 bench.

Covers the rubrics that are checkable without a judge. Rubrics needing human or
LLM judgement (persona gender nuance, proper-noun pinning, glossary fidelity)
are scored separately.

Usage: python3 rubric_score.py <results.csv> [--policy path/to/gu_term_policy.json]
"""
import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter

csv.field_size_limit(sys.maxsize)

GU_LO, GU_HI = "઀", "૿"


def script_of(ch):
    if GU_LO <= ch <= GU_HI:
        return "gujarati"
    if "a" <= ch.lower() <= "z":
        return "latin"
    if "ऀ" <= ch <= "ॿ":
        return "devanagari"
    try:
        return unicodedata.name(ch).split()[0].lower()
    except ValueError:
        return "unknown"


def skip(ch):
    return (ch.isspace() or ch.isdigit()
            or unicodedata.category(ch)[0] in ("P", "S"))


# R1: terminology. Loaded from gu_term_policy.json forbidden map.
def load_forbidden(path):
    try:
        pol = json.load(open(path, encoding="utf-8"))
    except Exception:
        return {}
    f = pol.get("forbidden", {})
    return f if isinstance(f, dict) else {}


# R2: persona gender — masculine self-reference forms Sarlaben must never use.
# Scoped to first-person clauses; a bare token match would fire on quoted farmer
# speech, so require the Gujarati first-person pronoun nearby.
MASC_SELF = [r"શકું", r"કરું", r"આવું", r"શકતો", r"કરતો", r"જાણતો"]
FEM_SELF = [r"શકતી", r"કરૂં", r"કરતી", r"જાણતી"]

# R3: the AI-sense rule — helpline "AI" must never render as insemination.
INSEMINATION = [r"કૃત્રિમ\s*બીજદાન", r"બીજદાન"]
AI_HELPLINE = [r"અમૂલ\s*એ\.?આઈ", r"એ\.?આઈ\.?\s*હેલ્પલાઇન"]

# R7: placeholder quantities and markdown scaffolding.
PLACEHOLDER = re.compile(r"(?<![\d\w])[-–—]{1,3}(?![\d\w])")
MARKDOWN = re.compile(r"(\*\*|^\s*[\*\-•]\s|^#{1,6}\s|\[.+?\]\(.+?\))", re.M)

# Char-set gap probe: characters #142 omits that our answers legitimately need.
GAP_CHARS = "°…×≥≤→←±½¼¾"


NATIVE = {"gu": {"gujarati"}, "hi": {"devanagari"}, "en": set()}


def load_langmap(path):
    if not path:
        return {}
    m = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r.get("id"):
            m[r["id"]] = (r.get("lang") or "gu").strip().lower()
    return m


def score(path, forbidden, langmap=None):
    langmap = langmap or {}
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    n = len(rows)
    ans = [(r, (r.get("answer") or r.get("answer_gu") or "")) for r in rows]
    nonempty = [(r, a) for r, a in ans if a.strip()]

    out = {"file": path, "rows": n, "nonempty": len(nonempty),
           "empty": n - len(nonempty),
           "errors": sum(1 for r, _ in ans if (r.get("error") or "").strip())}

    drift_rows, drift_detail, drift_scored = 0, Counter(), 0
    latin_only = 0
    for r, a in nonempty:
        # Score against the row's OWN expected script. A Hindi row is supposed to
        # be Devanagari; counting that as drift would be a scorer artifact.
        lang = langmap.get(r.get("qid", ""), "gu")
        # Only gu/hi rows can test script fidelity. BARE_GU_NATIVE forces the
        # Gujarati prompt for EVERY row, so an English-input row is answered in
        # Gujarati by design -- scoring it against Latin would be meaningless.
        if lang not in ("gu", "hi"):
            continue
        drift_scored += 1
        native = NATIVE[lang] | {"latin"}
        sc = Counter(script_of(c) for c in a if not skip(c))
        foreign = {k: v for k, v in sc.items() if k not in native}
        if foreign:
            drift_rows += 1
            drift_detail.update(foreign.keys())
        if sc.get("latin", 0) > 0 and sc.get("gujarati", 0) > 0:
            latin_only += 1
    out["drift_rows"] = drift_rows
    out["drift_scored_rows"] = drift_scored
    out["drift_pct"] = round(100 * drift_rows / max(drift_scored, 1), 2)
    out["drift_scripts"] = dict(drift_detail.most_common())
    out["rows_with_latin"] = latin_only

    viol = Counter()
    for r, a in nonempty:
        for bad in forbidden:
            if bad and bad in a:
                viol[bad] += 1
    out["forbidden_hits_total"] = sum(viol.values())
    out["forbidden_rows"] = sum(1 for r, a in nonempty
                                if any(b in a for b in forbidden if b))
    out["forbidden_top"] = dict(viol.most_common(10))

    masc = sum(1 for _, a in nonempty
               if re.search(r"હું[^.।]{0,40}(" + "|".join(MASC_SELF) + ")", a))
    fem = sum(1 for _, a in nonempty
              if re.search(r"હું[^.।]{0,40}(" + "|".join(FEM_SELF) + ")", a))
    out["masc_self_ref_rows"] = masc
    out["fem_self_ref_rows"] = fem

    ins = sum(1 for _, a in nonempty if any(re.search(p, a) for p in INSEMINATION))
    hl = sum(1 for _, a in nonempty if any(re.search(p, a) for p in AI_HELPLINE))
    out["insemination_wording_rows"] = ins
    out["ai_helpline_wording_rows"] = hl

    out["placeholder_rows"] = sum(1 for _, a in nonempty if PLACEHOLDER.search(a))
    out["markdown_rows"] = sum(1 for _, a in nonempty if MARKDOWN.search(a))

    out["gap_char_rows"] = sum(1 for _, a in nonempty
                               if any(c in a for c in GAP_CHARS))
    out["degree_rows"] = sum(1 for _, a in nonempty if "°" in a)

    lat = [float(r["latency_s"]) for r, _ in ans if r.get("latency_s")]
    tc = [int(r["num_tool_calls"] or 0) for r, _ in ans]
    out["latency_mean"] = round(sum(lat) / max(len(lat), 1), 2)
    out["latency_p90"] = round(sorted(lat)[int(0.9 * len(lat))], 2) if lat else 0
    out["tools_per_turn"] = round(sum(tc) / max(len(tc), 1), 3)
    out["zero_tool_rows"] = sum(1 for x in tc if x == 0)
    out["answer_chars_mean"] = round(
        sum(len(a) for _, a in ans) / max(n, 1))

    names = Counter()
    for r, _ in ans:
        for t in (r.get("tool_names") or "").split("|"):
            if t:
                names[t] += 1
    out["tool_mix"] = dict(names.most_common())
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--policy", default="")
    p.add_argument("--langmap", default="", help="question-set CSV with id,lang")
    a = p.parse_args()
    forb = load_forbidden(a.policy) if a.policy else {}
    lm = load_langmap(a.langmap)
    print(f"[policy] {len(forb)} forbidden terms | [langmap] {len(lm)} rows",
          file=sys.stderr)
    print(json.dumps([score(f, forb, lm) for f in a.files],
                     ensure_ascii=False, indent=2))
