#!/usr/bin/env python3
"""Cross-repo parity guard for the unified LLM pipeline core (``app/llm_core``).

WHAT THIS GUARDS
================
The chat repo (``amul-oan-api``) and the voice repo (``voice-oan-api``) each carry
their own copy of the unified LLM pipeline core under ``app/llm_core/`` (plus the
shared fallback walker at ``app/services/fallback.py``). The north-star is a single
merged repo; until then the two copies are kept in lockstep BY DISCIPLINE ALONE —
there is no build-time link, so an edit landed in one repo and forgotten in the
other drifts silently and is only discovered when the two behave differently in
prod. This script turns that discipline into a CI check.

It splits the pipeline files into two sets:

  * MUST-MATCH — infrastructure that has NO legitimate reason to differ between the
    repos. These are byte-compared; ANY difference fails CI (exit 1) and prints a
    unified diff. Keeping them identical is the whole point of the guard.

  * TOLERATED — files that legitimately differ because chat and voice genuinely
    have different pipeline shapes (e.g. chat has a ``suggestions`` step; voice has
    a ``non_meaningful`` step and RAW_OPENAI moderation). For these we still want
    to catch *unexpected* drift, so each file declares a WHITELIST of the regions
    that are ALLOWED to differ:
      - "region-whitelisted" files (``config_model.py``, ``resolver.py``): only the
        named regions (the ``Step`` enum / ``StepClientKind`` block; the
        ``STEP_CLIENT_KIND`` map) may differ. Each whitelisted region is collapsed
        to a single sentinel line in BOTH files and the REMAINDER is compared; a
        difference outside every whitelisted region is un-whitelisted drift and
        fails CI. If a whitelist anchor no longer matches (file restructured) the
        script says so loudly so the whitelist gets updated.
      - "whole-file" files (``legacy_shim.py``, ``split.py``, ``runtime.py``): these
        are pervasively repo-specific (per-repo env wiring, docstrings, boot glue).
        The whole file is whitelisted-as-divergent: the script prints an
        informational diff summary for human review but never fails on them.

WHY THE SPLIT IS SAFE
=====================
The MUST-MATCH set is chosen so that a divergence there is always a bug: the
circuit breaker (``health.py``), the concurrency gauge (``concurrency.py``), the
boot-config tracer (``trace.py``) and the fallback walker (``fallback.py``) encode
shared cross-cutting behavior. The TOLERATED set is exactly the files whose
per-repo divergence is a deliberate, documented product difference — and even those
are pinned down to the specific regions allowed to move.

USAGE
=====
    python scripts/check_pipeline_parity.py <chat_root> <voice_root>

Roots may also come from ``PARITY_CHAT_ROOT`` / ``PARITY_VOICE_ROOT``; if neither
args nor env are given, the two repos are assumed to sit side by side as
``../amul-oan-api`` and ``../voice-oan-api`` relative to this repo.

Exit code: 0 = in parity (only whitelisted regions differ); 1 = un-whitelisted
divergence in a MUST-MATCH or region-whitelisted file (CI failure); 2 = usage /
missing-file error. This file is intended to be IDENTICAL in both repos.
"""

from __future__ import annotations

import difflib
import os
import sys
from dataclasses import dataclass

# ── files that MUST stay byte-identical across the two repos ──────────────────
MUST_MATCH: list[str] = [
    "app/llm_core/health.py",       # P2 passive circuit breaker
    "app/llm_core/concurrency.py",  # P3 concurrency gauge / reorder
    "app/llm_core/trace.py",        # boot full-config tracer
    "app/services/fallback.py",     # OSS->managed fallback walker
]


@dataclass(frozen=True)
class Region:
    """A contiguous span that is ALLOWED to differ between the repos.

    Bounded by substring anchors: the region starts at the first line containing
    ``start`` and ends at the next line containing ``end``. ``include_end`` decides
    whether that end-anchor line is part of the region (True) or is shared code that
    must still be compared (False). The whole span is collapsed to one sentinel line
    so a region of unequal length on each side still compares equal.
    """

    name: str
    start: str
    end: str
    include_end: bool = False


# ── files that legitimately differ, pinned to whitelisted regions ─────────────
# Only these named regions may differ; the rest of the file must still match.
REGION_TOLERATED: dict[str, list[Region]] = {
    "app/llm_core/config_model.py": [
        # Per-repo prose only.
        Region("module docstring", '"""Config data model', "from __future__ import"),
        # THE deliberate difference: chat has SUGGESTIONS, voice has NON_MEANINGFUL.
        Region("Step enum", "class Step(str, Enum)", "class StepClientKind"),
        # Same members, per-repo inline comments (which steps map to which kind).
        Region("StepClientKind enum", "class StepClientKind", "class Tier"),
    ],
    "app/llm_core/resolver.py": [
        Region("module docstring", '"""Resolver', "from __future__ import"),
        # chat: MODERATION/SUGGESTIONS are AGENT-kind; voice: MODERATION/
        # NON_MEANINGFUL are RAW_OPENAI-kind. This map is the single seam that
        # encodes that product difference.
        Region("STEP_CLIENT_KIND map", "STEP_CLIENT_KIND: dict", "}", include_end=True),
    ],
}

# ── files that are pervasively repo-specific: informational diff only ─────────
WHOLE_FILE_TOLERATED: dict[str, str] = {
    "app/llm_core/legacy_shim.py": (
        "per-repo env->config synthesis (chat: suggestions tiers; voice: shared "
        "RAW_OPENAI moderation/non_meaningful clients) — pervasively different"
    ),
    "app/llm_core/split.py": (
        "per-repo variant<->profile helpers and router-seam wiring — pervasively "
        "different"
    ),
    "app/llm_core/runtime.py": (
        "per-repo boot glue: docstrings, validate_config shape, Provider import — "
        "the shared boot-posture assertion is kept in sync by hand"
    ),
}


def _read(path: str) -> list[str] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.readlines()
    except FileNotFoundError:
        return None


def _mask(lines: list[str], regions: list[Region]) -> tuple[list[str], list[str]]:
    """Collapse each whitelisted region to a single sentinel line.

    Returns ``(masked_lines, stale_region_names)`` — a region whose anchors don't
    match (start present but end never found) is left unmasked and reported stale.
    """
    out: list[str] = []
    stale: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        hit: Region | None = None
        for reg in regions:
            if reg.start in lines[i]:
                hit = reg
                break
        if hit is None:
            out.append(lines[i])
            i += 1
            continue
        end_idx: int | None = None
        for k in range(i + 1, n):
            if hit.end in lines[k]:
                end_idx = k
                break
        if end_idx is None:
            stale.append(hit.name)
            out.append(lines[i])
            i += 1
            continue
        last = end_idx if hit.include_end else end_idx - 1
        out.append(f"<<<WHITELISTED REGION: {hit.name}>>>\n")
        i = last + 1
    return out, stale


def _diff(a: list[str], b: list[str], rel: str) -> str:
    return "".join(
        difflib.unified_diff(a, b, fromfile=f"chat/{rel}", tofile=f"voice/{rel}")
    )


def _resolve_roots(argv: list[str]) -> tuple[str, str]:
    if len(argv) >= 2:
        return os.path.abspath(argv[0]), os.path.abspath(argv[1])
    env_chat = os.getenv("PARITY_CHAT_ROOT")
    env_voice = os.getenv("PARITY_VOICE_ROOT")
    if env_chat and env_voice:
        return os.path.abspath(env_chat), os.path.abspath(env_voice)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parent = os.path.dirname(repo)
    return os.path.join(parent, "amul-oan-api"), os.path.join(parent, "voice-oan-api")


def main(argv: list[str]) -> int:
    chat_root, voice_root = _resolve_roots(argv)
    print(f"chat_root  = {chat_root}")
    print(f"voice_root = {voice_root}")
    for root in (chat_root, voice_root):
        if not os.path.isdir(root):
            print(f"ERROR: not a directory: {root}", file=sys.stderr)
            return 2

    failures = 0

    print("\n== MUST-MATCH (byte-identical required) ==")
    for rel in MUST_MATCH:
        a = _read(os.path.join(chat_root, rel))
        b = _read(os.path.join(voice_root, rel))
        if a is None or b is None:
            print(f"  FAIL  {rel}: missing (chat={a is not None}, voice={b is not None})")
            failures += 1
            continue
        if a == b:
            print(f"  OK    {rel}")
        else:
            print(f"  FAIL  {rel}: DIVERGED")
            print(_diff(a, b, rel))
            failures += 1

    print("\n== TOLERATED: region-whitelisted (only named regions may differ) ==")
    for rel, regions in REGION_TOLERATED.items():
        a = _read(os.path.join(chat_root, rel))
        b = _read(os.path.join(voice_root, rel))
        if a is None or b is None:
            print(f"  FAIL  {rel}: missing (chat={a is not None}, voice={b is not None})")
            failures += 1
            continue
        ma, stale_a = _mask(a, regions)
        mb, stale_b = _mask(b, regions)
        stale = sorted(set(stale_a) | set(stale_b))
        if stale:
            print(f"  WARN  {rel}: whitelist anchors stale (region(s) not located): {stale}")
        if ma == mb:
            print(f"  OK    {rel}: differs ONLY inside whitelisted regions "
                  f"({', '.join(r.name for r in regions)})")
        else:
            print(f"  FAIL  {rel}: UN-WHITELISTED divergence (outside {', '.join(r.name for r in regions)})")
            print(_diff(ma, mb, rel + " [whitelisted regions masked]"))
            failures += 1

    print("\n== TOLERATED: whole-file divergence expected (informational only) ==")
    for rel, reason in WHOLE_FILE_TOLERATED.items():
        a = _read(os.path.join(chat_root, rel))
        b = _read(os.path.join(voice_root, rel))
        if a is None or b is None:
            print(f"  WARN  {rel}: missing (chat={a is not None}, voice={b is not None})")
            continue
        if a == b:
            print(f"  SAME  {rel}: identical (reason for tolerance: {reason})")
        else:
            n = len(list(difflib.unified_diff(a, b)))
            print(f"  DIFF  {rel}: ~{n} diff lines — expected ({reason})")

    print()
    if failures:
        print(f"PARITY CHECK FAILED: {failures} un-whitelisted divergence(s).")
        return 1
    print("PARITY CHECK PASSED: MUST-MATCH identical; tolerated files differ only where whitelisted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
