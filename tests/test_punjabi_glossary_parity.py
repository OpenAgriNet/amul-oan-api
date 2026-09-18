"""Punjabi glossary: loads, covers every master term, and feeds Punjabi (never
Gujarati) target terms into the translation mini glossary."""
import os
import re
import sys
import types
import importlib.util
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = Path(__file__).resolve().parents[1]

# Avoid importing agents/__init__.py and agents/tools/__init__.py during tests:
# load `agents.tools.terms` directly by file path and register minimal package stubs.
agents_pkg = types.ModuleType("agents")
agents_pkg.__path__ = [str(REPO_ROOT / "agents")]
tools_pkg = types.ModuleType("agents.tools")
tools_pkg.__path__ = [str(REPO_ROOT / "agents" / "tools")]
sys.modules.setdefault("agents", agents_pkg)
sys.modules.setdefault("agents.tools", tools_pkg)

terms_spec = importlib.util.spec_from_file_location(
    "agents.tools.terms",
    REPO_ROOT / "agents" / "tools" / "terms.py",
)
assert terms_spec and terms_spec.loader, "Failed to load agents.tools.terms spec"
terms_module = importlib.util.module_from_spec(terms_spec)
sys.modules["agents.tools.terms"] = terms_module
terms_spec.loader.exec_module(terms_module)

from agents.tools.terms import PA_TERM_PAIRS, TERM_PAIRS, get_mini_glossary_for_text

# Gurmukhi sits directly below the Gujarati block; keep the two ranges apart.
_GURMUKHI_SCRIPT = re.compile(r"[਀-੿]")
_GUJARATI_SCRIPT = re.compile(r"[઀-૿]")
_ANY_LETTER = re.compile(r"[^\W\d_]")


def _first_punjabi_pair():
    for pair in PA_TERM_PAIRS:
        if pair.en.strip() and pair.pa.strip():
            return pair
    raise AssertionError("No Punjabi glossary pair found")


def test_punjabi_glossary_runtime_loads_terms():
    assert len(PA_TERM_PAIRS) > 0, "Punjabi glossary should load from glossary_terms_punjabi.json"


def test_punjabi_glossary_covers_every_master_term():
    master = {pair.en.strip().lower() for pair in TERM_PAIRS}
    punjabi = {pair.en.strip().lower() for pair in PA_TERM_PAIRS}
    assert sorted(master - punjabi) == []


def test_punjabi_glossary_values_are_gurmukhi_script():
    # Digits/symbol-only values are fine; any value with letters must be Gurmukhi.
    wrong_script = [
        pair.en for pair in PA_TERM_PAIRS
        if _ANY_LETTER.search(pair.pa) and not _GURMUKHI_SCRIPT.search(pair.pa)
    ]
    assert wrong_script == []


def test_punjabi_glossary_never_carries_gujarati_script():
    # The Gurmukhi and Gujarati Unicode blocks are adjacent; a mis-set range
    # would silently let Gujarati through.
    leaked = [pair.en for pair in PA_TERM_PAIRS if _GUJARATI_SCRIPT.search(pair.pa)]
    assert leaked == []


def test_punjabi_mini_glossary_uses_punjabi_target_terms():
    pair = _first_punjabi_pair()
    text = f"Please explain {pair.en} for my dairy animal."
    mini = get_mini_glossary_for_text(text=text, target_lang="pa", threshold=0.9, max_terms=10)
    assert mini, "Expected Punjabi mini glossary entries"
    assert f"{pair.en} -> {pair.pa}" in mini


def test_punjabi_mini_glossary_never_injects_gujarati_terms():
    # Regression: unknown targets normalize to "gu", which would inject Gujarati
    # terms as mandatory rules into a Punjabi translation prompt.
    text = "My cow has mastitis and low milk fat. How much mineral mixture should I give?"
    mini = get_mini_glossary_for_text(text=text, target_lang="punjabi", threshold=0.9, max_terms=40)
    assert mini, "Expected Punjabi mini glossary entries"
    assert not _GUJARATI_SCRIPT.search(mini)
    assert _GURMUKHI_SCRIPT.search(mini)
