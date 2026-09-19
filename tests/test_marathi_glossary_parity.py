"""Marathi glossary: loads, covers every master term, and feeds Marathi (never
Gujarati or Hindi) target terms into the translation mini glossary."""
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

from agents.tools.terms import MR_TERM_PAIRS, TERM_PAIRS, get_mini_glossary_for_text

_GUJARATI_SCRIPT = re.compile(r"[઀-૿]")
_DEVANAGARI_SCRIPT = re.compile(r"[ऀ-ॿ]")
_ANY_LETTER = re.compile(r"[^\W\d_]")


def _first_marathi_pair():
    for pair in MR_TERM_PAIRS:
        if pair.en.strip() and pair.mr.strip():
            return pair
    raise AssertionError("No Marathi glossary pair found")


def test_marathi_glossary_runtime_loads_terms():
    assert len(MR_TERM_PAIRS) > 0, "Marathi glossary should load from glossary_terms_marathi.json"


def test_marathi_glossary_covers_every_master_term():
    master = {pair.en.strip().lower() for pair in TERM_PAIRS}
    marathi = {pair.en.strip().lower() for pair in MR_TERM_PAIRS}
    assert sorted(master - marathi) == []


def test_marathi_glossary_values_are_devanagari_script():
    # Digits/symbol-only values are fine; any value with letters must be Devanagari.
    # Latin-script drug names (e.g. Buparvaquone) are allowed only alongside Devanagari.
    wrong_script = [
        pair.en for pair in MR_TERM_PAIRS
        if _ANY_LETTER.search(pair.mr) and not _DEVANAGARI_SCRIPT.search(pair.mr)
    ]
    assert wrong_script == []


def test_marathi_glossary_never_carries_gujarati_script():
    leaked = [pair.en for pair in MR_TERM_PAIRS if _GUJARATI_SCRIPT.search(pair.mr)]
    assert leaked == []


def test_marathi_mini_glossary_uses_marathi_target_terms():
    pair = _first_marathi_pair()
    text = f"Please explain {pair.en} for my dairy animal."
    mini = get_mini_glossary_for_text(text=text, target_lang="mr", threshold=0.9, max_terms=10)
    assert mini, "Expected Marathi mini glossary entries"
    assert f"{pair.en} -> {pair.mr}" in mini


def test_marathi_mini_glossary_never_injects_gujarati_terms():
    # Regression: unknown targets normalize to "gu", which would inject Gujarati
    # terms as mandatory rules into a Marathi translation prompt.
    text = "My cow has mastitis and low milk fat. How much mineral mixture should I give?"
    mini = get_mini_glossary_for_text(text=text, target_lang="marathi", threshold=0.9, max_terms=40)
    assert mini, "Expected Marathi mini glossary entries"
    assert not _GUJARATI_SCRIPT.search(mini)


def test_marathi_mini_glossary_is_not_the_hindi_glossary():
    # Marathi and Hindi share Devanagari, so script alone cannot prove the right
    # index was used. These terms differ between the two glossaries.
    text = "The cow has mastitis after calving."
    mini_mr = get_mini_glossary_for_text(text=text, target_lang="mr", threshold=0.9, max_terms=40)
    mini_hi = get_mini_glossary_for_text(text=text, target_lang="hi", threshold=0.9, max_terms=40)
    assert mini_mr and mini_hi
    assert mini_mr != mini_hi
    assert "कासदाह" in mini_mr
    assert "थनैला" in mini_hi
