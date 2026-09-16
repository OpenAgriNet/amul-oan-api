"""Bengali glossary: loads, covers every master term, and feeds Bengali (never
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

from agents.tools.terms import BN_TERM_PAIRS, TERM_PAIRS, get_mini_glossary_for_text

_GUJARATI_SCRIPT = re.compile(r"[\u0A80-\u0AFF]")
_BENGALI_SCRIPT = re.compile(r"[\u0980-\u09FF]")
_ANY_LETTER = re.compile(r"[^\W\d_]")


def _first_bengali_pair():
    for pair in BN_TERM_PAIRS:
        if pair.en.strip() and pair.bn.strip():
            return pair
    raise AssertionError("No Bengali glossary pair found")


def test_bengali_glossary_runtime_loads_terms():
    assert len(BN_TERM_PAIRS) > 0, "Bengali glossary should load from glossary_terms_bengali.json"


def test_bengali_glossary_covers_every_master_term():
    master = {pair.en.strip().lower() for pair in TERM_PAIRS}
    bengali = {pair.en.strip().lower() for pair in BN_TERM_PAIRS}
    assert sorted(master - bengali) == []


def test_bengali_glossary_values_are_bengali_script():
    # Digits/symbol-only values are fine; any value with letters must be Bengali.
    wrong_script = [
        pair.en for pair in BN_TERM_PAIRS
        if _ANY_LETTER.search(pair.bn) and not _BENGALI_SCRIPT.search(pair.bn)
    ]
    assert wrong_script == []


def test_bengali_mini_glossary_uses_bengali_target_terms():
    pair = _first_bengali_pair()
    text = f"Please explain {pair.en} for my dairy animal."
    mini = get_mini_glossary_for_text(text=text, target_lang="bn", threshold=0.9, max_terms=10)
    assert mini, "Expected Bengali mini glossary entries"
    assert f"{pair.en} -> {pair.bn}" in mini


def test_bengali_mini_glossary_never_injects_gujarati_terms():
    # Regression: unknown targets used to normalize to "gu", which would inject
    # Gujarati terms as mandatory rules into a Bengali translation prompt.
    text = "My cow has mastitis and low milk fat. How much mineral mixture should I give?"
    mini = get_mini_glossary_for_text(text=text, target_lang="bengali", threshold=0.9, max_terms=40)
    assert mini, "Expected Bengali mini glossary entries"
    assert not _GUJARATI_SCRIPT.search(mini)
