import os
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

from agents.tools.terms import HI_TERM_PAIRS, TERM_PAIRS, get_mini_glossary_for_text


def _pick_pair_with_target(target: str):
    if target == "hi":
        for pair in HI_TERM_PAIRS:
            if pair.en.strip() and pair.hi.strip():
                return pair
    for pair in TERM_PAIRS:
        if pair.en.strip() and pair.gu.strip():
            return pair
    raise AssertionError("No glossary pair found for target")


def test_hindi_glossary_runtime_loads_openrouter_terms():
    assert len(HI_TERM_PAIRS) > 0, "Hindi glossary should load from glossary_terms_hindi_openrouter.json"


def test_hindi_mini_glossary_uses_hindi_target_terms():
    pair = _pick_pair_with_target("hi")
    text = f"Please explain {pair.en} for my dairy animal."
    mini = get_mini_glossary_for_text(text=text, target_lang="hi", threshold=0.9, max_terms=10)
    assert mini, "Expected Hindi mini glossary entries"
    assert f"{pair.en} -> {pair.hi}" in mini


def test_gujarati_mini_glossary_remains_unchanged():
    pair = _pick_pair_with_target("gu")
    text = f"Please explain {pair.en} for my dairy animal."
    mini = get_mini_glossary_for_text(text=text, target_lang="gu", threshold=0.9, max_terms=10)
    assert mini, "Expected Gujarati mini glossary entries"
    assert f"{pair.en} -> {pair.gu}" in mini


