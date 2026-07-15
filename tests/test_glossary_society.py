"""Regression: standalone Society glossary entry for compact farmer profile lines."""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from agents.tools.terms import get_mini_glossary_for_text
from app.services.translation import _format_translation_prompt

_FARMER_PROFILE_SOCIETY_LINE = (
    "- Society: Suchit Bajrang D.U.S.M.Ltd (Code: 2239)"
)


def test_standalone_society_in_mini_glossary_for_compact_profile_line():
    """Agent often emits 'Society:' (not 'Society name') on farmer profile lines."""
    glossary = get_mini_glossary_for_text(
        _FARMER_PROFILE_SOCIETY_LINE, threshold=0.90, max_terms=40
    )
    assert "Society -> સોસાયટી" in glossary


def test_standalone_society_injected_into_translation_prompt():
    glossary = get_mini_glossary_for_text(
        _FARMER_PROFILE_SOCIETY_LINE, threshold=0.90, max_terms=40
    )
    prompt = _format_translation_prompt(
        _FARMER_PROFILE_SOCIETY_LINE, "english", "gujarati", mini_glossary=glossary
    )
    assert "'Society' must be translated as 'સોસાયટી'" in prompt
