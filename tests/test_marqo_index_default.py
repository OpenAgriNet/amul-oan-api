"""Verify search_documents uses the centralized settings config path.

This is intentionally source-level to avoid importing heavy runtime deps while
still asserting the behaviorally important wiring in the actual tool function.
"""
from pathlib import Path


def test_search_documents_uses_settings_index_with_canonical_fallback():
    src = Path("agents/tools/search.py").read_text(encoding="utf-8")
    assert 'index_name = settings.marqo_index_name or "amul-veterinary-index"' in src


def test_search_documents_uses_settings_endpoint_path():
    src = Path("agents/tools/search.py").read_text(encoding="utf-8")
    assert "endpoint_url = settings.marqo_endpoint_url" in src
