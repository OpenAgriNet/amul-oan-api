import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("marqo")

from pydantic_ai import ModelRetry
from agents.tools.search import _validate_search_query


def test_validate_search_query_rejects_empty() -> None:
    with pytest.raises(ModelRetry) as exc:
        _validate_search_query("   ")
    assert "EMPTY_QUERY" in str(exc.value)


def test_validate_search_query_accepts_focused_agri_query() -> None:
    assert _validate_search_query("buffalo mastitis treatment") == "buffalo mastitis treatment"


def test_validate_search_query_normalizes_whitespace() -> None:
    assert _validate_search_query("  buffalo   mastitis \n treatment  ") == "buffalo mastitis treatment"
