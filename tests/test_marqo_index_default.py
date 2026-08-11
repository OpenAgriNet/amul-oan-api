"""Search must retain the canonical Marqo index fallback.

When MARQO_INDEX_NAME is unset, search_documents should still fall back to
`amul-veterinary-index` so all retrieval paths stay aligned.
"""
from app.config import settings


def test_marqo_index_default_is_canonical():
    resolved = settings.marqo_index_name or "amul-veterinary-index"
    assert resolved == "amul-veterinary-index"
