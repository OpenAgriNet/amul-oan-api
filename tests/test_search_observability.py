from unittest.mock import MagicMock
import inspect

from agents.tools.search import (
    SearchHit,
    _build_hit_provenance,
    _build_search_observability_output,
    _human_display_title,
    _infer_section_from_text,
    _is_opaque_doc_label,
    _safe_update_observation,
    search_documents,
)


def test_infer_section_from_markdown_heading():
    text = "Intro line\n\n## Heat Detection\n\nBody text"
    assert _infer_section_from_text(text) == "Heat Detection"


def test_build_hit_provenance_uses_explicit_fields():
    hit = {
        "doc_id": "289d641c2ed1a8c88d3475cfad59ea04",
        "filename": "doc-ef226cde5062",
        "_id": "marqo-1",
        "chunk_num": 78,
        "page_start": 63,
        "page_end": 64,
        "_score": 0.91,
        "text": "## Deworming Schedule\n\nBody",
    }
    prov = _build_hit_provenance(hit, rank=1)
    assert prov["doc_id"] == "doc-ef226cde5062"
    assert prov["internal_doc_id"] == "289d641c2ed1a8c88d3475cfad59ea04"
    assert prov["chunk_index"] == 78
    assert prov["page_range"] == "63-64"
    assert prov["section"] == "Deworming Schedule"
    # IDs stay in Langfuse payload; display title is humanized.
    assert "doc-ef226cde5062" not in prov["display_title"]
    assert "§" not in prov["display_title"]
    assert prov["display_title"] == "Deworming Schedule"


def test_build_hit_provenance_infers_section_when_missing():
    hit = {
        "name": "doc-xyz",
        "_id": "marqo-2",
        "text": "# Focus on Buffalo\n\nParagraph",
        "_score": 0.5,
    }
    prov = _build_hit_provenance(hit, rank=2)
    assert prov["section"] == "Focus on Buffalo"
    assert prov["rank"] == 2


def test_build_search_observability_output_deduplicates_doc_ids():
    hits = [
        {"doc_id": "doc-a", "_id": "1", "text": "# One", "_score": 0.9},
        {"doc_id": "doc-a", "_id": "2", "text": "# Two", "_score": 0.8},
        {"doc_id": "doc-b", "_id": "3", "text": "# Three", "_score": 0.7},
    ]
    payload = _build_search_observability_output(
        query="heat timing",
        index_name="test-index",
        search_mode="hybrid",
        final_top_k=3,
        hits=hits,
    )
    assert payload["hit_count"] == 3
    assert payload["unique_doc_count"] == 2
    assert payload["unique_doc_ids"] == ["doc-a", "doc-b"]
    assert len(payload["documents"]) == 3


def test_is_opaque_doc_label():
    assert _is_opaque_doc_label("doc-ef226cde5062")
    assert _is_opaque_doc_label("289d641c2ed1a8c88d3475cfad59ea04")
    assert _is_opaque_doc_label("")
    assert not _is_opaque_doc_label("Dairy Handbook")


def test_human_display_title_strips_ids_and_section_marker():
    assert _human_display_title(name="doc-ef226cde5062", section="Deworming Schedule") == (
        "Deworming Schedule"
    )
    assert _human_display_title(name="Dairy Handbook", section="Heat Timing") == (
        "Dairy Handbook — Heat Timing"
    )
    assert "§" not in _human_display_title(name="x", section="Y")
    assert _human_display_title(name="", document_number="12") == "Document #12"
    assert _human_display_title() == "Document"


def test_search_hit_str_keeps_header_and_fence_format():
    hit = SearchHit(
        name="Dairy Handbook",
        text="## Heat Timing\n\nBody paragraph",
        section="Heat Timing",
        document_number="",
        doc_id="doc-abc123def456",
        id="marqo-opaque-id",
    )
    rendered = str(hit)
    assert rendered.startswith("**Dairy Handbook — Heat Timing**\n```\n")
    assert rendered.rstrip().endswith("```")
    assert "doc-abc" not in rendered
    assert "marqo-opaque" not in rendered
    assert "§" not in rendered


def test_safe_update_observation_swallows_langfuse_errors():
    boom = MagicMock()
    boom.update.side_effect = RuntimeError("langfuse down")
    _safe_update_observation(boom, output={"hit_count": 1})
    boom.update.assert_called_once()
    _safe_update_observation(None, output={"hit_count": 0})


def test_no_parallel_search_documents_retrieval_span():
    """Review #109: provenance must fold into marqo_search, not a second span."""
    source = inspect.getsource(search_documents)
    assert "search_documents_retrieval" not in source
    assert "get_langfuse_client" not in source
    assert "marqo_search" in source
    assert "_safe_update_observation" in source
