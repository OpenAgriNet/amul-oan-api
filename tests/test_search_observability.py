from agents.tools.search import (
    _build_hit_provenance,
    _build_search_observability_output,
    _infer_section_from_text,
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
