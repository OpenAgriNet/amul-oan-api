import asyncio

from app.config import Settings
from agents.tools import search


def test_settings_accept_valid_marqo_numeric_overrides(monkeypatch):
    monkeypatch.setenv("MARQO_DEFAULT_FINAL_CHUNKS", "11")
    monkeypatch.setenv("MARQO_MAX_FINAL_CHUNKS", "31")
    monkeypatch.setenv("MARQO_HYBRID_ALPHA", "0.35")
    monkeypatch.setenv("MARQO_HYBRID_RRFK", "77")

    cfg = Settings()
    assert cfg.marqo_default_final_chunks == 11
    assert cfg.marqo_max_final_chunks == 31
    assert cfg.marqo_hybrid_alpha == 0.35
    assert cfg.marqo_hybrid_rrfk == 77


def test_settings_fallback_on_malformed_marqo_numeric_env(monkeypatch):
    monkeypatch.setenv("MARQO_DEFAULT_FINAL_CHUNKS", "not-an-int")
    monkeypatch.setenv("MARQO_MAX_FINAL_CHUNKS", "oops")
    monkeypatch.setenv("MARQO_HYBRID_ALPHA", "bad-float")
    monkeypatch.setenv("MARQO_HYBRID_RRFK", "nan-int")

    cfg = Settings()  # must not raise
    assert cfg.marqo_default_final_chunks == 8
    assert cfg.marqo_max_final_chunks == 20
    assert cfg.marqo_hybrid_alpha == 0.6
    assert cfg.marqo_hybrid_rrfk == 60


def test_settings_clamp_operational_numeric_bounds(monkeypatch):
    monkeypatch.setenv("FARMER_REFRESH_QUEUE_BATCH_SIZE", "-5")
    monkeypatch.setenv("SCHEME_OCR_MAX_FAILED_PAGE_RATIO", "1.5")
    monkeypatch.setenv("SCHEME_BANAS_MIN_RECORD_COVERAGE_RATIO", "-0.2")
    monkeypatch.setenv("SCHEME_HTTP_TIMEOUT_SECONDS", "0")

    cfg = Settings()
    assert cfg.farmer_refresh_queue_batch_size == 1
    assert cfg.scheme_ocr_max_failed_page_ratio == 1.0
    assert cfg.scheme_banas_min_record_coverage_ratio == 0.0
    assert cfg.scheme_http_timeout_seconds == 0.001


def test_search_documents_consumes_settings_endpoint_and_index(monkeypatch):
    captured: dict[str, tuple] = {}

    monkeypatch.setattr(search.settings, "enable_network", False)
    monkeypatch.setattr(search.settings, "marqo_endpoint_url", "http://marqo.test")
    monkeypatch.setattr(search.settings, "marqo_index_name", None)

    def _fake_caps(endpoint_url: str, index_name: str):
        captured["caps"] = (endpoint_url, index_name)
        return {"exists": False, "error": "not-found", "has_is_reference_filter": False}

    def _fake_search(endpoint_url: str, index_name: str, search_params):
        captured["search"] = (endpoint_url, index_name, search_params)
        return []

    monkeypatch.setattr(search, "_get_index_capabilities_sync", _fake_caps)
    monkeypatch.setattr(search, "_marqo_search_sync", _fake_search)

    out = asyncio.run(search.search_documents("mastitis", top_k=3))
    assert "No results found for" in out
    assert captured["caps"] == ("http://marqo.test", "amul-veterinary-index")
    assert captured["search"][0] == "http://marqo.test"
    assert captured["search"][1] == "amul-veterinary-index"
