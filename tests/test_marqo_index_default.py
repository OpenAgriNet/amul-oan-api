import asyncio

import pytest

from app.config import Settings
from agents.tools import search


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1", True),
        ("true", True),
        ("YES", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("NO", False),
        ("off", False),
    ],
)
def test_settings_accept_valid_marqo_boolean_overrides(monkeypatch, raw, expected):
    monkeypatch.setenv("MARQO_USE_E5_QUERY_PREFIX", raw)
    monkeypatch.setenv("MARQO_EXCLUDE_REFERENCE", raw)

    cfg = Settings()
    assert cfg.marqo_use_e5_query_prefix is expected
    assert cfg.marqo_exclude_reference is expected


@pytest.mark.parametrize("raw", ["invalid", ""])
def test_settings_fallback_on_invalid_marqo_use_e5_boolean(monkeypatch, raw):
    monkeypatch.setenv("MARQO_USE_E5_QUERY_PREFIX", raw)

    cfg = Settings()  # must not raise
    assert cfg.marqo_use_e5_query_prefix is True


@pytest.mark.parametrize("raw", ["invalid", ""])
def test_settings_fallback_on_invalid_marqo_exclude_reference_boolean(monkeypatch, raw):
    monkeypatch.setenv("MARQO_EXCLUDE_REFERENCE", raw)

    cfg = Settings()  # must not raise
    assert cfg.marqo_exclude_reference is True


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


def test_settings_vistaar_coords_fallback_on_malformed_values(monkeypatch):
    monkeypatch.setenv("VISTAAR_DEFAULT_LAT", "bad-lat")
    monkeypatch.setenv("VISTAAR_DEFAULT_LON", "bad-lon")

    cfg = Settings()  # must not raise
    assert cfg.vistaar_default_lat == 22.55
    assert cfg.vistaar_default_lon == 72.93


def test_settings_vistaar_coords_clamp_geographic_bounds(monkeypatch):
    monkeypatch.setenv("VISTAAR_DEFAULT_LAT", "999")
    monkeypatch.setenv("VISTAAR_DEFAULT_LON", "-999")

    cfg = Settings()
    assert cfg.vistaar_default_lat == 90.0
    assert cfg.vistaar_default_lon == -180.0


def test_settings_non_finite_floats_fallback_to_defaults(monkeypatch):
    monkeypatch.setenv("SCHEME_HTTP_TIMEOUT_SECONDS", "nan")
    monkeypatch.setenv("MARQO_HYBRID_ALPHA", "inf")

    cfg = Settings()
    assert cfg.scheme_http_timeout_seconds == 30.0
    assert cfg.marqo_hybrid_alpha == 0.6


def test_settings_normalize_backend_base_urls(monkeypatch):
    monkeypatch.setenv("AMULPASHUDHAN_BASE_URL", "https://example.test/root/")
    monkeypatch.setenv("HERDMAN_BASE_URL", "https://herdman.test/api///")
    monkeypatch.setenv("BANAS_MOBILE_BASE_URL", "https://banas.test/visit/")
    monkeypatch.setenv("CVCC_BASE_URL", "https://cvcc.test/path/")

    cfg = Settings()
    assert cfg.amulpashudhan_base_url == "https://example.test/root"
    assert cfg.herdman_base_url == "https://herdman.test/api"
    assert cfg.banas_mobile_base_url == "https://banas.test/visit"
    assert cfg.cvcc_base_url == "https://cvcc.test/path"


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
