"""Unit tests for the Beckn government-scheme discovery tool.

No network: httpx is monkeypatched. Follows the repo's synchronous
``asyncio.run`` test style (no pytest-asyncio dependency).
"""

import asyncio

import httpx

from agents.tools import beckn_search
from agents.tools.beckn_search import _extract_items, search_government_schemes


def _run(coro):
    return asyncio.run(coro)


# ---- _extract_items: both catalog shapes -----------------------------------

def test_extract_items_vistaar_shape():
    """Vistaar uses message.catalog.providers[]."""
    leg = {
        "message": {
            "catalog": {
                "providers": [
                    {
                        "descriptor": {"name": "SchemeFinder"},
                        "items": [
                            {
                                "id": "PMKISAN-101",
                                "descriptor": {
                                    "name": "Kisan Credit Card",
                                    "short_desc": "Credit for farmers",
                                },
                            }
                        ],
                    }
                ]
            }
        }
    }
    items = _extract_items(leg)
    assert items == [
        {
            "provider": "SchemeFinder",
            "name": "Kisan Credit Card",
            "description": "Credit for farmers",
            "id": "PMKISAN-101",
        }
    ]


def test_extract_items_mock_slash_shape():
    """MH mock uses the Beckn 1.x slash convention message.catalog['bpp/providers']."""
    leg = {
        "message": {
            "catalog": {
                "bpp/providers": [
                    {
                        "descriptor": {"name": "MH Weather"},
                        "items": [
                            {"id": "mh-w-basic", "descriptor": {"name": "Weather Advisory (MH)"}}
                        ],
                    }
                ]
            }
        }
    }
    items = _extract_items(leg)
    assert items == [
        {
            "provider": "MH Weather",
            "name": "Weather Advisory (MH)",
            "description": "",
            "id": "mh-w-basic",
        }
    ]


def test_extract_items_handles_none_and_empty():
    assert _extract_items(None) == []
    assert _extract_items({}) == []
    assert _extract_items({"message": {"catalog": {}}}) == []


# ---- fake httpx client -----------------------------------------------------

class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error", request=httpx.Request("POST", "http://x"), response=self
            )


class _FakeClient:
    def __init__(self, payload=None, status_code=200, raises=None):
        self._payload = payload
        self._status_code = status_code
        self._raises = raises

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, *args, **kwargs):
        if self._raises is not None:
            raise self._raises
        return _FakeResponse(self._payload, self._status_code)


def _patch_client(monkeypatch, *, cache_hit=False, cached_value=None, **client_kwargs):
    monkeypatch.setattr(beckn_search.settings, "beckn_enabled", True)
    monkeypatch.setattr(beckn_search.settings, "amul_bap_url", "https://bap.example")
    monkeypatch.setattr(
        beckn_search.httpx, "AsyncClient", lambda *a, **k: _FakeClient(**client_kwargs)
    )

    async def _fake_get(key):
        return (cache_hit, cached_value)

    async def _fake_set(key, value):
        return None

    monkeypatch.setattr(beckn_search, "get_cached_api_response", _fake_get)
    monkeypatch.setattr(beckn_search, "set_cached_api_response", _fake_set)


# ---- search_government_schemes ---------------------------------------------

def test_disabled_returns_clear_message(monkeypatch):
    monkeypatch.setattr(beckn_search.settings, "beckn_enabled", False)
    out = _run(search_government_schemes("KCC"))
    assert "not enabled" in out.lower()


def test_aggregates_both_legs(monkeypatch):
    payload = {
        "moa": {
            "message": {
                "catalog": {
                    "providers": [
                        {
                            "descriptor": {"name": "SchemeFinder"},
                            "items": [
                                {"id": "PMKISAN-101", "descriptor": {"name": "Kisan Credit Card"}}
                            ],
                        }
                    ]
                }
            }
        },
        "mh": {
            "message": {
                "catalog": {
                    "bpp/providers": [
                        {
                            "descriptor": {"name": "MH Market"},
                            "items": [{"id": "mh-m-prices", "descriptor": {"name": "Mandi Prices (MH)"}}],
                        }
                    ]
                }
            }
        },
    }
    _patch_client(monkeypatch, payload=payload)
    out = _run(search_government_schemes("KCC"))
    assert "Kisan Credit Card" in out
    assert "Mandi Prices (MH)" in out
    assert "vistaar_goi_schemes" in out


def test_one_leg_down_still_returns_other(monkeypatch):
    payload = {
        "moa": None,
        "mh": {
            "message": {
                "catalog": {
                    "bpp/providers": [
                        {"descriptor": {"name": "MH Market"}, "items": [{"id": "x", "descriptor": {"name": "Mandi Prices (MH)"}}]}
                    ]
                }
            }
        },
        "errors": {"moa": "bap_unreachable"},
    }
    _patch_client(monkeypatch, payload=payload)
    out = _run(search_government_schemes("KCC"))
    assert "Mandi Prices (MH)" in out
    assert "bap_unreachable" in out


def test_empty_both_legs_no_cache_returns_clean_message(monkeypatch):
    payload = {"moa": None, "mh": None, "errors": {"moa": "bap_unreachable", "mh": "timeout"}}
    _patch_client(monkeypatch, payload=payload)  # no cache hit
    out = _run(search_government_schemes("KCC"))
    assert "temporarily unavailable" in out.lower()


def test_http_error_no_cache_returns_clean_message(monkeypatch):
    _patch_client(monkeypatch, status_code=502)  # no cache hit
    out = _run(search_government_schemes("KCC"))
    assert "temporarily unavailable" in out.lower()


def test_timeout_no_cache_returns_clean_message(monkeypatch):
    _patch_client(monkeypatch, raises=httpx.TimeoutException("slow"))
    out = _run(search_government_schemes("KCC"))
    assert "temporarily unavailable" in out.lower()


def test_live_failure_falls_back_to_cache(monkeypatch):
    """When the live BAP errors, serve the last-known-good cached result."""
    cached = {
        "vistaar_goi_schemes": [
            {"provider": "SchemeFinder", "name": "Kisan Credit Card", "description": "", "id": "PMKISAN-101"}
        ],
        "maharashtra_network": [],
        "unavailable_networks": {"vistaar_goi": None, "maharashtra": "bap_unreachable"},
    }
    _patch_client(monkeypatch, raises=httpx.TimeoutException("slow"), cache_hit=True, cached_value=cached)
    out = _run(search_government_schemes("KCC"))
    assert "Kisan Credit Card" in out
    assert "served from cache" in out


def test_live_empty_falls_back_to_cache(monkeypatch):
    """A live response with no items also falls back to cache when available."""
    cached = {"vistaar_goi_schemes": [{"provider": "SchemeFinder", "name": "Kisan Credit Card", "description": "", "id": "PMKISAN-101"}], "maharashtra_network": [], "unavailable_networks": {}}
    _patch_client(
        monkeypatch,
        payload={"moa": None, "mh": None, "errors": {"moa": "bap_unreachable", "mh": "timeout"}},
        cache_hit=True,
        cached_value=cached,
    )
    out = _run(search_government_schemes("KCC"))
    assert "Kisan Credit Card" in out
    assert "served from cache" in out
