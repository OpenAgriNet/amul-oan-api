import pytest
from pydantic_ai import ModelRetry

from agents.voice.tools import search
from agents.tools.beckn import network


@pytest.mark.asyncio
async def test_beckn_voice_route_needs_no_marqo_endpoint(monkeypatch):
    monkeypatch.setenv("VET_SEARCH_BACKEND", "beckn")
    monkeypatch.delenv("MARQO_ENDPOINT_URL", raising=False)
    calls = []

    async def provider(query, top_k):
        calls.append((query, top_k))
        return "1. Veterinary passage"

    monkeypatch.setattr(network, "network_search_documents", provider)
    result = await search.search_documents(None, "cow mastitis", 12)
    assert result == "1. Veterinary passage"
    assert calls == [("cow mastitis", search._resolve_final_top_k(12))]


@pytest.mark.asyncio
async def test_invalid_voice_query_does_not_reach_provider(monkeypatch):
    monkeypatch.setenv("VET_SEARCH_BACKEND", "beckn")

    async def unexpected(*args):
        raise AssertionError("invalid query reached provider")

    monkeypatch.setattr(network, "network_search_documents", unexpected)
    with pytest.raises(ModelRetry, match="please try again"):
        await search.search_documents(None, "")


@pytest.mark.asyncio
async def test_voice_provider_errors_keep_retry_contract(monkeypatch):
    monkeypatch.setenv("VET_SEARCH_BACKEND", "beckn")

    async def broken(*args):
        raise RuntimeError("provider transport failed")

    monkeypatch.setattr(network, "network_search_documents", broken)
    with pytest.raises(ModelRetry):
        await search.search_documents(None, "cow mastitis")


@pytest.mark.asyncio
async def test_default_voice_route_remains_marqo(monkeypatch):
    monkeypatch.delenv("VET_SEARCH_BACKEND", raising=False)
    monkeypatch.delenv("MARQO_ENDPOINT_URL", raising=False)

    async def unexpected(*args):
        raise AssertionError("default route reached Beckn")

    monkeypatch.setattr(network, "network_search_documents", unexpected)
    with pytest.raises(ModelRetry):
        await search.search_documents(None, "cow mastitis")


@pytest.mark.asyncio
async def test_unknown_backend_does_not_silently_fall_back(monkeypatch):
    monkeypatch.setenv("VET_SEARCH_BACKEND", "typo")
    with pytest.raises(ModelRetry):
        await search.search_documents(None, "cow mastitis")
