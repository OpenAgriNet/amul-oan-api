"""Voice's opt-in BPP route, with both retrieval backends mocked."""
import asyncio
import os
from unittest.mock import AsyncMock

import pytest
from pydantic_ai import ModelRetry

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from agents.tools.beckn import network
from agents.voice.agent import voice_agent, voice_agent_signed_in
from agents.voice.tools import search


@pytest.fixture(autouse=True)
def isolated_backends(monkeypatch):
    for name in list(os.environ):
        if name.startswith(("MARQO_", "VET_SEARCH_")):
            monkeypatch.delenv(name, raising=False)

    def unexpected_marqo(*args, **kwargs):
        raise AssertionError("This test must not contact Marqo")

    monkeypatch.setattr(search, "_get_index_capabilities_sync", unexpected_marqo)
    monkeypatch.setattr(search, "_marqo_search_sync", unexpected_marqo)


@pytest.mark.parametrize("agent", [voice_agent, voice_agent_signed_in])
def test_registered_voice_tool_delegates_normalized_query(monkeypatch, agent):
    monkeypatch.setenv("VET_SEARCH_BACKEND", " BeCkN ")
    provider = AsyncMock(return_value="1. Veterinary reference\nEvidence\n(source: reference.pdf)")
    monkeypatch.setattr(network, "network_search_documents", provider)

    tool = agent._function_toolset.tools["search_documents"]
    result = asyncio.run(tool.function(None, "  calf\n  scours  ", 1))

    provider.assert_awaited_once_with("calf scours", 1)
    assert result == provider.return_value


@pytest.mark.parametrize("requested, expected", [(0, 12), (1, 1), (999, 20)])
def test_network_route_preserves_result_limit_contract(monkeypatch, requested, expected):
    monkeypatch.setenv("VET_SEARCH_BACKEND", "beckn")
    provider = AsyncMock(return_value="evidence")
    monkeypatch.setattr(network, "network_search_documents", provider)

    asyncio.run(search.search_documents(None, "mastitis cattle", requested))

    provider.assert_awaited_once_with("mastitis cattle", expected)


def test_network_route_respects_lower_deployment_cap(monkeypatch):
    monkeypatch.setenv("VET_SEARCH_BACKEND", "beckn")
    monkeypatch.setenv("MARQO_MAX_FINAL_CHUNKS", "4")
    provider = AsyncMock(return_value="evidence")
    monkeypatch.setattr(network, "network_search_documents", provider)

    asyncio.run(search.search_documents(None, "mastitis cattle", 12))

    provider.assert_awaited_once_with("mastitis cattle", 4)


@pytest.mark.parametrize("backend", [None, "marqo"])
def test_default_and_explicit_marqo_keep_direct_route(monkeypatch, backend):
    if backend is not None:
        monkeypatch.setenv("VET_SEARCH_BACKEND", backend)
    monkeypatch.setenv("MARQO_ENDPOINT_URL", "http://unused.invalid")
    monkeypatch.setattr(search, "_get_index_capabilities_sync", lambda *args: {"exists": True})
    calls = []

    def direct(endpoint, index, params):
        calls.append(params)
        return [{"name": "Reference", "text": "Evidence", "_id": "reference-1"}]

    monkeypatch.setattr(search, "_marqo_search_sync", direct)
    provider = AsyncMock()
    monkeypatch.setattr(network, "network_search_documents", provider)

    result = asyncio.run(search.search_documents(None, "mastitis cattle"))

    assert len(calls) == 1
    assert "Reference" in result
    provider.assert_not_awaited()


@pytest.mark.parametrize("query", ["", "I can only answer agricultural questions"])
def test_invalid_query_never_reaches_provider(monkeypatch, query):
    monkeypatch.setenv("VET_SEARCH_BACKEND", "beckn")
    provider = AsyncMock()
    monkeypatch.setattr(network, "network_search_documents", provider)

    with pytest.raises(ModelRetry):
        asyncio.run(search.search_documents(None, query))

    provider.assert_not_awaited()


@pytest.mark.parametrize("backend", ["qdrant", ""])
def test_unknown_backend_fails_without_fallback(monkeypatch, backend):
    monkeypatch.setenv("VET_SEARCH_BACKEND", backend)
    provider = AsyncMock()
    monkeypatch.setattr(network, "network_search_documents", provider)

    with pytest.raises(ModelRetry):
        asyncio.run(search.search_documents(None, "mastitis cattle"))

    provider.assert_not_awaited()


def test_raised_provider_error_uses_existing_retry_contract(monkeypatch):
    monkeypatch.setenv("VET_SEARCH_BACKEND", "beckn")
    provider = AsyncMock(side_effect=RuntimeError("controlled provider failure"))
    monkeypatch.setattr(network, "network_search_documents", provider)

    with pytest.raises(ModelRetry, match="Error searching documents, please try again"):
        asyncio.run(search.search_documents(None, "mastitis cattle"))

    provider.assert_awaited_once()


def test_provider_unavailability_is_not_an_empty_result_or_marqo_fallback(monkeypatch):
    monkeypatch.setenv("VET_SEARCH_BACKEND", "beckn")
    unavailable = "The document search is temporarily unavailable. Please try again shortly."
    provider = AsyncMock(return_value=unavailable)
    monkeypatch.setattr(network, "network_search_documents", provider)

    assert asyncio.run(search.search_documents(None, "mastitis cattle")) == unavailable
    provider.assert_awaited_once()
