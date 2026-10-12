import pytest
from pydantic_ai import ModelRetry

from agents.tools import search


@pytest.mark.asyncio
async def test_malformed_beckn_search_payload_becomes_model_retry(monkeypatch):
    async def malformed_provider(*args, **kwargs):
        raise TypeError("malformed provider tags")

    monkeypatch.setattr(search, "network_search_documents", malformed_provider)

    with pytest.raises(ModelRetry, match="please try again"):
        await search.search_documents("mastitis treatment")


@pytest.mark.asyncio
async def test_query_validation_model_retry_is_preserved(monkeypatch):
    async def unexpected_search(*args, **kwargs):
        raise AssertionError("invalid queries must not reach Beckn")

    monkeypatch.setattr(search, "network_search_documents", unexpected_search)

    with pytest.raises(ModelRetry, match="EMPTY_QUERY"):
        await search.search_documents("")
