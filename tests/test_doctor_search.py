import json

import httpx
import pytest

from agents.tools import doctor_search as ds


def _hit(doc, chunk, score, text="mastitis treatment", **extra):
    return {"doc_id": doc, "chunk_num": chunk, "_score": score, "text": text, "name_en": f"Doc {doc}", "filename": f"{doc}.pdf", **extra}


def _serve(monkeypatch, handler):
    real = httpx.AsyncClient
    monkeypatch.setattr(ds.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))


@pytest.mark.asyncio
async def test_without_url_the_doctor_searches_like_the_farmer(monkeypatch):
    monkeypatch.setattr(ds.settings, "doctor_search_url", "")

    async def farmer_search(query, top_k):
        return f"beckn:{query}:{top_k}"

    monkeypatch.setattr(ds, "search_documents", farmer_search)
    assert await ds.doctor_search_documents("mastitis treatment", 6) == "beckn:mastitis treatment:6"


@pytest.mark.asyncio
async def test_direct_search_reranks_caps_two_per_doc_and_formats(monkeypatch):
    monkeypatch.setattr(ds.settings, "doctor_search_url", "http://search.test")
    monkeypatch.setattr(ds.settings, "doctor_search_api_key", "k" * 32)
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        hits = [_hit("a", 1, 0.9), _hit("a", 2, 0.8), _hit("a", 3, 0.7), _hit("b", 1, 0.1, text="unrelated words")]
        return httpx.Response(200, json={"hits": hits})

    _serve(monkeypatch, handler)
    out = await ds.doctor_search_documents("mastitis treatment", 3)

    assert seen["auth"] == "Bearer " + "k" * 32
    assert seen["body"] == {"q": "query: mastitis treatment", "limit": 120, "alpha": 0.6, "rrf_k": 60, "exclude_reference": False}
    assert out.startswith("1. Doc a\nmastitis treatment\n(source: a.pdf)\n\n2. Doc a")
    assert "3. Doc b" in out  # third chunk of doc a is held back by the 2-per-doc cap


@pytest.mark.asyncio
async def test_service_down_reads_as_unavailable_not_empty(monkeypatch):
    monkeypatch.setattr(ds.settings, "doctor_search_url", "http://search.test")
    _serve(monkeypatch, lambda request: httpx.Response(503, json={"detail": "down"}))
    assert "temporarily unavailable" in await ds.doctor_search_documents("mastitis treatment")
