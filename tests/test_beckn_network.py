"""Tests for the Beckn network client (agents/tools/beckn_network.py).

Mocks the HTTP layer so no live services are needed; asserts the Beckn
on_search / on_confirm payloads are formatted into the same string contract the
direct tools return, and that errors (NACK) are surfaced cleanly.
"""
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load beckn_network directly by path so we don't trigger agents/tools/__init__
# (which imports pydantic_ai + every heavy tool). The module itself only needs
# httpx, app.config, and helpers.utils.
_spec = importlib.util.spec_from_file_location(
    "beckn_network", ROOT / "agents" / "tools" / "beckn_network.py"
)
bn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bn)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeAsyncClient:
    """Stands in for httpx.AsyncClient; returns a queued payload per POST."""
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None):
        self.calls.append((url, json))
        return _FakeResponse(self._payload)


def _seeker_payload(leg, items):
    return {"results": {leg: {"message": {"catalog": {"providers": [{"id": leg, "items": items}]}}}}}


VET_ITEMS = [
    {"id": "doc-1#3", "descriptor": {"name": "Mastitis care", "long_desc": "Treat with intramammary antibiotics after stripping."},
     "tags": [{"code": "source", "value": "Merck Vet Manual"}, {"code": "score", "value": "0.91"}]},
]
SCHEME_ITEMS = [
    {"id": "banas-cattle-insurance", "descriptor": {"name": "Cattle Insurance", "long_desc": "Cover for milch cattle."},
     "tags": [{"code": "union", "value": "banas"}, {"code": "category", "value": "insurance"}]},
    {"id": "kutch-mineral", "descriptor": {"name": "Mineral Mixture", "long_desc": "Free mineral mixture."},
     "tags": [{"code": "union", "value": "kutch"}, {"code": "category", "value": "input-support"}]},
]


@pytest.mark.asyncio
async def test_vet_search_formats_items_with_source():
    fake = _FakeAsyncClient(_seeker_payload("amulvet", VET_ITEMS))
    with patch.object(bn.httpx, "AsyncClient", return_value=fake):
        out = await bn.network_search_documents("mastitis", top_k=5)
    assert "Mastitis care" in out
    assert "intramammary antibiotics" in out
    assert "source: Merck Vet Manual" in out
    # scoped to the vet leg only
    assert fake.calls[0][1]["legs"] == ["amulvet"]


@pytest.mark.asyncio
async def test_vet_search_empty():
    fake = _FakeAsyncClient(_seeker_payload("amulvet", []))
    with patch.object(bn.httpx, "AsyncClient", return_value=fake):
        out = await bn.network_search_documents("nonsense")
    assert "No relevant documents" in out


@pytest.mark.asyncio
async def test_union_schemes_filters_by_union():
    fake = _FakeAsyncClient(_seeker_payload("amulschemes", SCHEME_ITEMS))
    with patch.object(bn.httpx, "AsyncClient", return_value=fake):
        out = await bn.network_union_schemes("insurance", union="banas")
    parsed = json.loads(out)
    assert all(s["union"] == "banas" for s in parsed)
    assert parsed[0]["scheme_title"] == "Cattle Insurance"


@pytest.mark.asyncio
async def test_ai_call_success_returns_ticket():
    payload = {"message": {"order": {"id": "AICALL-889231", "state": "ACTIVE"}}}
    fake = _FakeAsyncClient(payload)
    with patch.object(bn.httpx, "AsyncClient", return_value=fake):
        out = await bn.network_create_ai_call("12", "S1", "F1", "AIT-1", "cow")
    assert "AICALL-889231" in out
    assert "booked successfully" in out
    # confirm order carried the farmer + technician correctly
    order = fake.calls[0][1]["message"]["order"]
    assert order["items"][0]["id"] == "ait:AIT-1"
    assert {"code": "farmer_id", "value": "F1"} in order["fulfillment"]["customer"]["tags"]


@pytest.mark.asyncio
async def test_ai_call_nack_surfaces_error():
    payload = {"message": {"ack": {"status": "NACK"}}, "error": {"code": "40002", "message": "society not serviced"}}
    fake = _FakeAsyncClient(payload)
    with patch.object(bn.httpx, "AsyncClient", return_value=fake):
        out = await bn.network_create_ai_call("12", "S1", "F1", "AIT-1", "cow")
    assert "failed" in out.lower()
    assert "society not serviced" in out
