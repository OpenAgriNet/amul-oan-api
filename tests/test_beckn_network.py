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


def _leg_result(leg, items):
    return {"message": {"catalog": {"providers": [{"id": leg, "items": items}]}}}


def _seeker_payload(leg, items):
    return {"results": {leg: _leg_result(leg, items)}}


def _multi_leg_payload(by_leg):
    """A seeker response covering several legs at once: {leg: items}."""
    return {"results": {leg: _leg_result(leg, items) for leg, items in by_leg.items()}}


class _LegAwareAsyncClient(_FakeAsyncClient):
    """Like _FakeAsyncClient but answers only for the legs actually requested,
    the way the real seeker does — so a caller that stops asking for a leg
    stops receiving it."""
    def __init__(self, by_leg):
        super().__init__(None)
        self._by_leg = by_leg

    async def post(self, url, json=None):
        self.calls.append((url, json))
        asked = (json or {}).get("legs") or []
        return _FakeResponse(_multi_leg_payload(
            {leg: items for leg, items in self._by_leg.items() if leg in asked}
        ))


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
# Bharat Vistaar central schemes, served by the seeker's MOA leg.
VISTAAR_SCHEME_ITEMS = [
    {"id": "kcc", "descriptor": {"name": "Kisan Credit Card", "long_desc": "Short-term credit for farmers."},
     "tags": [{"code": "category", "value": "credit"}, {"code": "source", "value": "Bharat Vistaar"}]},
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
async def test_scheme_discovery_fans_out_to_both_legs_in_one_call():
    fake = _LegAwareAsyncClient({"amulschemes": SCHEME_ITEMS, "moa": VISTAAR_SCHEME_ITEMS})
    with patch.object(bn.httpx, "AsyncClient", return_value=fake):
        await bn.network_union_schemes("credit")
    # One fan-out request covering both legs, not two sequential searches.
    assert len(fake.calls) == 1
    assert fake.calls[0][1]["legs"] == [bn.SCHEMES_LEG, bn.VISTAAR_LEG]


@pytest.mark.asyncio
async def test_scheme_discovery_merges_union_and_vistaar_results():
    fake = _LegAwareAsyncClient({"amulschemes": SCHEME_ITEMS, "moa": VISTAAR_SCHEME_ITEMS})
    with patch.object(bn.httpx, "AsyncClient", return_value=fake):
        parsed = json.loads(await bn.network_union_schemes("credit"))
    titles = [s["scheme_title"] for s in parsed]
    assert "Cattle Insurance" in titles and "Kisan Credit Card" in titles
    by_title = {s["scheme_title"]: s for s in parsed}
    assert by_title["Cattle Insurance"]["source_network"] == "amul-union"
    assert by_title["Kisan Credit Card"]["source_network"] == "bharat-vistaar"
    # Central schemes have no owning union — the key must not be faked.
    assert by_title["Kisan Credit Card"].get("union") is None


@pytest.mark.asyncio
async def test_union_filter_does_not_drop_vistaar_schemes():
    fake = _LegAwareAsyncClient({"amulschemes": SCHEME_ITEMS, "moa": VISTAAR_SCHEME_ITEMS})
    with patch.object(bn.httpx, "AsyncClient", return_value=fake):
        parsed = json.loads(await bn.network_union_schemes("credit", union="banas"))
    unions = [s.get("union") for s in parsed if s["source_network"] == "amul-union"]
    assert unions == ["banas"]
    assert any(s["source_network"] == "bharat-vistaar" for s in parsed)


@pytest.mark.asyncio
async def test_scheme_discovery_survives_a_dead_vistaar_leg():
    # The seeker reports a failed leg as null; the union half must still answer.
    fake = _FakeAsyncClient({"results": {
        "amulschemes": _leg_result("amulschemes", SCHEME_ITEMS), "moa": None,
    }})
    with patch.object(bn.httpx, "AsyncClient", return_value=fake):
        parsed = json.loads(await bn.network_union_schemes("insurance"))
    assert {s["source_network"] for s in parsed} == {"amul-union"}


@pytest.mark.asyncio
async def test_scheme_discovery_empty_on_both_legs():
    fake = _FakeAsyncClient(_multi_leg_payload({"amulschemes": [], "moa": []}))
    with patch.object(bn.httpx, "AsyncClient", return_value=fake):
        out = await bn.network_union_schemes("nonsense")
    assert "No scheme data was found" in out


@pytest.mark.asyncio
async def test_single_leg_wrapper_still_scopes_to_one_leg():
    fake = _FakeAsyncClient(_seeker_payload("amulvet", VET_ITEMS))
    with patch.object(bn.httpx, "AsyncClient", return_value=fake):
        on_search = await bn._seeker_search("mastitis", bn.VET_LEG, user_id="u-1")
    assert fake.calls[0][1] == {"query": "mastitis", "legs": ["amulvet"], "user_id": "u-1"}
    assert bn._items(on_search)[0]["descriptor"]["name"] == "Mastitis care"


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


@pytest.mark.asyncio
async def test_ai_call_nack_is_marked_authoritative_no_booking():
    """A NACK is the BPP saying it did not book — the caller may release its
    reservation on it, so the flag must be set."""
    payload = {"message": {"ack": {"status": "NACK"}}, "error": {"code": "40002", "message": "society not serviced"}}
    fake = _FakeAsyncClient(payload)
    with patch.object(bn.httpx, "AsyncClient", return_value=fake):
        res = await bn.network_create_ai_call_result("12", "S1", "F1", "AIT-1", "cow")
    assert res.ok is False
    assert res.authoritative_no_booking is True


@pytest.mark.asyncio
async def test_ai_call_200_with_unparseable_body_is_not_success():
    """`ok` used to default to True, so a 200 with no ack and no order.id told
    the farmer "booked successfully. Ticket: None" and burned the session's TTL
    with nothing booked. It is a failure, and the message must neither claim
    success nor print a None ticket."""
    fake = _FakeAsyncClient({})
    with patch.object(bn.httpx, "AsyncClient", return_value=fake):
        res = await bn.network_create_ai_call_result("12", "S1", "F1", "AIT-1", "cow")
    assert res.ok is False, "a 200 with no order.id was reported as a successful booking"
    assert res.ticket is None
    assert "booked successfully" not in res.message.lower()
    assert "none" not in res.message.lower(), "the farmer was shown a None ticket"
    assert "could not be confirmed" in res.message.lower()
    # NOT authoritative: the BPP answered without refusing, so it may already
    # have called PashuGPT and sent the SMS. Same class as a read timeout —
    # the caller must hold the reservation.
    assert res.authoritative_no_booking is False


@pytest.mark.asyncio
async def test_ai_call_200_with_order_but_no_id_is_not_success():
    """Same for a partially-filled order envelope."""
    fake = _FakeAsyncClient({"message": {"order": {"state": "ACTIVE"}}})
    with patch.object(bn.httpx, "AsyncClient", return_value=fake):
        res = await bn.network_create_ai_call_result("12", "S1", "F1", "AIT-1", "cow")
    assert res.ok is False
    assert res.authoritative_no_booking is False
