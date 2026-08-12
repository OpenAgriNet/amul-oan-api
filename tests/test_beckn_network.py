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
# Bharat Vistaar central schemes, served by the seeker's MOA leg. The `id` is
# the scheme CODE, which is the ONLY thing that leg matches on.
VISTAAR_SCHEME_ITEMS = [
    {"id": "kcc", "descriptor": {"name": "Kisan Credit Card", "long_desc": "Short-term credit for farmers."},
     "tags": [{"code": "category", "value": "credit"}, {"code": "source", "value": "Bharat Vistaar"}]},
    {"id": "pmfby", "descriptor": {"name": "Pradhan Mantri Fasal Bima Yojana", "long_desc": "Crop insurance cover."},
     "tags": [{"code": "category", "value": "insurance"}, {"code": "source", "value": "Bharat Vistaar"}]},
]


class _RealisticSeekerClient(_FakeAsyncClient):
    """A seeker mock that models the legs' REAL, DISJOINT query vocabularies.

    This is the whole point of these tests. The previous mock answered every
    requested leg with its full item list whatever the query, so a fan-out that
    sent one shared string to both legs looked perfectly healthy — which is
    exactly how a tool that returned zero `bharat-vistaar` records on every live
    call shipped green. Measured on dev (12/12 reps):

        query "schemes" -> moa 0, amulschemes 15
        query "kcc"     -> moa 1, amulschemes 0

    So, faithfully:
      * `amulschemes` does loose free-text matching over scheme TITLES, and
        treats "schemes" / an empty query as "list the catalogue".
      * `moa` matches the exact scheme CODE and nothing else — no title, no
        synonym, no substring.

    A mock more permissive than the service proves nothing; this one is not.
    """

    def __init__(self, union_items=None, vistaar_items=None, errors=None, dead_legs=()):
        super().__init__(None)
        self._union = SCHEME_ITEMS if union_items is None else union_items
        self._vistaar = VISTAAR_SCHEME_ITEMS if vistaar_items is None else vistaar_items
        self._errors = errors or {}
        self._dead = set(dead_legs)

    def _match(self, leg, query):
        q = (query or "").strip().lower()
        if leg == bn.SCHEMES_LEG:
            if q in ("", "schemes"):
                return list(self._union)
            tokens = [t for t in q.split() if len(t) >= 3]
            return [
                it for it in self._union
                if any(t in it["descriptor"]["name"].lower() for t in tokens)
            ]
        if leg == bn.VISTAAR_LEG:
            # Exact code only. "schemes", "Kisan Credit Card", "crop insurance"
            # all return an empty catalogue here, as they do live.
            return [it for it in self._vistaar if it["id"] == q]
        return []

    async def post(self, url, json=None):
        self.calls.append((url, json))
        asked = (json or {}).get("legs") or []
        query = (json or {}).get("query", "")
        results, errors = {}, {}
        for leg in asked:
            if leg in self._dead:
                results[leg] = None
                if leg in self._errors:
                    errors[leg] = self._errors[leg]
            else:
                results[leg] = _leg_result(leg, self._match(leg, query))
        body = {"elapsed_ms": 5, "results": results}
        if errors:
            body["errors"] = errors
        return _FakeResponse(body)


def _queries_by_leg(calls):
    """{leg: query} actually sent, asserting one leg per request."""
    out = {}
    for _url, body in calls:
        legs = body["legs"]
        assert len(legs) == 1, f"expected one leg per request, got {legs}"
        out[legs[0]] = body["query"]
    return out


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
async def test_vet_search_dead_leg_is_not_reported_as_no_documents():
    """Same failure/empty distinction as scheme discovery: a leg the seeker
    reports as failed must not read as an empty knowledge base."""
    fake = _FakeAsyncClient({"results": {"amulvet": None}, "errors": {"amulvet": "timeout"}})
    with patch.object(bn.httpx, "AsyncClient", return_value=fake):
        out = await bn.network_search_documents("mastitis")
    assert "temporarily unavailable" in out.lower()
    assert "No relevant documents" not in out


@pytest.mark.asyncio
async def test_union_schemes_filters_by_union():
    fake = _FakeAsyncClient(_seeker_payload("amulschemes", SCHEME_ITEMS))
    with patch.object(bn.httpx, "AsyncClient", return_value=fake):
        out = await bn.network_union_schemes("insurance", union="banas")
    parsed = json.loads(out)
    assert all(s["union"] == "banas" for s in parsed)
    assert parsed[0]["scheme_title"] == "Cattle Insurance"


@pytest.mark.asyncio
async def test_mock_reproduces_the_disjoint_leg_vocabularies():
    """Guards the guard. If this ever passes with a permissive mock, every test
    below is vacuous, so pin the measured dev behaviour explicitly: one shared
    query string CANNOT satisfy both legs."""
    fake = _RealisticSeekerClient()
    # "schemes" — the string the old code sent — is empty on the central leg.
    assert fake._match(bn.SCHEMES_LEG, "schemes") == SCHEME_ITEMS
    assert fake._match(bn.VISTAAR_LEG, "schemes") == []
    # ...and a code is empty on the union leg.
    assert fake._match(bn.VISTAAR_LEG, "kcc") == [VISTAAR_SCHEME_ITEMS[0]]
    assert fake._match(bn.SCHEMES_LEG, "kcc") == []


@pytest.mark.asyncio
async def test_each_leg_gets_its_own_query_union_free_text_vistaar_code():
    fake = _RealisticSeekerClient()
    with patch.object(bn.httpx, "AsyncClient", return_value=fake):
        await bn.network_union_schemes("Kisan Credit Card")
    sent = _queries_by_leg(fake.calls)
    # The union leg keeps the farmer's words; the central leg gets the CODE.
    assert sent == {bn.SCHEMES_LEG: "Kisan Credit Card", bn.VISTAAR_LEG: "kcc"}


@pytest.mark.asyncio
async def test_scheme_discovery_returns_union_and_central_in_one_call():
    """The acceptance criterion: ONE call, BOTH sources. Impossible with a
    single shared query string — the union leg needs "crop insurance", the
    central leg needs "pmfby"."""
    fake = _RealisticSeekerClient()
    with patch.object(bn.httpx, "AsyncClient", return_value=fake):
        parsed = json.loads(await bn.network_union_schemes("crop insurance"))
    by_title = {s["scheme_title"]: s for s in parsed}
    assert by_title["Cattle Insurance"]["source_network"] == "amul-union"
    assert by_title["Pradhan Mantri Fasal Bima Yojana"]["source_network"] == "bharat-vistaar"
    # Central schemes have no owning union — the key must not be faked.
    assert by_title["Pradhan Mantri Fasal Bima Yojana"].get("union") is None


@pytest.mark.asyncio
async def test_gujarati_phrasing_reaches_the_central_leg():
    """Production chat is mostly Gujarati; "પાક વીમો" must resolve to pmfby."""
    fake = _RealisticSeekerClient()
    with patch.object(bn.httpx, "AsyncClient", return_value=fake):
        parsed = json.loads(await bn.network_union_schemes("પાક વીમો શું છે"))
    assert _queries_by_leg(fake.calls)[bn.VISTAAR_LEG] == "pmfby"
    assert any(s["source_network"] == "bharat-vistaar" for s in parsed)


@pytest.mark.asyncio
async def test_central_leg_is_skipped_when_no_scheme_code_resolves():
    """A bare "what schemes do I get" is not a central-scheme question. Sending
    the central leg a word it can never match is a guaranteed-empty ~2.2s round
    trip, so it must not be sent at all."""
    fake = _RealisticSeekerClient()
    with patch.object(bn.httpx, "AsyncClient", return_value=fake):
        parsed = json.loads(await bn.network_union_schemes(""))
    assert list(_queries_by_leg(fake.calls)) == [bn.SCHEMES_LEG]
    assert {s["source_network"] for s in parsed} == {"amul-union"}


@pytest.mark.asyncio
async def test_union_filter_does_not_drop_vistaar_schemes():
    fake = _RealisticSeekerClient()
    with patch.object(bn.httpx, "AsyncClient", return_value=fake):
        parsed = json.loads(await bn.network_union_schemes("crop insurance", union="banas"))
    unions = [s.get("union") for s in parsed if s["source_network"] == "amul-union"]
    assert unions == ["banas"]
    assert any(s["source_network"] == "bharat-vistaar" for s in parsed)


@pytest.mark.asyncio
async def test_dead_central_leg_is_flagged_unavailable_not_silently_dropped():
    """A failed leg must never render as a clean union-only answer — that is an
    infrastructure failure presented to the farmer as fact."""
    fake = _RealisticSeekerClient(errors={"moa": "timeout"}, dead_legs=["moa"])
    with patch.object(bn.httpx, "AsyncClient", return_value=fake):
        out = await bn.network_union_schemes("crop insurance")
    assert "Cattle Insurance" in out
    assert "temporarily unavailable" in out.lower()
    assert "central government schemes" in out


@pytest.mark.asyncio
async def test_dead_leg_with_no_error_entry_is_still_flagged():
    """The seeker omits `errors` for some failures and just nulls the result.
    A null on_search is not an empty catalogue."""
    fake = _RealisticSeekerClient(dead_legs=["moa"])
    with patch.object(bn.httpx, "AsyncClient", return_value=fake):
        out = await bn.network_union_schemes("crop insurance")
    assert "temporarily unavailable" in out.lower()


@pytest.mark.asyncio
async def test_all_legs_dead_never_claims_no_schemes_exist():
    fake = _RealisticSeekerClient(
        errors={"amulschemes": "timeout", "moa": "timeout"},
        dead_legs=["amulschemes", "moa"],
    )
    with patch.object(bn.httpx, "AsyncClient", return_value=fake):
        out = await bn.network_union_schemes("crop insurance")
    assert "temporarily unavailable" in out.lower()
    assert "No scheme data was found" not in out


@pytest.mark.asyncio
async def test_scheme_discovery_empty_on_both_legs_is_reported_as_empty():
    """A genuinely empty catalogue on healthy legs still says "none found" —
    the unavailable message must not swallow real misses."""
    fake = _RealisticSeekerClient(union_items=[], vistaar_items=[])
    with patch.object(bn.httpx, "AsyncClient", return_value=fake):
        out = await bn.network_union_schemes("crop insurance")
    assert "No scheme data was found" in out
    assert "temporarily unavailable" not in out.lower()


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


# ── the LLM-visible docstring must match what the tool actually does ─────────

def test_union_scheme_docstring_does_not_contradict_the_merge():
    """`get_union_scheme_data` used to promise "ONLY for dairy-union schemes /
    Do NOT use this for KCC, PM-KISAN, PMFBY" while, with ENABLE_NETWORK=true,
    that same call returns merged central schemes. The docstring is the model's
    contract and has to hold for BOTH flag states."""
    from agents.tools.union_schemes import get_union_scheme_data

    doc = " ".join((get_union_scheme_data.__doc__ or "").split())
    assert doc, "the tool has no LLM-visible description at all"
    lowered = doc.lower()
    assert "only for these dairy-union schemes" not in lowered
    assert "do not use this for central" not in lowered
    # It must positively acknowledge the merged behaviour.
    assert "central" in lowered
