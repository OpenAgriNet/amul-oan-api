"""Tests for the Bharat Vistaar scheme tool and the seeker-error handling in
agents/tools/vistaar.py.

Three things are pinned here:

  1. `get_vistaar_scheme_info` is constrained at the JSON-schema level, the way
     bharat-oan-api's own `get_scheme_info` is, so the model picks from an enum
     instead of inventing a code. This is asserted against a REAL
     `pydantic_ai.Tool(...)` built with this repo's flags
     (docstring_format='auto', require_parameter_descriptions=True), not a
     hand-rolled schema — a Literal that the tool machinery quietly widens to a
     bare string would otherwise look fine.
  2. The farmer never sees `Unknown scheme code '…'. Valid codes: …`.
     agrinet_system.md forbids exposing internal tool mechanics.
  3. A failed seeker leg is distinguished from an empty catalogue. Reading only
     `results.<leg>` and dropping `errors` is what made the `moa` timeout flap
     render as a confident "No mandi prices were found…".
"""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic_ai import Tool

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "vistaar_schemes_mod", ROOT / "agents" / "tools" / "vistaar.py"
)
vistaar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vistaar)

from agents.tools.scheme_codes import SCHEME_CODES  # noqa: E402

SCHEME_ITEM = {
    "descriptor": {"name": "Kisan Credit Card", "long_desc": "Short-term credit."},
    "tags": [],
}


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeAsyncClient:
    def __init__(self, payload, timeout=None):
        self._payload = payload
        self.calls = []
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None):
        self.calls.append((url, json))
        return _FakeResponse(self._payload)


def _seeker_ok(items):
    leg = vistaar.VISTAAR_LEG
    return {"results": {leg: {"message": {"catalog": {"providers": [{"items": items}]}}}}}


def _client_factory(payload, sink):
    """Captures the timeout httpx.AsyncClient was constructed with."""
    def _make(*a, **kw):
        client = _FakeAsyncClient(payload, timeout=kw.get("timeout"))
        sink.append(client)
        return client
    return _make


# ── 1. schema-level constraint ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tool_schema_publishes_an_enum_of_the_fifteen_codes():
    tool = Tool(
        vistaar.get_vistaar_scheme_info,
        takes_ctx=False,
        docstring_format="auto",
        require_parameter_descriptions=True,
    )
    tool_def = await tool.prepare_tool_def(None)
    prop = tool_def.parameters_json_schema["properties"]["scheme_code"]
    assert prop.get("enum") == list(SCHEME_CODES), (
        "the model is not constrained to the valid codes; it can invent one"
    )
    # require_parameter_descriptions=True means an undescribed arg fails to
    # build at all — assert the description survived alongside the enum.
    assert prop.get("description")


# ── 2. no internal codes leak to the farmer ───────────────────────────────────

@pytest.mark.asyncio
async def test_unresolvable_scheme_never_leaks_the_code_list():
    out = await vistaar.get_vistaar_scheme_info("some scheme I made up")
    lowered = out.lower()
    assert "unknown scheme code" not in lowered
    assert "valid codes" not in lowered
    # Not one internal code appears as a bare word.
    words = set(lowered.replace(",", " ").replace(".", " ").split())
    assert not (words & set(SCHEME_CODES)), out
    # It still helps the farmer, by NAME.
    assert "Kisan Credit Card" in out


@pytest.mark.asyncio
async def test_free_phrasing_is_resolved_rather_than_rejected():
    """The merged discovery path calls this with the farmer's own words."""
    clients = []
    with patch.object(vistaar.httpx, "AsyncClient",
                      _client_factory(_seeker_ok([SCHEME_ITEM]), clients)):
        out = await vistaar.get_vistaar_scheme_info("Kisan Credit Card")
    sent = clients[0].calls[0][1]["intent"]["item"]["descriptor"]["name"]
    assert sent == "kcc", "the BPP matches the code only; free text returns nothing"
    assert "Kisan Credit Card" in out


@pytest.mark.asyncio
async def test_empty_catalogue_message_names_the_scheme_not_the_code():
    clients = []
    with patch.object(vistaar.httpx, "AsyncClient", _client_factory(_seeker_ok([]), clients)):
        out = await vistaar.get_vistaar_scheme_info("kcc")
    assert "Kisan Credit Card" in out
    assert "'kcc'" not in out


# ── 3. failed leg is not an empty catalogue ───────────────────────────────────

@pytest.mark.asyncio
async def test_seeker_error_raises_rather_than_returning_no_items():
    leg = vistaar.VISTAAR_LEG
    payload = {"results": {leg: None}, "errors": {leg: "timeout"}}
    clients = []
    with patch.object(vistaar.httpx, "AsyncClient", _client_factory(payload, clients)):
        with pytest.raises(vistaar.VistaarLegUnavailable):
            await vistaar._vistaar_search({"category": {"descriptor": {"code": "schemes-agri"}}})


@pytest.mark.asyncio
async def test_dead_leg_reports_mandi_prices_unavailable_not_none_found():
    """The exact regression: a failed leg used to be byte-identical to a market
    with no arrivals, so the farmer was told there were no prices."""
    leg = vistaar.VISTAAR_LEG
    payload = {"results": {leg: None}, "errors": {leg: "timeout"}}
    clients = []
    with patch.object(vistaar.httpx, "AsyncClient", _client_factory(payload, clients)):
        out = await vistaar.get_vistaar_mandi_prices(None, "Tomato")
    assert "temporarily unavailable" in out.lower()
    assert "No mandi prices were found" not in out


@pytest.mark.asyncio
async def test_dead_leg_reports_scheme_info_unavailable_not_not_found():
    leg = vistaar.VISTAAR_LEG
    payload = {"results": {leg: None}, "errors": {leg: "timeout"}}
    clients = []
    with patch.object(vistaar.httpx, "AsyncClient", _client_factory(payload, clients)):
        out = await vistaar.get_vistaar_scheme_info("kcc")
    assert "temporarily unavailable" in out.lower()
    assert "No information was found" not in out


@pytest.mark.asyncio
async def test_null_leg_without_an_errors_entry_is_still_a_failure():
    """The seeker only emits `errors` for some failures; a null on_search with
    no error entry is still not a catalogue."""
    payload = {"results": {vistaar.VISTAAR_LEG: None}}
    clients = []
    with patch.object(vistaar.httpx, "AsyncClient", _client_factory(payload, clients)):
        out = await vistaar.get_vistaar_mandi_prices(None, "Tomato")
    assert "temporarily unavailable" in out.lower()


@pytest.mark.asyncio
async def test_healthy_but_empty_leg_still_says_none_found():
    """The unavailable message must not swallow genuine misses."""
    clients = []
    with patch.object(vistaar.httpx, "AsyncClient", _client_factory(_seeker_ok([]), clients)):
        out = await vistaar.get_vistaar_mandi_prices(None, "Tomato")
    assert "No mandi prices were found" in out
    assert "temporarily unavailable" not in out.lower()


# ── 4. one timeout knob ───────────────────────────────────────────────────────

def test_the_duplicate_vistaar_timeout_knob_is_gone():
    assert not hasattr(vistaar, "VISTAAR_TIMEOUT_S"), (
        "a second, larger timeout for the same hop means the shared budget is "
        "not the one in force"
    )


@pytest.mark.asyncio
async def test_calls_run_under_the_shared_network_budget():
    from app.config import settings
    clients = []
    with patch.object(vistaar.httpx, "AsyncClient", _client_factory(_seeker_ok([]), clients)):
        await vistaar.get_vistaar_mandi_prices(None, "Tomato")
    assert clients[0].timeout == settings.amul_network_timeout_s
    assert clients[0].timeout <= 35
