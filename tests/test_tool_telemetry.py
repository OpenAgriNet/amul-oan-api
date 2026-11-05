"""Tool and API steps say how they went (amul-telemetry telemetry/turn.v1.yaml).

A TOOL step carries ``amul.tool_status`` and, on a failure, level ERROR with the reason:
``reporting`` writes them on the current span (pydantic-ai's tool span) as the call
ends, from what the tool reported through ``report_tool``. An API step under a tool
sends ``ok`` and ``status_code``, level ERROR when the call did not succeed.
"""
import asyncio
import contextlib
import os
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import httpx
import pytest
from pydantic_ai import ModelRetry, Tool

from agents.tools import ai_call, bonus, bonus_backend, cattle_trade_backend, loan, telemetry, union_schemes, vet_offices, vistaar
from agents.tools import loan_eligibility as le
from agents.tools.beckn import network as beckn_network
from agents.tools.models.ai_call import AISpecies
from agents.tools.models.bonus import FarmerBonusAmountRequestModel
from agents.tools.registry import TOOLS
from agents.tools.search import search_documents
from agents.tools.telemetry import report_tool, reporting
from agents.doctor import DOCTOR_TOOLS
from app import observability


class _Client:
    """Records what a tool's span is told, as the Langfuse client would be asked."""

    def __init__(self):
        self.updates = []

    def update_current_span(self, **kwargs):
        self.updates.append(kwargs)


@pytest.fixture
def span(monkeypatch):
    client = _Client()
    monkeypatch.setattr(telemetry, "get_langfuse_client", lambda: client)
    return client


def _status(span):
    """The one update the call made: its status, cache, level and message."""
    (update,) = span.updates
    return (
        update["metadata"].get("amul.tool_status"),
        update["metadata"].get("amul.cache"),
        update.get("level"),
        update.get("status_message"),
    )


# ── the wrapper ──────────────────────────────────────────────────────────────


def test_a_tool_that_answered_is_a_success(span):
    @reporting
    async def tool(question: str) -> str:
        return "answer"

    assert asyncio.run(tool("q")) == "answer"
    assert _status(span) == ("success", None, None, None)


@pytest.mark.parametrize("result", ["", "  ", None, [], {}])
def test_a_tool_that_returned_nothing_is_empty(span, result):
    @reporting
    def tool():
        return result

    tool()
    assert _status(span) == ("empty", None, None, None)


@pytest.mark.parametrize(
    ("status", "message", "recorded"),
    [
        ("no_result", "no bonus records", ("no_result", None, None, None)),
        ("unavailable", "provider down", ("unavailable", None, "ERROR", "provider down")),
        ("error", "bad date range", ("error", None, "ERROR", "bad date range")),
        ("unavailable", None, ("unavailable", None, "ERROR", "unavailable")),
    ],
)
def test_a_reported_status_wins_and_a_failure_marks_the_span(span, status, message, recorded):
    @reporting
    async def tool() -> str:
        report_tool(status, message)
        return "told the model"

    asyncio.run(tool())
    assert _status(span) == recorded


def test_a_tool_that_raised_is_an_error_with_the_reason(span):
    @reporting
    async def tool() -> str:
        raise ModelRetry("INVALID_QUERY: say more")

    with pytest.raises(ModelRetry):
        asyncio.run(tool())
    assert _status(span) == ("error", None, "ERROR", "ModelRetry: INVALID_QUERY: say more")


def test_a_raise_overrides_what_the_tool_reported_and_the_message_is_bounded(span):
    @reporting
    def tool():
        report_tool("no_result", "nothing")
        raise RuntimeError("x" * 400)

    with pytest.raises(RuntimeError):
        tool()
    status, _, level, message = _status(span)
    assert (status, level, len(message)) == ("error", "ERROR", 300)


def test_the_cache_state_rides_along(span):
    @reporting
    async def tool() -> str:
        report_tool(cache="hit")
        return "cached"

    asyncio.run(tool())
    assert _status(span) == ("success", "hit", None, None)


def test_the_last_report_wins_and_the_report_does_not_leak_to_the_next_call(span):
    @reporting
    async def first() -> str:
        report_tool("unavailable", "first try")
        report_tool("no_result", "then nothing")
        return "x"

    @reporting
    async def second() -> str:
        return "y"

    asyncio.run(first())
    asyncio.run(second())
    assert [u["metadata"]["amul.tool_status"] for u in span.updates] == ["no_result", "success"]
    assert "level" not in span.updates[1]


def test_reporting_outside_a_reporting_tool_does_nothing(span):
    report_tool("unavailable", "no span to tell")
    assert span.updates == [] and telemetry._report.get() is None


@pytest.mark.parametrize("call", [lambda: report_tool("bad"), lambda: report_tool(cache="warm")])
def test_a_value_the_registry_does_not_list_is_refused(call):
    with pytest.raises(ValueError):
        call()


def test_without_a_langfuse_client_the_tool_runs_as_before(monkeypatch):
    monkeypatch.setattr(telemetry, "get_langfuse_client", lambda: None)

    @reporting
    async def tool() -> str:
        report_tool("unavailable", "down")
        return "answer"

    assert asyncio.run(tool()) == "answer"


def test_a_langfuse_failure_never_reaches_the_tool(monkeypatch):
    class _Broken:
        def update_current_span(self, **kwargs):
            raise RuntimeError("otel down")

    monkeypatch.setattr(telemetry, "get_langfuse_client", lambda: _Broken())

    @reporting
    async def tool() -> str:
        return "answer"

    assert asyncio.run(tool()) == "answer"


def test_every_registered_tool_reports_and_keeps_its_schema():
    """Chat's registry (and the doctor's) wraps every tool; the wrapper changes nothing
    the model sees: name, description, parameters, whether it takes the context."""
    assert TOOLS and DOCTOR_TOOLS
    for tool in [*TOOLS, *DOCTOR_TOOLS]:
        wrapped = tool.function
        assert getattr(wrapped, "__wrapped__", None) is not None, tool.name
        plain = Tool(wrapped.__wrapped__, takes_ctx=tool.takes_ctx, prepare=tool.prepare)
        assert (plain.name, plain.description, plain.function_schema.json_schema) == (
            tool.name, tool.description, tool.function_schema.json_schema,
        )
        assert asyncio.iscoroutinefunction(wrapped) == asyncio.iscoroutinefunction(wrapped.__wrapped__)


# ── the tools, at their failure sites ────────────────────────────────────────


def _deps(**fields):
    return SimpleNamespace(deps=SimpleNamespace(**fields), tool_call_id="call-1")


def test_search_documents_refuses_an_empty_query_as_an_error(span):
    with pytest.raises(ModelRetry):
        asyncio.run(reporting(search_documents)("   "))
    status, _, level, message = _status(span)
    assert (status, level) == ("error", "ERROR") and message.startswith("ModelRetry: INVALID_QUERY")


def test_the_vet_directory_that_did_not_load_is_unavailable(span, monkeypatch):
    monkeypatch.setattr(vet_offices, "_load_offices", lambda: ())

    asyncio.run(reporting(vet_offices.find_nearby_vet_offices)(_deps(farmer_village="", farmer_district="")))
    assert _status(span) == ("unavailable", None, "ERROR", "the office directory did not load")


def test_a_scheme_code_that_matches_nothing_is_no_result(span):
    asyncio.run(reporting(vistaar.get_vistaar_scheme_info)("something else"))
    assert _status(span) == ("no_result", None, None, "no central scheme matched")[:1] + (None, None, None)


@pytest.mark.parametrize(
    ("outcome", "recorded"),
    [
        (Exception("boom"), ("unavailable", "ERROR", "Bharat Vistaar weather: Exception: boom")),
        (([], SimpleNamespace(town="Anand")), ("no_result", None, None)),
        (([{"descriptor": {"name": "Day 1"}}], SimpleNamespace(town="Anand")), ("success", None, None)),
    ],
    ids=["raised", "no-items", "forecast"],
)
def test_the_weather_tool_tells_a_failed_leg_from_an_empty_forecast(span, monkeypatch, outcome, recorded):
    where = SimpleNamespace(location=SimpleNamespace(key="anand", display="Anand"), assumed=False, source="farmer")

    async def _resolve(ctx, location):
        return where, None

    async def _search(build, where_):
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(vistaar, "_resolve_search_location", _resolve)
    monkeypatch.setattr(vistaar, "_search_candidates", _search)

    asyncio.run(reporting(vistaar.get_vistaar_weather)(_deps(session_id="s")))
    status, _, level, message = _status(span)
    assert (status, level, message) == recorded


def test_a_place_the_tool_does_not_cover_is_no_result(span, monkeypatch):
    async def _resolve(ctx, location):
        return None, "That place is not covered."

    monkeypatch.setattr(vistaar, "_resolve_search_location", _resolve)

    asyncio.run(reporting(vistaar.get_vistaar_mandi_prices)(_deps(session_id="s"), "Onion", location="Nowhere"))
    assert _status(span)[0] == "no_result"


def test_a_bonus_lookup_without_a_signed_in_mobile_is_an_error(span):
    asyncio.run(reporting(bonus.get_farmer_bonus_amount)(_deps(mobile="")))
    assert _status(span) == ("error", None, "ERROR", "no signed-in mobile")


@pytest.mark.parametrize(
    ("outcome", "recorded"),
    [(le.DISABLED, ("unavailable", "ERROR")), (le.ERROR, ("unavailable", "ERROR")), (le.NO_PHONE, ("error", "ERROR")), (le.ELIGIBLE_OFFER, ("success", None))],
)
def test_the_loan_tool_reports_a_service_it_could_not_use(span, monkeypatch, outcome, recorded):
    async def _evaluate(**kwargs):
        return SimpleNamespace(outcome=outcome, phone=None, code=None, sms_status=None, loan_amount=None, milk_threshold=None)

    monkeypatch.setattr(le, "evaluate_and_issue", _evaluate)

    asyncio.run(reporting(loan.check_loan_eligibility)(_deps(mobile=None, farmer_accounts=[], session_id="s")))
    assert _status(span)[::2] == recorded


def test_union_schemes_without_a_union_is_an_error_and_a_failed_discovery_unavailable(span, monkeypatch):
    monkeypatch.setattr(union_schemes.settings, "scheme_require_union_auth", True)
    asyncio.run(reporting(union_schemes.get_union_scheme_data)(_deps(farmer_unions=[]), "insurance"))
    assert _status(span) == ("error", None, "ERROR", "farmer union not in the context")

    async def _discover(query, union=None):
        raise httpx.ConnectError("seeker down")

    span.updates.clear()
    monkeypatch.setattr(union_schemes, "network_union_schemes", _discover)
    asyncio.run(reporting(union_schemes.get_union_scheme_data)(_deps(farmer_unions=["banas"]), "insurance"))
    assert _status(span) == ("unavailable", None, "ERROR", "scheme discovery: ConnectError: seeker down")


@pytest.mark.parametrize(
    ("results", "errors", "recorded"),
    [
        ({}, {beckn_network.VET_LEG: "timeout"}, ("unavailable", "ERROR")),
        ({beckn_network.VET_LEG: {"message": {"catalog": {"providers": []}}}}, {}, ("no_result", None)),
    ],
    ids=["leg-failed", "no-items"],
)
def test_the_document_search_tells_a_dead_leg_from_an_empty_catalog(span, monkeypatch, results, errors, recorded):
    async def _legs(queries, intents=None):
        return results, errors

    monkeypatch.setattr(beckn_network, "_seeker_search_per_leg", _legs)

    asyncio.run(reporting(search_documents)("mastitis treatment"))
    assert _status(span)[::2] == recorded


# ── the API steps ────────────────────────────────────────────────────────────


class _Observation:
    def __init__(self, name, kwargs):
        self.name, self.kwargs, self.updates = name, kwargs, []

    def update(self, **kwargs):
        self.updates.append(kwargs)


def _observations(monkeypatch, module):
    opened = []

    @contextlib.contextmanager
    def _start(name, **kwargs):
        observation = _Observation(name, kwargs)
        opened.append(observation)
        yield observation

    monkeypatch.setattr(module, "start_observation", _start)
    return opened


@pytest.mark.parametrize(
    ("ok", "recorded"),
    [
        (True, [{"output": {"ok": True, "status_code": 200, "rows": 3}}]),
        (False, [{"output": {"ok": False, "status_code": 503, "rows": 3}, "level": "ERROR", "status_message": "HTTP 503"}]),
    ],
)
def test_record_api_result_sends_ok_the_status_code_and_the_failure(ok, recorded):
    observation = _Observation("api", {})
    observability.record_api_result(observation, ok=ok, status_code=200 if ok else 503, reason="HTTP 503", rows=3)
    assert observation.updates == recorded


def test_record_api_result_without_an_observation_or_a_working_one_is_quiet():
    class _Broken:
        def update(self, **kwargs):
            raise RuntimeError("gone")

    observability.record_api_result(None, ok=True)
    observability.record_api_result(_Broken(), ok=False, reason="x")


class _Http:
    """httpx.AsyncClient as the backends use it: one GET, answered or failing."""

    response = None
    error = None

    def __init__(self, timeout):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, params=None, headers=None):
        if self.error is not None:
            raise self.error
        return self.response


def _bonus_request():
    return FarmerBonusAmountRequestModel(unionCode="0001", societyCode="2004", farmerCode="0001")


@pytest.mark.parametrize(
    ("response", "error", "result", "recorded"),
    [
        (httpx.Response(200, json=[], request=httpx.Request("GET", "https://x")), None, [], {"output": {"ok": True, "status_code": 200}}),
        (httpx.Response(500, text="down", request=httpx.Request("GET", "https://x")), None, None,
         {"output": {"ok": False, "status_code": 500}, "level": "ERROR", "status_message": "HTTP 500"}),
        (None, httpx.ReadTimeout("slow"), None, {"output": {"ok": False}, "level": "ERROR", "status_message": "ReadTimeout: slow"}),
    ],
    ids=["200", "500", "timeout"],
)
def test_the_bonus_api_step_sends_ok_and_the_status_code(monkeypatch, response, error, result, recorded):
    _Http.response, _Http.error = response, error
    monkeypatch.setattr(bonus_backend.httpx, "AsyncClient", _Http)
    opened = _observations(monkeypatch, bonus_backend)

    assert asyncio.run(bonus_backend.get_farmer_bonus_amount_api(_bonus_request(), "token")) == result
    (observation,) = opened
    assert (observation.name, observation.updates) == ("get_farmer_bonus_amount_api", [recorded])


def test_the_cattle_rates_api_step_marks_a_failed_call(monkeypatch):
    _Http.response, _Http.error = httpx.Response(502, text="bad gateway", request=httpx.Request("GET", "https://x")), None
    monkeypatch.setattr(cattle_trade_backend.httpx, "AsyncClient", _Http)
    opened = _observations(monkeypatch, cattle_trade_backend)

    assert asyncio.run(cattle_trade_backend.get_cattle_trade_rates_api("token")) is None
    (observation,) = opened
    assert observation.updates == [{"output": {"ok": False, "status_code": 502}, "level": "ERROR", "status_message": "HTTP 502"}]


# ── the booking step ─────────────────────────────────────────────────────────

SPECIES = next(iter(AISpecies))


def _book(monkeypatch, outcome):
    """One network booking answered with ``outcome`` (a result, or an exception to raise),
    with the per-session guard off; returns the booking step as it was told."""
    async def _confirm(*args, **kwargs):
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(beckn_network, "network_create_ai_call_result", _confirm)
    opened = _observations(monkeypatch, ai_call)
    told = asyncio.run(ai_call._book_via_network("159", "00002", "5058", "TECH", SPECIES, "s1", {"union_code": "159"}, guard=False))
    (observation,) = opened
    return told, observation


def test_a_booked_visit_is_an_ok_api_step(monkeypatch):
    told, step = _book(monkeypatch, beckn_network.NetworkBookingResult(True, "T1", "Booked. Ticket: T1"))

    assert told == "Booked. Ticket: T1"
    assert (step.name, step.kwargs.get("as_type", "span")) == ("ai_call_booking", "span")
    assert step.updates == [{"output": {"ok": True, "ticket_number": "T1", "message": "Booked. Ticket: T1"}}]


def test_a_booking_the_network_refused_is_a_failed_api_step(monkeypatch):
    refused = beckn_network.NetworkBookingResult(False, None, "Booking failed on the network: rejected", authoritative_no_booking=True)
    told, step = _book(monkeypatch, refused)

    assert told == refused.message
    assert step.updates == [{
        "output": {"ok": False, "authoritative_no_booking": True, "message": refused.message},
        "level": "ERROR", "status_message": refused.message,
    }]


def test_a_booking_request_that_raised_is_a_failed_api_step(monkeypatch):
    told, step = _book(monkeypatch, httpx.ConnectError("bap down"))

    assert told == ai_call.NOT_SENT_MESSAGE
    (update,) = step.updates
    assert update["output"] == {"ok": False, "pre_send_failure": True, "refused": False, "message": ai_call.NOT_SENT_MESSAGE}
    assert (update["level"], update["status_message"]) == ("ERROR", "booking request: ConnectError: bap down")



# ── a tool that reports its failure, then raises ─────────────────────────────


def test_a_failure_the_tool_reported_survives_the_raise(span):
    @reporting
    async def tool() -> str:
        report_tool("unavailable", "document search: ConnectError: seeker down")
        raise ModelRetry("Error searching documents, please try again")

    with pytest.raises(ModelRetry):
        asyncio.run(tool())
    assert _status(span) == ("unavailable", None, "ERROR", "document search: ConnectError: seeker down")


def test_a_document_search_that_raised_keeps_the_upstream_reason(span, monkeypatch):
    from agents.tools import search as chat_search

    async def _down(query, top_k):
        raise httpx.ConnectError("seeker down")

    monkeypatch.setattr(chat_search, "network_search_documents", _down)
    with pytest.raises(ModelRetry):
        asyncio.run(reporting(search_documents)("mastitis treatment"))
    assert _status(span) == ("unavailable", None, "ERROR", "document search: ConnectError: seeker down")


def test_milk_with_one_account_failing_answers_and_is_unavailable(span, monkeypatch):
    from agents.tools import milk_collection as chat_milk
    from agents.tools.models.milk_collection import FarmerMilkCollectionResponseModel

    accounts = [SimpleNamespace(union_code="2021", society_code=str(i), farmer_code="1") for i in (1, 2)]

    async def _farmers(*args, **kwargs):
        return ["farmer"]

    async def _fetch(account, **kwargs):
        if account.society_code == "2":
            raise httpx.ReadTimeout("slow")
        return FarmerMilkCollectionResponseModel(result="success", milk=[], deduction=[])

    monkeypatch.setattr(chat_milk, "fetch_authenticated_farmers", _farmers)
    monkeypatch.setattr(chat_milk, "authenticated_accounts", lambda farmers: accounts)
    monkeypatch.setattr(chat_milk, "fetch_milk_collection", _fetch)
    told = asyncio.run(reporting(chat_milk.get_farmer_milk_collection_details)(
        _deps(mobile="9000000000", session_id="s"), "2026-10-01", "2026-10-07",
    ))
    assert told.startswith("Farmer milk collection details fetched successfully")
    assert _status(span) == ("unavailable", None, "ERROR", "milk collection provider failed for 1 of 2 accounts")


# ── the voice tools ──────────────────────────────────────────────────────────


def _reporting_codes():
    async def _a():
        return None

    return {reporting(lambda: None).__code__, reporting(_a).__code__}


def test_every_voice_tool_reports_and_keeps_its_schema():
    """Voice's registry wraps every tool too, under the nudge, and the model sees the same tool."""
    from agents.voice.tools import BASE_TOOLS, SIGNED_IN_FARMER_TOOLS

    codes = _reporting_codes()
    for tool in [*BASE_TOOLS, *SIGNED_IN_FARMER_TOOLS]:
        chain, fn = [], tool.function
        while fn is not None:
            chain.append(fn)
            fn = getattr(fn, "__wrapped__", None)
        assert any(f.__code__ in codes for f in chain), tool.name
        plain = Tool(chain[-1], takes_ctx=tool.takes_ctx, prepare=tool.prepare)
        assert (plain.name, plain.description, plain.function_schema.json_schema) == (
            tool.name, tool.description, tool.function_schema.json_schema,
        )


def _voice_accounts(*societies):
    from agents.deps import FarmerAccount

    return [FarmerAccount(union_code="2021", society_code=s, farmer_code="NA0002") for s in societies]


def test_voice_bonus_failures_are_unavailable(span, monkeypatch):
    from agents.voice.tools import bonus as voice_bonus

    async def _api(request, token):
        return None if request.society_code == "2" else []

    monkeypatch.setenv("PASHUGPT_TOKEN", "t")
    monkeypatch.setattr(voice_bonus, "get_farmer_bonus_amount_api", _api)
    ctx = _deps(mobile="9000000000", farmer_accounts=_voice_accounts("1", "2"))
    asyncio.run(reporting(voice_bonus.get_farmer_bonus_amount)(ctx))
    assert _status(span) == ("unavailable", None, "ERROR", "bonus provider failed for 1 of 2 accounts")

    span.updates.clear()
    monkeypatch.delenv("PASHUGPT_TOKEN")
    asyncio.run(reporting(voice_bonus.get_farmer_bonus_amount)(ctx))
    assert _status(span) == ("unavailable", None, "ERROR", "PASHUGPT_TOKEN is not set")


def test_voice_milk_provider_failure_is_unavailable_with_the_reason(span, monkeypatch):
    from agents.voice.tools import milk_collection as voice_milk

    async def _fetch(account, **kwargs):
        raise httpx.ConnectError("bap down")

    monkeypatch.setattr(voice_milk, "fetch_milk_collection", _fetch)
    told = asyncio.run(voice_milk.fetch_milk_summary_for_accounts(_voice_accounts("1"), "2026-10-01", "2026-10-07"))
    assert told == "Milk collection lookup failed. Unable to fetch details at the moment."
    assert span.updates == []  # outside a tool (outbound calls) nothing is told

    told = asyncio.run(reporting(voice_milk.fetch_milk_summary_for_accounts)(_voice_accounts("1"), "2026-10-01", "2026-10-07"))
    assert _status(span) == ("unavailable", None, "ERROR", "milk collection provider: ConnectError: bap down")


@pytest.mark.parametrize(("outcome", "recorded"), [("disabled", ("unavailable", "ERROR")), ("no_phone", ("error", "ERROR"))])
def test_voice_loan_reports_a_service_it_could_not_use(span, monkeypatch, outcome, recorded):
    from agents.voice.services import loan_eligibility as voice_le
    from agents.voice.tools import loan as voice_loan

    async def _evaluate(**kwargs):
        return SimpleNamespace(outcome=outcome, phone=None, code=None, sms_status=None, loan_amount=None, milk_threshold=None)

    async def _in_scope():
        return True

    monkeypatch.setattr(voice_le, "evaluate_and_issue", _evaluate)
    ctx = _deps(mobile=None, farmer_accounts=[], session_id="s", ensure_in_scope=_in_scope)
    asyncio.run(reporting(voice_loan.check_loan_eligibility)(ctx))
    assert _status(span)[::2] == recorded


def test_voice_union_schemes_cache_failure_is_unavailable(span, monkeypatch):
    from agents.voice.tools import union_schemes as voice_schemes

    async def _cache(union):
        raise voice_schemes.SchemeCacheError("redis down")

    monkeypatch.setattr(voice_schemes, "get_cached_scheme_records_for_union", _cache)
    asyncio.run(reporting(voice_schemes.get_union_scheme_data)(_deps(farmer_unions=["banas"])))
    assert _status(span) == ("unavailable", None, "ERROR", "scheme cache: SchemeCacheError: redis down")


def test_voice_search_that_could_not_reach_marqo_is_unavailable(span, monkeypatch):
    from agents.voice.tools import search as voice_search

    def _down(*args, **kwargs):
        raise ConnectionError("marqo down")

    monkeypatch.setenv("MARQO_ENDPOINT_URL", "http://marqo.invalid")
    monkeypatch.setenv("MARQO_SEARCH_MODE", "tensor")
    monkeypatch.setattr(voice_search, "_get_index_capabilities_sync", lambda url, index: {"exists": False})
    monkeypatch.setattr(voice_search, "_marqo_search_sync", _down)
    with pytest.raises(ModelRetry):
        asyncio.run(reporting(voice_search.search_documents)(_deps(), "mastitis treatment"))
    assert _status(span) == ("unavailable", None, "ERROR", "document search: ConnectionError: marqo down")


def test_voice_vet_directory_that_did_not_load_is_unavailable(span, monkeypatch):
    from agents.voice.tools import vet_offices as voice_vet

    monkeypatch.setattr(voice_vet, "_load_offices", lambda: ())
    asyncio.run(reporting(voice_vet.find_nearby_vet_offices)(_deps(farmer_village="", farmer_district="")))
    assert _status(span) == ("unavailable", None, "ERROR", "the office directory did not load")


def test_voice_booking_with_an_invented_code_is_an_error(span):
    from agents.voice.tools import ai_call as voice_ai_call

    async def _in_scope():
        return True

    ctx = _deps(session_id="s", farmer_unions=[], farmer_accounts=None, ensure_in_scope=_in_scope)
    asyncio.run(reporting(voice_ai_call.create_ai_call)(ctx, "MISSING", "1", "1", "TECH1", SPECIES))
    status, _, level, message = _status(span)
    assert (status, level) == ("error", "ERROR") and message.startswith("invalid ")
