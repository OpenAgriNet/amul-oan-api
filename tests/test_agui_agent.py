"""Agent intents, deterministic milk transitions, and the committed POST boundary."""
import copy
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from agents.deps import FarmerContext
from agents.tools.beckn.amul import AuthenticatedFarmerAccount
from agents.tools.milk_collection import (
    InteractiveMilk, prepare_interactive_milk, prepare_get_farmer_milk_collection_details,
)
from agents.tools.models.milk_collection import FarmerMilkCollectionResponseModel
from app.agui.milk import milk_result
from app.agui.models import ChatRunRequest
from app.agui.service import MilkStore, UIError, advance, events, new_record
from tests.test_beckn_operations import MemoryRedis

PERIOD = dict(fromdate="2026-11-01", todate="2026-11-10")
ACCOUNT = dict(option_id="A", label="Farmer A")


@pytest.fixture
def anyio_backend():
    return "asyncio"


def request(action=None, command=None, text="Milk records"):
    props = dict(version=1, source_lang="en", target_lang="en")
    if action: props["ui_action"] = action
    if command: props["command"] = command
    return ChatRunRequest.model_validate(dict(threadId="thread", runId=uuid4().hex, state={}, tools=[], context=[],
        messages=[] if action or command else [dict(id=uuid4().hex, role="user", content=text)], forwardedProps=props))


def pick(component, value):
    return request(action=dict(component_id=component["id"], revision=component["revision"], kind=component["kind"], value=value))


class Provider:
    def __init__(self, codes=("A",)):
        self.owned = [AuthenticatedFarmerAccount("U", "S", code) for code in codes]
        self.calls = []
        self.fail = False

    async def accounts(self, *args): return self.owned

    async def milk(self, account, period, *args):
        self.calls.append((account.farmer_code, period))
        if self.fail: raise RuntimeError("provider unavailable")
        return FarmerMilkCollectionResponseModel(milk=[dict(date="2026-11-01", qty=10, amount=400, fat=4, snf=8)])


async def start(provider, period=None):
    return await advance(new_record(), request(), "9000000000", provider, milk_intent=dict(period=period))


@pytest.mark.anyio
async def test_agent_intent_uses_clear_dates_and_owned_account_only():
    provider = Provider(("A", "B"))
    record, _ = await start(provider, PERIOD)
    choice = record["ui"]["components"][-1]
    assert choice["kind"] == "choice" and not provider.calls
    record, response = await advance(record, pick(choice, dict(option_id=choice["options"][1]["id"])), "9000000000", provider)
    assert provider.calls == [("B", PERIOD)]
    assert record["ui"]["components"][-1]["data"]["summary"]["total_amount_rupees"] == 400
    assert list(events(request(), response))[-1].type == "RUN_FINISHED"


@pytest.mark.anyio
async def test_unrelated_messages_preserve_pending_controls_even_at_capacity():
    provider = Provider()
    record, _ = await start(provider)
    record["ui"]["components"] *= 50
    before = copy.deepcopy(record["ui"]["components"])
    record, response = await advance(record, request(text="Weather tomorrow"), "9000000000", provider)
    assert record["ui"]["components"] == before and response["text"] == ""
    with pytest.raises(UIError) as exc:
        await advance(record, request(), "9000000000", provider, milk_intent=dict(period=PERIOD))
    assert exc.value.body["error"]["code"] == "SESSION_LIMIT" and not provider.calls


@pytest.mark.anyio
async def test_failed_lookup_requires_date_reselection_including_after_account_switch():
    provider = Provider()
    provider.fail = True
    record, response = await start(provider, PERIOD)
    assert response["error"]["error"]["code"] == "MILK_UNAVAILABLE"
    assert record["date_reselection_required"]
    record, _ = await advance(record, request(), "9000000000", provider, milk_intent=dict(period=PERIOD))
    assert record["period"] is None and record["ui"]["components"][-1]["kind"] == "date_range"
    record, _ = await advance(record, request(command=dict(kind="change_milk_account", state_revision=record["ui"]["state_revision"])), "9000000000", provider)
    choice = record["ui"]["components"][-1]
    record, _ = await advance(record, pick(choice, dict(option_id=choice["options"][0]["id"])), "9000000000", provider)
    assert record["ui"]["components"][-1]["kind"] == "date_range" and len(provider.calls) == 1
    provider.fail = False
    record, _ = await advance(record, pick(record["ui"]["components"][-1], PERIOD), "9000000000", provider)
    assert not record["date_reselection_required"] and len(provider.calls) == 2


@pytest.mark.parametrize("dates", [(None, None), ("2026-11-01", None), ("bad", "2026-11-10"), ("2026-01-01", "2026-03-01")])
def test_missing_or_invalid_dates_request_picker(dates):
    ui = InteractiveMilk()
    ui.stage(*dates)
    assert ui.request == dict(period=None)


def test_repeated_intents_coalesce_and_failed_attempt_cannot_bypass_picker():
    ui = InteractiveMilk()
    ui.stage(**PERIOD)
    ui.stage(**PERIOD)
    ui.stage("2026-11-02", "2026-11-03")
    assert ui.request == dict(period=PERIOD)
    failed = InteractiveMilk(failed=True)
    failed.stage(**PERIOD)
    assert failed.request == dict(period=None)


def displayed(count=25):
    return milk_result(ACCOUNT, PERIOD, FarmerMilkCollectionResponseModel(
        milk=[dict(date="2026-11-01" if i % 2 else "02-11-2026", qty=2, amount=10, fat=4, snf=8) for i in range(count)],
        deduction=[dict(date="2026-11-01", accountname="Feed", amount=20)]))


def test_followups_are_paginated_and_totals_are_calculated_before_paging():
    ui = InteractiveMilk(ACCOUNT, displayed())
    first = json.loads(ui.read())
    assert len(first["milk"]) == 20 and first["summary"]["total_amount_rupees"] == 250
    assert first["next_offset"] == 20 and not first["complete"]
    second = json.loads(ui.read(offset=20))
    assert second["complete"] and len(second["milk"]) == 5 and second["deductions"][0]["account_name"] == "Feed"
    filtered = json.loads(ui.read("2026-11-01", "2026-11-01"))
    assert filtered["summary"]["total_quantity_litres"] == 24
    assert "does not cover" in ui.read("2026-10-31", "2026-11-01")
    assert "No currently owned" in InteractiveMilk().read()


def test_followup_byte_limit_does_not_truncate_rows_or_provider_labels():
    data = displayed(1)
    data["deductions"][0]["account_name"] = "large" * 4000
    ui = InteractiveMilk(ACCOUNT, data)
    page = json.loads(ui.read())
    assert page["next_offset"] == 1 and not page["complete"]
    assert "too large" in ui.read(offset=1)
    assert len(ui.read().encode()) <= 16 * 1024


@pytest.mark.anyio
async def test_tool_gating_replaces_legacy_milk_only_on_interactive_farmer_turns():
    deps = FarmerContext(query="Milk", mobile="9000000000", farmer_unions=["kaira"], farmer_profile_status="found", interactive_milk=True)
    deps._milk_ui = InteractiveMilk()
    ctx, definition = SimpleNamespace(deps=deps), object()
    assert await prepare_interactive_milk(ctx, definition) is definition
    assert await prepare_get_farmer_milk_collection_details(ctx, definition) is None
    deps.persona = "doctor"
    assert await prepare_interactive_milk(ctx, definition) is None
    deps.persona, deps.interactive_milk = "farmer", False
    assert await prepare_interactive_milk(ctx, definition) is None
    assert await prepare_get_farmer_milk_collection_details(ctx, definition) is definition


@pytest.mark.anyio
async def test_real_agent_tool_schema_and_call_are_interactive_only():
    from agents.agrinet import agrinet_agent
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    deps = FarmerContext(query="Show my milk payment", mobile="9000000000", farmer_unions=["kaira"],
                         farmer_profile_status="found", interactive_milk=True)
    deps._milk_ui = InteractiveMilk()
    calls = []
    def model(messages, info):
        names = {t.name for t in info.function_tools}
        assert "request_milk_records" in names and "get_farmer_milk_collection_details" not in names
        calls.append(messages)
        if len(calls) == 1:
            return ModelResponse(parts=[ToolCallPart("request_milk_records", PERIOD, "milk-call")])
        return ModelResponse(parts=[TextPart("Requested your milk records.")])
    await agrinet_agent.run(deps.query, deps=deps, model=FunctionModel(model))
    assert len(calls) == 2 and deps._milk_ui.request == dict(period=PERIOD)


@pytest.mark.anyio
async def test_abandoned_model_attempt_discards_its_ui_intent(monkeypatch, materialized_tier):
    from app import llm_core
    from tests.test_chat_turn_contract import _Run

    deps = FarmerContext(query="Milk")
    deps._milk_ui = InteractiveMilk()
    calls = []
    class Agent:
        def iter(self, **kwargs):
            calls.append(kwargs["model"])
            assert deps._milk_ui.request is None
            if len(calls) == 1:
                deps._milk_ui.stage(**PERIOD)
                raise ConnectionError("first model unavailable")
            return _Run(["Weather answer without a milk request."])
    execution = await llm_core.context("fallback-milk")
    chain = [materialized_tier("oss", object()), materialized_tier("managed", object())]
    async def tiers(*args): return chain
    monkeypatch.setattr(type(execution), "_chain", tiers)
    output = [chunk async for chunk in execution.stream(Agent(), "Milk", message_history=[], deps=deps, new_messages=[])]
    assert len(calls) == 2 and output and deps._milk_ui.request is None


class Redis(MemoryRedis):
    def __init__(self):
        super().__init__()
        self.commits = 0
        self.lease_owned = True

    def lock(self, *args, **kwargs): return self
    async def acquire(self): return True
    async def owned(self): return self.lease_owned
    async def release(self): pass
    def pipeline(self, **kwargs): return Pipeline(self)


class Pipeline:
    def __init__(self, redis): self.redis, self.commands = redis, []
    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass
    async def watch(self, *keys): pass
    async def get(self, key): return await self.redis.get(key)
    def multi(self): pass
    def set(self, key, value, **kwargs): self.commands.append((key, value))
    def delete(self, key): self.commands.append((key, None))
    async def execute(self):
        for key, value in self.commands:
            if value is None: self.redis.values.pop(key, None)
            else: self.redis.values[key] = value
        self.redis.commits += 1


@pytest.mark.anyio
async def test_store_commits_history_and_ui_together_and_deduplicates_before_agent():
    redis, provider, req = Redis(), Provider(), request()
    store = MilkStore(redis, "test:", 7200, provider)
    calls = []
    async def agent(record, history, commit):
        calls.append(req.runId)
        return await commit(dict(intent=dict(period=PERIOD), text="Requested", message_id="agent", artifacts=[],
                                 history=[], suggestions_pending=True))
    response = await store.apply(req, "owner", "farmer", "9000000000", agent_reply=agent)
    assert redis.commits == 1 and "test:thread__SVA" in redis.values
    assert response["state"]["components"][-1]["kind"] == "chart"
    assert "test:suggestions_thread_en:pending" in redis.values
    before = dict(redis.values)
    with pytest.raises(UIError):
        await store.apply(req, "owner", "farmer", "9000000000", agent_reply=agent)
    assert len(calls) == 1 and redis.values == before


@pytest.mark.anyio
async def test_lost_lease_does_not_commit_history_or_ui():
    redis, provider = Redis(), Provider()
    redis.lease_owned = False
    store = MilkStore(redis, "test:", 7200, provider)
    async def agent(record, history, commit):
        return await commit(dict(intent=None, text="Weather", message_id="agent", artifacts=[], history=[], suggestions_pending=False))
    with pytest.raises(UIError) as exc:
        await store.apply(request(), "owner", "farmer", "9000000000", agent_reply=agent)
    assert exc.value.body["error"]["code"] == "SESSION_BUSY" and redis.commits == 0 and not redis.values


@pytest.mark.anyio
@pytest.mark.parametrize("commit_failure", [False, True])
async def test_existing_pipeline_stages_the_tool_and_finalizes_before_root_closes(monkeypatch, commit_failure):
    from app.agui.agent import agent_reply
    from app.services import chat
    from tests.test_chat_turn_contract import _Run, patch_turn

    seen = patch_turn(monkeypatch, agent_text="Requested.")
    async def context(*args): return SimpleNamespace(markdown="", found=True, unions=["kaira"], location={}), "found"
    async def empty(*args, **kwargs): return ""
    monkeypatch.setattr(chat, "_load_farmer_context", context)
    monkeypatch.setattr(chat, "get_session_shc_context", empty)
    monkeypatch.setattr(chat, "fetch_memory_context", empty)
    def run(**kwargs):
        from pydantic_ai.messages import ModelResponse, TextPart
        kwargs["deps"]._milk_ui.stage(**PERIOD)
        result = _Run(["Requested."], deps=kwargs["deps"])
        result.result = SimpleNamespace(new_messages=lambda: [ModelResponse(parts=[TextPart("Requested.")])])
        return result
    monkeypatch.setattr(chat.agrinet_agent, "iter", run)
    async def commit(reply):
        from redis.exceptions import WatchError
        assert seen["spans"][-1] != ("exit", "turn")
        assert reply["intent"] == dict(period=PERIOD) and reply["history"] is not None
        if commit_failure:
            raise WatchError("changed state")
        return dict(error=None)
    if commit_failure:
        from redis.exceptions import WatchError
        with pytest.raises(WatchError):
            await agent_reply(new_record(), [], commit, request=request(),
                              user_info=dict(sub="owner", phone="9000000000"), provider=Provider(), trace_id=uuid4().hex)
        assert seen["spans"][-1] == ("exit", "turn") and seen["history_writes"] == []
        return
    response = await agent_reply(new_record(), [], commit, request=request(),
                                 user_info=dict(sub="owner", phone="9000000000"), provider=Provider(), trace_id=uuid4().hex)
    assert response["background_tasks"] and seen["history_writes"] == []
    assert seen["spans"][-1] == ("exit", "turn")


@pytest.mark.anyio
async def test_post_round_trip_keeps_ordinary_chat_and_direct_controls(monkeypatch):
    import httpx
    from fastapi import FastAPI
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
    from pydantic_core import to_jsonable_python
    from app.agui import agent
    from app.routers import agui
    from app.services import chat

    monkeypatch.setattr(agui.settings, "environment", "development")
    monkeypatch.setattr(agui.settings, "chat_agui_preview_enabled", True)
    monkeypatch.setattr(agui.settings, "chat_agui_enabled", True)
    monkeypatch.setattr(chat, "get_langfuse_client", None)
    provider, redis = Provider(), Redis()
    store = MilkStore(redis, "test:", 7200, provider)
    agent_calls = []
    async def reply(record, history, commit, *, request, **kwargs):
        query = request.messages[0].content
        agent_calls.append(query)
        messages = [*history, ModelRequest(parts=[UserPromptPart(query)]), ModelResponse(parts=[TextPart("Answer")])]
        return await commit(dict(text="Agent answer", message_id=uuid4().hex, artifacts=[],
                                 intent=dict(period=None) if query == "Milk" else None,
                                 history=to_jsonable_python(messages), suggestions_pending=False))
    monkeypatch.setattr(agent, "agent_reply", reply)
    app = FastAPI()
    app.include_router(agui.router, prefix="/api")
    app.dependency_overrides[agui.get_current_user] = lambda: dict(sub="owner", phone="9000000000")
    app.dependency_overrides[agui.get_milk_store] = lambda: store
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        async def post(req):
            response = await client.post("/api/chat/", json=req.model_dump())
            assert response.status_code == 200 and response.headers["X-Chat-Format"] == "agui-v1"
            return [json.loads(line[5:]) for line in response.text.splitlines() if line.startswith("data:")]
        weather = await post(request(text="Weather"))
        assert next(e["snapshot"]["amul_ui"] for e in weather if e["type"] == "STATE_SNAPSHOT")["components"] == []
        milk = await post(request(text="Milk"))
        ui = next(e["snapshot"]["amul_ui"] for e in milk if e["type"] == "STATE_SNAPSHOT")
        result = await post(pick(ui["components"][-1], PERIOD))
        ui = next(e["snapshot"]["amul_ui"] for e in result if e["type"] == "STATE_SNAPSHOT")
        assert ui["components"][-1]["kind"] == "chart"
        assert agent_calls == ["Weather", "Milk"] and len(provider.calls) == 1
