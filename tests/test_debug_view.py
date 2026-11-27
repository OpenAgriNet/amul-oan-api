"""The debug view (app/debug_view.py, GET /api/debug/turns/): real pydantic-ai spans, fake Redis. No network."""

import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest  # noqa: E402
from fastapi import BackgroundTasks, FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator  # noqa: E402
from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.models.instrumented import InstrumentationSettings  # noqa: E402
from pydantic_ai.models.test import TestModel  # noqa: E402

from app import debug_view  # noqa: E402
from app.auth.jwt_auth import get_chat_user  # noqa: E402
from app.config import settings  # noqa: E402
from app.models.requests import ChatRequest  # noqa: E402
from app.routers import chat as chat_router  # noqa: E402
from app.routers import debug as debug_router  # noqa: E402
from app.routers import ui_flags  # noqa: E402

TRACE = 0x0123456789ABCDEF0123456789ABCDEF
TRACE_ID = format(TRACE, "032x")
JWT = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJhbm9uLTEifQ.c2lnbmF0dXJlLXZhbHVl"


class FixedTrace(RandomIdGenerator):
    def generate_trace_id(self) -> int:
        return TRACE


class FakeAsyncRedis:
    def __init__(self):
        self.values = {}
        self.lists = {}

    async def set(self, key, value, ex=None):
        self.values[key] = value

    async def get(self, key):
        return self.values.get(key)

    async def lrange(self, key, start, end):
        return list(self.lists.get(key, []))

    async def exists(self, key):
        return int(key in self.values)


class FakeSyncRedis:
    """What RedisSink.write uses: a pipeline that applies to the same store as FakeAsyncRedis."""

    def __init__(self, store: FakeAsyncRedis):
        self.store = store
        self.expires = {}

    def pipeline(self, transaction=False):
        ops = []
        store, expires = self.store, self.expires
        return SimpleNamespace(
            rpush=lambda key, value: ops.append(("rpush", key, value)),
            expire=lambda key, ttl: ops.append(("expire", key, ttl)),
            set=lambda key, value, ex=None: ops.append(("set", key, value)),
            execute=lambda: [
                store.lists.setdefault(op[1], []).append(op[2]) if op[0] == "rpush"
                else expires.__setitem__(op[1], op[2]) if op[0] == "expire"
                else store.values.__setitem__(op[1], op[2])
                for op in ops
            ],
        )


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(settings, "debug_view_enabled", True)
    monkeypatch.setattr(settings, "environment", "development")


def run_turn(*processors):
    """One turn as the app makes it: a root span, an agent with a search tool, an extra span with
    things that must not be kept, and one that fails. Returns the finished spans by name."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider(id_generator=FixedTrace())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    for processor in processors:
        provider.add_span_processor(processor)
    agent = Agent(
        TestModel(),
        name="amul",
        instructions="THE SYSTEM PROMPT",
        instrument=InstrumentationSettings(tracer_provider=provider),
    )

    @agent.tool_plain
    def search_documents(query: str) -> list[dict]:
        return [{"title": f"Green fodder {i}", "score": 0.9 - i / 10, "text": "x" * 3000} for i in range(60)]

    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("turn"):
        with tracer.start_as_current_span("moderation") as step:
            step.set_attribute("api_key", "sk-live-123")
            step.set_attribute("langfuse.trace.user.id", "9876543210")
            step.set_attribute("note", f"called with Bearer abcdefghijklmnop and {JWT}")
            step.set_attribute("big", "y" * 10_000)
            step.set_attribute("gen_ai.usage.input_tokens", 12)
        agent.run_sync("which fodder?")
        with pytest.raises(ValueError):
            with tracer.start_as_current_span("translation"):
                raise ValueError("translator down")
    return {span.name: span for span in exporter.get_finished_spans()}


def test_on_only_where_asked_and_never_outside_development(monkeypatch):
    for enabled, environment, expected in [
        (True, "development", True),
        (True, "production", False),
        (True, "staging", False),
        (False, "development", False),
    ]:
        monkeypatch.setattr(settings, "debug_view_enabled", enabled)
        monkeypatch.setattr(settings, "environment", environment)
        assert debug_view.active() is expected, (enabled, environment)


def test_secrets_and_user_ids_are_hidden_but_usage_counts_are_not():
    for key in ["api_key", "Authorization", "langfuse.trace.user.id", "x.user_id", "headers.cookie", "client_secret", "access_token", "db.password"]:
        assert debug_view.is_hidden_key(key), key
    for key in ["gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens", "tool_arguments", "gen_ai.tool.name", "session.id", "max_tokens"]:
        assert not debug_view.is_hidden_key(key), key


def test_a_tool_keeps_its_arguments_and_a_readable_result():
    spans = run_turn()
    tool = debug_view.compact(spans["running tool"])
    attributes = tool["attributes"]
    assert attributes["gen_ai.tool.name"] == "search_documents"
    assert json.loads(attributes["tool_arguments"]) == {"query": "a"}
    result = json.loads(attributes["tool_response"])  # pruned, still JSON
    assert result[0]["title"] == "Green fodder 0"
    assert len(attributes["tool_response"]) <= debug_view.MAX_VALUE_CHARS
    assert tool["parent"] and tool["status"] == "ok" and tool["end_ns"] >= tool["start_ns"]


def _search_result(count: int) -> str:
    """What the document search returns (agents/tools/beckn/network.py), with long texts."""
    items = []
    for i in range(1, count + 1):
        text = f"## Treatment:\nIsolate the animal.\n\n1. Wash the udder\n2. Call the vet\n{'Long advice. ' * 60}"
        items.append(f"{i}. Document {i}\n{text}" + (f"\n(source: book-{i}.pdf)" if i % 2 else ""))
    return "\n\n".join(items)


def test_a_search_tool_keeps_every_document_even_when_its_result_is_cut():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with provider.get_tracer("test").start_as_current_span("running tool") as span:
        span.set_attribute("gen_ai.tool.name", "search_documents")
        span.set_attribute("tool_response", _search_result(12))
    record = debug_view.compact(exporter.get_finished_spans()[0])
    documents = json.loads(record["attributes"][debug_view.DOCUMENTS_KEY])
    assert [d["title"] for d in documents] == [f"Document {i}" for i in range(1, 13)]
    assert [d["source"] for d in documents[:3]] == ["book-1.pdf", None, "book-3.pdf"]
    assert "(+" in record["attributes"]["tool_response"]  # the text itself was cut


def test_result_documents_reads_only_the_numbered_items():
    assert debug_view.result_documents("No relevant documents were found on the network for this query.") == []
    assert [d["title"] for d in debug_view.result_documents(_search_result(2))] == ["Document 1", "Document 2"]
    many = debug_view.result_documents(_search_result(debug_view.MAX_LIST_ITEMS + 5))
    assert len(many) == debug_view.MAX_LIST_ITEMS


def test_model_calls_keep_model_usage_and_output_but_not_the_prompt_or_history():
    spans = run_turn()
    model = debug_view.compact(next(s for name, s in spans.items() if name.startswith("chat ")))
    attributes = model["attributes"]
    assert attributes["gen_ai.request.model"] == "test"
    assert attributes["gen_ai.usage.input_tokens"] > 0
    assert "gen_ai.output.messages" in attributes
    for left_out in ["gen_ai.input.messages", "gen_ai.system_instructions", "gen_ai.tool.definitions", "model_request_parameters"]:
        assert left_out not in attributes
    run = debug_view.compact(spans["agent run"])
    assert "pydantic_ai.all_messages" not in run["attributes"] and "gen_ai.system_instructions" not in run["attributes"]
    assert "THE SYSTEM PROMPT" not in json.dumps([debug_view.compact(s) for s in spans.values()])


def test_secrets_are_dropped_tokens_masked_and_long_values_cut():
    step = debug_view.compact(run_turn()["moderation"])
    attributes = step["attributes"]
    assert "api_key" not in attributes and "langfuse.trace.user.id" not in attributes
    assert attributes["gen_ai.usage.input_tokens"] == 12
    assert "abcdefghijklmnop" not in attributes["note"] and JWT not in attributes["note"]
    assert "[masked token]" in attributes["note"] and "Bearer [masked]" in attributes["note"]
    assert attributes["big"].endswith(f"(+{10_000 - debug_view.MAX_VALUE_CHARS} chars)")


def test_a_failed_step_says_why():
    step = debug_view.compact(run_turn()["translation"])
    assert step["status"] == "error"
    assert step["error"] == "ValueError: translator down"


def test_only_watched_turns_are_kept_and_the_root_ends_the_turn():
    kept = []
    turns = debug_view._Watched()
    run_turn(debug_view.DebugSpanProcessor(lambda trace, record, ends: kept.append((trace, record["name"], ends)), turns))
    assert kept == []  # not watched: nothing
    turns.add(TRACE_ID)
    run_turn(debug_view.DebugSpanProcessor(lambda trace, record, ends: kept.append((trace, record["name"], ends)), turns))
    assert {trace for trace, _, _ in kept} == {TRACE_ID}
    assert [name for _, name, ends in kept if ends] == ["turn"]
    assert {"moderation", "agent run", "running tool", "translation"} <= {name for _, name, _ in kept}


def test_a_turn_keeps_at_most_max_spans(monkeypatch):
    monkeypatch.setattr(debug_view, "MAX_SPANS", 3)
    kept = []
    turns = debug_view._Watched()
    turns.add(TRACE_ID)
    run_turn(debug_view.DebugSpanProcessor(lambda trace, record, ends: kept.append(record), turns))
    assert len(kept) == 4 and kept[-1]["dropped"] is True


def test_a_broken_sink_never_breaks_tracing():
    turns = debug_view._Watched()
    turns.add(TRACE_ID)

    def broken(*_):
        raise RuntimeError("redis down")

    spans = run_turn(debug_view.DebugSpanProcessor(broken, turns))
    assert "turn" in spans


def test_written_turns_are_read_back_by_their_owner_alone(on):
    import asyncio

    store = FakeAsyncRedis()
    asyncio.run(debug_view.register_turn(TRACE_ID, owner="anon-1", session_id="s1", client=store))
    sink = debug_view.RedisSink(client=FakeSyncRedis(store))
    turns = debug_view._Watched()
    turns.add(TRACE_ID)
    run_turn(debug_view.DebugSpanProcessor(lambda *item: sink.write([item]), turns))

    turn = asyncio.run(debug_view.read_turn(TRACE_ID, "anon-1", client=store))
    assert turn["session_id"] == "s1" and turn["complete"] is True
    assert turn["backend"]["environment"] and turn["backend"]["release"]
    assert {span["name"] for span in turn["spans"]} >= {"turn", "moderation", "running tool"}
    assert asyncio.run(debug_view.read_turn(TRACE_ID, "anon-2", client=store)) is None
    assert asyncio.run(debug_view.read_turn(TRACE_ID, None, client=store)) is None
    assert asyncio.run(debug_view.read_turn("f" * 32, "anon-1", client=store)) is None


@pytest.mark.asyncio
async def test_nothing_is_registered_when_off_or_for_a_server_side_caller(monkeypatch):
    store = FakeAsyncRedis()
    monkeypatch.setattr(settings, "debug_view_enabled", False)
    monkeypatch.setattr(settings, "environment", "development")
    await debug_view.register_turn(TRACE_ID, owner="anon-1", session_id="s1", client=store)
    assert store.values == {}
    assert debug_view.owner_of({"sub": "919876543210", "auth_type": "api_key"}) is None
    assert debug_view.owner_of({"sub": "anon-1"}) == "anon-1"
    assert debug_view.owner_of({"phone": "1"}) is None


def test_install_needs_it_on_and_a_tracer_provider(monkeypatch, on):
    monkeypatch.setattr(debug_view, "_installed", False)
    assert debug_view.install(provider=SimpleNamespace()) is False  # tracing off: no add_span_processor
    added = []
    assert debug_view.install(provider=SimpleNamespace(add_span_processor=added.append), sink=lambda *_: None) is True
    assert len(added) == 1 and isinstance(added[0], debug_view.DebugSpanProcessor)
    monkeypatch.setattr(debug_view, "_installed", False)
    monkeypatch.setattr(settings, "environment", "production")
    assert debug_view.install(provider=SimpleNamespace(add_span_processor=added.append), sink=lambda *_: None) is False
    assert len(added) == 1


@pytest.mark.asyncio
async def test_the_ui_flag_follows_it(monkeypatch, on):
    assert (await ui_flags.ui_flags()).debug_view is True
    monkeypatch.setattr(settings, "environment", "production")
    assert (await ui_flags.ui_flags()).debug_view is False


@pytest.mark.asyncio
async def test_a_chat_turn_is_registered_for_its_asker(monkeypatch, on):
    store = FakeAsyncRedis()
    monkeypatch.setattr(debug_view, "_client", lambda client=None: store)

    async def _stream(**kwargs):
        yield "ok"

    monkeypatch.setattr(chat_router, "stream_chat_messages", _stream)
    monkeypatch.setattr(chat_router, "_get_message_history", AsyncMock(return_value=[]))
    monkeypatch.setattr(chat_router, "mask_private_numbers", AsyncMock(side_effect=lambda query, *_: query))
    response = await chat_router.chat_endpoint(
        BackgroundTasks(),
        request=ChatRequest(query="hi", session_id="s1", stream=False),
        user_info={"sub": "anon-1", "anonymous": True},
    )
    trace_id = response.headers["X-Trace-Id"]
    turn = await debug_view.read_turn(trace_id, "anon-1", client=store)
    assert turn is not None and turn["session_id"] == "s1"
    assert debug_view.watched.count(trace_id) is not None


def test_the_endpoint_gives_a_turn_to_its_owner_and_a_404_to_anyone_else(monkeypatch, on):
    store = FakeAsyncRedis()
    monkeypatch.setattr(debug_view, "_client", lambda client=None: store)
    monkeypatch.setattr(debug_router, "get_langfuse_client", lambda: None)
    import asyncio

    asyncio.run(debug_view.register_turn(TRACE_ID, owner="anon-1", session_id="s1"))
    user = {"value": {"sub": "anon-1"}}
    app = FastAPI()
    app.include_router(debug_router.router, prefix="/api")
    app.dependency_overrides[get_chat_user] = lambda: user["value"]
    client = TestClient(app)

    mine = client.get(f"/api/debug/turns/{TRACE_ID}")
    assert mine.status_code == 200
    assert mine.json()["trace_id"] == TRACE_ID and mine.json()["langfuse_url"] is None
    for other in [{"sub": "anon-2"}, {"sub": "anon-1", "auth_type": "api_key"}, {"phone": "1"}]:
        user["value"] = other
        assert client.get(f"/api/debug/turns/{TRACE_ID}").status_code == 404, other
    user["value"] = {"sub": "anon-1"}
    assert client.get("/api/debug/turns/not-a-trace").status_code == 404
    assert client.get(f"/api/debug/turns/{'a' * 32}").status_code == 404

    async def redis_down(*_):
        raise ConnectionError("redis down")

    monkeypatch.setattr(store, "get", redis_down)
    assert client.get(f"/api/debug/turns/{TRACE_ID}").status_code == 503


def test_the_app_has_no_debug_route_unless_it_is_on():
    import main

    paths = {getattr(route, "path", None) for route in main.app.routes}
    assert not any(path and path.startswith("/api/debug") for path in paths)
