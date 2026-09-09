"""Memory integration contracts using the real bot modules and a fake reader."""
import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic_ai import Tool

os.environ.setdefault("OPENAI_API_KEY", "test-key")
from agents.deps import FarmerContext, MEMORY_INSTRUCTIONS
from agents.agrinet import get_agrinet_instructions
from agents.tools import memory_tools as tools
from app.services import memory

REF = "11111111-1111-4111-8111-111111111111"


@pytest.fixture
def ctx():
    return SimpleNamespace(deps=FarmerContext(query="my earlier issue", mobile="test-farmer"))


@pytest.fixture
def http(monkeypatch):
    calls = []
    bodies = {"/profile": {"profile": {}}, "/search": {"hits": []}, "/keys": {"keys": []}}
    class Store:
        def __init__(self, client, farmer):
            self.farmer = farmer
        async def enabled(self):
            return not any(isinstance(v, dict) and v.get("memory_enabled") is False for v in bodies.values())
        async def result(self, suffix, **kwargs):
            calls.append({"farmer": self.farmer, "operation": suffix, **kwargs})
            value = bodies[suffix]
            if isinstance(value, Exception):
                raise value
            return value
        async def profile(self):
            return await self.result("/profile")
        async def keys(self):
            return await self.result("/keys")
        async def search(self, query, **kwargs):
            return await self.result("/search", query=query, **kwargs)
        async def list_entries(self, **kwargs):
            return await self.result("/entries", **kwargs)
        async def read(self, **kwargs):
            return await self.result("/read", **kwargs)
    monkeypatch.setattr(memory, "MEMORY_ENABLED", True)
    monkeypatch.setattr(tools, "MEMORY_ENABLED", True)
    monkeypatch.setattr(memory, "MemoryStore", Store)
    monkeypatch.setattr(tools, "MemoryStore", Store)
    return bodies, calls


def episode(**values):
    return {"id": REF, "headline": "Calf registration correction pending.", "status": "open",
            "source_ts": "2026-09-01T00:00:00Z", "chunk_count": 1,
            "contents": [{"start_chunk": 1, "end_chunk": 1, "text": "Receipt location and correction papers."}], **values}


def test_global_off_never_calls_memory(http, monkeypatch):
    _, calls = http
    monkeypatch.setattr(memory, "MEMORY_ENABLED", False)
    assert asyncio.run(memory.fetch_memory_context("test-farmer", "hello")) == ""
    assert calls == []


def test_keys_alone_do_not_create_context(http):
    bodies, _ = http
    bodies["/keys"] = {"keys": [{"filter_path": "metadata.scheme", "values_for_this_farmer": ["fodder"]}]}
    assert asyncio.run(memory.fetch_memory_context("test-farmer", "hello")) == ""


def test_explicit_disabled_flag_suppresses_all_automatic_reads(http):
    bodies, _ = http
    bodies.update({"/profile": {"memory_enabled": False, "profile": {"usual_topics": "ignore me"}},
                   "/search": {"memory_enabled": False, "hits": [episode()]},
                   "/keys": {"memory_enabled": False, "keys": [{"filter_path": "metadata.scheme"}]}})
    assert asyncio.run(memory.fetch_memory_context("test-farmer", "hello")) == ""


def test_overflow_preserves_complete_episode_and_sample_warning(http, monkeypatch):
    bodies, _ = http
    bodies["/search"] = {"hits": [episode(), episode(id="22222222-2222-4222-8222-222222222222", headline="x" * 300)]}
    monkeypatch.setattr(memory, "MEMORY_MAX_CHARS", 950)
    text = asyncio.run(memory.fetch_memory_context("test-farmer", "registration"))
    assert REF in text and "22222222-2222-4222-8222-222222222222" not in text
    assert "Receipt location and correction papers." in text
    assert "SEARCH SAMPLE ONLY" in text and "not everything on record" in text
    assert len(text) <= 950


def test_failed_search_keeps_independent_standing_context(http):
    bodies, _ = http
    bodies["/profile"] = {"profile": {"explicit_requests": "short practical answers"}}
    bodies["/search"] = ValueError("bad JSON")
    text = asyncio.run(memory.fetch_memory_context("test-farmer", "hello"))
    assert "short practical answers" in text


def test_total_lookup_deadline_returns_empty(monkeypatch, http):
    async def slow(*args):
        await asyncio.sleep(1)
    monkeypatch.setattr(memory, "MEMORY_ENABLED", True)
    monkeypatch.setattr(memory, "MEMORY_TIMEOUT_SECONDS", .01)
    for name in ("_standing_summary", "_relevant_entries", "_available_filters"):
        monkeypatch.setattr(memory, name, slow)
    assert asyncio.run(memory.fetch_memory_context("test-farmer", "hello")) == ""


def test_memory_instructions_are_conditional_and_preserve_shc(ctx):
    without = get_agrinet_instructions(ctx)
    original = ctx.deps.get_user_message()
    assert MEMORY_INSTRUCTIONS not in without
    ctx.deps.memory_context = "Prior registration issue."
    ctx.deps.soil_health_card_context = "Soil pH: 7.2"
    assert get_agrinet_instructions(ctx) == without + "\n\n" + MEMORY_INSTRUCTIONS
    message = ctx.deps.get_user_message()
    assert "Prior registration issue." in message and "Soil pH: 7.2" in message
    assert original in message


@pytest.mark.parametrize("mobile,persona,enabled", [(None, "farmer", True), ("test", "doctor", True), ("test", "farmer", False)])
def test_tools_hidden_without_enabled_farmer(ctx, monkeypatch, mobile, persona, enabled):
    ctx.deps.mobile = mobile
    ctx.deps.persona = persona
    monkeypatch.setattr(tools, "MEMORY_ENABLED", enabled)
    assert asyncio.run(tools.prepare_memory_tool(ctx, object())) is None


def test_all_memory_tool_schemas_build_with_installed_sdk():
    for fn in (tools.search_memories, tools.list_memories, tools.read_memory):
        tool = Tool(fn, takes_ctx=True, require_parameter_descriptions=True)
        assert "farmer_id" not in tool.function_schema.json_schema["properties"]
        assert "max_chars" not in tool.function_schema.json_schema["properties"]


def test_list_unresolved_uses_both_states_and_preserves_cursor(ctx, http):
    bodies, calls = http
    bodies["/entries"] = {"entries": [episode()], "next_cursor": "page-two"}
    text = asyncio.run(tools.list_memories(ctx, state="unresolved", filters={"scheme": "fodder"}))
    assert calls[-1]["status"] == ["open", "pending"]
    assert "page-two" in text and calls[-1]["filters"] == {"scheme": "fodder"}
    assert calls[-1]["farmer"] == "test-farmer"


def test_disabled_or_failed_lookup_does_not_claim_no_memories(ctx, http):
    bodies, _ = http
    bodies["/entries"] = {"memory_enabled": False, "entries": []}
    assert "Do not infer absence" in asyncio.run(tools.list_memories(ctx))
    bodies["/entries"] = ValueError("malformed response")
    assert "Do not infer that nothing is on record" in asyncio.run(tools.list_memories(ctx))


def test_chunks_are_complete_and_parallel_calls_share_budget(ctx, http, monkeypatch):
    bodies, _ = http
    text = "BEGIN " + "detail " * 120 + " END"
    bodies["/read"] = {"entry": episode(), "chunks": [{"number": 1, "text": text}], "continuation": None}
    monkeypatch.setattr(tools, "TURN_MAX_CHARS", 1500)
    async def both():
        return await asyncio.gather(tools.read_memory(ctx, REF), tools.read_memory(ctx, REF))
    results = asyncio.run(both())
    assert sum(text in result for result in results) == 1
    assert tools.BUDGET_REACHED in results
    assert ctx.deps.memory_tool_chars <= 1500


def test_invalid_reference_and_exhausted_budget_make_no_request(ctx, http, monkeypatch):
    _, calls = http
    assert "invalid" in asyncio.run(tools.read_memory(ctx, "../other-farmer"))
    ctx.deps.memory_tool_calls = tools.MAX_CALLS
    assert "budget" in asyncio.run(tools.list_memories(ctx)).lower()
    assert calls == []
