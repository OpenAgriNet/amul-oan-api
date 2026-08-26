from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

from app.services.suggestion_context import (
    extract_returned_docs,
    load_suggestion_banks,
    load_union_scheme_catalog,
    open_bank_domains,
    pick_candidates,
    tools_called_this_turn,
)


@dataclass
class _Msg:
    parts: list[object]


def _user(content: str) -> object:
    return SimpleNamespace(part_kind="user-prompt", content=content)


def _text(content: str) -> object:
    return SimpleNamespace(part_kind="text", content=content)


def _tool_call(call_id: str, tool_name: str) -> object:
    return SimpleNamespace(part_kind="tool-call", tool_call_id=call_id, tool_name=tool_name, args={})


def _tool_return(call_id: str, content: str, tool_name: str | None = None) -> object:
    payload = {"part_kind": "tool-return", "tool_call_id": call_id, "content": content}
    if tool_name:
        payload["tool_name"] = tool_name
    return SimpleNamespace(**payload)


def test_tools_called_this_turn_ignores_previous_turns():
    history = [
        _Msg(parts=[_user("book ai call")]),
        _Msg(parts=[_tool_call("1", "create_ai_call")]),
        _Msg(parts=[_tool_return("1", "ok")]),
        _Msg(parts=[_text("done")]),
        _Msg(parts=[_user("weather in anand")]),
        _Msg(parts=[_tool_call("2", "get_vistaar_weather")]),
        _Msg(parts=[_tool_return("2", "forecast payload")]),
    ]
    assert tools_called_this_turn(history) == ["get_vistaar_weather"]


def test_open_bank_domains_respects_tool_mapping_and_gates():
    banks = load_suggestion_banks()
    tools_called = [
        "create_health_call",
        "create_ai_call",
        "get_farmer_milk_collection_details",
        "get_vistaar_weather",
        "check_loan_eligibility",
    ]
    opened = open_bank_domains(
        tools_called,
        farmer_unions=["kutch"],
        enable_network=False,
        loan_feature_enabled=False,
        banks=banks,
    )
    assert opened == ["animal_health", "milk_quantity"]


def test_pick_candidates_only_from_opened_domains_and_tags():
    banks = load_suggestion_banks()

    ai_only = pick_candidates(
        ["ai_call"],
        banks,
        tools_called=["create_ai_call"],
        max_candidates=10,
    )
    assert ai_only
    assert all(candidate["domain"] == "ai_call" for candidate in ai_only)

    weather_only = pick_candidates(
        ["vistaar"],
        banks,
        tools_called=["get_vistaar_weather"],
        max_candidates=10,
    )
    assert weather_only
    assert all(candidate["tag"] == "weather" for candidate in weather_only)


def test_extract_returned_docs_from_latest_turn():
    old_search = "> Search Results for `old`\n\nold_chunk_1\n\n----\n\nold_chunk_2"
    new_search = (
        "> Search Results for `new`\n\nnew_chunk_1\n\n----\n\nnew_chunk_2\n\n----\n\nnew_chunk_3"
    )
    history = [
        _Msg(parts=[_user("old question")]),
        _Msg(parts=[_tool_call("old_search", "search_documents")]),
        _Msg(parts=[_tool_return("old_search", old_search)]),
        _Msg(parts=[_user("new question")]),
        _Msg(parts=[_tool_call("a", "search_documents")]),
        _Msg(parts=[_tool_call("b", "get_union_scheme_data")]),
        _Msg(parts=[_tool_call("c", "check_loan_eligibility")]),
        _Msg(parts=[_tool_return("a", new_search)]),
        _Msg(parts=[_tool_return("b", "scheme json payload")]),
        _Msg(parts=[_tool_return("c", "loan result payload")]),
    ]

    extracted = extract_returned_docs(history, max_search_chunks=2, max_chars=200)
    assert extracted["search_chunks"] == [
        "> Search Results for `new`\n\nnew_chunk_1",
        "new_chunk_2",
    ]
    assert extracted["scheme_tool_returns"] == [
        {"tool_name": "get_union_scheme_data", "content": "scheme json payload"}
    ]
    assert extracted["contextual_tool_returns"] == [
        {"tool_name": "check_loan_eligibility", "content": "loan result payload"}
    ]


def test_extract_search_chunks_splits_before_truncating():
    # First hit alone exceeds max_chars; splitting after truncate would drop hit 2.
    first = "A" * 1500
    second = "second_hit_content"
    payload = f"{first}\n\n----\n\n{second}"
    history = [
        _Msg(parts=[_user("q")]),
        _Msg(parts=[_tool_call("a", "search_documents")]),
        _Msg(parts=[_tool_return("a", payload)]),
    ]
    extracted = extract_returned_docs(history, max_search_chunks=2, max_chars=1200)
    assert len(extracted["search_chunks"]) == 2
    assert extracted["search_chunks"][0].startswith("A")
    assert extracted["search_chunks"][0].endswith("…")
    assert len(extracted["search_chunks"][0]) == 1201  # 1200 chars + ellipsis
    assert extracted["search_chunks"][1] == second


def test_extract_search_chunks_accumulates_across_multiple_returns():
    history = [
        _Msg(parts=[_user("q")]),
        _Msg(parts=[_tool_call("a", "search_documents")]),
        _Msg(parts=[_tool_return("a", "hit_a1\n\n----\n\nhit_a2")]),
        _Msg(parts=[_tool_call("b", "search_documents")]),
        _Msg(parts=[_tool_return("b", "hit_b1\n\n----\n\nhit_b2")]),
    ]

    extracted = extract_returned_docs(history, max_search_chunks=3, max_chars=200)
    assert extracted["search_chunks"] == ["hit_a1", "hit_a2", "hit_b1"]


def test_load_union_scheme_catalog_requires_scheme_tool(monkeypatch):
    async def fake_records(_union):
        return [{"scheme_title": "Cattle Insurance", "scheme_url": "https://example.com/a"}]

    monkeypatch.setattr(
        "app.services.scheme_ingestion.get_cached_scheme_records_for_union",
        fake_records,
    )

    # No scheme tool this turn → never inject cache.
    empty = asyncio.run(
        load_union_scheme_catalog(["create_ai_call"], ["banas"])
    )
    assert empty == []

    catalog = asyncio.run(
        load_union_scheme_catalog(["get_union_scheme_data"], ["banas"])
    )
    assert catalog == ["Banas: Cattle Insurance — https://example.com/a"]


def test_load_union_scheme_catalog_skips_unsupported_unions(monkeypatch):
    called = {"n": 0}

    async def fake_records(_union):
        called["n"] += 1
        return [{"scheme_title": "X", "scheme_url": "https://example.com/x"}]

    monkeypatch.setattr(
        "app.services.scheme_ingestion.get_cached_scheme_records_for_union",
        fake_records,
    )

    catalog = asyncio.run(
        load_union_scheme_catalog(["get_vistaar_scheme_info"], ["dudhsagar"])
    )
    assert catalog == []
    assert called["n"] == 0


def test_load_union_scheme_catalog_degrades_on_cache_error(monkeypatch):
    async def boom(_union):
        raise RuntimeError("redis down")

    monkeypatch.setattr(
        "app.services.scheme_ingestion.get_cached_scheme_records_for_union",
        boom,
    )

    catalog = asyncio.run(
        load_union_scheme_catalog(["get_union_scheme_data"], ["banas"])
    )
    assert catalog == []
