from types import SimpleNamespace

from app.services import chat
from app.tasks import suggestions


def _search_messages(tool_return: str, *, query: str = "milk production") -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            parts=[
                SimpleNamespace(
                    part_kind="tool-call",
                    tool_name="search_documents",
                    tool_call_id="search-1",
                    args={"query": query},
                ),
                SimpleNamespace(
                    part_kind="tool-return",
                    tool_call_id="search-1",
                    content=tool_return,
                ),
            ]
        )
    ]


def test_real_search_return_format_produces_usable_hybrid_evidence():
    evidence = "Milk production improves with balanced feed and clean water. " * 5
    tool_return = (
        "> Search Results for `milk production`\n\n"
        f"**Balanced feeding guide**\n```\n{evidence}\n```\n\n"
        "----\n\n"
        f"**Dairy management guide**\n```\n{evidence}Track daily milk yield.\n```\n"
    )

    payload = chat._extract_shadow_search_evidence(_search_messages(tool_return))
    payload["request_turn_id"] = "turn-1"

    assert payload["total_snippets"] == 2
    assert payload["parser_empty_calls"] == 0
    usable, reason = suggestions._is_shadow_retrieval_usable_for_turn(payload, "turn-1")
    assert usable is True
    assert reason == "ok"


def test_search_return_format_drift_is_detected_and_rejected():
    tool_return = (
        "> Search Results for `milk production`\n\n"
        "Balanced feeding guide: Milk production improves with balanced feed."
    )

    payload = chat._extract_shadow_search_evidence(_search_messages(tool_return))

    assert payload["total_snippets"] == 0
    assert payload["parser_empty_calls"] == 1
    usable, reason = suggestions._is_shadow_retrieval_usable(payload)
    assert usable is False
    assert reason == "insufficient_snippets"


def test_no_result_search_return_is_detected_and_rejected():
    tool_return = "No results found for `milk production`"

    payload = chat._extract_shadow_search_evidence(_search_messages(tool_return, query=""))

    assert payload["no_result_calls"] == 1
    assert payload["distilled_calls"][0]["query"] == "milk production"
    usable, reason = suggestions._is_shadow_retrieval_usable(payload)
    assert usable is False
    assert reason == "explicit_no_results"
