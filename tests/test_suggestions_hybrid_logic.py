import app.tasks.suggestions as sug


def _valid_shadow_payload(turn_id: str) -> dict:
    long_snippet = ("milk production guidance and cooperative payment details " * 8).strip()
    return {
        "request_turn_id": turn_id,
        "search_return_count": 2,
        "distilled_calls": [
            {
                "query": "milk payment",
                "no_results": False,
                "snippets": [long_snippet, long_snippet],
            }
        ],
    }


def test_shadow_retrieval_rejects_turn_id_mismatch():
    payload = _valid_shadow_payload("turn-a")
    usable, reason = sug._is_shadow_retrieval_usable_for_turn(payload, "turn-b")
    assert usable is False
    assert reason == "turn_id_mismatch"


def test_shadow_retrieval_accepts_matching_turn_id():
    payload = _valid_shadow_payload("turn-a")
    usable, reason = sug._is_shadow_retrieval_usable_for_turn(payload, "turn-a")
    assert usable is True
    assert reason == "ok"
