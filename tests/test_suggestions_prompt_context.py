import asyncio
import os
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""

import app.tasks.suggestions as sug


def test_create_suggestions_builds_grounded_prompt(monkeypatch):
    captured = {}

    async def fake_hist(_session_id):
        return []

    async def fake_set_cache(*_args, **_kwargs):
        return True

    async def fake_delete(*_args, **_kwargs):
        return None

    async def fake_run(message, model=None):
        captured["message"] = message
        captured["model"] = model
        return SimpleNamespace(output=["q1", "q2", "q3"])

    monkeypatch.setattr(sug, "_get_message_history", fake_hist)
    monkeypatch.setattr(sug, "trim_history", lambda *_a, **_k: [])
    monkeypatch.setattr(sug, "format_message_pairs", lambda *_a, **_k: [])
    monkeypatch.setattr(sug, "set_cache", fake_set_cache)
    monkeypatch.setattr(sug.cache, "delete", fake_delete)
    monkeypatch.setattr(sug.suggestions_agent, "run", fake_run)
    monkeypatch.setattr(sug.settings, "fallback_enabled", False)

    monkeypatch.setattr(sug, "tools_called_this_turn", lambda _h: ["create_ai_call", "search_documents"])
    monkeypatch.setattr(sug, "load_suggestion_banks", lambda: {"domains": {}})
    monkeypatch.setattr(sug, "open_bank_domains", lambda *_a, **_k: ["ai_call"])
    monkeypatch.setattr(
        sug,
        "pick_candidates",
        lambda *_a, **_k: [
            {
                "domain": "ai_call",
                "id": "ai_call_technician_arrival",
                "en": "When will the AI technician arrive?",
                "gu": "કૃત્રિમ બીજદાનના ટેકનિશિયન ક્યારે આવશે?",
                "hi": "एआई तकनीशियन कब आएंगे?",
                "tag": None,
            }
        ],
    )
    monkeypatch.setattr(
        sug,
        "extract_returned_docs",
        lambda *_a, **_k: {
            "search_chunks": ["chunk-1", "chunk-2"],
            "scheme_tool_returns": [{"tool_name": "get_union_scheme_data", "content": "scheme payload"}],
            "contextual_tool_returns": [{"tool_name": "check_loan_eligibility", "content": "loan payload"}],
        },
    )
    monkeypatch.setattr(
        sug,
        "capability_allowlist",
        lambda *_a, **_k: ["animal_health", "ai_call"],
    )

    async def fake_catalog(*_a, **_k):
        return ["Banas: Cattle Insurance — https://example.com/a"]

    monkeypatch.setattr(sug, "load_union_scheme_catalog", fake_catalog)

    out = asyncio.run(sug.create_suggestions("s1", "hi", "managed", farmer_unions=["banas"]))
    assert out == ["q1", "q2", "q3"]
    prompt = captured["message"]
    assert "**Candidate questions (capability-approved bank; may be en/gu/hi):**" in prompt
    assert "एआई तकनीशियन कब आएंगे?" in prompt
    assert "**Retrieved documents (doc-grounded questions only from these):**" in prompt
    assert "chunk-1" in prompt and "chunk-2" in prompt
    assert "**Scheme information (doc-grounded questions only from these):**" in prompt
    assert "[get_union_scheme_data] scheme payload" in prompt
    assert "**Union scheme catalog (cached; doc-grounded questions only from these):**" in prompt
    assert "Banas: Cattle Insurance — https://example.com/a" in prompt
    assert "**Other returned tool data (optional context; bank candidates remain valid):**" in prompt
    assert "[check_loan_eligibility] loan payload" in prompt
    assert "Capability allowlist" in prompt
    assert "Bank candidates do not need to be answerable from returned tool docs" in prompt
    assert "Conversation fallback" not in prompt


def test_create_suggestions_omits_scheme_catalog_when_empty(monkeypatch):
    captured = {}

    async def fake_hist(_session_id):
        return []

    async def fake_set_cache(*_args, **_kwargs):
        return True

    async def fake_delete(*_args, **_kwargs):
        return None

    async def fake_run(message, model=None):
        captured["message"] = message
        return SimpleNamespace(output=["q1"])

    async def fake_catalog(*_a, **_k):
        return []

    monkeypatch.setattr(sug, "_get_message_history", fake_hist)
    monkeypatch.setattr(sug, "trim_history", lambda *_a, **_k: [])
    monkeypatch.setattr(sug, "format_message_pairs", lambda *_a, **_k: [])
    monkeypatch.setattr(sug, "set_cache", fake_set_cache)
    monkeypatch.setattr(sug.cache, "delete", fake_delete)
    monkeypatch.setattr(sug.suggestions_agent, "run", fake_run)
    monkeypatch.setattr(sug.settings, "fallback_enabled", False)
    monkeypatch.setattr(sug, "tools_called_this_turn", lambda _h: ["create_ai_call"])
    monkeypatch.setattr(sug, "load_suggestion_banks", lambda: {"domains": {}})
    monkeypatch.setattr(sug, "open_bank_domains", lambda *_a, **_k: ["ai_call"])
    monkeypatch.setattr(sug, "pick_candidates", lambda *_a, **_k: [])
    monkeypatch.setattr(
        sug,
        "extract_returned_docs",
        lambda *_a, **_k: {
            "search_chunks": [],
            "scheme_tool_returns": [],
            "contextual_tool_returns": [],
        },
    )
    monkeypatch.setattr(sug, "capability_allowlist", lambda *_a, **_k: ["ai_call"])
    monkeypatch.setattr(sug, "load_union_scheme_catalog", fake_catalog)

    out = asyncio.run(sug.create_suggestions("s1", "en", "managed", farmer_unions=["banas"]))
    assert out == ["q1"]
    assert "Union scheme catalog" not in captured["message"]
    # No candidates and no docs → conversation fallback path.
    assert "**Conversation fallback:**" in captured["message"]


def test_create_suggestions_conversation_fallback_for_mr(monkeypatch):
    captured = {}

    async def fake_hist(_session_id):
        return []

    async def fake_set_cache(*_args, **_kwargs):
        return True

    async def fake_delete(*_args, **_kwargs):
        return None

    async def fake_run(message, model=None):
        captured["message"] = message
        return SimpleNamespace(output=["q1", "q2", "q3"])

    async def fake_catalog(*_a, **_k):
        return []

    monkeypatch.setattr(sug, "_get_message_history", fake_hist)
    monkeypatch.setattr(sug, "trim_history", lambda *_a, **_k: [])
    monkeypatch.setattr(
        sug,
        "format_message_pairs",
        lambda *_a, **_k: ["**User Message**:\nCow is sick\n\n**Assistant Message**:\nRest and water"],
    )
    monkeypatch.setattr(sug, "set_cache", fake_set_cache)
    monkeypatch.setattr(sug.cache, "delete", fake_delete)
    monkeypatch.setattr(sug.suggestions_agent, "run", fake_run)
    monkeypatch.setattr(sug.settings, "fallback_enabled", False)
    monkeypatch.setattr(sug, "tools_called_this_turn", lambda _h: [])
    monkeypatch.setattr(sug, "load_suggestion_banks", lambda: {"domains": {}})
    monkeypatch.setattr(sug, "open_bank_domains", lambda *_a, **_k: [])
    monkeypatch.setattr(sug, "pick_candidates", lambda *_a, **_k: [])
    monkeypatch.setattr(
        sug,
        "extract_returned_docs",
        lambda *_a, **_k: {
            "search_chunks": [],
            "scheme_tool_returns": [],
            "contextual_tool_returns": [],
        },
    )
    monkeypatch.setattr(
        sug,
        "capability_allowlist",
        lambda *_a, **_k: ["animal_health", "ai_call"],
    )
    monkeypatch.setattr(sug, "load_union_scheme_catalog", fake_catalog)

    out = asyncio.run(sug.create_suggestions("s1", "mr", "managed", farmer_unions=[]))
    assert out == ["q1", "q2", "q3"]
    prompt = captured["message"]
    assert "**Conversation fallback:**" in prompt
    assert "Marathi" in prompt or "mr" in prompt.lower() or "capability allowlist" in prompt.lower()
    assert "Cow is sick" in prompt
