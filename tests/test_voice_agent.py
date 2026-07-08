"""Inc 7.3 — the voice Agent builds, and the 7.2 tool registries have the
expected composition. Importing agents.voice constructs the agents at module
load, so this also guards against import/registry breakage on the voice path.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")


def test_voice_agents_build_with_prompt():
    from agents.voice import (
        voice_agent,
        voice_agent_signed_in,
        STATIC_VOICE_SYSTEM_PROMPT,
    )

    assert voice_agent is not None
    assert voice_agent_signed_in is not None
    assert isinstance(STATIC_VOICE_SYSTEM_PROMPT, str) and STATIC_VOICE_SYSTEM_PROMPT.strip()


def test_voice_tool_registry_composition():
    # Pins the per-tool decisions from Inc 7.2.
    from agents.tools import BASE_TOOLS, SIGNED_IN_FARMER_TOOLS

    # search_terms, search_documents, create_ai_call,
    # get_farmer_milk_collection_details, create_health_call, signal_conversation_state
    assert len(BASE_TOOLS) == 6
    # get_union_scheme_data only (profile/herd/tags intentionally disabled)
    assert len(SIGNED_IN_FARMER_TOOLS) == 1


def test_default_prompt_variant_is_mixed():
    from agents.voice import VOICE_SYSTEM_PROMPT_NAME

    assert VOICE_SYSTEM_PROMPT_NAME == "voice_system_translation_pipeline_en"
