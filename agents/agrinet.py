import os

from pydantic_ai import Agent, RunContext
from helpers.utils import get_prompt, get_today_date_str, get_today_datetime_str
from agents.models import LLM_MODEL, LLM_MODEL_NAME, LLM_PROVIDER
from agents.tools import TOOLS
from agents.tools.terms import get_ambiguity_hints_for_query
from pydantic_ai.settings import ModelSettings
from agents.deps import FarmerContext


def _agrinet_max_output_tokens() -> int:
    """Cap completion tokens so prompt + max_tokens stays under small-context vLLM models (e.g. Gemma 16k)."""
    override = os.getenv("AGRINET_MAX_TOKENS")
    if override and override.isdigit():
        return int(override)
    if LLM_PROVIDER == "vllm" and "gemma" in LLM_MODEL_NAME.lower():
        gemma_cap = os.getenv("AGRINET_MAX_TOKENS_VLLM_GEMMA", "2048")
        return int(gemma_cap) if gemma_cap.isdigit() else 2048
    return 4000


# ──────────────────────────────────────────────────────────────────────
# Prompt caching optimisation
# ──────────────────────────────────────────────────────────────────────
# The system prompt is now split into two layers:
#
#   1. STATIC system prompt  (role=system)  – identity, rules, tools.
#      Identical across every request → cached by the LLM provider.
#      Template files: agrinet_system_static.md / agrinet_system_translation_pipeline_static.md
#
#   2. DYNAMIC context block (role=user)   – date, farmer profile, ambiguity hints.
#      Injected per-request as a prefix to the user message.
#      Built by FarmerContext.get_dynamic_context_block().
#
# The old dynamic templates (agrinet_system.md, agrinet_system_translation_pipeline.md)
# are preserved for backward compatibility but are no longer used by default.
# ──────────────────────────────────────────────────────────────────────

# Pre-load static prompts at module level (they never change at runtime)
_STATIC_PROMPT = get_prompt("agrinet_system_static.md")
_STATIC_PROMPT_TRANSLATION = get_prompt("agrinet_system_translation_pipeline_static.md")


agrinet_agent = Agent(
    model=LLM_MODEL,
    name="Amul AI Agent",
    instrument=True,
    output_type=str,
    deps_type=FarmerContext,
    retries=5,
    tools=TOOLS,
    end_strategy='exhaustive',
    model_settings=ModelSettings(
        max_tokens=_agrinet_max_output_tokens(),
        parallel_tool_calls=True,
        request_limit=10,
    )
)

@agrinet_agent.instructions
def get_agrinet_instructions(ctx: RunContext):
    # Return the pre-loaded static prompt.
    # All dynamic content is now in the user message (see FarmerContext.get_dynamic_context_block).
    if ctx.deps.use_translation_pipeline:
        return _STATIC_PROMPT_TRANSLATION
    return _STATIC_PROMPT

