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

@agrinet_agent.system_prompt(dynamic=True)
def get_agrinet_system_prompt(ctx: RunContext):
    farmer_context = ctx.deps.get_farmer_context_string()
    ambiguity_hints = get_ambiguity_hints_for_query(ctx.deps.query)

    context = {
        'today_date': get_today_date_str(),
        'today_datetime': get_today_datetime_str(),
        'farmer_context': farmer_context if farmer_context else None,
        'ambiguity_hints': ambiguity_hints if ambiguity_hints else None,
    }

    if ctx.deps.use_translation_pipeline:
        return get_prompt("agrinet_system_translation_pipeline.md", context=context)
    return get_prompt("agrinet_system.md", context=context)
