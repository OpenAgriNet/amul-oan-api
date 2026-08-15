import os

from pydantic_ai import Agent, RunContext
from helpers.utils import get_prompt, get_today_date_str, get_today_datetime_str
from app.config import settings
from agents.tools import TOOLS
from agents.tools.terms import get_ambiguity_hints_for_query
from pydantic_ai.settings import ModelSettings
from agents.deps import FarmerContext


def _agrinet_max_output_tokens() -> int:
    """Cap completion tokens so prompt + max_tokens stays under small-context vLLM models (e.g. Gemma 16k)."""
    override = os.getenv("AGRINET_MAX_TOKENS")
    if override and override.isdigit():
        return int(override)
    provider = (settings.llm_provider or "openai").lower()
    model_name = settings.llm_model_name or "gpt-4.1"
    if provider == "vllm" and "gemma" in model_name.lower():
        gemma_cap = os.getenv("AGRINET_MAX_TOKENS_VLLM_GEMMA", "2048")
        return int(gemma_cap) if gemma_cap.isdigit() else 2048
    return 4000


agrinet_agent = Agent(
    # Model selection belongs to app.llm_core. Every execution path supplies the
    # resolved per-turn model explicitly; leaving this unset makes an omitted
    # model fail immediately instead of silently using a startup singleton.
    model=None,
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
    farmer_context = ctx.deps.get_farmer_context_string()
    ambiguity_hints = get_ambiguity_hints_for_query(ctx.deps.query)

    context = {
        'today_date': get_today_date_str(),
        'today_datetime': get_today_datetime_str(),
        'farmer_context': farmer_context if farmer_context else None,
        'ambiguity_hints': ambiguity_hints if ambiguity_hints else None,
        'response_max_chars': ctx.deps.get_response_max_chars(),
        'loan_max_amount': f"{int(settings.loan_max_amount):,}",
        'loan_interest_rate_pct': f"{int(settings.loan_interest_rate_pct)}",
        # Gates the Bharat Vistaar (mandi / weather / central scheme) prompt block
        # on the SAME flag that decides whether those tools are registered at all
        # (agents/tools/__init__.py). Advertising a tool the runtime has hidden is
        # not free: the model calls it, gets `Unknown tool name`, retries, and the
        # error enumerates the whole tool list back into context — two wasted LLM
        # round-trips on every turn, which is already pushing turns to 10-19 s
        # where it happens today.
        'network_tools_enabled': settings.enable_network,
    }

    if ctx.deps.use_translation_pipeline:
        return get_prompt("agrinet_system_translation_pipeline.md", context=context)
    return get_prompt("agrinet_system.md", context=context)
