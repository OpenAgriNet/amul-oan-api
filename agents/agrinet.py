import os

from pydantic_ai import Agent, RunContext
from helpers.utils import get_prompt, get_today_date_str, get_today_datetime_str
from agents.models import LLM_MODEL, LLM_MODEL_NAME, LLM_PROVIDER
from agents.tools import TOOLS
from agents.tools.terms import get_ambiguity_hints_for_query
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits
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


def _env_int(name: str) -> int | None:
    """Parse an optional non-negative int env var; None when unset/blank/invalid."""
    raw = os.getenv(name)
    return int(raw) if raw and raw.isdigit() else None


def agrinet_usage_limits() -> UsageLimits:
    """Per-run cap on the agent's tool-calling loop.

    pydantic-ai enforces these via the ``usage_limits=`` argument to
    ``run``/``run_stream``/``iter`` — NOT via ``ModelSettings`` (a ``request_limit``
    placed there is silently dropped, leaving only the framework default of 50
    model requests / unlimited tool calls). Callers must pass this object.

    Tunable via env (a turn that hits the cap raises ``UsageLimitExceeded``):
    * ``AGENT_REQUEST_LIMIT``     — max model requests per run   (default 10; ``0`` = unlimited)
    * ``AGENT_TOOL_CALLS_LIMIT``  — max tool calls per run       (default unset = unlimited; ``0`` = unlimited)
    """
    rl = _env_int("AGENT_REQUEST_LIMIT")
    tcl = _env_int("AGENT_TOOL_CALLS_LIMIT")
    return UsageLimits(
        # Unset -> default 10; explicit 0 -> None (no cap).
        request_limit=(10 if rl is None else (rl or None)),
        tool_calls_limit=(tcl or None),
    )


# Built once at import (env is fixed for the process lifetime, mirroring max_tokens above).
AGENT_USAGE_LIMITS = agrinet_usage_limits()


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
    }

    if ctx.deps.use_translation_pipeline:
        return get_prompt("agrinet_system_translation_pipeline.md", context=context)
    return get_prompt("agrinet_system.md", context=context)
