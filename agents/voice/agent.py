import os
from typing import Optional
from pydantic_ai import Agent
from helpers.utils import get_prompt, get_logger
from dotenv import load_dotenv
from app.config import settings
from agents.voice.tools import BASE_TOOLS, SIGNED_IN_FARMER_TOOLS
from pydantic_ai.settings import ModelSettings
from agents.deps import FarmerContext

logger = get_logger(__name__)

load_dotenv()

# Selectable prompt variants. Set VOICE_AGENT_PROMPT_VERSION, or voice's pipeline
# settings (agent_prompt_version), to switch.
#   mixed    — the current, accreted prompt (default; legacy tests pin this one)
#   gpt-5.1  — variant tuned to the GPT-5.1 prompting guide
#   gemma4   — variant tuned to the Gemma 4 prompting guide
VOICE_PROMPT_VARIANTS = {
    "mixed": "voice_system_translation_pipeline_en",
    "gpt-5.1": "voice_system_translation_pipeline_gpt5_1_en",
    "gemma4": "voice_system_translation_pipeline_gemma4_en",
}
DEFAULT_VOICE_PROMPT_VERSION = "mixed"


def _resolve_voice_prompt_name(version: Optional[str] = None) -> str:
    requested = (version or os.getenv("VOICE_AGENT_PROMPT_VERSION") or DEFAULT_VOICE_PROMPT_VERSION).strip().lower()
    if requested not in VOICE_PROMPT_VARIANTS:
        logger.warning(
            "Unknown VOICE_AGENT_PROMPT_VERSION=%r; falling back to %r",
            requested,
            DEFAULT_VOICE_PROMPT_VERSION,
        )
        requested = DEFAULT_VOICE_PROMPT_VERSION
    return VOICE_PROMPT_VARIANTS[requested]


def _render_voice_prompt(prompt_name: str) -> str:
    return get_prompt(
        prompt_name,
        context={
            "loan_max_amount": f"{int(settings.loan_max_amount):,}",
            "loan_interest_rate_pct": f"{int(settings.loan_interest_rate_pct)}",
            "creation_date_words": settings.voice_profile_creation_date_words,
            "service_channels_words": settings.voice_profile_service_channels,
            "helpline_number_words": settings.voice_profile_helpline_number_words,
        },
    )


VOICE_SYSTEM_PROMPT_NAME = _resolve_voice_prompt_name()
STATIC_VOICE_SYSTEM_PROMPT = _render_voice_prompt(VOICE_SYSTEM_PROMPT_NAME)

def _build_voice_agent(name: str, tools, instructions: str = STATIC_VOICE_SYSTEM_PROMPT):
    return Agent(
        # Model selection belongs to app.llm_core: every run is given the
        # turn's model, as chat's agents are.
        model=None,
        name=name,
        output_type=str,
        deps_type=FarmerContext,
        retries=3,
        tools=tools,
        instructions=instructions,
        end_strategy='exhaustive',
        model_settings=ModelSettings(
            max_tokens=3600,
            temperature=0.0,
            parallel_tool_calls=True,
       )
    )


voice_agent = _build_voice_agent("Voice Agent", BASE_TOOLS)
voice_agent_signed_in = _build_voice_agent(
    "Voice Agent Signed In",
    [*BASE_TOOLS, *SIGNED_IN_FARMER_TOOLS],
)

# Voice's pipeline settings load after this module, so a prompt version they name
# that differs from the one above gets its own pair, built on first use.
_agents_by_prompt: dict = {}


def voice_agents(version: Optional[str] = None) -> tuple[Agent, Agent]:
    """(voice_agent, voice_agent_signed_in) on a prompt version; None keeps
    VOICE_AGENT_PROMPT_VERSION's, the pair above."""
    prompt_name = _resolve_voice_prompt_name(version) if version else VOICE_SYSTEM_PROMPT_NAME
    if prompt_name == VOICE_SYSTEM_PROMPT_NAME:
        return voice_agent, voice_agent_signed_in
    if prompt_name not in _agents_by_prompt:
        instructions = _render_voice_prompt(prompt_name)
        _agents_by_prompt[prompt_name] = (
            _build_voice_agent("Voice Agent", BASE_TOOLS, instructions),
            _build_voice_agent("Voice Agent Signed In", [*BASE_TOOLS, *SIGNED_IN_FARMER_TOOLS], instructions),
        )
    return _agents_by_prompt[prompt_name]
