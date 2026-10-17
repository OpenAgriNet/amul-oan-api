import os
from pydantic_ai import Agent
from helpers.utils import get_prompt, get_logger
from dotenv import load_dotenv
from app.config import settings
from agents.voice.tools import BASE_TOOLS, SIGNED_IN_FARMER_TOOLS
from pydantic_ai.settings import ModelSettings
from agents.deps import FarmerContext

logger = get_logger(__name__)

load_dotenv()

# Selectable prompt variants. Set VOICE_AGENT_PROMPT_VERSION to switch.
#   mixed    — the current, accreted prompt (default; legacy tests pin this one)
#   gpt-5.1  — variant tuned to the GPT-5.1 prompting guide
#   gemma4   — variant tuned to the Gemma 4 prompting guide
VOICE_PROMPT_VARIANTS = {
    "mixed": "voice_system_translation_pipeline_en",
    "gpt-5.1": "voice_system_translation_pipeline_gpt5_1_en",
    "gemma4": "voice_system_translation_pipeline_gemma4_en",
}
DEFAULT_VOICE_PROMPT_VERSION = "mixed"


def _resolve_voice_prompt_name() -> str:
    requested = (os.getenv("VOICE_AGENT_PROMPT_VERSION") or DEFAULT_VOICE_PROMPT_VERSION).strip().lower()
    if requested not in VOICE_PROMPT_VARIANTS:
        logger.warning(
            "Unknown VOICE_AGENT_PROMPT_VERSION=%r; falling back to %r",
            requested,
            DEFAULT_VOICE_PROMPT_VERSION,
        )
        requested = DEFAULT_VOICE_PROMPT_VERSION
    return VOICE_PROMPT_VARIANTS[requested]


VOICE_SYSTEM_PROMPT_NAME = _resolve_voice_prompt_name()
STATIC_VOICE_SYSTEM_PROMPT = get_prompt(
    VOICE_SYSTEM_PROMPT_NAME,
    context={
        "loan_max_amount": f"{int(settings.loan_max_amount):,}",
        "loan_interest_rate_pct": f"{int(settings.loan_interest_rate_pct)}",
        "creation_date_words": settings.voice_profile_creation_date_words,
        "service_channels_words": settings.voice_profile_service_channels,
        "helpline_number_words": settings.voice_profile_helpline_number_words,
    },
)

def _build_voice_agent(name: str, tools):
    return Agent(
        # Model selection belongs to app.llm_core: every run is given the
        # turn's model, as chat's agents are.
        model=None,
        name=name,
        output_type=str,
        deps_type=FarmerContext,
        retries=3,
        tools=tools,
        instructions=STATIC_VOICE_SYSTEM_PROMPT,
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
