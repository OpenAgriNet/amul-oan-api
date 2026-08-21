import os

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.settings import ModelSettings

from agents.agrinet import _agrinet_max_output_tokens
from agents.deps import FarmerContext
from agents.tools.search import search_documents
from helpers.utils import get_prompt, get_today_date_str


def _doctor_max_output_tokens() -> int:
    override = os.getenv("DOCTOR_MAX_TOKENS")
    if override and override.isdigit():
        return int(override)
    return min(_agrinet_max_output_tokens(), 1200)


def _doctor_request_limit() -> int:
    """Allow enough model turns for treatment completeness searches plus synthesis."""
    override = os.getenv("DOCTOR_REQUEST_LIMIT")
    if override and override.isdigit() and int(override) > 0:
        return int(override)
    return 10


# Clinical answers may retrieve evidence, but cannot invoke farmer-service,
# booking, financial, milk-collection, or network tools.
DOCTOR_TOOLS = [
    Tool(
        search_documents,
        takes_ctx=False,
        docstring_format="auto",
        require_parameter_descriptions=True,
    )
]


doctor_agent = Agent(
    model=None,
    name="Amul Doctor Agent",
    instrument=True,
    output_type=str,
    deps_type=FarmerContext,
    retries=5,
    tools=DOCTOR_TOOLS,
    end_strategy="exhaustive",
    model_settings=ModelSettings(
        max_tokens=_doctor_max_output_tokens(),
        parallel_tool_calls=False,
        request_limit=_doctor_request_limit(),
    ),
)


@doctor_agent.instructions
def get_doctor_instructions(ctx: RunContext) -> str:
    return get_prompt(
        "doctor_system_translation_pipeline.md",
        context={"today_date": get_today_date_str()},
    )
