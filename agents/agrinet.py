from pydantic_ai import Agent, RunContext
from helpers.utils import get_prompt, get_today_date_str
from agents.models import LLM_MODEL
from agents.tools import TOOLS
from pydantic_ai.settings import ModelSettings
from agents.deps import FarmerContext


agrinet_agent = Agent(
    model=LLM_MODEL,
    name="Amul Vistaar Agent",
    instrument=True,
    output_type=str,
    deps_type=FarmerContext,
    retries=5,
    tools=TOOLS,
    end_strategy='exhaustive',
    model_settings=ModelSettings(
        max_tokens=8192,
        parallel_tool_calls=True,
        request_limit=10,
   )
)

@agrinet_agent.system_prompt(dynamic=True)
def get_agrinet_system_prompt(ctx: RunContext):
    prompt_file = f'agrinet_system_{ctx.deps.lang_code}'
    return get_prompt(prompt_file, context={'today_date': get_today_date_str()})