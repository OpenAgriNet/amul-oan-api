import functools
import inspect

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.settings import ModelSettings

from agents.deps import FarmerContext
from agents.models import LLM_MODEL
from agents.tools.ai_call import create_ai_call
from agents.tools.common import fire_tool_call_nudge
from agents.tools.feedback import signal_conversation_state
from agents.tools.search import search_documents
from agents.tools.terms import search_terms
from helpers.utils import get_logger, get_prompt, get_today_date_str

logger = get_logger(__name__)


def _with_nudge_signal(func):
    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            fire_tool_call_nudge()
            return await func(*args, **kwargs)

        return wrapper

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        fire_tool_call_nudge()
        return func(*args, **kwargs)

    return wrapper


VOICE_TOOLS = [
    Tool(_with_nudge_signal(search_terms), takes_ctx=False),
    Tool(_with_nudge_signal(search_documents), takes_ctx=False),
    Tool(_with_nudge_signal(create_ai_call), takes_ctx=False, docstring_format="auto", require_parameter_descriptions=True),
    Tool(_with_nudge_signal(signal_conversation_state), takes_ctx=True, docstring_format="auto"),
]


voice_agent = Agent(
    model=LLM_MODEL,
    name="Voice Agent",
    instrument=True,
    output_type=str,
    deps_type=FarmerContext,
    retries=3,
    tools=VOICE_TOOLS,
    end_strategy="exhaustive",
    model_settings=ModelSettings(
        max_tokens=8192,
        parallel_tool_calls=True,
    ),
)


@voice_agent.system_prompt(dynamic=True)
def get_voice_system_prompt(ctx: RunContext[FarmerContext]) -> str:
    deps = ctx.deps
    farmer_context = deps.get_farmer_context_string()
    prompt_name = (
        "voice_system_translation_pipeline_en"
        if deps.use_translation_pipeline
        else f"voice_system_{deps.target_lang if deps.target_lang else 'gu'}"
    )
    logger.info(
        "Voice prompt selected=%s translation_pipeline=%s",
        prompt_name,
        deps.use_translation_pipeline,
    )
    return get_prompt(
        prompt_name,
        context={
            "today_date": get_today_date_str(),
            "farmer_context": farmer_context if farmer_context else None,
        },
    )

