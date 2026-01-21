import os
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from typing import List
from helpers.utils import get_prompt
from dotenv import load_dotenv
from agents.models import LLM_MODEL

load_dotenv()

suggestions_agent = Agent(
    name="Suggestions Agent",
    model=LLM_MODEL,
    system_prompt=get_prompt('suggestions_system'),
    instrument=True,
    output_type=List[str],
    retries=1,
    end_strategy='exhaustive',
    model_settings=ModelSettings(
        parallel_tool_calls=False, # Prevent multiple tool calls
    )
)