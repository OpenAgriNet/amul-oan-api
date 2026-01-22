from pydantic import BaseModel, Field
from typing import Literal
from pydantic_ai import Agent
from helpers.utils import get_prompt
from pydantic_ai.models import ModelSettings
from pydantic_ai import NativeOutput
from agents.models import LLM_MODEL


class QueryModerationResult(BaseModel):
    """Moderation result of the query."""
    category: Literal["valid_agricultural",
                      "invalid_language", 
                      "invalid_non_agricultural",
                      "invalid_external_reference",
                      "invalid_compound_mixed",
                      "unsafe_illegal",
                      "political_controversial",
                      "cultural_sensitive",
                      "role_obfuscation"] = Field(..., description="Moderation category of the user's message.")
    action: str = Field(..., description="Action to take on the query, always in English.")

    def __str__(self):
        category_str = self.category.replace("_", " ").title()
        return f"**Moderation Recommendation:** {self.action} ({category_str})"

moderation_agent = Agent(
    model=LLM_MODEL,
    name="Moderation Agent",
    system_prompt=get_prompt('moderation_system'),
    instrument=True,
    output_type=NativeOutput(QueryModerationResult),
    retries=2,
    model_settings=ModelSettings(
        #  temperature=0.1,
        openai_reasoning_effort='low',
        # max_tokens=128,
        timeout=5 # NOTE: Added timeout to avoid infinite loops
    )
)
