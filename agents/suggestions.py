import os
from pydantic_ai import Agent
from typing import List
from helpers.utils import get_prompt
from dotenv import load_dotenv
from agents.models import LLM_MODEL
load_dotenv()

# Suggestions are follow-up questions derived purely from the conversation — no
# retrieval needed. The agent previously carried the search_documents tool (with a
# prompt saying "do not call tools"); gpt-5.1 honoured that, but gemma ignores it
# and runs multi-hop RAG (2x search + ~5k-token prefill ~6s), tripping the OSS
# fallback budget. Removing the tool enforces the no-RAG intent in code so it's a
# single fast generation on any model.
suggestions_agent = Agent(
    name="Suggestions Agent",
    model=LLM_MODEL,
    instructions=get_prompt('suggestions_system'),
    instrument=True,
    output_type=List[str],
    retries=1,
)
