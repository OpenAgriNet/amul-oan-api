import os
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from dotenv import load_dotenv

load_dotenv()

# Get configurations from environment variables
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'vllm').lower()
LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME', 'gpt-oss-20b')

if LLM_PROVIDER == 'vllm':
    LLM_MODEL = OpenAIChatModel(
        LLM_MODEL_NAME,
        provider=OpenAIProvider(
            base_url=os.getenv('VLLM_BASE_URL'), 
            api_key="dummy"
        ),
    )
elif LLM_PROVIDER == 'openai':
    LLM_MODEL = OpenAIResponsesModel(
        LLM_MODEL_NAME,
        provider=OpenAIProvider(
            api_key=os.getenv('OPENAI_API_KEY'),
        ),
    )
else:
    raise ValueError(f"Invalid LLM_PROVIDER: {LLM_PROVIDER}. Must be one of: 'gemini', 'qwen', 'openai'")

