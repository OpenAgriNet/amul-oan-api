import os
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from dotenv import load_dotenv

load_dotenv()

# Get configurations from environment variables
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'openai').lower()
LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME', 'gpt-4.1')

if LLM_PROVIDER == 'vllm':
    LLM_MODEL = OpenAIModel(
        LLM_MODEL_NAME,
        provider=OpenAIProvider(
            base_url=os.getenv('VLLM_ENDPOINT_URL'), 
            api_key="dummy"
        ),
    )
elif LLM_PROVIDER == 'openai':
    LLM_MODEL = OpenAIModel(
        LLM_MODEL_NAME,
        provider=OpenAIProvider(
            api_key=os.getenv('OPENAI_API_KEY'),
        ),
    )
else:
    raise ValueError(f"Invalid LLM_PROVIDER: {LLM_PROVIDER}. Must be one of: 'gemini', 'qwen', 'openai'")

