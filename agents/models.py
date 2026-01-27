import os
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.gemini import GeminiModel
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
            base_url=os.getenv('INFERENCE_ENDPOINT_URL'),
            api_key=os.getenv('INFERENCE_API_KEY'),
        ),
    )
elif LLM_PROVIDER == 'openai':
    LLM_MODEL = OpenAIModel(
        LLM_MODEL_NAME,
        provider=OpenAIProvider(
            api_key=os.getenv('OPENAI_API_KEY'),
        ),
    )
elif LLM_PROVIDER == 'anthropic':
    # AnthropicModel reads ANTHROPIC_API_KEY from environment automatically
    LLM_MODEL = AnthropicModel(LLM_MODEL_NAME)
elif LLM_PROVIDER == 'gemini':
    # GeminiModel reads GEMINI_API_KEY from environment automatically
    # Can also use GOOGLE_API_KEY as an alias
    api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    if api_key:
        os.environ['GEMINI_API_KEY'] = api_key
    LLM_MODEL = GeminiModel(
        LLM_MODEL_NAME,
        provider='google-gla',  # Use Generative Language API (can also use 'google-vertex' for Vertex AI)
    )
else:
    raise ValueError(f"Invalid LLM_PROVIDER: {LLM_PROVIDER}. Must be one of: 'vllm', 'openai', 'anthropic', 'gemini'")

