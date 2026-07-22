import os
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.openai import OpenAIProvider
from dotenv import load_dotenv

load_dotenv()

# Get configurations from environment variables
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'openai').lower()
LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME', 'gpt-4.1')

# LLM_MODEL is the construction-time default model for every pydantic-ai agent
# (agrinet / moderation / suggestions / voice). Every request passes an explicit
# ``model=`` resolved by ``app.llm_core`` (the unified pipeline is the only
# runtime path), so this default is only a fallback and is never used per-turn.
# The OSS/legacy split + get_model_for_variant/provider_for_variant/
# oss_model_available selectors were removed at P4 — model selection now lives
# entirely in app/llm_core (factory + resolver + weighted-profile split).
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
else:
    raise ValueError(f"Invalid LLM_PROVIDER: {LLM_PROVIDER}. Must be one of: 'vllm', 'openai', 'anthropic'")
