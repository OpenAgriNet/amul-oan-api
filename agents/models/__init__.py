import os
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
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
else:
    raise ValueError(f"Invalid LLM_PROVIDER: {LLM_PROVIDER}. Must be one of: 'vllm', 'openai', 'anthropic'")


# --- OSS pipeline (open-source models via vLLM) ---------------------------------
# Built additively alongside the legacy LLM_MODEL so a per-request sticky split can
# route a configurable % of sessions to the OSS path (mirrors the dev pipeline:
# vLLM gemma agent + translategemma pre/post-translation).
#
# This block must never raise at import: if OSS env is absent, OSS_LLM_MODEL stays
# None and get_model_for_variant() transparently falls back to the legacy model, so
# legacy behaviour is byte-identical when the split is disabled (OSS_PIPELINE_PCT=0).
OSS_LLM_MODEL_NAME = os.getenv('OSS_LLM_MODEL_NAME', 'gemma-4-31b-it')
OSS_INFERENCE_ENDPOINT_URL = os.getenv('OSS_INFERENCE_ENDPOINT_URL')
OSS_INFERENCE_API_KEY = os.getenv('OSS_INFERENCE_API_KEY', 'dummy')

OSS_LLM_MODEL = None
if OSS_INFERENCE_ENDPOINT_URL:
    try:
        OSS_LLM_MODEL = OpenAIModel(
            OSS_LLM_MODEL_NAME,
            provider=OpenAIProvider(
                base_url=OSS_INFERENCE_ENDPOINT_URL,
                api_key=OSS_INFERENCE_API_KEY,
            ),
        )
    except Exception:  # pragma: no cover - never break startup on OSS misconfig
        OSS_LLM_MODEL = None


def oss_model_available() -> bool:
    """True when an OSS vLLM model object was successfully constructed."""
    return OSS_LLM_MODEL is not None


def get_model_for_variant(variant: str):
    """Return the pydantic-ai model object for a resolved pipeline variant.

    'oss' -> the vLLM model when configured, else the legacy model (fail-safe).
    anything else -> the legacy startup model (unchanged behaviour).
    """
    if variant == 'oss' and OSS_LLM_MODEL is not None:
        return OSS_LLM_MODEL
    return LLM_MODEL


def provider_for_variant(variant: str) -> str:
    """Effective provider for a resolved variant: OSS is OpenAI-compatible (vLLM)."""
    if variant == 'oss' and OSS_LLM_MODEL is not None:
        return 'vllm'
    return LLM_PROVIDER

