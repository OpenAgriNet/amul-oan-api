import os
from typing import Literal, cast

from openai import APIStatusError, AsyncAzureOpenAI, AsyncStream
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.openai import (
    NOT_GIVEN,
    OpenAIModelSettings,
    chat,
    get_user_agent,
    ModelRequestParameters,
    ModelResponse,
    ModelMessage,
)
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.openai import OpenAIProvider
from dotenv import load_dotenv
from openai.types.chat import ChatCompletionChunk

from pydantic_ai import ModelHTTPError

from voice.app.model_boundary_capture import capture_model_boundary_payload

load_dotenv()


class BoundaryCaptureOpenAIModel(OpenAIModel):
    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: OpenAIModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> chat.ChatCompletion | AsyncStream[ChatCompletionChunk]:
        tools = self._get_tools(model_request_parameters)

        if not tools:
            tool_choice: Literal["none", "required", "auto"] | None = None
        elif not model_request_parameters.allow_text_output:
            tool_choice = "required"
        else:
            tool_choice = "auto"

        openai_messages = await self._map_messages(messages, model_request_parameters)
        extra_headers = model_settings.get("extra_headers", {})
        extra_headers.setdefault("User-Agent", get_user_agent())

        request_kwargs = {
            "model": self._model_name,
            "messages": openai_messages,
            "n": 1,
            "parallel_tool_calls": model_settings.get("parallel_tool_calls", NOT_GIVEN),
            "tools": tools or NOT_GIVEN,
            "tool_choice": tool_choice or NOT_GIVEN,
            "stream": stream,
            "stream_options": {"include_usage": True} if stream else NOT_GIVEN,
            "stop": model_settings.get("stop_sequences", NOT_GIVEN),
            "max_completion_tokens": model_settings.get("max_tokens", NOT_GIVEN),
            "temperature": model_settings.get("temperature", NOT_GIVEN),
            "top_p": model_settings.get("top_p", NOT_GIVEN),
            "timeout": model_settings.get("timeout", NOT_GIVEN),
            "seed": model_settings.get("seed", NOT_GIVEN),
            "presence_penalty": model_settings.get("presence_penalty", NOT_GIVEN),
            "frequency_penalty": model_settings.get("frequency_penalty", NOT_GIVEN),
            "logit_bias": model_settings.get("logit_bias", NOT_GIVEN),
            "reasoning_effort": model_settings.get("openai_reasoning_effort", NOT_GIVEN),
            "user": model_settings.get("openai_user", NOT_GIVEN),
            "extra_headers": extra_headers,
            "extra_body": model_settings.get("extra_body"),
        }
        capture_payload = {
            "model_name": self.model_name,
            "provider": self.system,
            "stream": stream,
            "tool_choice": tool_choice,
            "allow_text_output": model_request_parameters.allow_text_output,
            "payload": {
                key: value
                for key, value in request_kwargs.items()
                if value is not NOT_GIVEN and value is not None
            },
        }
        capture_model_boundary_payload(capture_payload)

        try:
            return await self.client.chat.completions.create(**request_kwargs)
        except APIStatusError as e:
            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise


# Get configurations from environment variables
LLM_PROVIDER    = os.getenv('LLM_PROVIDER', 'openai').lower()
LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME')

if LLM_PROVIDER == 'vllm':
    LLM_MODEL = BoundaryCaptureOpenAIModel(
        LLM_MODEL_NAME,
        provider=OpenAIProvider(
            base_url=os.getenv('INFERENCE_ENDPOINT_URL'), 
            api_key=os.getenv('INFERENCE_API_KEY'),  
        ),
    )
elif LLM_PROVIDER == 'openai':
    LLM_MODEL = BoundaryCaptureOpenAIModel(
        LLM_MODEL_NAME,
        provider=OpenAIProvider(
            api_key=os.getenv('OPENAI_API_KEY'),
        ),
    )
elif LLM_PROVIDER == 'anthropic':
    # AnthropicModel reads ANTHROPIC_API_KEY from the environment
    # and uses the Anthropic SDK under the hood.
    if not LLM_MODEL_NAME:
        raise ValueError("LLM_MODEL_NAME environment variable is required when using 'anthropic' provider")
    LLM_MODEL = AnthropicModel(LLM_MODEL_NAME)
elif LLM_PROVIDER == 'azure-openai':
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
    azure_api_version = os.getenv('AZURE_OPENAI_API_VERSION')
    azure_deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
    
    if not azure_endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required")
    if not azure_api_key:
        raise ValueError("AZURE_OPENAI_API_KEY environment variable is required")
    if not azure_api_version:
        raise ValueError("AZURE_OPENAI_API_VERSION environment variable is required")
    if not azure_deployment_name:
        raise ValueError("AZURE_OPENAI_DEPLOYMENT_NAME environment variable is required")
    
    azure_client = AsyncAzureOpenAI(
        azure_endpoint=azure_endpoint.rstrip('/'),
        api_version=azure_api_version,
        api_key=azure_api_key,
    )
    
    LLM_MODEL = BoundaryCaptureOpenAIModel(
        azure_deployment_name,
        provider=OpenAIProvider(openai_client=azure_client),
    )
else:
    raise ValueError(
        f"Invalid LLM_PROVIDER: {LLM_PROVIDER}. Must be one of: 'vllm', 'openai', 'azure-openai', 'anthropic'"
    )
