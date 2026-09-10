"""Provider factory — the single seam that turns an inert :class:`Tier` into a
live handle. Superset of both repos' model construction.

* ``build_handle(tier, kind)`` is ``@lru_cache``'d on the frozen ``(tier, kind)``
  key. Nothing is built until the execution walker reaches that tier.
* Providers: vllm + openai (OpenAI-compatible ``base_url``), azure-openai
  (``AsyncAzureOpenAI`` + ``OpenAIProvider(openai_client=...)``), anthropic
  (``AnthropicModel``), gemini (``GeminiModel(provider='google-gla')``), and
  translategemma (an aiohttp text-completion *descriptor*, not a client).
* The httpx **boundary-capture** hook (adopted from voice's
  ``_build_openai_compatible_model``) now covers every OpenAI-compatible client
  kind — chat's legacy path never had it. 600s read / 5s connect matches the
  OpenAI SDK default so long streaming agent runs are not aborted.

Legality is enforced per client kind; pretranslation supports native Anthropic
and OpenAI-compatible clients, while TranslateGemma uses its own descriptor.

NOTE (pydantic-ai version): the chat repo is pinned to pydantic-ai 0.2.4, whose
model class is ``OpenAIModel`` and whose Gemini arm is
``GeminiModel(provider='google-gla')``. Voice runs 1.x (``OpenAIChatModel`` /
``GoogleModel`` + ``GoogleProvider``). The public factory API is identical across
repos; only these two construction lines differ — the per-repo delta the merge
will reconcile last.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

import httpx
from openai import AsyncOpenAI, AsyncAzureOpenAI
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

from helpers.utils import get_logger
from app.config import get_config_value, settings
from app.llm_core.config_model import Provider, Step, Tier, StepClientKind

# Version-tolerant OpenAI model class: the deploy target pins pydantic-ai 1.x
# (``OpenAIChatModel``, matching voice); older local envs expose ``OpenAIModel``.
# Same construction either way — only the class name moved.
try:  # pydantic-ai 1.x
    from pydantic_ai.models.openai import OpenAIChatModel as _OpenAIModel
except ImportError:  # pragma: no cover - older pydantic-ai
    from pydantic_ai.models.openai import OpenAIModel as _OpenAIModel

logger = get_logger(__name__)

STEP_CLIENT_KIND: dict[Step, StepClientKind] = {
    Step.AGENT: StepClientKind.AGENT,
    Step.MODERATION: StepClientKind.AGENT,
    Step.SUGGESTIONS: StepClientKind.AGENT,
    Step.PRE_TRANSLATION: StepClientKind.PRE_TRANSLATION,
    Step.POST_TRANSLATION: StepClientKind.TRANSLATEGEMMA,
}

# Boundary-capture hook — best-effort; a failed capture must never drop a request.
try:  # pragma: no cover - import guard
    from app.model_boundary_capture import (
        boundary_capture_enabled,
        capture_model_boundary_payload,
    )
except Exception:  # pragma: no cover
    def boundary_capture_enabled() -> bool:  # type: ignore
        return False

    def capture_model_boundary_payload(payload):  # type: ignore
        return None


# ── httpx boundary-capture (adopted from voice) ───────────────────────────────
async def _capture_request_hook(request: httpx.Request) -> None:
    if not boundary_capture_enabled():
        return
    try:
        if not request.url.path.endswith("/chat/completions"):
            return
        raw = request.content
        if not raw:
            return
        body = json.loads(raw.decode("utf-8"))
        capture_model_boundary_payload(
            {
                "model_name": body.get("model"),
                "provider": "openai-compatible",
                "stream": bool(body.get("stream", False)),
                "tool_choice": body.get("tool_choice"),
                "url": str(request.url),
                "payload": body,
            }
        )
    except Exception as exc:  # pragma: no cover - capture is best-effort
        logger.debug("Model boundary capture hook failed: %s", exc)


def _capture_http_client() -> httpx.AsyncClient:
    """Long-lived AsyncClient with the boundary-capture event hook attached.

    Pin an explicit 600s read/write/pool timeout (5s connect) to match the
    OpenAI SDK default and pydantic-ai's cached client. A bare AsyncClient
    inherits httpx's 5s default, which would abort long streaming agent runs.
    """
    return httpx.AsyncClient(
        event_hooks={"request": [_capture_request_hook]},
        timeout=httpx.Timeout(600.0, connect=5.0),
    )


def _build_openai_compatible_model(
    model_name: str,
    *,
    base_url: Optional[str],
    api_key: Optional[str],
):
    """OpenAI-compatible pydantic-ai model (vLLM or OpenAI) with the boundary
    hook. ``base_url=None`` targets OpenAI proper."""
    return _OpenAIModel(
        model_name,
        provider=OpenAIProvider(
            base_url=base_url,
            api_key=api_key,
            http_client=_capture_http_client(),
        ),
    )


@dataclass(frozen=True)
class TGDescriptor:
    """TranslateGemma is ``/completions``-over-aiohttp, not an OpenAI client —
    so it builds an inert descriptor the translation service consumes
    directly, never a client object."""

    completions_url: str
    model_id: str
    endpoint: str


def _key(tier: Tier) -> Optional[str]:
    """Read the named secret at handle-build time (never store it in config)."""
    if not tier.api_key_env:
        return None
    return get_config_value(tier.api_key_env)


# ── low-level builders ────────────────────────────────────────────────────────
def _build_agent_model(tier: Tier) -> Any:
    """AGENT kind -> pydantic-ai Model."""
    if tier.provider in (Provider.VLLM, Provider.OPENAI):
        # (D) A vLLM/OSS tier MUST carry its endpoint. The legacy
        # ``_get_oss_pretranslation_client`` RAISED when the OSS endpoint was
        # absent (so moderation/non_meaningful caught it and failed OPEN); an
        # endpoint-less vLLM tier here would otherwise silently build an
        # OpenAI-default client labeled vLLM (base_url=None -> OpenAI proper),
        # flipping behaviour. Fail loudly instead to preserve the legacy semantics.
        if tier.provider is Provider.VLLM and not tier.endpoint:
            raise ValueError(
                "vLLM/OSS agent tier requires an endpoint; refusing to build an "
                "OpenAI-default client for a vLLM-labeled tier"
            )
        base_url = tier.endpoint if tier.provider is Provider.VLLM else None
        return _build_openai_compatible_model(tier.model, base_url=base_url, api_key=_key(tier))

    if tier.provider is Provider.AZURE:
        endpoint = tier.endpoint
        api_key = _key(tier)
        api_version = tier.api_version
        if not endpoint:
            raise ValueError("azure-openai tier requires endpoint")
        if not api_key:
            raise ValueError("azure-openai tier requires api_key_env to be set")
        if not api_version:
            raise ValueError("azure-openai tier requires api_version")
        azure_client = AsyncAzureOpenAI(
            azure_endpoint=endpoint.rstrip("/"),
            api_version=api_version,
            api_key=api_key,
            http_client=_capture_http_client(),
        )
        # tier.model is the Azure deployment name.
        return _OpenAIModel(tier.model, provider=OpenAIProvider(openai_client=azure_client))

    if tier.provider is Provider.ANTHROPIC:
        return AnthropicModel(tier.model, provider=AnthropicProvider(api_key=_key(tier)))

    if tier.provider is Provider.GEMINI:
        # Rebased off the dead-file ``feat/adding-google-as-model-provider`` arm.
        # Deploy target (pydantic-ai 1.x): GoogleModel + GoogleProvider(api_key=).
        # Older local envs: GeminiModel(provider='google-gla'), reading
        # GEMINI_API_KEY / GOOGLE_API_KEY (the chat branch's exact behaviour).
        api_key = _key(tier) or get_config_value("GEMINI_API_KEY") or get_config_value("GOOGLE_API_KEY")
        try:  # pydantic-ai 1.x
            from pydantic_ai.models.google import GoogleModel
            from pydantic_ai.providers.google import GoogleProvider

            return GoogleModel(tier.model, provider=GoogleProvider(api_key=api_key))
        except ImportError:  # pragma: no cover - older pydantic-ai
            from pydantic_ai.models.gemini import GeminiModel
            from pydantic_ai.providers.google_gla import GoogleGLAProvider

            return GeminiModel(tier.model, provider=GoogleGLAProvider(api_key=api_key))

    raise ValueError(f"provider {tier.provider} is not valid for an AGENT step")


def _build_pretranslation(tier: Tier) -> Any:
    """Build the provider-native async client used by translation adapters."""
    if tier.provider not in (
        Provider.VLLM,
        Provider.OPENAI,
        Provider.AZURE,
        Provider.GEMINI,
    ):
        raise ValueError(
            f"provider {tier.provider} is not an OpenAI-compatible raw client; "
            "anthropic/translategemma are built by their dedicated paths"
        )
    if tier.provider is Provider.GEMINI:
        from google import genai

        api_key = _key(tier) or get_config_value("GOOGLE_API_KEY")
        return genai.Client(api_key=api_key).aio
    if tier.provider is Provider.AZURE:
        endpoint = tier.endpoint
        api_key = _key(tier)
        if not endpoint or not api_key or not tier.api_version:
            raise ValueError("azure-openai raw client requires endpoint, api_version and api_key_env")
        return AsyncAzureOpenAI(
            azure_endpoint=endpoint.rstrip("/"),
            api_version=tier.api_version,
            api_key=api_key,
            http_client=_capture_http_client(),
        )
    # (D) Same guard as the agent builder: a vLLM/OSS PRE_TRANSLATION tier without an
    # endpoint must RAISE (mirrors legacy ``_get_oss_pretranslation_client``), not
    # silently fall through to an OpenAI-default client (base_url=None) that is
    # labeled vLLM — which would flip pre-translation/moderation from fail-OPEN
    # (OSS unconfigured) to actually calling OpenAI.
    if tier.provider is Provider.VLLM and not tier.endpoint:
        raise ValueError(
            "vLLM/OSS pretranslation tier requires an endpoint; refusing to build an "
            "OpenAI-default client for a vLLM-labeled tier"
        )
    base_url = tier.endpoint if tier.provider is Provider.VLLM else None
    return AsyncOpenAI(api_key=_key(tier), base_url=base_url, http_client=_capture_http_client())


def _build_translategemma(tier: Tier) -> TGDescriptor:
    """TRANSLATEGEMMA kind -> aiohttp text-completion descriptor."""
    if tier.provider is not Provider.TRANSLATEGEMMA:
        raise ValueError(f"provider {tier.provider} is not valid for a TRANSLATEGEMMA step")
    if not tier.endpoint:
        raise ValueError("translategemma tier requires endpoint")
    endpoint = tier.endpoint.rstrip("/")
    return TGDescriptor(
        completions_url=f"{endpoint}/completions",
        model_id=tier.model,
        endpoint=endpoint,
    )


@lru_cache(maxsize=256)
def build_handle(tier: Tier, kind: StepClientKind) -> Any:
    """Build (and cache) the live handle for a tier under a client kind.

    Deferred: nothing is constructed until a tier is attempted, then it is cached
    on the frozen ``(tier, kind)`` key so repeated resolutions reuse one client.
    """
    if kind is StepClientKind.AGENT:
        # anthropic/gemini legality is enforced inside _build_agent_model.
        if tier.provider is Provider.TRANSLATEGEMMA:
            raise ValueError("translategemma is only valid for a TRANSLATEGEMMA step")
        return _build_agent_model(tier)
    if kind is StepClientKind.PRE_TRANSLATION:
        if tier.provider is Provider.ANTHROPIC:
            from anthropic import AsyncAnthropic

            return AsyncAnthropic(api_key=_key(tier))
        return _build_pretranslation(tier)
    if kind is StepClientKind.TRANSLATEGEMMA:
        return _build_translategemma(tier)
    raise ValueError(f"unknown step client kind: {kind}")


def tier_client_kind(step_client_kind: StepClientKind, tier: Tier) -> StepClientKind:
    """Per-tier client kind for a step.

    Every step uses its single fixed kind EXCEPT POST_TRANSLATION, whose chain is
    mixed-provider: the TranslateGemma primary builds as an aiohttp
    text-completion :class:`TGDescriptor`, while a cross-provider LLM overflow tier
    (openai/vllm/azure/anthropic/gemini) builds as a provider-native async client. The step
    client kind for POST_TRANSLATION is ``TRANSLATEGEMMA`` (the primary's kind), so
    only a NON-TranslateGemma tier under that step is redirected to ``PRE_TRANSLATION``."""
    if step_client_kind is StepClientKind.TRANSLATEGEMMA and tier.provider is not Provider.TRANSLATEGEMMA:
        return StepClientKind.PRE_TRANSLATION
    return step_client_kind
