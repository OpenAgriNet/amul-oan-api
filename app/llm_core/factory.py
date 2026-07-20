"""Provider factory — the single seam that turns an inert :class:`Tier` into a
live handle. Superset of both repos' model construction.

* ``build_handle(tier, kind)`` is ``@lru_cache``'d on the frozen ``(tier, kind)``
  key, reproducing the legacy "build once at import" property without import-time
  eagerness — nothing is built until a tier is actually resolved.
* Providers: vllm + openai (OpenAI-compatible ``base_url``), azure-openai
  (``AsyncAzureOpenAI`` + ``OpenAIProvider(openai_client=...)``), anthropic
  (``AnthropicModel``), gemini (``GeminiModel(provider='google-gla')``), and
  translategemma (an aiohttp text-completion *descriptor*, not a client).
* The httpx **boundary-capture** hook (adopted from voice's
  ``_build_openai_compatible_model``) now covers every OpenAI-compatible client
  kind — chat's legacy path never had it. 600s read / 5s connect matches the
  OpenAI SDK default so long streaming agent runs are not aborted.

Legality (enforced): translategemma only for the TRANSLATEGEMMA client kind;
anthropic / gemini only for the AGENT kind (raw-openai steps can't use them).

NOTE (pydantic-ai version): the chat repo is pinned to pydantic-ai 0.2.4, whose
model class is ``OpenAIModel`` and whose Gemini arm is
``GeminiModel(provider='google-gla')``. Voice runs 1.x (``OpenAIChatModel`` /
``GoogleModel`` + ``GoogleProvider``). The public factory API is identical across
repos; only these two construction lines differ — the per-repo delta the merge
will reconcile last.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

import httpx
from openai import AsyncOpenAI, AsyncAzureOpenAI
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.openai import OpenAIProvider

from helpers.utils import get_logger
from app.llm_core.config_model import Provider, Tier, StepClientKind

# Version-tolerant OpenAI model class: the deploy target pins pydantic-ai 1.x
# (``OpenAIChatModel``, matching voice); older local envs expose ``OpenAIModel``.
# Same construction either way — only the class name moved.
try:  # pydantic-ai 1.x
    from pydantic_ai.models.openai import OpenAIChatModel as _OpenAIModel
except ImportError:  # pragma: no cover - older pydantic-ai
    from pydantic_ai.models.openai import OpenAIModel as _OpenAIModel

logger = get_logger(__name__)

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
    so it materializes to an inert descriptor the translation service consumes
    directly, never a client object."""

    completions_url: str
    model_id: str
    endpoint: str


def _key(tier: Tier) -> Optional[str]:
    """Read the named secret env var at materialize time (never stored)."""
    if not tier.api_key_env:
        return None
    return os.getenv(tier.api_key_env)


# ── low-level builders ────────────────────────────────────────────────────────
def _build_agent_model(tier: Tier) -> Any:
    """AGENT kind -> pydantic-ai Model."""
    if tier.provider in (Provider.VLLM, Provider.OPENAI):
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
        # AnthropicModel reads ANTHROPIC_API_KEY from the environment.
        return AnthropicModel(tier.model)

    if tier.provider is Provider.GEMINI:
        # Rebased off the dead-file ``feat/adding-google-as-model-provider`` arm.
        # Deploy target (pydantic-ai 1.x): GoogleModel + GoogleProvider(api_key=).
        # Older local envs: GeminiModel(provider='google-gla'), reading
        # GEMINI_API_KEY / GOOGLE_API_KEY (the chat branch's exact behaviour).
        api_key = _key(tier) or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        try:  # pydantic-ai 1.x
            from pydantic_ai.models.google import GoogleModel
            from pydantic_ai.providers.google import GoogleProvider
            return GoogleModel(tier.model, provider=GoogleProvider(api_key=api_key))
        except ImportError:  # pragma: no cover - older pydantic-ai
            from pydantic_ai.models.gemini import GeminiModel
            if api_key:
                os.environ["GEMINI_API_KEY"] = api_key
            return GeminiModel(tier.model, provider="google-gla")

    raise ValueError(f"provider {tier.provider} is not valid for an AGENT step")


def _build_raw_openai(tier: Tier) -> AsyncOpenAI:
    """RAW_OPENAI kind -> AsyncOpenAI client (pre-translation)."""
    if tier.provider not in (Provider.VLLM, Provider.OPENAI, Provider.AZURE):
        raise ValueError(
            f"provider {tier.provider} is not an OpenAI-compatible raw client; "
            "anthropic/gemini/translategemma are not valid for a RAW_OPENAI step"
        )
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

    Deferred: nothing is constructed until a tier is resolved, then it is cached
    on the frozen ``(tier, kind)`` key so repeated resolutions reuse one client.
    """
    if kind is StepClientKind.AGENT:
        # anthropic/gemini legality is enforced inside _build_agent_model.
        if tier.provider is Provider.TRANSLATEGEMMA:
            raise ValueError("translategemma is only valid for a TRANSLATEGEMMA step")
        return _build_agent_model(tier)
    if kind is StepClientKind.RAW_OPENAI:
        return _build_raw_openai(tier)
    if kind is StepClientKind.TRANSLATEGEMMA:
        return _build_translategemma(tier)
    raise ValueError(f"unknown step client kind: {kind}")


@dataclass(frozen=True)
class MaterializedTier:
    """Drop-in successor to ``fallback.Attempt`` (fallback.py:133). Carries the
    live handle plus telemetry labels + per-attempt timeout (seconds)."""

    kind: str
    handle: Any
    model_name: str
    provider: str
    endpoint: str
    timeout: Optional[float]

    @property
    def model(self) -> Any:
        """Back-compat: existing walkers read ``attempt.model``."""
        return self.handle


def materialize(step_client_kind: StepClientKind, tiers: list[Tier]) -> list[MaterializedTier]:
    """Build a live handle per tier, preserving order (primary first). Never
    empty when ``tiers`` is non-empty (StepConfig guarantees ``min_length=1``)."""
    out: list[MaterializedTier] = []
    for tier in tiers:
        handle = build_handle(tier, step_client_kind)
        # kind label mirrors fallback.Attempt exactly: a vLLM (self-hosted) tier is
        # "oss"; every other provider is "managed". The run/stream closures branch
        # only on ``kind == "oss"`` and moderation maps non-"oss" -> managed OpenAI,
        # so this reproduces attempt_chain's [oss, managed] / [managed] labels and
        # keeps ``emit`` telemetry (from_variant/to_variant) byte-identical.
        out.append(
            MaterializedTier(
                kind="oss" if tier.provider is Provider.VLLM else "managed",
                handle=handle,
                model_name=tier.model,
                provider=tier.provider.value,
                endpoint=tier.endpoint or "managed",
                timeout=(tier.timeout_ms / 1000.0) if tier.timeout_ms is not None else None,
            )
        )
    return out
