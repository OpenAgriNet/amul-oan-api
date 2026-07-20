"""Config data model for the unified LLM pipeline (P0).

Four inert-config concepts — ``Tier`` / ``StepConfig`` / ``NamedProfile`` /
``PipelineConfig`` — plus the enums that discriminate provider, api-style, LLM
step, and step-client-kind. Nothing here builds a client or reads a secret; a
``Tier`` merely *names* the secret env var (``api_key_env``) so keys never enter
the config file. The factory turns tiers into live handles at resolve time.

Kept import-clean (stdlib + pydantic only) so the voice repo can mirror the same
public API and the eventual repo-merge stays a mechanical convergence.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class Provider(str, Enum):
    VLLM = "vllm"
    OPENAI = "openai"
    AZURE = "azure-openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    TRANSLATEGEMMA = "translategemma"


class ApiStyle(str, Enum):
    CHAT = "chat"
    TEXT_COMPLETION = "text_completion"


class Step(str, Enum):
    """LLM steps the config must cover. Chat has 5 (no non_meaningful)."""

    PRE_TRANSLATION = "pre_translation"
    MODERATION = "moderation"
    AGENT = "agent"
    SUGGESTIONS = "suggestions"
    POST_TRANSLATION = "post_translation"


class StepClientKind(str, Enum):
    """How the engine consumes a materialized tier at a call site."""

    AGENT = "agent"            # pydantic-ai Model (agent loop, chat moderation, suggestions)
    RAW_OPENAI = "raw_openai"  # AsyncOpenAI client (pre-translation)
    TRANSLATEGEMMA = "translategemma"  # aiohttp text-completion descriptor (post-translation)


class Tier(BaseModel):
    """One inert tier in a step's chain (primary first, fallbacks after).

    Frozen so it is hashable and usable as an ``lru_cache`` key in the factory.
    ``api_key_env`` names the secret env var; the VALUE is read at materialize
    time via ``os.getenv`` and never stored on the model or in any file.
    """

    provider: Provider
    model: str
    endpoint: Optional[str] = None
    api_key_env: Optional[str] = None
    api_style: ApiStyle = ApiStyle.CHAT
    timeout_ms: Optional[int] = None
    api_version: Optional[str] = None
    max_tokens: Optional[int] = None
    label: Optional[str] = None

    model_config = {"frozen": True}


class Triggers(BaseModel):
    """Composable pre-flight trigger config. Health / concurrency gates are
    stubs in P0 (declared, never evaluated) — wired in P2/P3."""

    ttft_deadline_ms: Optional[int] = None
    health_check: bool = False
    concurrency_gate: Optional[dict] = None


class StepConfig(BaseModel):
    tiers: list[Tier] = Field(min_length=1)
    triggers: Triggers = Triggers()

    model_config = {"frozen": True}


class NamedProfile(BaseModel):
    name: str
    weight: int = Field(ge=0, le=100)
    steps: dict[Step, StepConfig] = {}


class PipelineConfig(BaseModel):
    profiles: list[NamedProfile]
    defaults: dict[Step, StepConfig] = {}
    sticky_ttl_s: int = 604800
    fallback_enabled: bool = False

    @model_validator(mode="after")
    def _validate(self) -> "PipelineConfig":
        if not self.profiles:
            raise ValueError("PipelineConfig requires at least one profile")
        names = [p.name for p in self.profiles]
        if len(names) != len(set(names)):
            raise ValueError(f"profile names must be unique, got {names}")
        total = sum(p.weight for p in self.profiles)
        if total != 100:
            raise ValueError(f"profile weights must sum to 100, got {total}")
        return self

    def by_name(self, name: str) -> Optional[NamedProfile]:
        for p in self.profiles:
            if p.name == name:
                return p
        return None

    def step_config(self, profile: NamedProfile, step: Step) -> Optional[StepConfig]:
        """Resolve a step's config for a profile, falling back to defaults."""
        return profile.steps.get(step) or self.defaults.get(step)
