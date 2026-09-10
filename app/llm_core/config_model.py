"""Config data model for the unified LLM pipeline (P0).

Four inert-config concepts — ``Tier`` / ``StepConfig`` / ``NamedProfile`` /
``PipelineConfig`` — plus the enums that discriminate provider, LLM step, and
step-client-kind. Nothing here builds a client or reads a secret; a
``Tier`` merely *names* the secret env var (``api_key_env``) so keys never enter
the config file. The factory turns tiers into live handles at resolve time.

Kept import-clean (stdlib + pydantic only) so the voice repo can mirror the same
public API and the eventual repo-merge stays a mechanical convergence.
"""

from __future__ import annotations

from dataclasses import dataclass
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


class AdmissionPolicy(str, Enum):
    AUTO = "auto"
    NONE = "none"
    MANAGED = "managed"


class Step(str, Enum):
    """LLM steps the config must cover. Chat has 5 (no non_meaningful)."""

    PRE_TRANSLATION = "pre_translation"
    MODERATION = "moderation"
    AGENT = "agent"
    SUGGESTIONS = "suggestions"
    POST_TRANSLATION = "post_translation"


class StepClientKind(str, Enum):
    """How the engine consumes a tier at a call site."""

    AGENT = "agent"            # pydantic-ai Model (agent loop, chat moderation, suggestions)
    PRE_TRANSLATION = "pre_translation"  # provider-native raw client
    TRANSLATEGEMMA = "translategemma"  # aiohttp text-completion descriptor (post-translation)


class Tier(BaseModel):
    """One inert tier in a step's chain (primary first, fallbacks after).

    Frozen so it is hashable and usable as an ``lru_cache`` key in the factory.
    ``api_key_env`` names the secret; the value is read when a handle is built via
    the centralized secret provider and never stored in pipeline config.
    """

    provider: Provider
    model: str
    endpoint: Optional[str] = None
    api_key_env: Optional[str] = None
    timeout_ms: Optional[int] = None
    # Distinct FIRST-token deadline (ms) — bounds only the wait for the first
    # streamed token, independent of ``timeout_ms`` (the overall/total per-attempt
    # cap). Set for the post-translation TranslateGemma tier so a saturated-but-
    # ALIVE TG overflows in single-digit seconds instead of blocking a voice turn
    # for the full 60s total. ``None`` -> the consumer falls back to ``timeout_ms``.
    ttft_ms: Optional[int] = None
    api_version: Optional[str] = None
    admission: AdmissionPolicy = AdmissionPolicy.AUTO
    label: Optional[str] = None

    model_config = {"frozen": True}


class ConcurrencyGate(BaseModel):
    """Explicit config for the P3 concurrency-gauge trigger on a step.

    ``metrics_url`` is the vLLM Prometheus ``/metrics`` URL, given **explicitly**
    — never derived by regex-stripping ``/v1`` off the inference endpoint (that is
    bh-voice-prod's fragile derivation; the plan §2 hardens it out). ``max_concurrency``
    is the in-flight (``num_requests_running + num_requests_waiting``) threshold
    at/above which this step's vLLM tier is DEPRIORITIZED (reordered toward the
    back) so the managed tier is tried first under load. The gauge only reorders;
    it never drops a tier.

    ``overflow_tier`` (M3) makes the concurrency-overflow target SEPARATELY
    configurable — a specifically chosen overflow model, independent of the
    session-% profile's own chain (req #3: "the overflow model is set separately —
    no mixing of variables from the session-% selection"). It is a full ``Tier``
    (provider/model/endpoint/api_key_env/timeout). When set AND the shed roll
    fires, the shed request is routed to THAT tier (moved to the FRONT of the
    chain, original tiers following as further fallback) rather than merely
    deprioritizing the saturated vLLM tier behind the profile's managed fallback.
    Left ``None`` (the default), behaviour is byte-identical to the reorder-behind-
    managed shed — the overflow is the profile's own fallback, unchanged.
    """

    metrics_url: str
    max_concurrency: int = 10
    overflow_tier: Optional[Tier] = None

    model_config = {"frozen": True}


class Triggers(BaseModel):
    """Composable pre-flight trigger config."""

    concurrency_gate: Optional[ConcurrencyGate] = None

    model_config = {"frozen": True}


class StepConfig(BaseModel):
    tiers: list[Tier] = Field(min_length=1)
    triggers: Triggers = Triggers()

    model_config = {"frozen": True}


@dataclass(frozen=True)
class StepPlan:
    """The immutable set of tiers that can execute for one configured step."""

    step: Step
    tiers: tuple[Tier, ...]
    concurrency_gate: Optional[ConcurrencyGate]

    @property
    def candidates(self) -> tuple[Tier, ...]:
        """Every reachable tier, including a separately configured overflow."""
        overflow = (
            self.concurrency_gate.overflow_tier
            if self.concurrency_gate is not None
            else None
        )
        if overflow is None or overflow in self.tiers:
            return self.tiers
        return (*self.tiers, overflow)


class ProfileCapabilities(BaseModel):
    """Optional application-policy overrides for a named profile."""

    requires_translation: Optional[bool] = None
    history_max_tokens: Optional[int] = Field(default=None, gt=0)

    model_config = {"frozen": True}


class NamedProfile(BaseModel):
    name: str
    weight: int = Field(ge=0, le=100)
    capabilities: Optional[ProfileCapabilities] = None
    steps: dict[Step, StepConfig] = {}

    model_config = {"frozen": True}


class PipelineConfig(BaseModel):
    profiles: list[NamedProfile]
    defaults: dict[Step, StepConfig] = {}
    fallback_enabled: bool = False

    model_config = {"frozen": True}

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

    def step_plan(self, profile: NamedProfile, step: Step) -> Optional[StepPlan]:
        """Normalize one step to the tiers that can execute under current policy."""
        configured = self.step_config(profile, step)
        if configured is None:
            return None
        if not self.fallback_enabled:
            return StepPlan(step, (configured.tiers[0],), None)
        return StepPlan(
            step,
            tuple(configured.tiers),
            configured.triggers.concurrency_gate,
        )
