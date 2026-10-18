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
from typing import Literal, Optional

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
    """LLM steps a config can cover. Chat configures the first five;
    ``non_meaningful`` is voice's, and a chat config simply leaves it out."""

    PRE_TRANSLATION = "pre_translation"
    MODERATION = "moderation"
    AGENT = "agent"
    SUGGESTIONS = "suggestions"
    POST_TRANSLATION = "post_translation"
    NON_MEANINGFUL = "non_meaningful"


class StepClientKind(str, Enum):
    """How the engine consumes a tier at a call site."""

    AGENT = "agent"            # pydantic-ai Model (agent loop, chat moderation, suggestions)
    PRE_TRANSLATION = "pre_translation"  # provider-native raw client
    TRANSLATEGEMMA = "translategemma"  # aiohttp text-completion descriptor (post-translation)
    RAW_OPENAI = "raw_openai"  # bare OpenAI-compatible client (voice's moderation / classifier calls)


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
    # Post-translation only: constrain this vLLM tier's decoding to the target
    # language's script (vLLM structured outputs) and give it the strict
    # translation-only prompt written for that grammar. Gujarati is the only
    # target with a grammar; other targets translate unconstrained.
    target_script_only: bool = False
    api_version: Optional[str] = None
    admission: AdmissionPolicy = AdmissionPolicy.AUTO
    label: Optional[str] = None

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def _script_only_needs_vllm(self) -> "Tier":
        if self.target_script_only and self.provider is not Provider.VLLM:
            raise ValueError(
                "target_script_only needs a vllm tier (it sends vLLM structured_outputs)"
            )
        return self


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
            return StepPlan((configured.tiers[0],), None)
        tiers = tuple(configured.tiers)
        gate = configured.triggers.concurrency_gate
        # The gauge describes the selected vLLM primary. A vLLM tier that is only
        # reachable after a healthy managed primary must not reorder that primary.
        if tiers[0].provider is not Provider.VLLM:
            gate = None
        return StepPlan(
            tiers,
            gate,
        )

    def effective_capabilities(
        self,
        profile: NamedProfile,
        *,
        history_default_tokens: int = 80_000,
        history_vllm_gemma_tokens: int = 10_000,
    ) -> ProfileCapabilities:
        """Resolve application policy once from the active agent plan + overrides."""
        plan = self.step_plan(profile, Step.AGENT)
        if plan is None:
            raise ValueError(f"profile={profile.name} has no agent plan")
        is_vllm = [tier for tier in plan.candidates if tier.provider is Provider.VLLM]
        configured = profile.capabilities
        return ProfileCapabilities(
            history_max_tokens=(
                configured.history_max_tokens
                if configured is not None and configured.history_max_tokens is not None
                else history_vllm_gemma_tokens
                if any("gemma" in tier.model.lower() for tier in is_vllm)
                else history_default_tokens
            ),
        )


class SurfaceSettings(BaseModel):
    """A surface's own settings: the ``settings:`` block of its pipeline file.

    For what has to differ between chat and voice when one process serves both.
    Read only when the deployment loads one file per surface (``PIPELINE_CONFIG_DIR``).
    A setting left out is None, and its reader falls back to the env var named
    beside it, so a file without the block changes nothing.
    """

    #: How long the surface's conversation history is kept after its last turn
    #: (HISTORY_CACHE_TTL_SECONDS).
    history_ttl_seconds: Optional[int] = Field(default=None, gt=0)
    #: The Langfuse environment the surface's traces land in
    #: (LANGFUSE_TRACING_ENVIRONMENT).
    langfuse_environment: Optional[str] = None

    model_config = {"frozen": True, "extra": "forbid"}


class VoiceSettings(SurfaceSettings):
    """Voice's settings, with the knobs only voice reads."""

    trace_user_id_mode: Optional[Literal["hash", "phone"]] = None  # VOICE_TRACE_USER_ID_MODE
    trace_text_mode: Optional[Literal["none", "preview_hash", "full"]] = None  # VOICE_TRACE_TEXT_MODE
    nudge_timeout_seconds: Optional[float] = Field(default=None, gt=0)  # NUDGE_TIMEOUT_SECONDS
    outbound_intro_enabled: Optional[bool] = None  # OUTBOUND_INTRO_ENABLED
    #: One of agents.voice.agent.VOICE_PROMPT_VARIANTS (VOICE_AGENT_PROMPT_VERSION).
    agent_prompt_version: Optional[Literal["mixed", "gpt-5.1", "gemma4"]] = None
    pretranslation_timeout_seconds: Optional[float] = Field(default=None, gt=0)  # OPENAI_PRETRANSLATION_TIMEOUT_SECONDS
