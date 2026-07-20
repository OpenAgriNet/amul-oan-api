"""Unified LLM pipeline core (P0).

Isolated, import-clean subpackage with a parallel public API to the voice repo's
future ``llm_core`` so the eventual repo-merge is a mechanical convergence.
Repo-specific wiring (chat's call sites, the flag branch) stays outside the core.

P0 = ZERO behaviour change: everything here is inert until ``LLM_CORE_ENABLED``
is turned on, and even then resolves the same provider/base_url/model/timeout as
the legacy singletons for the current env (see ``runtime.self_check``).
"""

from app.llm_core.config_model import (
    ApiStyle,
    ConcurrencyGate,
    NamedProfile,
    PipelineConfig,
    Provider,
    Step,
    StepClientKind,
    StepConfig,
    Tier,
    Triggers,
)
from app.llm_core.factory import (
    MaterializedTier,
    TGDescriptor,
    build_handle,
    materialize,
)
from app.llm_core.legacy_shim import synthesize_from_env
from app.llm_core import runtime, resolver, split, concurrency

__all__ = [
    "ApiStyle",
    "ConcurrencyGate",
    "NamedProfile",
    "PipelineConfig",
    "Provider",
    "Step",
    "StepClientKind",
    "StepConfig",
    "Tier",
    "Triggers",
    "MaterializedTier",
    "TGDescriptor",
    "build_handle",
    "materialize",
    "synthesize_from_env",
    "runtime",
    "resolver",
    "split",
    "concurrency",
]
