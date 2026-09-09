"""Unified LLM configuration and execution boundary."""

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
from app.llm_core import runtime, resolver, split, concurrency, trace
from app.llm_core.execution import (
    begin_trace,
    ModelInfo,
    primary_info,
    profile,
    run,
    run_adapter,
    stream,
    stream_adapter,
)

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
    "trace",
    "ModelInfo",
    "begin_trace",
    "primary_info",
    "profile",
    "run",
    "run_adapter",
    "stream",
    "stream_adapter",
]
