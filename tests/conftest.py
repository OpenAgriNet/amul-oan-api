import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def make_materialized_tier(kind, handle, *, model_name=None, provider=None,
                           endpoint=None, timeout=None):
    """Build a lazy execution target with a supplied fake handle."""
    from app.llm_core.config_model import AdmissionPolicy, Provider, StepClientKind, Tier
    from app.llm_core.execution import ExecutionTarget

    provider_name = provider or ("vllm" if kind == "oss" else "openai")
    tier = Tier(
        provider=Provider(provider_name),
        model=model_name or ("gemma-test" if kind == "oss" else "gpt-test"),
        endpoint=endpoint if endpoint is not None else (
            "http://oss:8020/v1" if kind == "oss" else None
        ),
        timeout_ms=int(timeout * 1000) if timeout is not None else None,
        admission=AdmissionPolicy.MANAGED if kind == "managed" else AdmissionPolicy.NONE,
    )
    target = ExecutionTarget(tier, StepClientKind.AGENT)
    object.__setattr__(target, "handle", handle)
    return target


def install_variant_chain(*, oss_handle=None, managed_handle=None,
                          oss_timeout=None, managed_timeout=None,
                          oss_endpoint="http://oss:8020/v1"):
    """Build a controlled lazy target chain for walker tests."""
    def _chain(profile_name):
        if profile_name == "oss":
            return [
                make_materialized_tier("oss", oss_handle, timeout=oss_timeout,
                                       endpoint=oss_endpoint),
                make_materialized_tier("managed", managed_handle, timeout=managed_timeout),
            ]
        return [make_materialized_tier("managed", managed_handle, timeout=managed_timeout)]

    return _chain


@pytest.fixture
def materialized_tier():
    """A lazy execution-target builder with an injected fake handle."""
    return make_materialized_tier


@pytest.fixture
def install_chain():
    """Return a controlled OSS-then-managed chain for walker tests."""
    def _install(**kw):
        return install_variant_chain(**kw)("oss")

    return _install


def pytest_configure(config):
    # Registers the marker used by the ported voice integration tests (live-model
    # / live-endpoint regressions; skipped by default via per-test env-var gates).
    config.addinivalue_line(
        "markers",
        "integration: live-model / live-endpoint regressions; skipped by default "
        "via per-test env-var gates (see individual test modules).",
    )
