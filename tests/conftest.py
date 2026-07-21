import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def make_materialized_tier(kind, handle, *, model_name=None, provider=None,
                           endpoint=None, timeout=None):
    """Build a MaterializedTier for tests (lazy import so env-before-import test
    modules stay in control of import order)."""
    from app.llm_core.factory import MaterializedTier
    return MaterializedTier(
        kind=kind,
        handle=handle,
        model_name=model_name or ("gemma-test" if kind == "oss" else "gpt-test"),
        provider=provider or ("vllm" if kind == "oss" else "openai"),
        endpoint=endpoint or ("http://oss:8020/v1" if kind == "oss" else "managed"),
        timeout=timeout,
    )


def install_variant_chain(monkeypatch, fb, *, oss_handle=None, managed_handle=None,
                          oss_timeout=None, managed_timeout=None,
                          oss_endpoint="http://oss:8020/v1"):
    """Replace ``fallback._resolve_chain`` (the seam the walkers read) with a
    controlled, variant-keyed MaterializedTier chain: variant 'oss' -> [oss,
    managed], anything else -> [managed]. Post-P4 the real chain comes from the
    config-driven weighted-profile resolver (exercised in test_split); the walker
    tests only need a deterministic chain to drive the classify/first-token-commit
    logic, so they stub this seam."""
    def _chain(variant):
        if variant == "oss":
            return [
                make_materialized_tier("oss", oss_handle, timeout=oss_timeout,
                                       endpoint=oss_endpoint),
                make_materialized_tier("managed", managed_handle, timeout=managed_timeout),
            ]
        return [make_materialized_tier("managed", managed_handle, timeout=managed_timeout)]

    async def _resolve_chain(*, pipeline, session_id, variant):
        return _chain(variant)

    monkeypatch.setattr(fb, "_resolve_chain", _resolve_chain)
    return _chain


@pytest.fixture
def materialized_tier():
    """The MaterializedTier builder, as a fixture (keeps import lazy)."""
    return make_materialized_tier


@pytest.fixture
def install_chain(monkeypatch):
    """Install a controlled variant-keyed chain on ``fallback._resolve_chain``.
    Usage: ``install_chain(oss_timeout=0.05)`` inside a test/fixture."""
    from app.services import fallback as fb

    def _install(**kw):
        return install_variant_chain(monkeypatch, fb, **kw)

    return _install


def pytest_configure(config):
    # Registers the marker used by the ported voice integration tests (live-model
    # / live-endpoint regressions; skipped by default via per-test env-var gates).
    config.addinivalue_line(
        "markers",
        "integration: live-model / live-endpoint regressions; skipped by default "
        "via per-test env-var gates (see individual test modules).",
    )
