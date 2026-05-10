"""Tests for parallel Firebase Cloud Messaging token verification.

The Firebase Admin SDK is heavy and requires real service-account
credentials, so these tests mock the per-app verification primitive
(:func:`app.auth.fcm_auth._verify_against_app_sync`) and the lazy
initializer. That keeps the suite hermetic while still exercising the
concurrency, first-success, and short-circuit semantics we care about.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from app.auth import fcm_auth


@pytest.fixture(autouse=True)
def _stub_firebase(monkeypatch: pytest.MonkeyPatch):
    """Pretend two Firebase apps are already initialized.

    The actual app objects are unused — every test stubs out
    ``_verify_against_app_sync``, which is the only function that
    touches them.
    """
    monkeypatch.setattr(fcm_auth, "_firebase_initialized", True)
    monkeypatch.setattr(
        fcm_auth, "_firebase_apps", {"default": object(), "secondary": object()}
    )


def _make_verifier(per_app_results: dict[str, tuple[bool, float]]):
    """Build a fake ``_verify_against_app_sync`` from a result spec.

    ``per_app_results`` maps app name to ``(returns_true, sleep_seconds)``.
    """

    def _fake(fcm_token: str, app_name: str, app: object) -> bool:
        result, delay = per_app_results[app_name]
        if delay:
            time.sleep(delay)
        return result

    return _fake


# ---------------------------------------------------------------------------
# Sync path (back-compat)
# ---------------------------------------------------------------------------


def test_sync_verify_returns_true_when_any_app_accepts(monkeypatch):
    monkeypatch.setattr(
        fcm_auth,
        "_verify_against_app_sync",
        _make_verifier({"default": (False, 0), "secondary": (True, 0)}),
    )
    assert fcm_auth.verify_fcm_token("token") is True


def test_sync_verify_returns_false_when_all_apps_reject(monkeypatch):
    monkeypatch.setattr(
        fcm_auth,
        "_verify_against_app_sync",
        _make_verifier({"default": (False, 0), "secondary": (False, 0)}),
    )
    assert fcm_auth.verify_fcm_token("token") is False


# ---------------------------------------------------------------------------
# Async path (new)
# ---------------------------------------------------------------------------


def test_async_verify_returns_true_when_any_app_accepts(monkeypatch):
    monkeypatch.setattr(
        fcm_auth,
        "_verify_against_app_sync",
        _make_verifier({"default": (False, 0), "secondary": (True, 0)}),
    )
    assert asyncio.run(fcm_auth.verify_fcm_token_async("token")) is True


def test_async_verify_returns_false_when_all_apps_reject(monkeypatch):
    monkeypatch.setattr(
        fcm_auth,
        "_verify_against_app_sync",
        _make_verifier({"default": (False, 0), "secondary": (False, 0)}),
    )
    assert asyncio.run(fcm_auth.verify_fcm_token_async("token")) is False


def test_async_verify_returns_false_when_no_apps_configured(monkeypatch):
    monkeypatch.setattr(fcm_auth, "_firebase_apps", {})
    assert asyncio.run(fcm_auth.verify_fcm_token_async("token")) is False


def _time_coroutine(coro):
    """Run *coro* to completion and return ``(result, elapsed)``.

    Uses a manually-managed event loop so the timing window covers only
    the coroutine itself. Closing the loop or relying on
    :func:`asyncio.run` would block on the default thread-pool shutdown
    (Python 3.10+), which would hide the very short-circuit / parallel
    behaviour we want to assert: pending background threads are abandoned
    intentionally because their results are no longer needed.
    """
    loop = asyncio.new_event_loop()
    try:
        start = time.perf_counter()
        result = loop.run_until_complete(coro)
        elapsed = time.perf_counter() - start
        return result, elapsed
    finally:
        # Close the loop after the timing window so its executor-shutdown
        # wait (which can block until orphaned threads finish their sleep)
        # doesn't pollute the measurement.
        loop.close()


def test_async_verify_runs_per_app_checks_concurrently(monkeypatch):
    """Wall-clock for two 200ms checks in parallel must be well below 400ms."""
    monkeypatch.setattr(
        fcm_auth,
        "_verify_against_app_sync",
        _make_verifier(
            {"default": (False, 0.20), "secondary": (False, 0.20)}
        ),
    )

    result, elapsed = _time_coroutine(fcm_auth.verify_fcm_token_async("token"))

    assert result is False
    # Sequential would be ~0.40s; with parallel execution we want <0.30s.
    # The 0.30s ceiling absorbs thread-pool / scheduler overhead on slower
    # CI runners while still proving the calls didn't run sequentially.
    assert elapsed < 0.30, f"verifications did not run in parallel (took {elapsed:.3f}s)"


def test_async_verify_returns_on_first_success_without_waiting_for_slow_app(
    monkeypatch,
):
    """A slow rejecting app must not delay the response when a fast one accepts."""
    monkeypatch.setattr(
        fcm_auth,
        "_verify_against_app_sync",
        _make_verifier(
            {
                "default": (False, 0.50),  # slow rejector
                "secondary": (True, 0.01),  # fast acceptor
            }
        ),
    )

    result, elapsed = _time_coroutine(fcm_auth.verify_fcm_token_async("token"))

    assert result is True
    # Pure sequential would block on the 0.50s rejector before trying the
    # fast acceptor; first-success returns once secondary completes.
    assert elapsed < 0.20, (
        f"first-success short-circuit not honoured (took {elapsed:.3f}s)"
    )
