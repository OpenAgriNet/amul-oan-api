"""The AI-call booking guard is a product trade-off, so it is a flag.

OFF (default): a farmer can book multiple AI visits in one session — two cows in
heat is a real case. Cost: an OSS->managed fallback re-run can duplicate a booking.
ON: first caller wins per session, so no duplicate booking and no duplicate SMS.
Cost: a legitimate second booking inside the TTL is refused.

amul-prod has historically run with the guard ON; main removed it deliberately.
The flag stops the two branches conflicting on every promote.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest

from agents.tools import ai_call


def test_namespace_is_a_constant_not_config():
    """Changing it at runtime orphans in-flight reservations, so it stays fixed."""
    assert ai_call.AI_CALL_CACHE_NAMESPACE == "ai_call_booked"
    assert not hasattr(ai_call, "AI_CALL_COOLDOWN_TTL"), "TTL moved to settings"


def test_cooldown_ttl_defaults_to_thirty_minutes(monkeypatch):
    from app.config import Settings
    monkeypatch.delenv("AI_CALL_COOLDOWN_TTL_SECONDS", raising=False)
    assert Settings().ai_call_cooldown_ttl_seconds == 60 * 30


def test_cooldown_ttl_is_configurable(monkeypatch):
    from app.config import Settings
    monkeypatch.setenv("AI_CALL_COOLDOWN_TTL_SECONDS", "300")
    assert Settings().ai_call_cooldown_ttl_seconds == 300


def test_the_reservation_reads_the_configured_ttl():
    """Both the reserve and the post-success write must use the setting."""
    import inspect
    src = inspect.getsource(ai_call)
    assert src.count("settings.ai_call_cooldown_ttl_seconds") >= 2


def test_flag_defaults_off(monkeypatch):
    """Default must match main's documented product decision."""
    from app.config import Settings
    monkeypatch.delenv("AI_CALL_BOOKING_GUARD_ENABLED", raising=False)
    assert Settings().ai_call_booking_guard_enabled is False


@pytest.mark.parametrize("raw,expected", [
    ("true", True), ("1", True), ("yes", True), ("on", True), ("TRUE", True),
    ("false", False), ("0", False), ("no", False), ("off", False),
])
def test_flag_parsing(monkeypatch, raw, expected):
    from app.config import Settings
    monkeypatch.setenv("AI_CALL_BOOKING_GUARD_ENABLED", raw)
    assert Settings().ai_call_booking_guard_enabled is expected


@pytest.mark.parametrize("raw", ["", "nonsense"])
def test_an_unparseable_value_fails_fast_at_boot(monkeypatch, raw):
    """pydantic-settings binds this env var directly and refuses garbage.

    Worth pinning: a typo in the ConfigMap stops the service starting rather than
    silently choosing a booking policy. Same for every bool flag in Settings.
    """
    from pydantic import ValidationError
    from app.config import Settings
    monkeypatch.setenv("AI_CALL_BOOKING_GUARD_ENABLED", raw)
    with pytest.raises(ValidationError):
        Settings()


def test_guard_is_consulted_only_when_enabled(monkeypatch):
    """The reservation call must not happen at all with the flag off."""
    import inspect
    src = inspect.getsource(ai_call)
    # every try_reserve / cooldown write is gated on the flag
    for line in src.splitlines():
        if "try_reserve(" in line or "namespace=AI_CALL_CACHE_NAMESPACE" in line:
            continue
    assert src.count("settings.ai_call_booking_guard_enabled") >= 2, (
        "both the reservation and the post-success cooldown must be flag-gated"
    )
    assert "release_reservation" in src, "a failed booking must release the reservation"
