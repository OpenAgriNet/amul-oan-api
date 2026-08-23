"""Coverage for /user/me profile router behavior."""
import asyncio
import os
from unittest.mock import AsyncMock

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from app.models.farmer_transport import FarmerDataEnvelope, FarmerRecord
from app.routers import user as user_router


def _envelope() -> FarmerDataEnvelope:
    return FarmerDataEnvelope.from_records(
        [
            FarmerRecord(
                farmerName="Ramesh",
                farmerCode="F1",
                societyCode="S1",
                unionCode="U1",
            )
        ],
        lookup_status="found",
    )


def test_user_profile_anonymous_when_no_user_info():
    out = asyncio.run(user_router.get_user_profile(user_info={}))
    assert out == {"status": "anonymous", "farmer": None}


def test_user_profile_anonymous_when_phone_missing():
    out = asyncio.run(user_router.get_user_profile(user_info={"sub": None}))
    assert out == {"status": "anonymous", "farmer": None}


def test_user_profile_anonymous_when_anon_token():
    out = asyncio.run(
        user_router.get_user_profile(
            user_info={"phone": "anon-1234", "anonymous": False}
        )
    )
    assert out == {"status": "anonymous", "farmer": None}


def test_user_profile_ok_on_envelope(monkeypatch):
    monkeypatch.setattr(
        user_router,
        "get_or_fetch_farmer_data",
        AsyncMock(return_value=_envelope()),
    )
    out = asyncio.run(user_router.get_user_profile(user_info={"phone": "9876543210"}))
    assert out["status"] == "ok"
    assert out["farmer"] is not None
    assert out["farmer"]["lookupStatus"] == "found"
    assert len(out["farmer"]["farmers"]) == 1


def test_user_profile_not_found_when_cache_service_returns_none(monkeypatch):
    monkeypatch.setattr(
        user_router,
        "get_or_fetch_farmer_data",
        AsyncMock(return_value=None),
    )
    out = asyncio.run(user_router.get_user_profile(user_info={"phone": "9876543210"}))
    assert out == {"status": "not_found", "farmer": None}


def test_user_profile_error_when_cache_service_raises(monkeypatch):
    async def boom(_phone):
        raise RuntimeError("cache down")

    monkeypatch.setattr(user_router, "get_or_fetch_farmer_data", boom)
    out = asyncio.run(user_router.get_user_profile(user_info={"phone": "9876543210"}))
    assert out == {"status": "error", "farmer": None}
