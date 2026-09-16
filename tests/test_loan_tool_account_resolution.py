"""Coverage for loan tool account resolution via Layer 2 farmer envelope."""
import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from agents.deps import FarmerAccount, FarmerContext
from agents.tools import loan as loan_tool
from app.models.farmer_transport import FarmerDataEnvelope, FarmerRecord


def _ctx(
    *,
    mobile: str | None = "9876543210",
    accounts: list[FarmerAccount] | None = None,
):
    return SimpleNamespace(
        deps=FarmerContext(
            query="loan",
            mobile=mobile,
            farmer_accounts=accounts or [],
        )
    )


def _envelope() -> FarmerDataEnvelope:
    return FarmerDataEnvelope.from_records(
        [
            FarmerRecord(
                unionCode="1",
                societyCode="S1",
                farmerCode="F1",
                farmerName="Ramesh",
            ),
            FarmerRecord(
                unionCode="1",
                societyCode="S2",
                farmerCode="F2",
                farmerName="Sita",
            ),
        ],
        lookup_status="found",
    )


def test_resolve_accounts_returns_existing_without_fetch(monkeypatch):
    monkeypatch.setattr(loan_tool.settings, "loan_check_milk_enabled", True)
    existing = [
        FarmerAccount(
            union_code="1",
            society_code="S1",
            farmer_code="F1",
            farmer_name="Ramesh",
        )
    ]
    ctx = _ctx(accounts=existing)
    out = asyncio.run(loan_tool._resolve_accounts(ctx))
    assert out == existing


def test_resolve_accounts_returns_empty_when_no_mobile(monkeypatch):
    monkeypatch.setattr(loan_tool.settings, "loan_check_milk_enabled", True)
    out = asyncio.run(loan_tool._resolve_accounts(_ctx(mobile=None)))
    assert out == []


def test_resolve_accounts_returns_empty_when_milk_check_disabled(monkeypatch):
    monkeypatch.setattr(loan_tool.settings, "loan_check_milk_enabled", False)
    out = asyncio.run(loan_tool._resolve_accounts(_ctx()))
    assert out == []


def test_resolve_accounts_fetches_from_layer2_when_needed(monkeypatch):
    monkeypatch.setattr(loan_tool.settings, "loan_check_milk_enabled", True)
    fetch = AsyncMock(return_value=_envelope())
    monkeypatch.setattr(
        "agents.services.farmer_cache.get_or_fetch_farmer_data",
        fetch,
    )
    out = asyncio.run(loan_tool._resolve_accounts(_ctx()))
    assert [(a.union_code, a.society_code, a.farmer_code) for a in out] == [
        ("1", "S1", "F1"),
        ("1", "S2", "F2"),
    ]
    fetch.assert_awaited_once_with("9876543210")


def test_resolve_accounts_returns_empty_on_layer2_exception(monkeypatch):
    monkeypatch.setattr(loan_tool.settings, "loan_check_milk_enabled", True)

    async def boom(_mobile):
        raise RuntimeError("cache read failed")

    monkeypatch.setattr("agents.services.farmer_cache.get_or_fetch_farmer_data", boom)
    out = asyncio.run(loan_tool._resolve_accounts(_ctx()))
    assert out == []
