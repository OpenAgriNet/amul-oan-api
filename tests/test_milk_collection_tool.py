from types import SimpleNamespace

import pytest

from agents.tools import milk_collection as milk
from agents.tools.models.farmer import FarmerModel
from agents.tools.models.milk_collection import FarmerMilkCollectionResponseModel


def _ctx(mobile="9000000000"):
    return SimpleNamespace(
        deps=SimpleNamespace(mobile=mobile, session_id="session-1", farmer_unions=["kaira"]),
        tool_call_id="tool-1",
    )


def _farmer():
    return FarmerModel.model_validate({
        "unionCode": "0201",
        "societyCode": "001066",
        "farmerCode": "000123",
        "unionName": "Kaira",
    })


@pytest.mark.asyncio
async def test_milk_collection_uses_authenticated_beckn_account(monkeypatch):
    calls = []

    async def farmers(mobile, **kwargs):
        assert mobile == "9000000000"
        return [_farmer()]

    async def fetch(account, **kwargs):
        calls.append((account, kwargs))
        return FarmerMilkCollectionResponseModel.model_validate({
            "result": "success",
            "milk": [{"date": "2026-04-01", "qty": 10, "fat": 6, "snf": 9, "amount": 500}],
            "deduction": [{"date": "2026-04-01", "accountname": "Feed", "amount": 100}],
        })

    monkeypatch.setattr(milk, "fetch_authenticated_farmers", farmers)
    monkeypatch.setattr(milk, "fetch_milk_collection", fetch)
    result = await milk.get_farmer_milk_collection_details(
        _ctx(), "2026-04-01", "2026-04-01"
    )

    assert "fetched successfully" in result
    assert "| 2026-04-01 | - | 10.00 | 6.00 | 9.00 | 500.00 |" in result
    assert calls[0][0].farmer_code == "000123"
    assert calls[0][1]["tool_call_id"] == "tool-1:account-0"


@pytest.mark.asyncio
async def test_missing_mobile_returns_before_beckn(monkeypatch):
    async def unexpected(*args, **kwargs):
        raise AssertionError("Beckn must not be called")

    monkeypatch.setattr(milk, "fetch_authenticated_farmers", unexpected)
    result = await milk.get_farmer_milk_collection_details(
        _ctx(None), "2026-04-01", "2026-04-01"
    )
    assert "signed-in farmer profile" in result


@pytest.mark.asyncio
async def test_invalid_date_returns_before_beckn(monkeypatch):
    async def unexpected(*args, **kwargs):
        raise AssertionError("Beckn must not be called")

    monkeypatch.setattr(milk, "fetch_authenticated_farmers", unexpected)
    result = await milk.get_farmer_milk_collection_details(
        _ctx(), "01-04-2026", "2026-04-01"
    )
    assert "YYYY-MM-DD" in result


@pytest.mark.asyncio
async def test_all_beckn_failures_return_temporary_failure(monkeypatch):
    async def farmers(*args, **kwargs):
        return [_farmer()]

    async def failed(*args, **kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(milk, "fetch_authenticated_farmers", farmers)
    monkeypatch.setattr(milk, "fetch_milk_collection", failed)
    result = await milk.get_farmer_milk_collection_details(
        _ctx(), "2026-04-01", "2026-04-01"
    )
    assert result == (
        "Milk collection lookup failed.\n\n"
        "Unable to fetch milk collection details at the moment."
    )
