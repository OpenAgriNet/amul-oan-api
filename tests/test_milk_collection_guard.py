import asyncio
import inspect
from types import SimpleNamespace

from agents.tools import milk_collection as milk


def _ctx(unions, mobile="9000000000"):
    return SimpleNamespace(deps=SimpleNamespace(
        farmer_unions=unions,
        mobile=mobile,
        session_id="session",
    ))


def test_prepare_hides_tool_without_resolved_farmer_or_authenticated_mobile():
    sentinel = object()
    assert asyncio.run(milk.prepare_get_farmer_milk_collection_details(_ctx([]), sentinel)) is None
    assert asyncio.run(milk.prepare_get_farmer_milk_collection_details(_ctx(["banas"], None), sentinel)) is None


def test_prepare_shows_tool_for_authenticated_resolved_farmer():
    sentinel = object()
    assert asyncio.run(
        milk.prepare_get_farmer_milk_collection_details(_ctx(["banas"]), sentinel)
    ) is sentinel


def test_model_facing_signature_has_no_identity_parameters():
    parameters = inspect.signature(milk.get_farmer_milk_collection_details).parameters
    assert list(parameters) == ["ctx", "fromdate", "todate"]
    assert "union_code" not in parameters
    assert "society_code" not in parameters
    assert "farmer_code" not in parameters


def test_partial_rows_are_none_safe(monkeypatch):
    from app.models.farmer import FarmerModel
    from app.models.milk_collection import (
        FarmerMilkCollectionResponseModel,
        MilkCollectionRecordModel,
        DeductionRecordModel,
    )

    monkeypatch.setattr(milk.settings, "enable_network", True)
    monkeypatch.setattr(milk.settings, "beckn_callback_transactions_enabled", True)

    async def farmers(mobile, **kwargs):
        return [FarmerModel.model_validate({
            "unionCode": "U", "societyCode": "S", "farmerCode": "F"
        })]

    async def callback(account, **kwargs):
        return FarmerMilkCollectionResponseModel(
            milk=[MilkCollectionRecordModel(
                date="2026-01-01", shift="M", qty=12.0, fat=None, snf=None, amount=None,
            )],
            deduction=[DeductionRecordModel(date=None, account_name=None, amount=None)],
        )

    monkeypatch.setattr(milk, "fetch_authenticated_farmers", farmers)
    monkeypatch.setattr(milk, "fetch_milk_collection", callback)

    out = asyncio.run(milk.get_farmer_milk_collection_details(
        _ctx(["banas"]), "2026-01-01", "2026-01-10"
    ))
    assert "successfully" in out.lower()
    assert "-" in out
