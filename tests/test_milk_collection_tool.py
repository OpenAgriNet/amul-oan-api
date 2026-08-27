import asyncio
from types import SimpleNamespace

from agents.tools import milk_collection as milk
from app.models.farmer import FarmerModel
from app.models.milk_collection import FarmerMilkCollectionResponseModel


def _ctx():
    return SimpleNamespace(
        deps=SimpleNamespace(
            mobile="9000000000",
            session_id="session-1",
            farmer_unions=["kaira"],
        ),
        tool_call_id="tool-1",
    )


def _farmer():
    return FarmerModel.model_validate({
        "unionCode": "0201",
        "societyCode": "001066",
        "farmerCode": "000123",
        "unionName": "Kaira",
    })


def _direct(monkeypatch):
    monkeypatch.setattr(milk.settings, "enable_network", False)
    monkeypatch.setattr(milk.settings, "beckn_callback_transactions_enabled", False)
    async def farmers(mobile):
        assert mobile == "9000000000"
        return [_farmer()]
    monkeypatch.setattr(milk, "get_farmer_data_by_mobile", farmers)


class TestMilkCollectionTool:
    def test_success_uses_signed_account_and_formats_markdown(self, monkeypatch):
        _direct(monkeypatch)
        monkeypatch.setenv("PASHUGPT_TOKEN", "test-token")

        async def fake_api(request, token):
            assert token == "test-token"
            assert request.to_query_params() == {
                "unionCode": "0201",
                "societyCode": "001066",
                "farmerCode": "000123",
                "fromdate": "2026-04-01",
                "todate": "2026-04-01",
            }
            return FarmerMilkCollectionResponseModel.model_validate({
                "result": "success",
                "milk": [{"date": "2026-04-01", "qty": 10, "fat": 6, "snf": 9, "amount": 500}],
                "deduction": [{"date": "2026-04-01", "accountname": "Feed", "amount": 100}],
            })

        monkeypatch.setattr(milk, "get_farmer_milk_collection_details_api", fake_api)
        result = asyncio.run(milk.get_farmer_milk_collection_details(
            _ctx(), "2026-04-01", "2026-04-01"
        ))

        assert result.startswith("Farmer milk collection details fetched successfully:\n\n")
        assert "| 2026-04-01 | - | 10.00 | 6.00 | 9.00 | 500.00 |" in result
        assert "| 2026-04-01 | Feed | 100.00 |" in result

    def test_missing_authenticated_mobile_never_calls_backend(self, monkeypatch):
        called = False
        async def unexpected(*args, **kwargs):
            nonlocal called
            called = True
        monkeypatch.setattr(milk, "get_farmer_data_by_mobile", unexpected)
        ctx = _ctx()
        ctx.deps.mobile = None

        result = asyncio.run(milk.get_farmer_milk_collection_details(
            ctx, "2026-04-01", "2026-04-01"
        ))

        assert "signed-in farmer profile" in result
        assert called is False

    def test_invalid_date_returns_before_profile_or_provider_lookup(self, monkeypatch):
        called = False
        async def unexpected(*args, **kwargs):
            nonlocal called
            called = True
        monkeypatch.setattr(milk, "get_farmer_data_by_mobile", unexpected)

        result = asyncio.run(milk.get_farmer_milk_collection_details(
            _ctx(), "01-04-2026", "2026-04-01"
        ))

        assert "YYYY-MM-DD" in result
        assert called is False

    def test_all_provider_failures_return_temporary_failure(self, monkeypatch):
        _direct(monkeypatch)
        monkeypatch.setenv("PASHUGPT_TOKEN", "test-token")
        async def failed(request, token):
            return None
        monkeypatch.setattr(milk, "get_farmer_milk_collection_details_api", failed)

        result = asyncio.run(milk.get_farmer_milk_collection_details(
            _ctx(), "2026-04-01", "2026-04-01"
        ))

        assert result == "Milk collection lookup failed.\n\nUnable to fetch milk collection details at the moment."

    def test_callback_mode_uses_beckn_and_not_direct_provider(self, monkeypatch):
        monkeypatch.setattr(milk.settings, "enable_network", True)
        monkeypatch.setattr(milk.settings, "beckn_callback_transactions_enabled", True)
        calls = []

        async def farmers(mobile, **kwargs):
            return [_farmer()]

        async def callback(account, **kwargs):
            calls.append((account, kwargs))
            return FarmerMilkCollectionResponseModel(milk=[], deduction=[])

        async def direct(*args, **kwargs):
            raise AssertionError("direct provider must not be called in callback mode")

        monkeypatch.setattr(milk, "fetch_authenticated_farmers", farmers)
        monkeypatch.setattr(milk, "fetch_milk_collection", callback)
        monkeypatch.setattr(milk, "get_farmer_milk_collection_details_api", direct)

        result = asyncio.run(milk.get_farmer_milk_collection_details(
            _ctx(), "2026-04-01", "2026-04-01"
        ))

        assert "fetched successfully" in result
        assert calls[0][0].farmer_code == "000123"
        assert calls[0][1]["tool_call_id"] == "tool-1:account-0"
