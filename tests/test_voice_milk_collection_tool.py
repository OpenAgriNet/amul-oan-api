import asyncio
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.voice.models.milk_collection import FarmerMilkCollectionResponseModel
from agents.deps import FarmerAccount, FarmerContext
from agents.voice.tools.milk_collection import get_farmer_milk_collection_details


def _ctx(accounts=None):
    """Minimal RunContext stand-in carrying FarmerContext deps."""
    deps = FarmerContext(query="milk", farmer_accounts=accounts or [])
    return SimpleNamespace(deps=deps)


class TestMilkCollectionTool:
    def test_success_returns_labelled_summary(self, monkeypatch):
        monkeypatch.setenv("PASHUGPT_TOKEN", "test-token")

        async def _fake_api(request, token):
            assert token == "test-token"
            assert request.to_query_params() == {
                "unionCode": "0201",
                "societyCode": "001066",
                "farmerCode": "000123",
                "fromdate": "2026-04-01",
                "todate": "2026-04-01",
            }
            return FarmerMilkCollectionResponseModel.model_validate(
                {
                    "result": "success",
                    "milk": [{"date": "2026-04-01", "shift": "M", "qty": 10, "fat": 6, "snf": 9, "amount": 500}],
                    "deduction": [{"date": "2026-04-01", "accountname": "Feed", "amount": 100}],
                }
            )

        monkeypatch.setattr(
            "agents.voice.tools.milk_collection.get_farmer_milk_collection_details_api",
            _fake_api,
        )

        # Single account in context: no per-account header, labelled fields.
        accounts = [FarmerAccount(union_code="0201", society_code="001066", farmer_code="000123")]
        result = asyncio.run(
            get_farmer_milk_collection_details(
                _ctx(accounts), "0201", "001066", "000123", "2026-04-01", "2026-04-01"
            )
        )

        assert "Milk collection details fetched successfully" in result
        assert "quantity 10 liters" in result
        assert "fat 6, SNF 9, amount 500 rupees" in result
        assert "Feed: amount 100 rupees" in result
        assert "Account —" not in result  # single account => no header

    def test_multi_account_fans_out_over_all_accounts(self, monkeypatch):
        monkeypatch.setenv("PASHUGPT_TOKEN", "test-token")

        async def _fake_api(request, token):
            # farmer 0006 has two morning records; 1006 is empty.
            if request.farmer_code == "0006":
                return FarmerMilkCollectionResponseModel.model_validate(
                    {"milk": [
                        {"date": "03-06-2026", "shift": "M", "qty": 2.38, "fat": 7.2, "snf": 9.1, "amount": 146.47},
                        {"date": "03-06-2026", "shift": "M", "qty": 9.68, "fat": 4.2, "snf": 8.5, "amount": 355.93},
                    ], "deduction": []}
                )
            return FarmerMilkCollectionResponseModel.model_validate({"milk": [], "deduction": []})

        monkeypatch.setattr(
            "agents.voice.tools.milk_collection.get_farmer_milk_collection_details_api",
            _fake_api,
        )

        accounts = [
            FarmerAccount(union_code="2017", society_code="1", farmer_code="1006", society_name="LALAVADA"),
            FarmerAccount(union_code="2017", society_code="1", farmer_code="0006", society_name="LALAVADA"),
        ]
        result = asyncio.run(
            get_farmer_milk_collection_details(
                _ctx(accounts), "2017", "1", "1006", "2026-06-03", "2026-06-03"
            )
        )

        # Both accounts present, each labelled; the populated account's records surface.
        assert "farmer code 1006" in result
        assert "farmer code 0006" in result
        assert "quantity 2.38 liters" in result
        assert "quantity 9.68 liters" in result

    def test_refuses_instead_of_using_supplied_codes_when_no_accounts_in_context(self, monkeypatch):
        monkeypatch.setenv("PASHUGPT_TOKEN", "test-token")
        seen = {}

        async def _fake_api(request, token):
            seen["codes"] = request.to_query_params()
            return FarmerMilkCollectionResponseModel.model_validate(
                {"milk": [{"date": "2026-04-01", "shift": "M", "qty": 5, "fat": 4, "snf": 8, "amount": 200}], "deduction": []}
            )

        monkeypatch.setattr(
            "agents.voice.tools.milk_collection.get_farmer_milk_collection_details_api",
            _fake_api,
        )

        # Empty context -> refuse. The codes the model supplies here cannot have
        # come from anywhere real: it is told to copy them out of a farmer block
        # that is empty on precisely these turns (issue #282).
        result = asyncio.run(
            get_farmer_milk_collection_details(
                _ctx([]), "2021", "1066", "123", "2026-04-01", "2026-04-01"
            )
        )
        assert "codes" not in seen, "the backend must not be reached without context accounts"
        assert result == (
            "Milk collection lookup failed. The farmer account details are not available."
        )

    def test_missing_token_returns_clear_failure_and_does_not_call_backend(self, monkeypatch):
        monkeypatch.delenv("PASHUGPT_TOKEN", raising=False)

        async def _unexpected_api(request, token):
            raise AssertionError("backend should not be called")

        monkeypatch.setattr(
            "agents.voice.tools.milk_collection.get_farmer_milk_collection_details_api",
            _unexpected_api,
        )

        result = asyncio.run(
            get_farmer_milk_collection_details(
                _ctx([FarmerAccount(union_code="2021", society_code="1066", farmer_code="123")]),
                "2021", "1066", "123", "2026-04-01", "2026-04-01",
            )
        )

        assert result == "Milk collection lookup failed. Service is not configured."

    def test_invalid_date_returns_validation_failure(self, monkeypatch):
        monkeypatch.setenv("PASHUGPT_TOKEN", "test-token")

        async def _unexpected_api(request, token):
            raise AssertionError("backend should not be called")

        monkeypatch.setattr(
            "agents.voice.tools.milk_collection.get_farmer_milk_collection_details_api",
            _unexpected_api,
        )

        accounts = [FarmerAccount(union_code="2021", society_code="1066", farmer_code="123")]
        result = asyncio.run(
            get_farmer_milk_collection_details(
                _ctx(accounts), "2021", "1066", "123", "01-04-2026", "2026-04-01"
            )
        )

        assert result.startswith("Milk collection lookup failed.")
        assert "YYYY-MM-DD" in result

    def test_backend_none_returns_temporary_failure(self, monkeypatch):
        monkeypatch.setenv("PASHUGPT_TOKEN", "test-token")

        async def _fake_api(request, token):
            return None

        monkeypatch.setattr(
            "agents.voice.tools.milk_collection.get_farmer_milk_collection_details_api",
            _fake_api,
        )

        result = asyncio.run(
            get_farmer_milk_collection_details(
                _ctx([FarmerAccount(union_code="2021", society_code="1066", farmer_code="123")]),
                "2021", "1066", "123", "2026-04-01", "2026-04-01",
            )
        )

        assert result == "Milk collection lookup failed. Unable to fetch details at the moment."
