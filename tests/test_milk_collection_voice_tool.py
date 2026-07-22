import asyncio
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.models.milk_collection import FarmerMilkCollectionResponseModel
from agents.deps import FarmerAccount, FarmerContext
from agents.tools.milk_collection import get_farmer_milk_collection_details_voice


def _ctx(accounts=None):
    """Minimal RunContext stand-in carrying FarmerContext deps."""
    deps = FarmerContext(query="milk", farmer_accounts=accounts or [])
    return SimpleNamespace(deps=deps)


class TestMilkCollectionVoiceTool:
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
            "agents.tools.milk_collection.get_farmer_milk_collection_details_api",
            _fake_api,
        )

        accounts = [FarmerAccount(union_code="0201", society_code="001066", farmer_code="000123")]
        result = asyncio.run(
            get_farmer_milk_collection_details_voice(
                _ctx(accounts), "0201", "001066", "000123", "2026-04-01", "2026-04-01"
            )
        )

        assert "Milk collection details fetched successfully" in result
        assert "quantity 10 liters" in result
        assert "fat 6, SNF 9, amount 500 rupees" in result
        assert "Feed: amount 100 rupees" in result
        assert "Account —" not in result

    def test_multi_account_fans_out_over_all_accounts(self, monkeypatch):
        monkeypatch.setenv("PASHUGPT_TOKEN", "test-token")

        async def _fake_api(request, token):
            if request.farmer_code == "0006":
                return FarmerMilkCollectionResponseModel.model_validate(
                    {"milk": [
                        {"date": "03-06-2026", "shift": "M", "qty": 2.38, "fat": 7.2, "snf": 9.1, "amount": 146.47},
                        {"date": "03-06-2026", "shift": "M", "qty": 9.68, "fat": 4.2, "snf": 8.5, "amount": 355.93},
                    ], "deduction": []}
                )
            return FarmerMilkCollectionResponseModel.model_validate({"milk": [], "deduction": []})

        monkeypatch.setattr(
            "agents.tools.milk_collection.get_farmer_milk_collection_details_api",
            _fake_api,
        )

        accounts = [
            FarmerAccount(union_code="2017", society_code="1", farmer_code="1006", society_name="LALAVADA"),
            FarmerAccount(union_code="2017", society_code="1", farmer_code="0006", society_name="LALAVADA"),
        ]
        result = asyncio.run(
            get_farmer_milk_collection_details_voice(
                _ctx(accounts), "2017", "1", "1006", "2026-06-03", "2026-06-03"
            )
        )

        assert "farmer code 1006" in result
        assert "farmer code 0006" in result
        assert "quantity 2.38 liters" in result
        assert "quantity 9.68 liters" in result
