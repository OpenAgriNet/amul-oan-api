import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.models.milk_collection import FarmerMilkCollectionResponseModel
from agents.tools.milk_collection import get_farmer_milk_collection_details


class TestMilkCollectionTool:
    def test_success_returns_formatted_json_with_aliases(self, monkeypatch):
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
                    "milk": [{"date": "2026-04-01", "qty": 10, "fat": 6, "snf": 9, "amount": 500}],
                    "deduction": [{"date": "2026-04-01", "accountname": "Feed", "amount": 100}],
                }
            )

        monkeypatch.setattr(
            "agents.tools.milk_collection.get_farmer_milk_collection_details_api",
            _fake_api,
        )

        result = asyncio.run(
            get_farmer_milk_collection_details(
                "0201",
                "001066",
                "000123",
                "2026-04-01",
                "2026-04-01",
            )
        )

        # Merged tool renders chat's markdown (not voice's JSON). Assert the EXACT
        # rendered rows so every field is verified — qty/fat/snf/amount on the milk
        # row and the full deduction row (stronger than the old single-field JSON
        # check), plus the success header and both section headings.
        assert result.startswith("Farmer milk collection details fetched successfully:\n\n")
        assert "### Milk Collection" in result
        assert "| 2026-04-01 | - | 10.00 | 6.00 | 9.00 | 500.00 |" in result
        assert "### Deductions" in result
        assert "| 2026-04-01 | Feed | 100.00 |" in result

    def test_missing_token_returns_clear_failure_and_does_not_call_backend(self, monkeypatch):
        monkeypatch.delenv("PASHUGPT_TOKEN", raising=False)

        async def _unexpected_api(request, token):
            raise AssertionError("backend should not be called")

        monkeypatch.setattr(
            "agents.tools.milk_collection.get_farmer_milk_collection_details_api",
            _unexpected_api,
        )

        result = asyncio.run(
            get_farmer_milk_collection_details(
                "2021",
                "1066",
                "123",
                "2026-04-01",
                "2026-04-01",
            )
        )

        assert result == "Milk collection lookup failed.\n\nPASHUGPT_TOKEN is not configured."

    def test_invalid_date_returns_validation_failure_and_does_not_call_backend(self, monkeypatch):
        monkeypatch.setenv("PASHUGPT_TOKEN", "test-token")

        async def _unexpected_api(request, token):
            raise AssertionError("backend should not be called")

        monkeypatch.setattr(
            "agents.tools.milk_collection.get_farmer_milk_collection_details_api",
            _unexpected_api,
        )

        result = asyncio.run(
            get_farmer_milk_collection_details(
                "2021",
                "1066",
                "123",
                "01-04-2026",
                "2026-04-01",
            )
        )

        assert result.startswith("Milk collection lookup failed.")
        assert "YYYY-MM-DD" in result

    def test_backend_none_returns_temporary_failure(self, monkeypatch):
        monkeypatch.setenv("PASHUGPT_TOKEN", "test-token")

        async def _fake_api(request, token):
            return None

        monkeypatch.setattr(
            "agents.tools.milk_collection.get_farmer_milk_collection_details_api",
            _fake_api,
        )

        result = asyncio.run(
            get_farmer_milk_collection_details(
                "2021",
                "1066",
                "123",
                "2026-04-01",
                "2026-04-01",
            )
        )

        assert result == "Milk collection lookup failed.\n\nUnable to fetch milk collection details at the moment."
