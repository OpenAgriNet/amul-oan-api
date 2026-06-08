import os
import sys

import pytest
from pydantic import ValidationError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.models.milk_collection import (
    DeductionRecordModel,
    FarmerMilkCollectionRequestModel,
    FarmerMilkCollectionResponseModel,
)


class TestMilkCollectionModel:
    def test_milk_collection_request_query_params_preserve_aliases(self):
        request = FarmerMilkCollectionRequestModel(
            unionCode="0201",
            societyCode="001066",
            farmerCode="000123",
            fromdate="2026-04-01",
            todate="2026-04-10",
        )

        assert request.to_query_params() == {
            "unionCode": "0201",
            "societyCode": "001066",
            "farmerCode": "000123",
            "fromdate": "2026-04-01",
            "todate": "2026-04-10",
        }

    def test_milk_collection_request_rejects_invalid_date_format(self):
        with pytest.raises(ValidationError):
            FarmerMilkCollectionRequestModel(
                unionCode="2021",
                societyCode="1066",
                farmerCode="123",
                fromdate="2026-04-01",
                todate="01-04-2026",
            )

    def test_milk_collection_request_rejects_reversed_date_range(self):
        request = FarmerMilkCollectionRequestModel(
            unionCode="2021",
            societyCode="1066",
            farmerCode="123",
            fromdate="2026-04-10",
            todate="2026-04-01",
        )

        with pytest.raises(ValueError, match="todate must be on or after fromdate"):
            request.validate_date_range()

    def test_milk_collection_request_rejects_range_over_31_days(self):
        request = FarmerMilkCollectionRequestModel(
            unionCode="2021",
            societyCode="1066",
            farmerCode="123",
            fromdate="2026-04-01",
            todate="2026-05-03",
        )

        with pytest.raises(ValueError, match="date range cannot exceed 31 days"):
            request.validate_date_range()

    def test_milk_collection_response_parses_milk_and_deductions(self):
        response = FarmerMilkCollectionResponseModel.model_validate(
            {
                "result": "success",
                "milk": [
                    {
                        "date": "2026-04-01",
                        "shift": "M",
                        "qty": 10.5,
                        "fat": 6.1,
                        "snf": 8.7,
                        "amount": 420.5,
                    }
                ],
                "deduction": [
                    {
                        "date": "2026-04-01",
                        "accountname": "Cattle feed",
                        "amount": 100,
                    }
                ],
            }
        )

        assert response.milk[0].qty == 10.5
        assert response.deduction[0].account_name == "Cattle feed"
        assert response.model_dump(by_alias=True)["deduction"][0]["accountname"] == "Cattle feed"

    def test_milk_collection_response_defaults_missing_lists_to_empty(self):
        response = FarmerMilkCollectionResponseModel.model_validate({"result": "success"})

        assert response.milk == []
        assert response.deduction == []

    def test_deduction_record_accepts_python_field_name(self):
        record = DeductionRecordModel(account_name="Advance", amount=50)

        assert record.model_dump(by_alias=True)["accountname"] == "Advance"
