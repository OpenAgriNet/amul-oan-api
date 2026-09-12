import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.models.bonus import (
    FarmerBonusAmountRecordModel,
    FarmerBonusAmountRequestModel,
)


class TestBonusModel:
    def test_bonus_request_query_params_preserve_aliases_and_leading_zeros(self):
        request = FarmerBonusAmountRequestModel(
            unionCode="0001",
            societyCode="2004",
            farmerCode="0001",
        )

        assert request.to_query_params() == {
            "unionCode": "0001",
            "societyCode": "2004",
            "farmerCode": "0001",
        }
        assert request.union_code == "0001"

    def test_bonus_record_parses_api_doc_sample(self):
        record = FarmerBonusAmountRecordModel.model_validate(
            {
                "societyCode": "2004",
                "societyName": "DEMO_2004",
                "societyNameLocal": "ડૅમૉ_૨૦૦૪",
                "farmerCode": "0001",
                "farmerName": "FARMER1",
                "farmerLocalName": "ફર્મૅર્૧",
                "bonusAmount": 1273.1,
                "fromDate": "2026-04-01T00:00:00",
                "toDate": "2026-04-01T00:00:00",
            }
        )

        assert record.bonus_amount == 1273.1
        assert record.society_name == "DEMO_2004"
        assert record.from_date == "2026-04-01T00:00:00"
        assert record.model_dump(by_alias=True)["bonusAmount"] == 1273.1

    def test_bonus_record_accepts_python_field_names(self):
        record = FarmerBonusAmountRecordModel(
            bonus_amount=10,
            farmer_name="X",
            society_code="2004",
        )

        assert record.model_dump(by_alias=True)["bonusAmount"] == 10
        assert record.model_dump(by_alias=True)["farmerName"] == "X"

    def test_bonus_record_ignores_unknown_keys_and_defaults_missing(self):
        record = FarmerBonusAmountRecordModel.model_validate(
            {"bonusAmount": 1, "unknown": True}
        )

        assert record.bonus_amount == 1
        assert record.society_name is None
        assert record.from_date is None
