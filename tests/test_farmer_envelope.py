"""Characterization tests for collect_farmer_accounts.

Pins the behavior that survived the move out of app/services/voice.py, so the
remaining extractions from that module have something to move against.
"""
from agents.tools.models.farmer_transport import FarmerDataEnvelope, FarmerRecord
from agents.tools.models.farmer import FarmerModel
from agents.tools.farmer_envelope import collect_farmer_accounts


def _envelope(*records: dict) -> FarmerDataEnvelope:
    return FarmerDataEnvelope(farmers=[FarmerRecord(**r) for r in records], source="test")


def test_none_envelope_yields_no_accounts():
    assert collect_farmer_accounts(None) == []


def test_camel_and_snake_case_keys_both_resolve():
    camel = _envelope({"unionCode": "1", "societyCode": "2", "farmerCode": "3"})
    snake = _envelope({"union_code": "1", "society_code": "2", "farmer_code": "3"})
    assert [(a.union_code, a.society_code, a.farmer_code) for a in collect_farmer_accounts(camel)] \
        == [("1", "2", "3")]
    assert [(a.union_code, a.society_code, a.farmer_code) for a in collect_farmer_accounts(snake)] \
        == [("1", "2", "3")]


def test_records_missing_any_code_are_dropped():
    env = _envelope(
        {"unionCode": "1", "societyCode": "2"},                    # no farmerCode
        {"unionCode": "1", "farmerCode": "3"},                     # no societyCode
        {"unionCode": "1", "societyCode": "2", "farmerCode": "3"},  # complete
    )
    assert len(collect_farmer_accounts(env)) == 1


def test_duplicate_accounts_are_deduped_on_the_three_codes():
    env = _envelope(
        {"unionCode": "1", "societyCode": "2", "farmerCode": "3", "farmerName": "A"},
        {"unionCode": "1", "societyCode": "2", "farmerCode": "3", "farmerName": "B"},
        {"unionCode": "1", "societyCode": "2", "farmerCode": "4"},
    )
    accounts = collect_farmer_accounts(env)
    assert len(accounts) == 2
    assert accounts[0].farmer_name == "A", "first occurrence wins"


def test_normalized_farmer_dump_preserves_transport_camel_case():
    farmer = FarmerModel.model_validate({
        "farmerName": "Farmer One",
        "mobileNumber": "9000000000",
        "unionName": "Banas",
        "unionCode": "U1",
        "societyName": "Society One",
        "societyCode": "S1",
        "farmerCode": "F1",
        "subDistrict": "Deesa",
        "avgMilkPerDayCow": 4.5,
        "cow": 2,
    })

    record = FarmerRecord.model_validate(farmer.model_dump()).model_dump()

    assert record["farmerName"] == "farmer one"
    assert record["mobileNumber"] == "9000000000"
    assert record["unionName"] == "banas"
    assert record["unionCode"] == "U1"
    assert record["societyName"] == "society one"
    assert record["societyCode"] == "S1"
    assert record["subDistrict"] == "deesa"
    assert record["avgMilkPerDayCow"] == 4.5
    assert record["cow"] == 2
    assert "union_name" not in record
    assert "mobile_number" not in record
