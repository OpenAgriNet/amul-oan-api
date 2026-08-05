"""Characterization tests for collect_farmer_accounts.

Pins the behavior that survived the move out of app/services/voice.py, so the
remaining extractions from that module have something to move against.
"""
from agents.models.farmer import FarmerDataEnvelope, FarmerRecord
from agents.services.farmer_envelope import collect_farmer_accounts


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
