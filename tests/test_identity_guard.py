"""The identity guard must cover every tool the model passes codes to.

Chat leaks far less than voice — the farmer context is nearly always resolved
before the turn — but the tool signatures are identical and the guard was on
create_ai_call only. See voice-oan-api#282 for the full attribution.
"""
from agents.tools.identity_guard import invalid_code_field, invalid_technician_id


def test_invented_codes_are_rejected():
    # Values actually observed in prod inputs, both channels.
    for bogus in ("MISSING", "UNKNOWN", "unknown", "not_provided", "not_available",
                  "NOT_PROVIDED", "UNION_CODE_FROM_CONTEXT", "UNION_CODE_PLACEHOLDER",
                  "Sanjaybhai", "", "   "):
        assert invalid_code_field(bogus, "00731", "0554") == "union_code"
        assert invalid_code_field("159", bogus, "0554") == "society_code"
        assert invalid_code_field("159", "00731", bogus) == "farmer_code"


def test_real_codes_pass():
    # Real codes are not always numeric, but they always carry a digit:
    # 28,089 successful bookings, 22 unions / 2,032 societies / 2,548 farmers.
    for good in ("159", "2021", "6666", "B2021", "M001", "NA4192", "00731", "0554"):
        assert invalid_code_field(good, good, good) is None


def test_technician_id_shape():
    assert invalid_technician_id("QYNWSGoELy1qwA7YfjyJcA==") is False
    assert invalid_technician_id("T55667") is True
    assert invalid_technician_id("") is True
