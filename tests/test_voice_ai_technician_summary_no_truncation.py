"""Regression tests for AMUL-39.

Voice silently capped the AI technician context at 5 groups and 5 technicians
per group, so the agent could neither offer nor recognise technicians past the
cap. Chat (agents/farmer_context.py) never capped, which is why the same query
worked there.
"""

from app.voice.farmer import _build_ai_technician_summary, _dedupe_technicians
from agents.voice.models.farmer import FarmerDataEnvelope


# Names are distinguished by letters, not digits: digits are stripped from
# technician names before they reach the prompt (they are internal society/route
# codes — see test_ait_name_codes.py), so a digit-numbered fixture would collapse
# into one indistinguishable name and stop testing truncation at all.
_NAME_SUFFIXES = ["Alpha", "Bravo", "Charlie", "Delta", "Echo",
                  "Foxtrot", "Golf", "Hotel", "India", "Juliett"]


def _technician_name(index: int) -> str:
    return f"Technician {_NAME_SUFFIXES[(index - 1) % len(_NAME_SUFFIXES)]}"


def _technician(index: int) -> dict:
    return {
        "userId": f"AIT{index:03d}",
        "fullName": _technician_name(index),
        "mobileNumber": f"90000000{index:02d}",
    }


def _group(index: int, technicians: list[dict]) -> dict:
    return {
        "farmerName": f"Farmer {index}",
        "farmerCode": f"FC{index:03d}",
        "societyName": f"Society {index}",
        "societyCode": f"SC{index:03d}",
        "unionCode": "BANAS",
        "technicians": technicians,
    }


def _envelope(groups: list[dict]) -> FarmerDataEnvelope:
    envelope = FarmerDataEnvelope()
    envelope.aiTechnicians = groups
    return envelope


class TestNoTruncation:
    def test_all_technicians_listed_beyond_legacy_cap(self):
        technicians = [_technician(i) for i in range(1, 9)]
        summary = _build_ai_technician_summary(_envelope([_group(1, technicians)]))

        for technician in technicians:
            assert technician["fullName"] in summary, (
                f"{technician['fullName']} missing — technician list is truncated"
            )
            assert technician["userId"] in summary

    def test_all_groups_listed_beyond_legacy_cap(self):
        groups = [_group(i, [_technician(i)]) for i in range(1, 8)]
        summary = _build_ai_technician_summary(_envelope(groups))

        for index in range(1, 8):
            assert f"Society {index}" in summary, (
                f"Society {index} missing — technician groups are truncated"
            )

    def test_sixth_technician_is_present(self):
        """The narrowest expression of the bug: index 5 used to be dropped."""
        technicians = [_technician(i) for i in range(1, 7)]
        summary = _build_ai_technician_summary(_envelope([_group(1, technicians)]))

        assert _technician_name(6) in summary


class TestDedupe:
    def test_duplicate_rows_collapse_by_user_id(self):
        duplicated = [_technician(1), _technician(1), _technician(2)]
        assert len(_dedupe_technicians(duplicated)) == 2

    def test_falls_back_to_name_and_mobile_without_id(self):
        row = {"userId": None, "fullName": "Ramesh Patel", "mobileNumber": "9000000001"}
        assert len(_dedupe_technicians([row, dict(row)])) == 1

    def test_distinct_technicians_are_kept(self):
        assert len(_dedupe_technicians([_technician(1), _technician(2)])) == 2

    def test_order_is_preserved(self):
        deduped = _dedupe_technicians([_technician(3), _technician(1), _technician(3)])
        assert [t["userId"] for t in deduped] == ["AIT003", "AIT001"]

    def test_duplicates_do_not_consume_visible_slots(self):
        """Pre-fix, 6 duplicate rows of 3 technicians surfaced as 3 of 5 slots."""
        rows = [_technician(i) for i in (1, 1, 2, 2, 3, 3)]
        summary = _build_ai_technician_summary(_envelope([_group(1, rows)]))

        for index in (1, 2, 3):
            assert _technician_name(index) in summary
        assert summary.count("AIT001") == 1


class TestUnchangedBehaviour:
    def test_empty_group_still_reports_none_available(self):
        summary = _build_ai_technician_summary(_envelope([_group(1, [])]))
        assert "none available for this farmer group" in summary

    def test_no_groups_reports_unavailable(self):
        summary = _build_ai_technician_summary(_envelope([]))
        assert "not available in the current signed-in context" in summary

    def test_none_envelope_returns_empty_string(self):
        assert _build_ai_technician_summary(None) == ""
