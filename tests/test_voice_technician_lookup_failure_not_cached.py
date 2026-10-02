"""A failed AI-technician lookup must not be cached as "no technicians".

get_ai_technicians_by_society_api returns None on failure and [] when the
society genuinely has none. Both were flattened to [] in the cached envelope,
so a transient upstream blip was indistinguishable from an empty society and
persisted for the life of the envelope: the "aiTechnicians" key was present
(so the missing_ai_technicians check could not fire) and the envelope was not
stale by age.
"""

import pytest

from agents.voice.models.farmer import FarmerDataEnvelope
from agents.voice.services.farmer_cache import _has_failed_technician_lookup
from app.voice.farmer import _build_ai_technician_summary


def _group(*, technicians: list[dict], lookup_failed: bool | None) -> dict:
    group = {
        "farmerName": "Farmer 1",
        "farmerCode": "FC001",
        "societyName": "Society 1",
        "societyCode": "SC001",
        "unionCode": "BANAS",
        "technicians": technicians,
    }
    if lookup_failed is not None:
        group["lookupFailed"] = lookup_failed
    return group


def _envelope(groups: list[dict]) -> FarmerDataEnvelope:
    envelope = FarmerDataEnvelope()
    envelope.aiTechnicians = groups
    return envelope


class TestFailedLookupDetection:
    def test_failed_lookup_is_detected(self):
        envelope = _envelope([_group(technicians=[], lookup_failed=True)])
        assert _has_failed_technician_lookup(envelope) is True

    def test_genuinely_empty_society_is_not_a_failure(self):
        envelope = _envelope([_group(technicians=[], lookup_failed=False)])
        assert _has_failed_technician_lookup(envelope) is False

    def test_successful_lookup_is_not_a_failure(self):
        group = _group(
            technicians=[{"userId": "AIT001", "fullName": "A", "mobileNumber": "9"}],
            lookup_failed=False,
        )
        assert _has_failed_technician_lookup(_envelope([group])) is False

    def test_legacy_envelope_without_the_flag_is_not_a_failure(self):
        """Pre-flag cache entries omit the key; treating them as failed would
        refresh them forever."""
        envelope = _envelope([_group(technicians=[], lookup_failed=None)])
        assert _has_failed_technician_lookup(envelope) is False

    def test_one_failed_group_among_several_is_detected(self):
        groups = [
            _group(technicians=[{"userId": "AIT001"}], lookup_failed=False),
            _group(technicians=[], lookup_failed=True),
        ]
        assert _has_failed_technician_lookup(_envelope(groups)) is True

    def test_no_groups_is_not_a_failure(self):
        assert _has_failed_technician_lookup(_envelope([])) is False


class TestPromptWording:
    def test_failed_lookup_does_not_claim_none_exist(self):
        summary = _build_ai_technician_summary(
            _envelope([_group(technicians=[], lookup_failed=True)])
        )
        assert "temporarily unavailable" in summary
        assert "none available for this farmer group" not in summary

    def test_genuinely_empty_society_still_says_none_available(self):
        summary = _build_ai_technician_summary(
            _envelope([_group(technicians=[], lookup_failed=False)])
        )
        assert "none available for this farmer group" in summary
        assert "temporarily unavailable" not in summary

    def test_legacy_envelope_keeps_the_none_available_wording(self):
        summary = _build_ai_technician_summary(
            _envelope([_group(technicians=[], lookup_failed=None)])
        )
        assert "none available for this farmer group" in summary


class TestFetchMarksFailure:
    @pytest.mark.asyncio
    async def test_api_none_marks_lookup_failed(self, monkeypatch):
        import agents.voice.services.farmer_cache as fc

        monkeypatch.setenv("PASHUGPT_TOKEN", "test-token")

        async def _api_returns_none(*args, **kwargs):
            return None

        monkeypatch.setattr(fc, "get_ai_technicians_by_society_api", _api_returns_none)

        record = fc.FarmerRecord(unionCode="BANAS", societyCode="SC001")
        groups = await fc._fetch_ai_technicians([record])

        assert groups[0]["lookupFailed"] is True
        assert groups[0]["technicians"] == []

    @pytest.mark.asyncio
    async def test_api_empty_list_does_not_mark_failure(self, monkeypatch):
        import agents.voice.services.farmer_cache as fc

        monkeypatch.setenv("PASHUGPT_TOKEN", "test-token")

        async def _api_returns_empty(*args, **kwargs):
            return []

        monkeypatch.setattr(fc, "get_ai_technicians_by_society_api", _api_returns_empty)

        record = fc.FarmerRecord(unionCode="BANAS", societyCode="SC001")
        groups = await fc._fetch_ai_technicians([record])

        assert groups[0]["lookupFailed"] is False
        assert groups[0]["technicians"] == []
