"""A failed AI-technician lookup must not be cached as "no technicians".

The lookup returns None on failure and [] when the society genuinely has none.
Both used to be flattened to [] in the cached envelope, so a transient upstream
blip was indistinguishable from an empty society and persisted for the life of
the envelope: the "aiTechnicians" key was present (so the missing_ai_technicians
check could not fire) and the envelope was not stale by age.
"""
import asyncio
from datetime import datetime, timezone

import agents.tools.farmer_cache as fc
from agents.tools.models.farmer_transport import FarmerDataEnvelope, FarmerRecord
from app.voice.farmer import _build_ai_technician_summary


def _group(*, technicians, failed=None, flag="techniciansLookupFailed") -> dict:
    group = {
        "farmerName": "Farmer 1",
        "farmerCode": "FC001",
        "societyName": "Society 1",
        "societyCode": "SC001",
        "unionCode": "BANAS",
        "technicians": technicians,
    }
    if failed is not None:
        group[flag] = failed
    return group


def _envelope(groups: list[dict]) -> FarmerDataEnvelope:
    envelope = FarmerDataEnvelope()
    envelope.aiTechnicians = groups
    return envelope


class TestFailedLookupDetection:
    def test_failed_lookup_is_detected(self):
        assert fc.technician_lookup_failed(_group(technicians=None, failed=True)) is True

    def test_genuinely_empty_society_is_not_a_failure(self):
        assert fc.technician_lookup_failed(_group(technicians=[], failed=False)) is False

    def test_successful_lookup_is_not_a_failure(self):
        group = _group(
            technicians=[{"userId": "AIT001", "fullName": "A", "mobileNumber": "9"}],
            failed=False,
        )
        assert fc.technician_lookup_failed(group) is False

    def test_legacy_envelope_without_the_flag_is_not_a_failure(self):
        """Pre-flag cache entries omit the key; treating them as failed would
        refresh them forever."""
        assert fc.technician_lookup_failed(_group(technicians=[])) is False

    def test_voice_oan_api_flag_is_read_too(self):
        """voice-oan-api writes the same Redis keys with ``lookupFailed``."""
        group = _group(technicians=[], failed=True, flag="lookupFailed")
        assert fc.technician_lookup_failed(group) is True


class TestFailedLookupIsRetried:
    def _read(self, monkeypatch, groups):
        raw = FarmerDataEnvelope(
            farmers=[FarmerRecord(farmerName="Farmer 1")],
            aiTechnicians=groups,
            fetchedAt=datetime.now(timezone.utc).isoformat(),
            lookupStatus="found",
        ).model_dump()

        class _Cache:
            async def get(self, key, namespace=None):
                return raw

        monkeypatch.setattr(fc, "cache", _Cache())
        return asyncio.run(fc.get_cached_farmer_data("9876543210"))

    def test_one_failed_group_among_several_marks_the_record_stale(self, monkeypatch):
        envelope = self._read(monkeypatch, [
            _group(technicians=[{"userId": "AIT001"}], failed=False),
            _group(technicians=None, failed=True),
        ])
        assert envelope.stale is True
        assert envelope.staleReason == "ai_technician_lookup_failed"

    def test_successful_lookups_leave_a_fresh_record_fresh(self, monkeypatch):
        envelope = self._read(monkeypatch, [_group(technicians=[], failed=False)])
        assert envelope.stale is False


class TestPromptWording:
    def test_failed_lookup_does_not_claim_none_exist(self):
        summary = _build_ai_technician_summary(
            _envelope([_group(technicians=None, failed=True)])
        )
        assert "temporarily unavailable" in summary
        assert "none available for this farmer group" not in summary

    def test_voice_oan_api_failed_lookup_does_not_claim_none_exist(self):
        summary = _build_ai_technician_summary(
            _envelope([_group(technicians=[], failed=True, flag="lookupFailed")])
        )
        assert "temporarily unavailable" in summary

    def test_genuinely_empty_society_still_says_none_available(self):
        summary = _build_ai_technician_summary(
            _envelope([_group(technicians=[], failed=False)])
        )
        assert "none available for this farmer group" in summary
        assert "temporarily unavailable" not in summary

    def test_legacy_envelope_keeps_the_none_available_wording(self):
        summary = _build_ai_technician_summary(_envelope([_group(technicians=[])]))
        assert "none available for this farmer group" in summary


class TestFetchMarksFailure:
    def _fetch(self, monkeypatch, search):
        monkeypatch.setattr(fc, "search_ai_technicians", search)
        record = FarmerRecord(unionCode="BANAS", societyCode="SC001")
        return asyncio.run(fc._fetch_ai_technicians([record]))

    def test_failed_search_marks_lookup_failed(self, monkeypatch):
        async def _fails(**kwargs):
            raise RuntimeError("upstream down")

        groups = self._fetch(monkeypatch, _fails)
        assert groups[0]["techniciansLookupFailed"] is True
        assert groups[0]["technicians"] is None

    def test_empty_society_does_not_mark_failure(self, monkeypatch):
        async def _empty(**kwargs):
            return []

        groups = self._fetch(monkeypatch, _empty)
        assert groups[0]["techniciansLookupFailed"] is False
        assert groups[0]["technicians"] == []
