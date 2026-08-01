"""
Unit tests for farmer/animal backends: normalization, merge, herdman shape mapping.
"""
import pytest

from agents.tools.farmer_animal_backends import (
    merge_animal_data,
    merge_farmer_records,
    normalize_phone,
    normalize_tag,
)


class TestNormalizePhone:
    def test_digits_only(self):
        assert normalize_phone("9415787824") == "9415787824"

    def test_with_spaces_and_plus91(self):
        assert normalize_phone("+91 94157 87824") == "9415787824"

    def test_leading_91_stripped(self):
        assert normalize_phone("919415787824") == "9415787824"

    def test_empty_returns_original(self):
        assert normalize_phone("") == ""


class TestNormalizeTag:
    def test_strip_whitespace(self):
        assert normalize_tag("  105183817302  ") == "105183817302"

    def test_empty(self):
        assert normalize_tag("") == ""


class TestMergeFarmerRecords:
    def test_dedupe_by_society_and_code(self):
        a = {"societyName": "S1", "farmerCode": "001", "farmerName": "A"}
        b = {"societyName": "S1", "farmerCode": "001", "farmerName": "B"}
        assert len(merge_farmer_records([a, b])) == 1

    def test_keep_different_society_or_code(self):
        a = {"societyName": "S1", "farmerCode": "001"}
        b = {"societyName": "S2", "farmerCode": "001"}
        c = {"societyName": "S1", "farmerCode": "002"}
        assert len(merge_farmer_records([a, b, c])) == 3

    def test_empty_list(self):
        assert merge_farmer_records([]) == []


class TestMergeAnimalData:
    def test_primary_only(self):
        primary = {"tagNumber": "123", "breed": "Murrah"}
        assert merge_animal_data(primary, None) == primary

    def test_fallback_only(self):
        fallback = {"tagNumber": "123", "breed": "HF Cross"}
        assert merge_animal_data(None, fallback) == fallback

    def test_merge_fills_missing(self):
        primary = {"tagNumber": "123", "breed": "Murrah", "lastPD": None}
        fallback = {"tagNumber": "123", "lastPD": "2025-06-24"}
        out = merge_animal_data(primary, fallback)
        assert out["breed"] == "Murrah"
        assert out["lastPD"] == "2025-06-24"

    def test_both_empty(self):
        assert merge_animal_data(None, None) == {}
