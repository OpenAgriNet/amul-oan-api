"""Unit tests for the voice runtime context (compact farmer summary).

Covers Fix-1: counts always surface (with len(tags) fallback), cow/buffalo/milking
always surface, all tags inline — so the voice agent answers herd questions from
context without needing the dropped get_herd_summary / list_animal_tags /
get_farmer_profile tools.
"""
# Import the app entry first so the pre-existing tools<->farmer_cache cycle
# resolves in the same order the running app establishes it.
import asyncio

import app.voice.farmer as voice

from agents.tools.models.farmer_transport import FarmerDataEnvelope
from app.voice.models.union import UnionName
from helpers.gujarati_numbers import mask_tag_identifier


def _envelope(record: dict) -> FarmerDataEnvelope:
    return FarmerDataEnvelope.from_records([record], source="api", lookup_status="found")


def test_summary_falls_back_to_tag_count_when_totalanimals_null():
    """Turn-A fix: a record missing totalAnimals (the incident shape) still
    answers 'you have N animals' by deriving N from the tag list."""
    record = {"farmerName": "X", "tagNo": "100000000001,100000000002,100000000003"}
    summary = voice._build_compact_farmer_summary(_envelope(record))
    assert "- Total animals: 3" in summary


def test_summary_includes_cow_buffalo_milking():
    record = {
        "farmerName": "X",
        "totalAnimals": 5,
        "cow": 3,
        "buffalo": 2,
        "totalMilkingAnimals": 4,
        "tagNo": "100000000001,100000000002,100000000003,100000000004,100000000005",
    }
    summary = voice._build_compact_farmer_summary(_envelope(record))
    assert "- Total animals: 5" in summary
    assert "- Cows: 3" in summary
    assert "- Buffaloes: 2" in summary
    assert "- Milking animals: 4" in summary


def test_summary_shows_all_tags_no_truncation():
    """list_animal_tags tool dropped — context must list all tags inline."""
    tags_csv = ",".join(f"10000000{i:04d}" for i in range(12))  # 12 tags
    record = {"farmerName": "X", "totalAnimals": 12, "tagNo": tags_csv}
    summary = voice._build_compact_farmer_summary(_envelope(record))
    assert "more)" not in summary  # no truncation marker
    # All 12 masked tag strings appear (distinct last-4-digit verbalizations)
    masked_count = sum(
        1 for i in range(12)
        if mask_tag_identifier(f"10000000{i:04d}") in summary
    )
    assert masked_count == 12


def test_summary_handles_record_with_no_count_and_no_tags():
    """Edge case: a record with neither totalAnimals nor tags omits the line
    rather than asserting a zero count we can't substantiate."""
    record = {"farmerName": "X", "societyName": "S"}
    summary = voice._build_compact_farmer_summary(_envelope(record))
    assert "Total animals" not in summary


def test_farmer_location_uses_one_complete_record():
    envelope = FarmerDataEnvelope.from_records(
        [{"village": "Bagodara", "district": "Ahmedabad"}],
        source="api",
        lookup_status="found",
    )
    assert voice._collect_farmer_location(envelope) == ("Bagodara", "Ahmedabad")


def test_farmer_location_is_withheld_when_accounts_disagree():
    envelope = FarmerDataEnvelope.from_records(
        [
            {"village": "Bagodara", "district": "Ahmedabad"},
            {"village": "Vijapur", "district": "Mehsana"},
        ],
        source="api",
        lookup_status="found",
    )
    assert voice._collect_farmer_location(envelope) == (None, None)


def test_farmer_location_never_combines_fields_from_different_records():
    envelope = FarmerDataEnvelope.from_records(
        [
            {"village": "Bagodara"},
            {"district": "Mehsana"},
        ],
        source="api",
        lookup_status="found",
    )
    assert voice._collect_farmer_location(envelope) == (None, None)


def test_farmer_location_accepts_compatible_partial_records():
    envelope = FarmerDataEnvelope.from_records(
        [
            {"village": "Bagodara"},
            {"village": "BAGODARA", "district": "Ahmedabad"},
        ],
        source="api",
        lookup_status="found",
    )
    assert voice._collect_farmer_location(envelope) == ("BAGODARA", "Ahmedabad")


def test_signed_in_farmer_tools_drops_brittle_three():
    """The three brittle voice-only farmer tools are commented out; scheme + bonus remain."""
    from agents.voice.tools import SIGNED_IN_FARMER_TOOLS
    from agents.voice.tools.bonus import prepare_get_farmer_bonus_amount

    assert len(SIGNED_IN_FARMER_TOOLS) == 2, (
        f"expected get_union_scheme_data + get_farmer_bonus_amount; "
        f"got {len(SIGNED_IN_FARMER_TOOLS)} tools"
    )
    names = {getattr(t.function, "__name__", "") for t in SIGNED_IN_FARMER_TOOLS}
    assert names == {"get_union_scheme_data", "get_farmer_bonus_amount"}
    bonus_tool = next(
        t
        for t in SIGNED_IN_FARMER_TOOLS
        if getattr(t.function, "__name__", "") == "get_farmer_bonus_amount"
    )
    assert bonus_tool.prepare is prepare_get_farmer_bonus_amount
    assert bonus_tool.takes_ctx is True


def test_scheme_summary_includes_sumul_sursagar_and_sabar(monkeypatch):
    async def fake_records(union_name):
        if union_name == UnionName.SUMUL.value:
            return [{"scheme_title": "Sumul Shed Subsidy", "scheme_url": "https://sumul.example/a.pdf"}]
        if union_name == UnionName.SURENDRANAGAR.value:
            return [{"scheme_title": "Sursagar Insurance", "scheme_url": "https://sursagar.example/b.pdf"}]
        if union_name == UnionName.SABAR.value:
            return [{"scheme_title": "Sabar Cattle Insurance", "scheme_url": "https://sabar.example/c.pdf"}]
        return []

    monkeypatch.setattr(voice, "get_cached_scheme_records_for_union", fake_records)

    sumul_out = asyncio.run(voice._build_union_scheme_summary(["sumul"]))
    assert "Sumul Shed Subsidy" in sumul_out

    sursagar_out = asyncio.run(voice._build_union_scheme_summary(["sursagar"]))
    assert "Sursagar Insurance" in sursagar_out

    sabar_out = asyncio.run(voice._build_union_scheme_summary(["sabar"]))
    assert "Sabar Cattle Insurance" in sabar_out

    assert asyncio.run(voice._build_union_scheme_summary(["dudhsagar"])) == ""


def test_scheme_context_unions_follow_cache_sources():
    from app.voice.scheme_ingestion import SUPPORTED_SCHEME_UNIONS

    assert voice.SUPPORTED_SCHEME_CONTEXT_UNIONS == SUPPORTED_SCHEME_UNIONS
    assert UnionName.SABAR.value in voice.SUPPORTED_SCHEME_CONTEXT_UNIONS
