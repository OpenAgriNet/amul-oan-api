"""Unit tests for the voice runtime context (compact farmer summary).

Covers Fix-1: counts always surface (with len(tags) fallback), cow/buffalo/milking
always surface, all tags inline — so the voice agent answers herd questions from
context without needing the dropped get_herd_summary / list_animal_tags /
get_farmer_profile tools.
"""
# Import the app entry first so the pre-existing tools<->farmer_cache cycle
# resolves in the same order the running app establishes it.
import app.services.voice as voice

from agents.models.farmer import FarmerDataEnvelope
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


def test_signed_in_farmer_tools_drops_brittle_three():
    """The three brittle voice-only farmer tools are commented out; only the
    union-scheme tool remains in SIGNED_IN_FARMER_TOOLS."""
    from agents.tools import SIGNED_IN_FARMER_TOOLS

    assert len(SIGNED_IN_FARMER_TOOLS) == 1, (
        f"expected only get_union_scheme_data; got {len(SIGNED_IN_FARMER_TOOLS)} tools"
    )
