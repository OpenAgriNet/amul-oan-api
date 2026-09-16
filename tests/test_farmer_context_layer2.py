"""Tests for Layer-2-first chat farmer context migration (step 3)."""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.models.farmer import FarmerModel
from app.models.farmer_transport import FarmerDataEnvelope, FarmerRecord


@pytest.fixture(autouse=True)
def _layer2_flags_off(monkeypatch):
    monkeypatch.setattr(
        "agents.farmer_context.settings.farmer_layer2_chat_context_enabled",
        False,
    )
    monkeypatch.setattr(
        "agents.farmer_context.settings.farmer_layer2_fallback_to_legacy_enabled",
        True,
    )
    monkeypatch.setattr(
        "agents.farmer_context.settings.enable_network",
        False,
    )
    monkeypatch.setattr(
        "agents.farmer_context.settings.beckn_callback_transactions_enabled",
        False,
    )


def _envelope_found(**farmer_fields) -> FarmerDataEnvelope:
    record = FarmerRecord(
        farmerName="Test Farmer",
        unionName="kaira",
        unionCode="1",
        societyCode="S1",
        societyName="Test Society",
        farmerCode="F1",
        district="Anand",
        tagNo="T1",
        **farmer_fields,
    )
    envelope = FarmerDataEnvelope.from_records([record], lookup_status="found")
    envelope.aiTechnicians = [{
        "farmerCode": "F1",
        "societyCode": "S1",
        "unionCode": "1",
        "technicians": [{
            "userId": "ait-1",
            "fullName": "Ramesh Patel",
            "mobileNumber": "9999999999",
        }],
    }]
    return envelope


def test_flag_off_uses_legacy_path(monkeypatch):
    import agents.farmer_context as fc

    legacy = AsyncMock(return_value=("legacy-md", ["kaira"], {"district": "anand"}))
    layer2 = AsyncMock()
    monkeypatch.setattr(fc, "_get_farmer_context_bundle_legacy", legacy)
    monkeypatch.setattr(fc, "_get_farmer_context_bundle_layer2", layer2)

    result = asyncio.run(fc.get_farmer_context_bundle_by_mobile("9876543210"))
    assert result == ("legacy-md", ["kaira"], {"district": "anand"})
    legacy.assert_awaited_once_with("9876543210")
    layer2.assert_not_called()


def test_flag_on_uses_layer2_path(monkeypatch):
    import agents.farmer_context as fc

    monkeypatch.setattr(
        "agents.farmer_context.settings.farmer_layer2_chat_context_enabled",
        True,
    )
    layer2 = AsyncMock(return_value=("layer2-md", ["kaira"], {"district": "anand"}))
    legacy = AsyncMock()
    monkeypatch.setattr(fc, "_get_farmer_context_bundle_layer2", layer2)
    monkeypatch.setattr(fc, "_get_farmer_context_bundle_legacy", legacy)

    result = asyncio.run(fc.get_farmer_context_bundle_by_mobile("9876543210"))
    assert result == ("layer2-md", ["kaira"], {"district": "anand"})
    layer2.assert_awaited_once_with("9876543210")
    legacy.assert_not_called()


def test_flag_on_not_found_does_not_fallback(monkeypatch):
    import agents.farmer_context as fc

    monkeypatch.setattr(
        "agents.farmer_context.settings.farmer_layer2_chat_context_enabled",
        True,
    )
    layer2 = AsyncMock(return_value=(
        "# Farmer Context\n\nNo farmer information found for mobile number `9876543210`.",
        [],
        {},
    ))
    legacy = AsyncMock()
    monkeypatch.setattr(fc, "_get_farmer_context_bundle_layer2", layer2)
    monkeypatch.setattr(fc, "_get_farmer_context_bundle_legacy", legacy)

    markdown, unions, location = asyncio.run(
        fc.get_farmer_context_bundle_by_mobile("9876543210")
    )
    assert "No farmer information found" in markdown
    assert unions == [] and location == {}
    legacy.assert_not_called()


def test_flag_on_unusable_envelope_falls_back_to_legacy(monkeypatch):
    import agents.farmer_context as fc

    monkeypatch.setattr(
        "agents.farmer_context.settings.farmer_layer2_chat_context_enabled",
        True,
    )
    layer2 = AsyncMock(return_value=None)
    legacy = AsyncMock(return_value=("legacy-md", ["kaira"], {}))
    monkeypatch.setattr(fc, "_get_farmer_context_bundle_layer2", layer2)
    monkeypatch.setattr(fc, "_get_farmer_context_bundle_legacy", legacy)

    result = asyncio.run(fc.get_farmer_context_bundle_by_mobile("9876543210"))
    assert result == ("legacy-md", ["kaira"], {})
    legacy.assert_awaited_once_with("9876543210")


def test_flag_on_unusable_without_fallback_returns_not_found(monkeypatch):
    import agents.farmer_context as fc

    monkeypatch.setattr(
        "agents.farmer_context.settings.farmer_layer2_chat_context_enabled",
        True,
    )
    monkeypatch.setattr(
        "agents.farmer_context.settings.farmer_layer2_fallback_to_legacy_enabled",
        False,
    )
    layer2 = AsyncMock(return_value=None)
    legacy = AsyncMock()
    monkeypatch.setattr(fc, "_get_farmer_context_bundle_layer2", layer2)
    monkeypatch.setattr(fc, "_get_farmer_context_bundle_legacy", legacy)

    markdown, unions, location = asyncio.run(
        fc.get_farmer_context_bundle_by_mobile("9876543210")
    )
    assert "No farmer information found" in markdown
    assert unions == [] and location == {}
    legacy.assert_not_called()


def test_layer2_uses_cached_technicians_without_live_api(monkeypatch):
    import agents.farmer_context as fc

    envelope = _envelope_found()
    api_calls = {"n": 0}

    async def fake_api(*args, **kwargs):
        api_calls["n"] += 1
        return []

    async def fake_schemes(lines, unions):
        return None

    async def fake_animal(tag, *args, **kwargs):
        return tag, None, None, None

    monkeypatch.setattr(
        "agents.services.farmer_cache.get_or_fetch_farmer_data",
        AsyncMock(return_value=envelope),
    )
    monkeypatch.setattr(fc, "_append_union_scheme_summary_markdown", fake_schemes)
    monkeypatch.setattr(fc, "get_ai_technicians_by_society_cached", fake_api)
    monkeypatch.setattr(fc, "_get_animal_context_bundle", fake_animal)

    markdown, unions, location = asyncio.run(
        fc._get_farmer_context_bundle_layer2("9876543210")
    )
    assert unions == ["kaira"]
    assert location["district"] == "anand"
    assert "Ramesh Patel" in markdown
    assert "ait-1" in markdown
    assert api_calls["n"] == 0


def test_layer2_passes_normalized_phone_to_cache_fetch(monkeypatch):
    import agents.farmer_context as fc

    envelope = _envelope_found()
    fetch = AsyncMock(return_value=envelope)
    monkeypatch.setattr("agents.services.farmer_cache.get_or_fetch_farmer_data", fetch)
    monkeypatch.setattr(
        fc,
        "_append_union_scheme_summary_markdown",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        fc,
        "_get_animal_context_bundle",
        AsyncMock(return_value=("T1", None, None, None)),
    )

    asyncio.run(fc._get_farmer_context_bundle_layer2("+91 9876543210"))

    fetch.assert_awaited_once_with("9876543210")


def test_layer2_get_or_fetch_none_returns_none(monkeypatch):
    import agents.farmer_context as fc

    monkeypatch.setattr(
        "agents.services.farmer_cache.get_or_fetch_farmer_data",
        AsyncMock(return_value=None),
    )

    result = asyncio.run(fc._get_farmer_context_bundle_layer2("9876543210"))
    assert result is None


def test_beckn_callback_priority_over_layer2_flag(monkeypatch):
    """When both Beckn callbacks and Layer 2 are enabled, callback routing wins."""
    import agents.farmer_context as fc

    monkeypatch.setattr(fc.settings, "enable_network", True)
    monkeypatch.setattr(fc.settings, "beckn_callback_transactions_enabled", True)
    monkeypatch.setattr(fc.settings, "farmer_layer2_chat_context_enabled", True)

    beckn = AsyncMock(return_value=("beckn-md", ["kaira"], {"district": "anand"}))
    layer2 = AsyncMock(return_value=("layer2-md", ["kaira"], {}))
    monkeypatch.setattr(fc, "_get_farmer_context_bundle_beckn", beckn)
    monkeypatch.setattr(fc, "_get_farmer_context_bundle_layer2", layer2)

    result = asyncio.run(fc.get_farmer_context_bundle_by_mobile("9876543210"))

    assert result == ("beckn-md", ["kaira"], {"district": "anand"})
    beckn.assert_awaited_once_with("9876543210")
    layer2.assert_not_called()


def test_technician_group_prefers_exact_farmer_code_over_earlier_society_match():
    """Regression: same society/union groups must not steal an exact farmerCode match.

    If society/union fallback returns inside the same loop as farmerCode matching,
    Group A (wrong farmer) is returned before Group B (exact code) is reached.
    """
    import agents.farmer_context as fc

    farmer = FarmerModel(
        farmerName="Target Farmer",
        unionName="kaira",
        unionCode="1",
        societyCode="S1",
        farmerCode="222",
    )
    group_a = {
        "farmerCode": "111",
        "societyCode": "S1",
        "unionCode": "1",
        "technicians": [{
            "userId": "ait-wrong",
            "fullName": "Wrong Tech",
            "mobileNumber": "9000000001",
        }],
    }
    group_b = {
        "farmerCode": "222",
        "societyCode": "S1",
        "unionCode": "1",
        "technicians": [{
            "userId": "ait-correct",
            "fullName": "Correct Tech",
            "mobileNumber": "9000000002",
        }],
    }

    selected = fc._technician_group_for_farmer(farmer, [group_a, group_b])
    assert selected is group_b
    assert selected["technicians"][0]["userId"] == "ait-correct"


def test_technician_group_falls_back_to_society_union_when_no_farmer_code_match():
    import agents.farmer_context as fc

    farmer = FarmerModel(
        farmerName="Target Farmer",
        unionName="kaira",
        unionCode="1",
        societyCode="S1",
        farmerCode="999",
    )
    other_society = {
        "farmerCode": "111",
        "societyCode": "S2",
        "unionCode": "1",
        "technicians": [{"userId": "ait-other"}],
    }
    same_society = {
        "farmerCode": "222",
        "societyCode": "S1",
        "unionCode": "1",
        "technicians": [{"userId": "ait-fallback"}],
    }

    selected = fc._technician_group_for_farmer(farmer, [other_society, same_society])
    assert selected is same_society
    assert selected["technicians"][0]["userId"] == "ait-fallback"


def test_technician_group_exact_farmer_code_requires_same_union_society():
    import agents.farmer_context as fc

    farmer = FarmerModel(
        farmerName="Target Farmer",
        unionName="kaira",
        unionCode="2",
        societyCode="S2",
        farmerCode="1",
    )
    wrong_scope_same_code = {
        "farmerCode": "1",
        "societyCode": "S1",
        "unionCode": "1",
        "technicians": [{"userId": "ait-wrong"}],
    }
    right_scope_same_code = {
        "farmerCode": "1",
        "societyCode": "S2",
        "unionCode": "2",
        "technicians": [{"userId": "ait-correct"}],
    }

    selected = fc._technician_group_for_farmer(farmer, [wrong_scope_same_code, right_scope_same_code])
    assert selected is right_scope_same_code
    assert selected["technicians"][0]["userId"] == "ait-correct"


def test_cached_technician_failure_does_not_emit_hard_none_found(monkeypatch):
    import agents.farmer_context as fc

    farmer = FarmerModel(
        farmerName="Target Farmer",
        unionName="kaira",
        unionCode="1",
        societyCode="S1",
        farmerCode="F1",
    )
    ai_groups = [{
        "farmerCode": "F1",
        "societyCode": "S1",
        "unionCode": "1",
        "technicians": None,
        "techniciansLookupFailed": True,
    }]
    lines = []

    lookup = AsyncMock(return_value=(None, "AI technician details could not be fetched right now."))
    monkeypatch.setattr(
        fc,
        "_get_ai_technicians_for_farmer",
        lookup,
    )

    asyncio.run(fc._append_ai_technicians_markdown_with_cache(lines, farmer, ai_groups))

    markdown = "\n".join(lines)
    assert "AI technician details could not be fetched right now." in markdown
    assert "No AI technicians were found for this society." not in markdown
    lookup.assert_awaited_once_with(farmer, force_refresh=True)


def test_live_verify_bypasses_cached_technician_helper(monkeypatch):
    import agents.farmer_context as fc

    farmer = FarmerModel(
        farmerName="Target Farmer",
        unionName="kaira",
        unionCode="1",
        societyCode="S1",
    )
    cached_lookup = AsyncMock(side_effect=AssertionError("cached helper must be bypassed"))
    refresh_lookup = AsyncMock(return_value=[])
    monkeypatch.setattr(fc, "get_ai_technicians_by_society_cached", cached_lookup)
    monkeypatch.setattr(fc, "get_ai_technicians_by_society_refresh", refresh_lookup)

    lines, error = asyncio.run(fc._get_ai_technicians_for_farmer(farmer, force_refresh=True))

    assert lines == []
    assert error is None
    cached_lookup.assert_not_called()
    refresh_lookup.assert_awaited_once()


def test_legacy_empty_cache_does_not_trigger_refresh_verification(monkeypatch):
    """Legacy/default path trusts cached empty technician results."""
    import agents.farmer_context as fc

    farmer = FarmerModel(
        farmerName="Target Farmer",
        unionName="kaira",
        unionCode="1",
        societyCode="S1",
    )
    cached_lookup = AsyncMock(return_value=[])
    refresh_lookup = AsyncMock(return_value=[])
    monkeypatch.setattr(fc, "get_ai_technicians_by_society_cached", cached_lookup)
    monkeypatch.setattr(fc, "get_ai_technicians_by_society_refresh", refresh_lookup)

    lines, error = asyncio.run(fc._get_ai_technicians_for_farmer(farmer))

    assert error is None
    assert lines == []
    cached_lookup.assert_awaited_once()
    refresh_lookup.assert_not_called()


def test_legacy_markdown_does_not_hard_none_found_on_stale_empty_cache(monkeypatch):
    import agents.farmer_context as fc

    farmer = FarmerModel(
        farmerName="Target Farmer",
        unionName="kaira",
        unionCode="1",
        societyCode="S1",
    )
    monkeypatch.setattr(fc, "get_ai_technicians_by_society_cached", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        fc,
        "get_ai_technicians_by_society_refresh",
        AsyncMock(side_effect=AssertionError("refresh should not be used for cached empty list")),
    )

    lines: list[str] = []
    asyncio.run(fc._append_ai_technicians_markdown(lines, farmer))
    markdown = "\n".join(lines)

    assert "No AI technicians were found for this society." in markdown


def test_layer2_records_preserve_legacy_merge_behavior(monkeypatch):
    import agents.farmer_context as fc

    records = [
        FarmerRecord(farmerName="Ramesh", societyName="Alpha Society", farmerCode="F1"),
        FarmerRecord(farmerName="Ramesh", societyName="Alpha Society", farmerCode="F1"),
    ]
    sentinel = [FarmerModel(farmerName="Merged Farmer")]
    from unittest.mock import Mock
    merge = Mock(return_value=sentinel)
    monkeypatch.setattr(fc, "merge_farmer_data", merge)
    farmers = fc._farmer_records_to_models(records)

    assert farmers is sentinel
    assert merge.call_count == 1
    assert len(merge.call_args.args[0]) == 2


def test_layer2_integration_sarhad_skips_technicians(monkeypatch):
    import agents.farmer_context as fc

    envelope = FarmerDataEnvelope.from_records([
        FarmerRecord(
            unionName="sarhad",
            unionCode="12",
            societyCode="S1",
            farmerName="Kutch Farmer",
        )
    ], lookup_status="found")
    envelope.aiTechnicians = [{
        "unionCode": "12",
        "societyCode": "S1",
        "technicians": [{
            "userId": "ait-1",
            "fullName": "Should Not Appear",
            "mobileNumber": "9999999999",
        }],
    }]

    api_calls = {"n": 0}

    async def fake_api(*args, **kwargs):
        api_calls["n"] += 1
        return []

    async def fake_schemes(lines, unions):
        return None

    monkeypatch.setattr(
        "agents.farmer_context.settings.farmer_layer2_chat_context_enabled",
        True,
    )
    monkeypatch.setattr(
        "agents.services.farmer_cache.get_or_fetch_farmer_data",
        AsyncMock(return_value=envelope),
    )
    monkeypatch.setattr(fc, "_append_union_scheme_summary_markdown", fake_schemes)
    monkeypatch.setattr(fc, "get_ai_technicians_by_society_cached", fake_api)
    monkeypatch.setattr(
        fc,
        "_get_animal_context_bundle",
        AsyncMock(return_value=("T1", None, None, None)),
    )

    markdown, unions, _ = asyncio.run(
        fc.get_farmer_context_bundle_by_mobile("9876543210")
    )
    assert unions == ["sarhad"]
    assert api_calls["n"] == 0
    assert "Should Not Appear" not in markdown
    assert "Available AI technicians" not in markdown


@pytest.mark.parametrize(
    "layer2_enabled,fallback_enabled,layer2_result,expect_legacy,expected_marker",
    [
        (
            False,
            True,
            ("layer2-md", ["kaira"], {"district": "anand"}),
            True,
            "legacy-md",
        ),
        (
            True,
            True,
            ("layer2-md", ["kaira"], {"district": "anand"}),
            False,
            "layer2-md",
        ),
        (
            True,
            True,
            None,
            True,
            "legacy-md",
        ),
        (
            True,
            False,
            None,
            False,
            "No farmer information found",
        ),
    ],
)
def test_routing_matrix_for_chat_migration_flags(
    monkeypatch,
    layer2_enabled,
    fallback_enabled,
    layer2_result,
    expect_legacy,
    expected_marker,
):
    import agents.farmer_context as fc

    monkeypatch.setattr(
        "agents.farmer_context.settings.farmer_layer2_chat_context_enabled",
        layer2_enabled,
    )
    monkeypatch.setattr(
        "agents.farmer_context.settings.farmer_layer2_fallback_to_legacy_enabled",
        fallback_enabled,
    )

    legacy = AsyncMock(return_value=("legacy-md", ["legacy"], {}))
    layer2 = AsyncMock(return_value=layer2_result)
    monkeypatch.setattr(fc, "_get_farmer_context_bundle_legacy", legacy)
    monkeypatch.setattr(fc, "_get_farmer_context_bundle_layer2", layer2)

    markdown, _unions, _location = asyncio.run(
        fc.get_farmer_context_bundle_by_mobile("9876543210")
    )
    assert expected_marker in markdown

    if layer2_enabled:
        layer2.assert_awaited_once_with("9876543210")
    else:
        layer2.assert_not_called()

    if expect_legacy:
        legacy.assert_awaited_once_with("9876543210")
    else:
        legacy.assert_not_called()
