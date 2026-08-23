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
    monkeypatch.setattr(fc, "get_ai_technicians_by_society_api", fake_api)
    monkeypatch.setattr(fc, "_get_animal_context_bundle", fake_animal)

    markdown, unions, location = asyncio.run(
        fc._get_farmer_context_bundle_layer2("9876543210")
    )
    assert unions == ["kaira"]
    assert location["district"] == "anand"
    assert "Ramesh Patel" in markdown
    assert "ait-1" in markdown
    assert api_calls["n"] == 0


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
    monkeypatch.setattr(fc, "get_ai_technicians_by_society_api", fake_api)
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
