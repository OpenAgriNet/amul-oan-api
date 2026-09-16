import pytest

from agents import farmer_context
from agents.tools.models.animal import AnimalModel
from agents.tools.models.farmer import FarmerModel


@pytest.mark.asyncio
async def test_chat_context_uses_directed_beckn_farmer_animal_and_banas_callbacks(monkeypatch):
    calls = []

    async def farmers(mobile, **kwargs):
        calls.append(("farmer", mobile))
        return [FarmerModel.model_validate({
            "unionName": "Banas",
            "unionCode": "U-BANAS",
            "societyCode": "S1",
            "farmerCode": "F1",
            "farmerName": "Farmer One",
            "tagNo": "TAG-OWNED",
        })]

    async def animal(tag, **kwargs):
        calls.append(("animal", tag, kwargs["union_code"]))
        return AnimalModel.model_validate({"tagNumber": tag, "breed": "Gir"})

    async def visits(tag, **kwargs):
        calls.append(("banas", tag, kwargs["union_code"]))
        return []

    async def technicians(**kwargs):
        calls.append(("ait", kwargs["union_code"], kwargs["society_code"]))
        return []

    monkeypatch.setattr(farmer_context, "fetch_authenticated_farmers", farmers)
    monkeypatch.setattr(farmer_context, "fetch_animal_profile", animal)
    monkeypatch.setattr(farmer_context, "fetch_banas_visits", visits)
    monkeypatch.setattr(farmer_context, "search_ai_technicians", technicians)

    markdown, unions, _location = (
        await farmer_context.get_farmer_context_bundle_by_mobile("9000000000")
    )

    assert unions == ["banas"]
    assert "TAG-OWNED" in markdown
    assert "gir" in markdown
    assert calls == [
        ("farmer", "9000000000"),
        ("ait", "U-BANAS", "S1"),
        ("animal", "TAG-OWNED", "U-BANAS"),
        ("banas", "TAG-OWNED", "U-BANAS"),
    ]


@pytest.mark.asyncio
async def test_chat_context_reports_explicit_farmer_not_found(monkeypatch):
    async def farmers(_mobile, **_kwargs):
        return []

    monkeypatch.setattr(farmer_context, "fetch_authenticated_farmers", farmers)

    markdown, unions, location = (
        await farmer_context.get_farmer_context_bundle_by_mobile("9000000000")
    )

    assert "No farmer information found" in markdown
    assert unions == []
    assert location == {}


@pytest.mark.asyncio
async def test_chat_context_preserves_farmer_and_animal_when_visit_lookup_fails(monkeypatch):
    async def farmers(mobile, **kwargs):
        return [FarmerModel.model_validate({
            "unionName": "Banas",
            "unionCode": "U-BANAS",
            "societyCode": "S1",
            "farmerCode": "F1",
            "farmerName": "Farmer One",
            "district": "Banaskantha",
            "tagNo": "TAG-OWNED",
        })]

    async def animal(tag, **kwargs):
        return AnimalModel.model_validate({"tagNumber": tag, "breed": "Gir"})

    async def visits(tag, **kwargs):
        raise RuntimeError("visit provider unavailable")

    async def technicians(**kwargs):
        return []

    monkeypatch.setattr(farmer_context, "fetch_authenticated_farmers", farmers)
    monkeypatch.setattr(farmer_context, "fetch_animal_profile", animal)
    monkeypatch.setattr(farmer_context, "fetch_banas_visits", visits)
    monkeypatch.setattr(farmer_context, "search_ai_technicians", technicians)

    markdown, unions, location = (
        await farmer_context.get_farmer_context_bundle_by_mobile("9000000000")
    )

    assert "farmer one" in markdown
    assert "gir" in markdown
    assert unions == ["banas"]
    assert location["district"] == "banaskantha"
