import pytest

from agents import farmer_context
from app.models.animal import AnimalModel
from app.models.farmer import FarmerModel


@pytest.mark.asyncio
async def test_chat_context_uses_directed_beckn_farmer_animal_and_banas_callbacks(monkeypatch):
    monkeypatch.setattr(farmer_context.settings, "enable_network", True)
    monkeypatch.setattr(
        farmer_context.settings, "beckn_callback_transactions_enabled", True
    )
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

    async def direct(*args, **kwargs):
        raise AssertionError("direct provider client must not run in callback mode")

    monkeypatch.setattr(farmer_context, "fetch_authenticated_farmers", farmers)
    monkeypatch.setattr(farmer_context, "fetch_animal_profile", animal)
    monkeypatch.setattr(farmer_context, "fetch_banas_visits", visits)
    monkeypatch.setattr(farmer_context, "search_ai_technicians", technicians)
    monkeypatch.setattr(farmer_context, "get_farmer_data_by_mobile", direct)
    monkeypatch.setattr(farmer_context, "get_animal_data_by_tag", direct)
    monkeypatch.setattr(farmer_context, "fetch_banas_operated_visit", direct)

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
