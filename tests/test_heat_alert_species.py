"""The alert animal's species when the alert's codes match no account of the mobile."""

import asyncio
from types import SimpleNamespace

from app.services import heat_alert_call as hac


def test_species_comes_from_the_pashu_aadhaar_without_an_account(monkeypatch):
    """Cowfit sends no union code, so 8 of 14 alerts matched no account: their species was never looked up."""

    async def no_cache(mobile):
        return None

    async def profile(tag_id, *, union_code):
        assert (tag_id, union_code) == ("105147666373", None)
        return SimpleNamespace(animal_type="Cow")

    monkeypatch.setattr(hac, "get_farmer_data_cached_only", no_cache)
    monkeypatch.setattr(hac, "fetch_animal_profile", profile)
    event = SimpleNamespace(id="e", pashuaadhar_no="105147666373", tag_no="1", farmer_contact="9999999999")

    assert asyncio.run(hac._species(event, None)) == "cow"
