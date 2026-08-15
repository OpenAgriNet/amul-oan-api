"""Farmer-location plumbing and the prompt guidance that routes to it.

Two halves of the same feature:

* The district has always been IN the farmer record and rendered into the prompt
  markdown — it was just never exposed as *data*, which is why the mandi and
  weather tools hardcoded Anand for every farmer in India.
* No system prompt mentioned Vistaar, Beckn or mandi at all (zero hits across
  all five files), so tool docstrings were the only routing signal and nothing
  told the model it could pass a location, or what to do when a place is not
  covered.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.deps import FarmerContext  # noqa: E402
from agents.farmer_context import _collect_farmer_location  # noqa: E402
from app.models.farmer import FarmerModel  # noqa: E402
from helpers.utils import get_prompt  # noqa: E402


class TestFarmerLocationCollection:
    def test_the_district_is_lifted_out_of_the_farmer_record(self):
        farmers = [FarmerModel(state="Gujarat", district="Junagadh", village="Vadal")]
        assert _collect_farmer_location(farmers) == {
            "district": "junagadh", "village": "vadal", "state": "gujarat",
        }

    def test_a_record_without_a_district_contributes_nothing(self):
        assert _collect_farmer_location([FarmerModel()]) == {}
        assert _collect_farmer_location([]) == {}

    def test_village_and_state_come_from_the_same_record_as_the_district(self):
        # Mixing a district from one account with a village from another would
        # invent a place. A mobile with several accounts is normal.
        farmers = [
            FarmerModel(village="Orphan Village"),
            FarmerModel(district="Kutch", village="Bhirandiyara", state="Gujarat"),
        ]
        assert _collect_farmer_location(farmers)["village"] == "bhirandiyara"

    def test_farmer_context_exposes_the_district_to_tools(self):
        assert FarmerContext(query="q", farmer_district=" Junagadh ").get_farmer_district() == "Junagadh"
        assert FarmerContext(query="q").get_farmer_district() is None
        assert FarmerContext(query="q", farmer_district="").get_farmer_district() is None

    @pytest.mark.asyncio
    async def test_the_bundle_returns_the_location_as_its_third_element(self, monkeypatch):
        import agents.farmer_context as fc

        async def _fake_get(mobile):
            return [FarmerModel(district="Banas Kantha", village="Dama", state="Gujarat")]

        monkeypatch.setattr(fc, "get_farmer_data_by_mobile", _fake_get)
        _, _, location = await fc.get_farmer_context_bundle_by_mobile("9876543210")
        assert location["district"] == "banas kantha"

    @pytest.mark.asyncio
    async def test_an_unknown_mobile_still_returns_three_elements(self, monkeypatch):
        # The no-farmer early return is a separate code path and has silently
        # skipped new fields before.
        import agents.farmer_context as fc

        async def _fake_get(mobile):
            return None

        monkeypatch.setattr(fc, "get_farmer_data_by_mobile", _fake_get)
        markdown, unions, location = await fc.get_farmer_context_bundle_by_mobile("1")
        assert unions == [] and location == {}
        assert "No farmer information found" in markdown


class TestPromptGuidance:
    PROMPTS = ("agrinet_system.md", "agrinet_system_translation_pipeline.md")

    @staticmethod
    def _render(name, network):
        return get_prompt(name, context={
            "today_date": "13-08-2026", "today_datetime": "13-08-2026 10:00",
            "farmer_context": None, "ambiguity_hints": None,
            "response_max_chars": None, "loan_max_amount": "5,000",
            "loan_interest_rate_pct": "7", "network_tools_enabled": network,
        })

    @pytest.mark.parametrize("name", PROMPTS)
    def test_the_location_argument_is_documented_for_the_model(self, name):
        rendered = self._render(name, True)
        assert "get_vistaar_mandi_prices" in rendered
        assert 'location="Junagadh"' in rendered
        assert "never coordinates" in rendered
        assert "not covered" in rendered

    @pytest.mark.parametrize("name", PROMPTS)
    def test_it_says_not_to_ask_the_farmer_where_they_are(self, name):
        assert "do **not** ask the farmer where they are" in self._render(name, True).lower()

    @pytest.mark.parametrize("name", PROMPTS)
    def test_price_and_weather_are_steered_away_from_document_search(self, name):
        # agrinet_system.md rule 3 and the pipeline prompt's routing rule 3 both
        # sent `market` / `weather` to search_documents first, which describes a
        # mandi question exactly.
        rendered = self._render(name, True)
        assert "do not search first" in rendered
        assert "live data" in rendered

    @pytest.mark.parametrize("name", PROMPTS)
    def test_the_block_disappears_when_the_network_flag_is_off(self, name):
        # Advertising a tool the runtime has hidden costs two LLM round-trips a
        # turn: the model calls it, gets `Unknown tool name`, and retries — and
        # the error enumerates the whole tool list back into context.
        rendered = self._render(name, False).lower()
        for tool in ("get_vistaar_mandi_prices", "get_vistaar_weather", "get_vistaar_scheme_info"):
            assert tool not in rendered, f"{tool} advertised while the network flag is off"

    def test_the_pipeline_prompt_stops_routing_market_to_search_when_tools_are_on(self):
        on = self._render("agrinet_system_translation_pipeline.md", True)
        off = self._render("agrinet_system_translation_pipeline.md", False)
        assert "`crop`, `market`, `weather`: use `search_documents` before" in off
        assert "`crop`, `market`, `weather`: use `search_documents` before" not in on

    def test_the_agent_passes_the_flag_into_the_prompt_context(self):
        # Without this line the block renders as absent regardless of the flag,
        # and the whole guidance silently does nothing.
        source = (ROOT / "agents" / "agrinet.py").read_text()
        assert "'network_tools_enabled': settings.enable_network" in source
