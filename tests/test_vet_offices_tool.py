import asyncio
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.deps import FarmerContext
from agents.tools import vet_offices
from agents.tools.vet_offices import find_nearby_vet_offices


def _ctx(village=None, district=None):
    deps = FarmerContext(
        query="nearest vet hospital",
        farmer_village=village,
        farmer_district=district,
    )
    return SimpleNamespace(deps=deps)


def _run(village=None, district=None, taluka=""):
    return asyncio.run(find_nearby_vet_offices(_ctx(village, district), taluka=taluka))


class TestAsset:
    def test_lookup_is_registered_for_the_chat_agent(self):
        from agents.tools.registry import TOOLS

        registered = {tool.name: tool for tool in TOOLS}
        assert registered["find_nearby_vet_offices"].takes_ctx is True

    def test_asset_loads_and_holds_only_treatable_offices(self):
        offices = vet_offices._load_offices()
        assert len(offices) > 3000
        assert {o["category"] for o in offices} <= set(vet_offices.CATEGORY_RANK)

    def test_asset_keeps_the_sheets_own_spellings(self):
        # The transcription is deliberately faithful: the department's sheet is
        # not ours to rewrite, so both sides of every variant pair survive in
        # the data and are reconciled at lookup time instead.
        districts = {o["district"] for o in vet_offices._load_offices()}
        assert {"Kutch", "Kachchh"} <= districts
        assert {"Dohad", "Dahod"} <= districts
        assert {"Vadpdara", "Vadodara"} <= districts

    def test_district_variants_resolve_to_one_district_at_lookup_time(self):
        assert vet_offices._district_key("Kutch") == vet_offices._district_key("Kachchh")
        assert vet_offices._district_key("Dohad") == vet_offices._district_key("Dahod")
        assert vet_offices._district_key("Vadpdara") == vet_offices._district_key("Vadodara")
        assert vet_offices._district_key("BHAVNAGAR") == vet_offices._district_key("Bhavnagar")
        assert vet_offices._district_key("Sabar Kantha") == vet_offices._district_key("Sabarkantha")

    def test_genuinely_different_districts_stay_apart(self):
        # The pairs a fuzzy threshold would have wrongly merged.
        assert vet_offices._district_key("Banaskantha") != vet_offices._district_key("Sabarkantha")
        assert vet_offices._district_key("Anand") != vet_offices._district_key("Patan")
        assert vet_offices._district_key("Botad") != vet_offices._district_key("Dahod")


class TestLocationFromProfile:
    def test_village_in_profile_needs_no_taluka_from_the_caller(self):
        result = _run(village="Bagodara", district="Ahmedabad")
        assert "Bavla" in result
        assert "Ask the user" not in result

    def test_own_village_office_is_offered_first(self):
        result = _run(village="Bagodara", district="Ahmedabad")
        first = result.split("2.")[0]
        assert "Bagodara" in first

    def test_misspelled_village_still_matches(self):
        # Profiles and the sheet disagree on transliteration constantly.
        assert "Bavla" in _run(village="Bavala", district="Ahmedabad")

    def test_exact_village_in_another_district_is_not_used(self):
        # Vijapur is an exact directory key, but only in Mehsana. An exact global
        # hit must not override a profile district that the directory recognizes.
        result = _run(village="Vijapur", district="Ahmedabad")
        assert "Ask the user" in result
        assert "District: Mehsana" not in result


class TestSpokenTalukaFallback:
    def test_asks_for_taluka_when_profile_has_no_usable_location(self):
        result = _run(district="Mehsana")
        assert "Ask the user" in result
        assert "Mehsana district" in result

    def test_asks_for_taluka_when_profile_is_empty(self):
        assert "Ask the user" in _run()

    def test_spoken_taluka_resolves_the_lookup(self):
        result = _run(district="Mehsana", taluka="Vijapur")
        assert "Vijapur" in result
        assert "Ask the user" not in result

    def test_spoken_taluka_wins_over_the_profile_village(self):
        result = _run(village="Bagodara", district="Ahmedabad", taluka="Vijapur")
        assert "Vijapur" in result

    def test_unknown_place_name_asks_again_instead_of_guessing(self):
        assert "Ask the user" in _run(district="Mehsana", taluka="Zzzqqxyz")

    def test_caller_may_answer_with_a_village_instead_of_a_taluka(self):
        # We ask for "taluka or village", so the answer has to work either way —
        # otherwise their reply loops straight back into the same question.
        result = _run(district="Mehsana", taluka="Bamanava")
        assert "Vijapur" in result
        assert "Ask the user" not in result

    def test_exact_village_beats_a_fuzzy_taluka_in_another_district(self):
        # Regression: "Bagodara" (a village in Bavla, Ahmedabad) scored 80
        # against Amreli's "Bagsara" taluka, and with no district to scope by,
        # the taluka index was tried first and won.
        result = _run(taluka="Bagodara")
        assert "Ahmedabad" in result
        assert "Amreli" not in result

    def test_a_name_that_is_both_taluka_and_village_returns_both_keys(self):
        # Vijapur is a taluka of Mehsana and a village inside it: the taluka
        # scopes the search, the village floats their own office to the top.
        assert vet_offices._resolve_spoken_place("Vijapur", "Mehsana") == (
            "vijapur",
            "vijapur",
        )

    def test_non_latin_place_name_asks_for_the_english_spelling(self):
        result = _run(district="Mehsana", taluka="વિજાપુર")
        assert "English letters" in result
        # Not the generic retry prompt: that would ask the same question again.
        assert "Ask the user which" not in result


class TestDistrictCollisions:
    """Place names repeat across Gujarat; the caller's district has to win."""

    def _districts(self, result):
        return {
            line.split("District:")[1].strip()
            for line in result.splitlines()
            if "District:" in line
        }

    def test_repeated_village_name_stays_in_the_callers_district(self):
        # "Kerala" is a village in both Ahmedabad and Morbi.
        assert self._districts(_run(village="Kerala", district="Ahmedabad")) == {"Ahmedabad"}
        assert self._districts(_run(village="Kerala", district="Morbi")) == {"Morbi"}

    def test_repeated_taluka_name_stays_in_the_callers_district(self):
        # Kalol is a taluka of both Gandhinagar and Panchmahal.
        assert self._districts(_run(district="Gandhinagar", taluka="Kalol")) == {"Gandhinagar"}
        assert self._districts(_run(district="Panchmahal", taluka="Kalol")) == {"Panchmahal"}

    def test_city_taluka_stays_in_the_callers_district(self):
        # "City" is the taluka for seven different districts.
        assert self._districts(_run(district="Rajkot", taluka="City")) == {"Rajkot"}

    def test_village_missing_from_the_sheet_does_not_match_another_district(self):
        result = _run(village="Zzzqqxyzpur", district="Kachchh")
        assert "Ask the user" in result


class TestOutput:
    def test_result_is_capped_and_carries_the_gujarati_office_type(self):
        result = _run(district="Ahmedabad", taluka="Bavla")
        assert result.count("Official Gujarati office type") <= vet_offices.MAX_RESULTS
        assert "પશુ" in result

    def test_no_contact_details_are_promised(self):
        result = _run(district="Ahmedabad", taluka="Bavla")
        assert "no phone number or street address" in result

    def test_result_does_not_claim_distance_the_directory_cannot_measure(self):
        result = _run(district="Ahmedabad", taluka="Bavla")
        assert "listed for the user's location" in result
        assert "Nearest veterinary offices" not in result
        assert "never as the nearest or closest office" in result


class TestRanking:
    def test_polyclinic_outranks_a_sub_centre_in_the_same_taluka(self):
        offices = [
            {"village": "A", "category": "ICDP Sub-Centre"},
            {"village": "B", "category": "Veterinary Polyclinic"},
        ]
        assert vet_offices._rank(offices, "")[0]["category"] == "Veterinary Polyclinic"

    def test_own_village_outranks_a_better_office_elsewhere(self):
        offices = [
            {"village": "Elsewhere", "category": "Veterinary Polyclinic"},
            {"village": "Home", "category": "ICDP Sub-Centre"},
        ]
        assert vet_offices._rank(offices, "home")[0]["village"] == "Home"
