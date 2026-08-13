"""Tests for the static Gujarat district → coordinates table.

Two classes of failure this guards against, both of which are silent in
production:

1. **A coordinate typo.** A transposed lat/lon or a dropped digit still looks
   like a coordinate and still returns *some* market — just the wrong one, in
   the wrong state, with no error anywhere. The bounding-box test is the only
   thing standing between a fat-fingered literal and a farmer in Bhuj being
   quoted Madhya Pradesh prices.
2. **A normalisation gap.** District strings arrive lowercased from two backends
   in two spellings (`kachchh` vs `kutch`, `banas kantha`, `the dangs`). A
   lookup miss does not raise — it falls through to the Anand default, i.e. the
   exact behaviour this table was built to remove, for exactly the districts it
   was extended to cover.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.tools.districts import (  # noqa: E402
    DEFAULT_LOCATION,
    DISTRICTS,
    GUJARAT_BBOX,
    covered_district_names,
    normalize_place,
    resolve_place,
    unknown_place_message,
)

MIN_LAT, MAX_LAT, MIN_LON, MAX_LON = GUJARAT_BBOX

# Every district of Gujarat. The earlier 28-entry draft omitted the last five,
# which is where the Dudhdhara, Sumul and Vasudhara farmers are — the omission
# was invisible because a missing district silently resolved to Anand.
ALL_GUJARAT_DISTRICTS = {
    "ahmedabad", "amreli", "anand", "aravalli", "banaskantha", "bharuch",
    "bhavnagar", "botad", "chhotaudaipur", "dahod", "dang", "devbhoomidwarka",
    "gandhinagar", "girsomnath", "jamnagar", "junagadh", "kheda", "kutch",
    "mahisagar", "mehsana", "morbi", "narmada", "navsari", "panchmahal",
    "patan", "porbandar", "rajkot", "sabarkantha", "surat", "surendranagar",
    "tapi", "vadodara", "valsad",
}


class TestCoverage:
    def test_all_thirty_three_gujarat_districts_are_present(self):
        assert set(DISTRICTS) == ALL_GUJARAT_DISTRICTS
        assert len(DISTRICTS) == 33

    def test_the_five_late_added_districts_are_there(self):
        # Narmada, Tapi and Dang are Dudhdhara / Sumul / Vasudhara country; a
        # union->district hop would strand those farmers 55-67 km from any anchor.
        for key in ("narmada", "tapi", "dang", "chhotaudaipur", "devbhoomidwarka"):
            assert key in DISTRICTS, f"{key} missing"

    def test_every_district_has_at_least_one_candidate(self):
        for key, loc in DISTRICTS.items():
            assert loc.candidates, f"{key} has no candidates"
            assert loc.primary is loc.candidates[0]

    def test_the_default_is_anand(self):
        assert DEFAULT_LOCATION.key == "anand"


class TestCoordinatesAreSane:
    @pytest.mark.parametrize("key", sorted(ALL_GUJARAT_DISTRICTS))
    def test_every_coordinate_sits_inside_gujarat(self, key):
        for candidate in DISTRICTS[key].candidates:
            assert MIN_LAT <= candidate.lat <= MAX_LAT, (
                f"{key}/{candidate.town} lat {candidate.lat} outside Gujarat"
            )
            assert MIN_LON <= candidate.lon <= MAX_LON, (
                f"{key}/{candidate.town} lon {candidate.lon} outside Gujarat"
            )

    def test_lat_and_lon_are_not_transposed(self):
        # Gujarat's lat and lon ranges do not overlap, so a swap always lands
        # outside the box — which is what makes the box test able to catch it.
        assert MAX_LAT < MIN_LON

    def test_no_two_districts_share_a_primary_coordinate(self):
        seen: dict[tuple[float, float], str] = {}
        for key, loc in DISTRICTS.items():
            point = (loc.primary.lat, loc.primary.lon)
            assert point not in seen, f"{key} duplicates {seen.get(point)}"
            seen[point] = key

    def test_large_districts_carry_more_than_one_candidate(self):
        # A ~50 km catchment does not cover Kutch (Bhuj->Rapar is 106 km) or
        # Banaskantha (Deesa->Vav is 69 km). One point per district was the bug.
        assert len(DISTRICTS["kutch"].candidates) >= 2
        assert len(DISTRICTS["banaskantha"].candidates) >= 2

    def test_banaskantha_leads_with_deesa_not_palanpur(self):
        # Measured: Deesa + Onion -> 10 rows from Deesa Veg Yard; Palanpur +
        # Onion -> 3 rows from Abu Road APMC, Rajasthan. Banas is Amul's largest
        # union, so the order of these two candidates is not cosmetic.
        assert DISTRICTS["banaskantha"].primary.town == "Deesa"
        assert "Palanpur" in [c.town for c in DISTRICTS["banaskantha"].candidates]


class TestVerificationHonesty:
    def test_only_actually_probed_coordinates_claim_verified(self):
        """`verified` means observed returning live BV rows — nothing else.

        BV has been down since 2026-08-13, so no coordinate added after that can
        legitimately carry the flag. This list is the 2026-08-12 sweep; adding to
        it requires evidence from scripts/validate_district_coords.py, not
        confidence.
        """
        observed = {
            "Anand", "Rajkot", "Junagadh", "Deesa", "Gandhinagar", "Ahmedabad",
            "Mehsana", "Patan", "Dahod", "Porbandar", "Jamnagar", "Morbi",
            "Bhavnagar", "Botad", "Amreli", "Surat", "Navsari", "Valsad", "Bhuj",
            "Godhra", "Modasa", "Himatnagar", "Nadiad", "Vadodara",
            "Surendranagar", "Veraval", "Lunawada",
        }
        claimed = {
            c.town
            for loc in DISTRICTS.values()
            for c in loc.candidates
            if c.verified
        }
        assert claimed == observed

    def test_most_coordinates_are_still_provisional(self):
        total = sum(len(loc.candidates) for loc in DISTRICTS.values())
        verified = sum(c.verified for loc in DISTRICTS.values() for c in loc.candidates)
        assert 0 < verified < total, (
            "either nothing is verified or everything claims to be; both are wrong"
        )


class TestNormalisation:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            # The two backend spellings that silently defeated the lookup.
            ("kachchh", "kutch"),
            ("kutch", "kutch"),
            ("KACHCHH", "kutch"),
            ("banas kantha", "banaskantha"),
            ("Banas-Kantha", "banaskantha"),
            ("banaskantha", "banaskantha"),
            ("the dangs", "dang"),
            ("The Dangs", "dang"),
            ("chhotaudepur", "chhotaudaipur"),
            ("Chhota Udepur", "chhotaudaipur"),
            ("devbhumi dwarka", "devbhoomidwarka"),
            ("Devbhoomi Dwarka", "devbhoomidwarka"),
            # Ordinary case/spacing/suffix noise.
            ("  ANAND  ", "anand"),
            ("Anand District", "anand"),
            ("mahesana", "mehsana"),
            ("baroda", "vadodara"),
            ("kaira", "kheda"),
            ("gir somnath", "girsomnath"),
        ],
    )
    def test_known_spelling_variants_resolve(self, raw, expected):
        resolved = resolve_place(raw)
        assert resolved is not None, f"{raw!r} fell through to the default"
        assert resolved.key == expected

    def test_towns_resolve_to_their_district(self):
        for town, district in [
            ("Palanpur", "banaskantha"),
            ("Gandhidham", "kutch"),
            ("Jetpur", "rajkot"),
            ("Vapi", "valsad"),
            ("Khambhat", "anand"),
            ("Rajpipla", "narmada"),
        ]:
            resolved = resolve_place(town)
            assert resolved is not None and resolved.key == district, town

    def test_normalize_place_handles_blank_input(self):
        assert normalize_place(None) == ""
        assert normalize_place("   ") == ""
        assert resolve_place(None) is None
        assert resolve_place("") is None

    def test_an_uncovered_place_returns_none_rather_than_a_default(self):
        # Returning DEFAULT_LOCATION here would make "prices in Timbuktu" answer
        # with Anand's prices and never say so.
        for name in ("Timbuktu", "Nashik", "Delhi", "qwertyuiop"):
            assert resolve_place(name) is None, name

    def test_a_raw_coordinate_string_does_not_resolve(self):
        # The model must never be able to smuggle coordinates in through the
        # place-name argument; a hallucinated lat/lon fails silently as zero rows.
        assert resolve_place("22.55,72.93") is None
        assert resolve_place("22.55") is None


class TestUnknownPlaceMessage:
    def test_names_the_place_and_offers_alternatives(self):
        message = unknown_place_message("Timbuktu")
        assert "Timbuktu" in message
        assert "do not have market coverage" in message
        # It must offer something, or the farmer has nowhere to go next.
        assert "Anand" in message or "Did you mean" in message

    def test_a_near_miss_suggests_the_right_district(self):
        assert "Junagadh" in unknown_place_message("Junagad")

    def test_it_never_implies_another_location_was_used(self):
        message = unknown_place_message("Timbuktu").lower()
        for lie in ("instead", "showing prices for", "using"):
            assert lie not in message

    def test_covered_district_names_are_human_readable(self):
        names = covered_district_names()
        assert len(names) == 33
        assert "Banaskantha" in names and "Gir Somnath" in names
        assert names == sorted(names)
