"""Per-farmer location for the Bharat Vistaar mandi/weather tools.

⚠️ **About the mock.** The reason this integration shipped broken twice is a mock
that was more permissive than the real BPP: it answered whatever it was asked,
so tests passed while the live service returned zero rows. `_FakeBpp` below
therefore models the measured behaviour, not a convenient one:

  * **GPS selects the market; `descriptor.name` is ignored entirely.** Measured:
    name=Junagadh + Anand's gps → 0 rows; name=Anand + Junagadh's gps → 10 rows
    from Junagadh APMC. The mock reads only the gps, so a change that sends the
    right name with the wrong point fails here exactly as it would live.
  * **It is a radius search of ~50 km returning 1–5 markets**, not
    nearest-market and not district containment. Results crossing district and
    state lines are normal.
  * **Market selection is per-commodity** — the BPP returns nearby markets that
    trade the queried commodity, which is why Junagadh + Cotton answers from
    Jetpur APMC in Rajkot district.
  * **A hard cap of 10 rows per response**, newest arrival first.

What is unit-tested here is *deterministic*: argument shaping, resolution order,
stickiness, fallback and rendering. Whether the model chooses to pass
`location="Junagadh"` when a farmer says "જુનાગઢમાં" is live-model behaviour and
is deliberately NOT asserted — the tests below always pass the argument
explicitly, and the routing signal for the model lives in the tool docstring and
the system prompt.
"""
import importlib.util
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "vistaar_loc", ROOT / "agents" / "tools" / "vistaar.py"
)
vistaar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vistaar)

from agents.deps import FarmerContext  # noqa: E402
from agents.tools import session_location  # noqa: E402
from agents.tools.districts import DISTRICTS  # noqa: E402

_IST = timezone(timedelta(hours=5, minutes=30))
RADIUS_KM = 50.0
ROW_CAP = 10


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


class _Market:
    def __init__(self, name, district, state, lat, lon, commodities):
        self.name, self.district, self.state = name, district, state
        self.lat, self.lon = lat, lon
        self.commodities = {c.casefold() for c in commodities}


# A deliberately sparse market map. Sparse is the point: the real catalogue has
# holes (south Gujarat trades no cereals at any coordinate), and a fallback that
# only works against a dense mock is not a fallback.
MARKETS = [
    _Market("Anand APMC", "Anand", "Gujarat", 22.556, 72.955, ["Onion", "Wheat"]),
    _Market("Junagadh APMC", "Junagadh", "Gujarat", 21.522, 70.458, ["Wheat", "Groundnut"]),
    _Market("Jetpur APMC", "Rajkot", "Gujarat", 21.754, 70.619, ["Cotton", "Onion"]),
    _Market("Deesa Veg Yard", "Banaskantha", "Gujarat", 24.260, 72.180, ["Onion"]),
    # Cross-STATE, and legitimately so: Palanpur -> Abu Road is 48.8 km.
    _Market("Abu Road APMC", "Sirohi", "Rajasthan", 24.480, 72.780, ["Onion"]),
    # 106 km from Bhuj — outside Kutch's HQ catchment, inside Bhachau's.
    _Market("Rapar APMC", "Kutch", "Gujarat", 23.571, 70.645, ["Wheat"]),
]


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeBpp:
    """Radius-search, per-commodity, 10-row-capped, GPS-only market selection."""

    def __init__(self):
        self.searches = []  # (lat, lon, commodity, descriptor_name)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None):
        intent = json["intent"]
        category = (intent.get("category") or {}).get("descriptor", {}).get("code")
        if category != "price-discovery":
            # Weather: one forecast block per point, always available.
            location = intent["fulfillment"]["stops"][0]["location"]
            self.searches.append((location["lat"], location["lon"], None, None))
            return _FakeResponse(self._wrap([
                {"descriptor": {"name": "Forecast", "long_desc": "Rain 0 mm"}}
            ]))

        location = intent["fulfillment"]["end"]["location"]
        lat, lon = (float(v) for v in location["gps"].split(","))
        commodity = intent["item"]["descriptor"]["name"]
        # NOTE: descriptor.name is recorded but NEVER used to choose a market.
        self.searches.append((lat, lon, commodity, (location.get("descriptor") or {}).get("name")))

        tags = {t["code"]: t["value"] for t in intent.get("tags", [])}
        to_date = datetime.strptime(tags["to_date"], "%d-%m-%Y").date()
        from_date = datetime.strptime(tags["from_date"], "%d-%m-%Y").date()
        span = (to_date - from_date).days

        hits = [
            m for m in MARKETS
            if commodity.casefold() in m.commodities
            and _haversine_km(lat, lon, m.lat, m.lon) <= RADIUS_KM
        ]
        rows = []
        for offset in range(span + 1):
            arrival = to_date - timedelta(days=offset)
            for market in hits:
                rows.append(self._item(commodity, market, arrival.strftime("%d-%m-%Y")))
        return _FakeResponse(self._wrap(rows[:ROW_CAP]))  # hard 10-row cap

    @staticmethod
    def _item(commodity, market, arrival):
        def t(code, value):
            return {"descriptor": {"code": code}, "value": value}
        return {
            "descriptor": {"name": commodity},
            "tags": [{"descriptor": {"code": "attributes"}, "list": [
                t("Arrival Date", arrival), t("Market", market.name),
                t("District", market.district), t("State", market.state),
                t("Modal Price", "2000"), t("Min Price", "1800"),
                t("Max Price", "2200"), t("Price Unit", "Rs./Qtl"),
            ]}],
        }

    @staticmethod
    def _wrap(items):
        leg = vistaar.VISTAAR_LEG
        return {"results": {leg: {"message": {"catalog": {
            "providers": [{"id": leg, "items": items}]}}}}}


class _FakeCache:
    """Stand-in for the shared Redis, so stickiness runs through the real code."""

    def __init__(self):
        self.store = {}
        self.ttls = {}

    async def get(self, key, namespace=None):
        return self.store.get((namespace, key))

    async def set(self, key, value, ttl=None, namespace=None):
        self.store[(namespace, key)] = value
        self.ttls[(namespace, key)] = ttl


@pytest.fixture
def bpp():
    fake = _FakeBpp()
    with patch.object(vistaar.httpx, "AsyncClient", return_value=fake):
        yield fake


@pytest.fixture
def fake_cache():
    cache = _FakeCache()
    with patch.object(session_location, "cache", cache):
        yield cache


def ctx(district=None, session_id="s-1", village=None, state=None):
    return SimpleNamespace(deps=FarmerContext(
        query="what is the price of onion",
        session_id=session_id,
        farmer_district=district,
        farmer_village=village,
        farmer_state=state,
    ))


# ── Case 1: explicit override ────────────────────────────────────────────────


class TestExplicitLocation:
    @pytest.mark.asyncio
    async def test_explicit_location_beats_the_profile_district(self, bpp, fake_cache):
        out = await vistaar.get_vistaar_mandi_prices(ctx(district="anand"), "Wheat", "Junagadh")
        lat, lon, commodity, _ = bpp.searches[0]
        assert (lat, lon) == (DISTRICTS["junagadh"].primary.lat,
                              DISTRICTS["junagadh"].primary.lon)
        assert "Junagadh APMC" in out
        assert "Anand APMC" not in out

    @pytest.mark.asyncio
    async def test_the_gps_moves_not_just_the_name(self, bpp, fake_cache):
        # The whole point: the BPP ignores descriptor.name, so a change that
        # only relabelled the intent would answer from Anand while claiming
        # Junagadh — invisible in the prose, visible here.
        await vistaar.get_vistaar_mandi_prices(ctx(district="anand"), "Wheat", "Junagadh")
        lat, lon, _, name = bpp.searches[0]
        assert name == "Junagadh"
        assert round(lat, 3) == 21.522 and round(lon, 3) == 70.458

    @pytest.mark.asyncio
    async def test_a_town_name_resolves_to_its_district(self, bpp, fake_cache):
        out = await vistaar.get_vistaar_mandi_prices(ctx(district="anand"), "Onion", "Palanpur")
        assert bpp.searches, "no search was issued"
        assert "Deesa Veg Yard" in out or "Abu Road" in out

    @pytest.mark.asyncio
    async def test_a_cross_state_result_is_reported_with_its_state(self, bpp, fake_cache):
        # Palanpur legitimately returns a Rajasthan market. Reporting it as the
        # farmer's local mandi is the failure; reporting it plainly is correct.
        out = await vistaar.get_vistaar_mandi_prices(
            ctx(district="banaskantha"), "Onion", "Banaskantha"
        )
        assert "Deesa Veg Yard, Banaskantha, Gujarat" in out


# ── Case 2: profile district missing ─────────────────────────────────────────


class TestProfileDistrict:
    @pytest.mark.asyncio
    async def test_the_farmers_own_district_is_used_by_default(self, bpp, fake_cache):
        await vistaar.get_vistaar_mandi_prices(ctx(district="junagadh"), "Wheat")
        lat, lon, _, _ = bpp.searches[0]
        assert round(lat, 3) == 21.522, "fell back to Anand instead of the farmer's district"

    @pytest.mark.asyncio
    async def test_a_backend_spelling_variant_still_resolves(self, bpp, fake_cache):
        # 'kachchh' is what one of the two farmer backends actually returns.
        await vistaar.get_vistaar_mandi_prices(ctx(district="kachchh"), "Wheat")
        lat, lon, _, _ = bpp.searches[0]
        assert round(lat, 3) == DISTRICTS["kutch"].primary.lat

    @pytest.mark.asyncio
    async def test_missing_district_answers_anyway_and_says_whose_prices(self, bpp, fake_cache):
        out = await vistaar.get_vistaar_mandi_prices(ctx(district=None), "Onion")
        # Answer-then-invite: the prices are there…
        assert "Anand APMC" in out
        # …AND the farmer is told they are Anand's, with a way to correct it.
        assert "Anand-area prices" in out
        assert "do not have your district on file" in out
        assert "Tell me your district" in out

    @pytest.mark.asyncio
    async def test_the_invite_appears_only_when_the_location_was_assumed(self, bpp, fake_cache):
        out = await vistaar.get_vistaar_mandi_prices(ctx(district="anand"), "Onion")
        assert "Anand APMC" in out
        assert "do not have your district on file" not in out

    @pytest.mark.asyncio
    async def test_an_unmapped_district_string_degrades_to_the_default(self, bpp, fake_cache):
        # A district we cannot map must behave like no district: answer, and say
        # so. Silently pretending it is theirs is the thing we are removing.
        out = await vistaar.get_vistaar_mandi_prices(ctx(district="nowhere-land"), "Onion")
        assert "Anand APMC" in out
        assert "do not have your district on file" in out

    @pytest.mark.asyncio
    async def test_it_does_not_block_with_a_clarifying_question(self, bpp, fake_cache):
        # A stalled tool call has previously run past the 60s nginx window and
        # dropped a voice call. Answer first; invite second.
        out = await vistaar.get_vistaar_mandi_prices(ctx(district=None), "Onion")
        assert bpp.searches, "it asked instead of answering"
        assert "modal 2000" in out


# ── Case 3: unresolvable place ───────────────────────────────────────────────


class TestUnresolvablePlace:
    @pytest.mark.asyncio
    async def test_an_uncovered_place_is_refused_not_substituted(self, bpp, fake_cache):
        out = await vistaar.get_vistaar_mandi_prices(ctx(district="junagadh"), "Wheat", "Timbuktu")
        assert "do not have market coverage" in out
        assert "Timbuktu" in out
        # Critically: it did NOT quietly answer from the profile district.
        assert "Junagadh APMC" not in out
        assert bpp.searches == [], "an unresolvable place must not reach the BPP"

    @pytest.mark.asyncio
    async def test_the_refusal_names_something_we_do_cover(self, bpp, fake_cache):
        out = await vistaar.get_vistaar_mandi_prices(ctx(), "Wheat", "Junagad")
        assert "Junagadh" in out

    @pytest.mark.asyncio
    async def test_raw_coordinates_are_refused(self, bpp, fake_cache):
        # A hallucinated lat/lon returns zero rows and looks like a real answer,
        # so coordinates must not be expressible at all.
        out = await vistaar.get_vistaar_mandi_prices(ctx(), "Wheat", "21.5, 70.4")
        assert "do not have market coverage" in out
        assert bpp.searches == []

    @pytest.mark.asyncio
    async def test_a_refusal_is_not_remembered_as_the_session_location(self, bpp, fake_cache):
        await vistaar.get_vistaar_mandi_prices(ctx(district="junagadh"), "Wheat", "Timbuktu")
        assert fake_cache.store == {}


# ── Stickiness ───────────────────────────────────────────────────────────────


class TestSessionStickiness:
    @pytest.mark.asyncio
    async def test_a_stated_location_is_reused_next_turn(self, bpp, fake_cache):
        await vistaar.get_vistaar_mandi_prices(ctx(district="anand"), "Wheat", "Junagadh")
        await vistaar.get_vistaar_mandi_prices(ctx(district="anand"), "Groundnut")
        second_lat = bpp.searches[1][0]
        assert round(second_lat, 3) == 21.522, (
            "the follow-up question fell back to the profile district"
        )

    @pytest.mark.asyncio
    async def test_a_new_explicit_location_replaces_the_sticky_one(self, bpp, fake_cache):
        await vistaar.get_vistaar_mandi_prices(ctx(district="anand"), "Wheat", "Junagadh")
        await vistaar.get_vistaar_mandi_prices(ctx(district="anand"), "Onion", "Banaskantha")
        await vistaar.get_vistaar_mandi_prices(ctx(district="anand"), "Onion")
        assert round(bpp.searches[2][0], 3) == DISTRICTS["banaskantha"].primary.lat

    @pytest.mark.asyncio
    async def test_stickiness_does_not_leak_across_sessions(self, bpp, fake_cache):
        await vistaar.get_vistaar_mandi_prices(ctx(district="anand", session_id="a"), "Wheat", "Junagadh")
        await vistaar.get_vistaar_mandi_prices(ctx(district="anand", session_id="b"), "Wheat")
        assert round(bpp.searches[1][0], 3) == DISTRICTS["anand"].primary.lat

    @pytest.mark.asyncio
    async def test_weather_shares_the_sticky_location_with_mandi(self, bpp, fake_cache):
        await vistaar.get_vistaar_mandi_prices(ctx(district="anand"), "Wheat", "Junagadh")
        out = await vistaar.get_vistaar_weather(ctx(district="anand"))
        assert "Junagadh" in out

    @pytest.mark.asyncio
    async def test_it_stores_a_district_key_with_a_ttl_not_coordinates(self, bpp, fake_cache):
        await vistaar.get_vistaar_mandi_prices(ctx(district="anand"), "Wheat", "Junagadh")
        (namespace, key), value = next(iter(fake_cache.store.items()))
        assert namespace == session_location.MANDI_LOCATION_NAMESPACE
        assert key == "s-1"
        # A district key, never a coordinate: the table stays the single source
        # of truth, so a corrected coordinate takes effect immediately.
        assert value == "junagadh"
        assert fake_cache.ttls[(namespace, key)] == session_location.MANDI_LOCATION_TTL_S
        assert 0 < session_location.MANDI_LOCATION_TTL_S <= 24 * 3600

    @pytest.mark.asyncio
    async def test_an_anonymous_session_still_works(self, bpp, fake_cache):
        out = await vistaar.get_vistaar_mandi_prices(
            ctx(district="anand", session_id=None), "Wheat", "Junagadh"
        )
        assert "Junagadh APMC" in out
        assert fake_cache.store == {}

    @pytest.mark.asyncio
    async def test_a_dead_cache_does_not_break_the_lookup(self, bpp):
        class _Broken:
            async def get(self, *a, **kw):
                raise RuntimeError("redis down")

            async def set(self, *a, **kw):
                raise RuntimeError("redis down")

        with patch.object(session_location, "cache", _Broken()):
            out = await vistaar.get_vistaar_mandi_prices(ctx(district="anand"), "Wheat", "Junagadh")
        assert "Junagadh APMC" in out


# ── Candidate fallback ───────────────────────────────────────────────────────


class TestCandidateFallback:
    @pytest.mark.asyncio
    async def test_zero_rows_at_the_hq_falls_through_to_the_next_candidate(self, bpp, fake_cache):
        # Kutch: Rapar APMC is 106 km from Bhuj — outside the ~50 km catchment —
        # but 45 km from Bhachau. With one point per district, every Sarhad
        # farmer asking about wheat got "no prices found".
        out = await vistaar.get_vistaar_mandi_prices(ctx(district="kutch"), "Wheat")
        assert "Rapar APMC" in out
        assert len(bpp.searches) >= 2
        assert round(bpp.searches[0][0], 3) == DISTRICTS["kutch"].candidates[0].lat
        assert round(bpp.searches[1][0], 3) == DISTRICTS["kutch"].candidates[1].lat

    @pytest.mark.asyncio
    async def test_the_happy_path_makes_exactly_one_call(self, bpp, fake_cache):
        # No parallel fan-out: the upstream is a single non-redundant sandbox,
        # and pre-firing every candidate would double its load on every success.
        await vistaar.get_vistaar_mandi_prices(ctx(district="banaskantha"), "Onion")
        assert len(bpp.searches) == 1

    @pytest.mark.asyncio
    async def test_the_walk_is_capped(self, bpp, fake_cache):
        # Nothing trades Cotton near Banaskantha; it must give up, not grind
        # through the list at ~2.2 s each.
        out = await vistaar.get_vistaar_mandi_prices(ctx(district="banaskantha"), "Cotton")
        assert "No mandi prices were found" in out
        assert len(bpp.searches) <= vistaar.MANDI_MAX_CANDIDATES

    @pytest.mark.asyncio
    async def test_a_dead_leg_does_not_burn_the_candidate_list(self, fake_cache):
        # Infrastructure failure is not zero rows. Walking coordinates during an
        # outage costs 3 x 2.2 s to reach the same "temporarily unavailable".
        calls = []

        class _DeadLeg(_FakeBpp):
            async def post(self, url, json=None):
                calls.append(json)
                return _FakeResponse(
                    {"results": {vistaar.VISTAAR_LEG: None},
                     "errors": {vistaar.VISTAAR_LEG: "timeout"}}
                )

        with patch.object(vistaar.httpx, "AsyncClient", return_value=_DeadLeg()):
            out = await vistaar.get_vistaar_mandi_prices(ctx(district="kutch"), "Wheat")
        assert "temporarily unavailable" in out.lower()
        assert "No mandi prices were found" not in out
        assert len(calls) == 1, "an outage must not be retried across coordinates"


# ── Provenance in the rendered output ────────────────────────────────────────


class TestProvenance:
    @pytest.mark.asyncio
    async def test_every_row_names_market_district_and_state(self, bpp, fake_cache):
        out = await vistaar.get_vistaar_mandi_prices(ctx(district="rajkot"), "Cotton")
        rows = [line for line in out.splitlines() if line.startswith("- ")]
        assert rows
        for row in rows:
            assert "Jetpur APMC, Rajkot, Gujarat" in row

    @pytest.mark.asyncio
    async def test_the_bpps_own_item_name_is_shown(self, bpp, fake_cache):
        # Commodity matching is fuzzy: asking for "Onion" can return the real,
        # different commodity "Onion Green". Omitting the name made that
        # invisible, so a farmer could be quoted the wrong crop's price.
        class _Fuzzy(_FakeBpp):
            async def post(self, url, json=None):
                json["intent"]["item"]["descriptor"]["name"] = "Onion"
                response = await super().post(url, json)
                for item in response.json()["results"][vistaar.VISTAAR_LEG][
                    "message"]["catalog"]["providers"][0]["items"]:
                    item["descriptor"]["name"] = "Onion Green"
                return response

        with patch.object(vistaar.httpx, "AsyncClient", return_value=_Fuzzy()):
            out = await vistaar.get_vistaar_mandi_prices(ctx(district="anand"), "Onion")
        assert "Onion Green" in out, "a fuzzy BPP match must not be invisible"

    @pytest.mark.asyncio
    async def test_the_header_names_the_town_and_district_searched(self, bpp, fake_cache):
        out = await vistaar.get_vistaar_mandi_prices(ctx(district="kutch"), "Wheat")
        assert "near Bhachau, Kutch" in out, (
            "the header must name the candidate actually used, not the HQ we tried first"
        )

    @pytest.mark.asyncio
    async def test_an_empty_result_names_the_place_searched(self, bpp, fake_cache):
        # Nothing near Kutch trades onions, at any of its three candidates. A
        # zero-row answer is legitimate; an unattributed one is not.
        out = await vistaar.get_vistaar_mandi_prices(ctx(district="kutch"), "Onion")
        assert "No mandi prices were found" in out
        assert "Kutch" in out


# ── Weather ──────────────────────────────────────────────────────────────────


class TestWeather:
    @pytest.mark.asyncio
    async def test_weather_uses_the_farmers_district(self, bpp, fake_cache):
        await vistaar.get_vistaar_weather(ctx(district="kutch"))
        lat, lon, _, _ = bpp.searches[0]
        assert (lat, lon) == (DISTRICTS["kutch"].primary.lat, DISTRICTS["kutch"].primary.lon)

    @pytest.mark.asyncio
    async def test_weather_takes_an_explicit_location(self, bpp, fake_cache):
        out = await vistaar.get_vistaar_weather(ctx(district="anand"), "Rajkot")
        lat, _, _, _ = bpp.searches[0]
        assert lat == DISTRICTS["rajkot"].primary.lat
        assert "Rajkot" in out

    @pytest.mark.asyncio
    async def test_weather_refuses_an_uncovered_place(self, bpp, fake_cache):
        out = await vistaar.get_vistaar_weather(ctx(district="anand"), "Timbuktu")
        assert "do not have market coverage" in out
        assert bpp.searches == []

    @pytest.mark.asyncio
    async def test_weather_says_whose_area_it_assumed(self, bpp, fake_cache):
        out = await vistaar.get_vistaar_weather(ctx(district=None))
        assert "Anand" in out
        assert "do not have your district on file" in out


# ── Tool registration ────────────────────────────────────────────────────────


class TestToolRegistration:
    @pytest.mark.asyncio
    async def test_both_tools_build_with_required_parameter_descriptions(self):
        # require_parameter_descriptions=True: ONE undocumented parameter makes
        # the agent lose ALL tools at import, so this is the guard on the new
        # `location` argument's Args: entry.
        from pydantic_ai import Tool

        from agents.tools.vistaar import get_vistaar_mandi_prices, get_vistaar_weather

        for fn, expected in (
            (get_vistaar_mandi_prices,
             {"commodity_name", "location", "price_date", "price_date_to"}),
            (get_vistaar_weather, {"location"}),
        ):
            tool = Tool(fn, takes_ctx=True, docstring_format="auto",
                        require_parameter_descriptions=True)
            tool_def = await tool.prepare_tool_def(None)
            properties = tool_def.parameters_json_schema["properties"]
            assert set(properties) == expected, fn.__name__
            for name, prop in properties.items():
                assert prop.get("description"), f"{fn.__name__}.{name} undocumented"

    @pytest.mark.asyncio
    async def test_no_coordinate_parameter_is_exposed_to_the_model(self):
        from pydantic_ai import Tool

        from agents.tools.vistaar import get_vistaar_mandi_prices

        tool = Tool(get_vistaar_mandi_prices, takes_ctx=True, docstring_format="auto",
                    require_parameter_descriptions=True)
        tool_def = await tool.prepare_tool_def(None)
        for name in tool_def.parameters_json_schema["properties"]:
            assert name not in {"lat", "lon", "latitude", "longitude", "gps"}

    def test_the_chat_registry_passes_context_to_both_tools(self):
        # takes_ctx=False here would make every farmer Anand again, silently.
        import os
        from unittest.mock import patch as _patch

        from app.config import settings

        with _patch.object(settings, "enable_network", True):
            import importlib

            import agents.tools as tools_pkg
            importlib.reload(tools_pkg)
            by_name = {t.name: t for t in tools_pkg.TOOLS}
            try:
                assert by_name["get_vistaar_mandi_prices"].takes_ctx is True
                assert by_name["get_vistaar_weather"].takes_ctx is True
                assert by_name["get_vistaar_scheme_info"].takes_ctx is False
            finally:
                importlib.reload(tools_pkg)
