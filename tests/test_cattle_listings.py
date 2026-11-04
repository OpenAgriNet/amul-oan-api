"""Verified cattle listings: place/species filters, stable listing numbers, Redis refresh rule."""

import asyncio
import json
import time
from decimal import Decimal
from types import SimpleNamespace

from agents.deps import FarmerContext
from agents.tools import cattle_listings_cache as cache
from agents.tools import cattle_trade
from agents.tools.models.cattle_trade import VerifiedAnimalDetailsModel


def _animal(district, village, cattle_type, breed, mobile, price=50000, pregnancy="Pregnant"):
    return dict(
        districtName=district, villageName=village, farmerName="Seller", farmerMobileNumber=mobile,
        cattleType=cattle_type, breed=breed, lactation=None, age="1.3", milkingStatus=None,
        pregnancyStatus=pregnancy, milkPerDay=0.0, cattlePrice=price,
    )


ANIMALS = [
    _animal("Vav-Tharad", "Dhanera", "Buffalo", "Mehsana", "9000000001", 150000),
    _animal("Banaskantha", "Deesa", "Buffalo", "Other Buffalo", "9000000002", 10000),
    # Identical except the seller: each must get its own number.
    _animal("Banaskantha", "Dhedhal", "Cow", "Kakrej", "9000000003", 5100, "Non Pregnant"),
    _animal("Banaskantha", "Dhedhal", "Cow", "Kakrej", "9000000004", 5100, "Non Pregnant"),
    _animal("KHEDA", "Kathlal", "Cow", "Shankar", "9000000005", 65000, "Non Pregnant"),
]


def _details(animals, from_date="2026-10-05T11:49:39.6101754+05:30"):
    cows = [a for a in animals if a["cattleType"] == "Cow"]
    buffaloes = [a for a in animals if a["cattleType"] == "Buffalo"]
    payload = dict(
        fromDate=from_date, toDate="2026-11-04T11:49:39+05:30",
        totalCows=len(cows), totalBuffaloes=len(buffaloes),
        cows=[dict(breedName=a["breed"], count=1, animals=[a]) for a in cows],
        buffaloes=[dict(breedName=a["breed"], count=1, animals=[a]) for a in buffaloes],
    )
    return VerifiedAnimalDetailsModel.model_validate(json.loads(json.dumps(payload), parse_float=Decimal))


def _ask(monkeypatch, farmer_district=None, **filters):
    snapshot = cache.build_snapshot(_details(ANIMALS), None, time.time())

    async def snap():
        return snapshot

    monkeypatch.setattr(cattle_trade, "get_snapshot", snap)
    ctx = SimpleNamespace(deps=FarmerContext(query="q", farmer_district=farmer_district))
    return asyncio.run(cattle_trade.get_verified_cattle_listings(ctx, **filters)), snapshot


def test_banaskantha_covers_vav_tharad_and_generic_species_means_any(monkeypatch):
    answer, _ = _ask(monkeypatch, district="Banas Kantha", species="cattle")
    assert "Matches (in Banaskantha and Vav-Tharad districts): 4." in answer
    assert "Dhanera, Vav-Tharad" in answer and "Kathlal" not in answer


def test_no_place_defaults_to_the_farmers_district_and_counts_elsewhere(monkeypatch):
    answer, _ = _ask(monkeypatch, farmer_district="kheda", pregnancy_status="not pregnant")
    assert "Searched the farmer's district, Kheda." in answer
    assert "Matches (not pregnant): 1." in answer and "2 more match elsewhere." in answer


def test_listings_hide_the_seller_until_one_is_picked_by_number(monkeypatch):
    answer, snapshot = _ask(monkeypatch, breed="Kankrej")
    assert "900000000" not in answer
    numbers = {listing.farmer_mobile_number: listing.number for listing in snapshot.listings}
    picked, _ = _ask(monkeypatch, listing_number=numbers["9000000004"])
    assert "mobile 9000000004" in picked and "9000000003" not in picked


def test_listing_numbers_survive_a_refresh_and_are_never_reused():
    first = cache.build_snapshot(_details(ANIMALS), None, 1.0)
    # One animal sold, one new animal listed in a place that sorts first.
    second = cache.build_snapshot(
        _details(ANIMALS[1:] + [_animal("Anand", "Dali", "Buffalo", "Mehsana", "9000000006")]), first, 2.0
    )
    before = {listing.farmer_mobile_number: listing.number for listing in first.listings}
    after = {listing.farmer_mobile_number: listing.number for listing in second.listings}
    assert len(set(before.values())) == len(ANIMALS)
    assert all(after[mobile] == before[mobile] for mobile in after if mobile in before)
    assert after["9000000006"] == len(ANIMALS) + 1


class _Redis:
    """The four Redis calls the cache makes, with expiry."""

    def __init__(self):
        self.data, self.now = {}, 0.0

    async def get(self, key):
        value, expires = self.data.get(key, (None, None))
        return value if expires is None or expires > self.now else None

    async def set(self, key, value, ex=None, nx=False):
        if nx and await self.get(key) is not None:
            return None
        self.data[key] = (value, self.now + ex if ex else None)
        return True

    async def exists(self, key):
        return int(await self.get(key) is not None)


def _refresh_world(monkeypatch, responses):
    redis, calls = _Redis(), []

    async def fetch(token):
        calls.append(token)
        return responses.pop(0)

    monkeypatch.setattr(cache, "redis_client", redis)
    monkeypatch.setattr(cache, "get_verified_animal_details_api", fetch)
    monkeypatch.setattr(cache, "get_config_value", lambda key: "token")
    return redis, calls


def test_refresh_rule_fresh_stale_idle_and_failure(monkeypatch):
    sentinel = _details([], from_date="0001-01-01T00:00:00")
    redis, calls = _refresh_world(monkeypatch, [_details(ANIMALS), sentinel, _details(ANIMALS[:1])])
    hour, idle = cache.settings.cattle_listings_refresh_seconds, cache.settings.cattle_listings_idle_seconds

    def at(seconds, call):
        redis.now = seconds
        return asyncio.run(call(now=seconds))

    assert len(at(0, cache.get_snapshot).listings) == 5 and len(calls) == 1  # cold: fetched
    assert at(600, cache.get_snapshot) is not None and len(calls) == 1  # fresh: served from Redis
    assert at(1200, cache.refresh_if_in_use) is False and len(calls) == 1  # in use but fresh
    # Old copy + provider sentinel: the last good copy is kept and served.
    assert len(at(hour + 60, cache.get_snapshot).listings) == 5 and len(calls) == 2
    # The lock left behind by the failed fetch holds other pods off for its TTL.
    assert at(hour + 90, cache.refresh_if_in_use) is False and len(calls) == 2
    # Background tick while the tool is in use refreshes the old copy.
    assert at(hour + 200, cache.refresh_if_in_use) is True and len(calls) == 3
    # No request for the idle window: the refresher stops calling the provider.
    assert at(hour + 60 + idle + 1, cache.refresh_if_in_use) is False and len(calls) == 3
