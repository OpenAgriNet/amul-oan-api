"""find_animals (agents/tools/herd.py): the farmer's own animals for chat and voice.

AMUL-68: voice read a day-old animal list with no names, said "I do not see an
animal named ... in your records", and nothing told it the list was old. The tool
says how much of the herd it has and how old it is, refreshes old records itself
(publishing each animal as it lands), and never lets missing data read as "no
such animal".

Self-contained: asyncio.run, an in-memory Redis and stubbed Beckn calls.
"""
import asyncio
import copy
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import agents.tools.farmer_cache as fc
import agents.tools.herd as herd
from agents.deps import FarmerContext
from agents.tools.farmer import FarmerFetchOutcome
from agents.tools.models.animal import AnimalModel
from agents.tools.models.farmer_transport import AnimalRecord, FarmerDataEnvelope, FarmerRecord
from app.config import settings
from app.core.cache import build_cache_key
from app.voice.farmer import _append_animal_records

PHONE = "9876543210"
TAGS = ["102030401111", "102030402222", "102030403333"]


class _Redis:
    def __init__(self):
        self.kv, self.sets, self.hashes, self.ttls = {}, {}, {}, {}

    async def set(self, key, value, ex=None, nx=False):
        if nx and key in self.kv:
            return None
        self.kv[key] = value
        return True

    async def get(self, key):
        return self.kv.get(key)

    async def exists(self, key):
        return int(key in self.kv or key in self.hashes)

    async def delete(self, key):
        self.kv.pop(key, None)
        self.hashes.pop(key, None)

    async def ttl(self, key):
        return self.ttls.get(key, 3600)

    async def expire(self, key, seconds):
        return True

    async def hset(self, key, field, value):
        self.hashes.setdefault(key, {})[field] = value

    async def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    async def sadd(self, key, member):
        self.sets.setdefault(key, set()).add(member)

    async def spop(self, key, count):
        members = self.sets.get(key, set())
        return [members.pop() for _ in range(min(count, len(members)))]


class _Cache:
    def __init__(self, redis):
        self.redis = redis
        self.ttls = []

    async def get(self, key, namespace=None):
        return copy.deepcopy(self.redis.kv.get(build_cache_key(key, namespace)))

    async def set(self, key, value, ttl=None, namespace=None):
        self.ttls.append(ttl)
        self.redis.kv[build_cache_key(key, namespace)] = copy.deepcopy(value)


class _Beckn:
    """Upstream now: types, names and stages per tag; each profile call can wait on a gate."""

    def __init__(self):
        self.names = {TAGS[0]: "Lakshmi"}
        self.types = {TAGS[0]: "Cow", TAGS[1]: "Buffalo", TAGS[2]: "Buffalo"}
        self.pregnancy = {TAGS[0]: "non pregnant", TAGS[1]: "pregnant", TAGS[2]: "non pregnant"}
        self.calls = []
        self.gates = {}

    async def farmer(self, phone):
        record = FarmerRecord.model_validate({
            "farmerName": "Ramesh", "farmerCode": "F1", "societyName": "Anand",
            "societyCode": "S1", "unionCode": "U1", "tagNo": ",".join(TAGS),
        })
        return [record], FarmerFetchOutcome.FOUND

    async def technicians(self, *, union_code, society_code, force_refresh=False):
        return []

    async def animal(self, tag, *, union_code=None, **kwargs):
        self.calls.append((tag, kwargs.get("force_refresh", False)))
        if tag in self.gates:
            await self.gates[tag].wait()
        return AnimalModel.model_validate({
            "tagNumber": tag,
            "animalType": self.types[tag],
            "animalName": self.names.get(tag),
            "pregnancyStage": self.pregnancy[tag],
        })


@pytest.fixture
def herd_env(monkeypatch):
    redis = _Redis()
    cache = _Cache(redis)
    beckn = _Beckn()
    monkeypatch.setattr(fc, "redis_client", redis)
    monkeypatch.setattr(fc, "cache", cache)
    monkeypatch.setattr(fc, "fetch_farmer_info_with_outcome", beckn.farmer)
    monkeypatch.setattr(fc, "search_ai_technicians", beckn.technicians)
    monkeypatch.setattr(fc, "fetch_animal_profile", beckn.animal)
    monkeypatch.setattr(settings, "voice_route_enabled", True)
    monkeypatch.setattr(herd, "WAIT_SECONDS", 0.5)
    monkeypatch.setattr(herd, "POLL_SECONDS", 0.01)
    return redis, cache, beckn


def _store(cache_redis, *, animals_age_hours, names=False):
    """A found record cached with every animal, as fetched ``animals_age_hours`` ago."""
    record = {
        "farmerName": "Ramesh", "farmerCode": "F1", "societyName": "Anand",
        "societyCode": "S1", "unionCode": "U1", "tagNo": ",".join(TAGS),
        "animals": [
            {"tagNumber": tag, "animalType": t.lower(), "animalName": ("lakshmi" if names and tag == TAGS[0] else None)}
            for tag, t in (("102030401111", "Cow"), ("102030402222", "Buffalo"), ("102030403333", "Buffalo"))
        ],
    }
    now = datetime.now(timezone.utc)
    envelope = FarmerDataEnvelope(
        farmers=[FarmerRecord.model_validate(record)],
        aiTechnicians=[],
        fetchedAt=(now - timedelta(hours=1)).isoformat(),
        source="api",
        lookupStatus="found",
        animalsFetchedAt=(now - timedelta(hours=animals_age_hours)).isoformat(),
    )
    cache_redis.kv[build_cache_key(fc._cache_key(PHONE), fc.FARMER_CACHE_NAMESPACE)] = envelope.model_dump()


def _ctx():
    return SimpleNamespace(deps=FarmerContext(query="q", mobile=PHONE, farmer_profile_status="found"))


def _cached_names(redis):
    raw = redis.kv[build_cache_key(fc._cache_key(PHONE), fc.FARMER_CACHE_NAMESPACE)]
    return [a.get("animalName") for f in raw["farmers"] for a in f["animals"]]


def test_old_records_are_refreshed_and_the_named_animal_is_found(herd_env):
    """The AMUL-68 state: animals a day old, no names. The tool refreshes them now,
    finds the name added upstream since, and stores it, keeping the record's TTL."""
    redis, cache, beckn = herd_env
    _store(redis, animals_age_hours=29)
    redis.ttls[build_cache_key(fc._cache_key(PHONE), fc.FARMER_CACHE_NAMESPACE)] = 500

    async def run():
        out = json.loads(await herd.find_animals(_ctx(), name="lakshmi"))
        await asyncio.gather(*fc._animal_refreshes)
        return out

    out = asyncio.run(run())

    assert [m["tag"] for m in out["matches"]] == [TAGS[0]]
    assert out["matches"][0]["name"] == "lakshmi"
    assert out["herd"]["names_set"] == 1
    assert _cached_names(redis) == ["lakshmi", None, None]
    assert cache.ttls == [500]
    assert sorted(tag for tag, _ in beckn.calls) == sorted(TAGS)


def test_a_running_refresh_reports_progress_and_never_says_not_registered(herd_env):
    """Mid-refresh the tool answers with what has landed, and its note tells the
    agent not to conclude the animal is missing."""
    redis, _, beckn = herd_env
    _store(redis, animals_age_hours=29)

    async def run():
        beckn.gates = {TAGS[0]: asyncio.Event(), TAGS[2]: asyncio.Event()}
        out = json.loads(await herd.find_animals(_ctx(), name="gauri"))
        for event in beckn.gates.values():
            event.set()
        await asyncio.gather(*fc._animal_refreshes)
        return out

    out = asyncio.run(run())

    assert out["herd"]["refresh"].startswith("running: 1 of 3 tags fetched")
    assert out["matches"] == []
    assert "Do not say the animal is not registered" in out["note"]
    assert out["herd"]["records_current"] is False


def test_a_refresh_already_running_is_joined_not_repeated(herd_env):
    """The worker holds the farmer's lock: the tool reads its progress and starts nothing."""
    redis, _, beckn = herd_env
    _store(redis, animals_age_hours=29)

    async def run():
        await redis.set(fc._refresh_lock_key(PHONE), "1", nx=True)
        await fc._start_animal_progress(PHONE, 3)
        record = AnimalRecord(tagNumber=TAGS[1], animalType="buffalo", pregnancyStage="pregnant")
        await fc._record_animal_progress(PHONE, TAGS[1], record.model_dump_json())
        out = json.loads(await herd.find_animals(_ctx(), pregnancy_stage="pregnant"))
        return out, await fc.start_animal_refresh(PHONE, force=True)

    out, second_start = asyncio.run(run())

    assert second_start is False
    assert beckn.calls == []
    assert [m["tag"] for m in out["matches"]] == [TAGS[1]]
    assert out["herd"]["refresh"].startswith("running: 1 of 3 tags fetched")


def test_current_records_without_names_say_a_name_cannot_identify_the_animal(herd_env):
    redis, _, beckn = herd_env
    _store(redis, animals_age_hours=1)

    out = json.loads(asyncio.run(herd.find_animals(_ctx(), name="lakshmi")))

    assert beckn.calls == []
    assert out["herd"]["records_current"] is True
    assert out["herd"]["names_set"] == 0
    assert out["matches"] == []
    assert "None of this farmer's animals has a name" in out["note"]
    assert "does not mean the animal is not registered" in out["note"]


def test_refresh_on_request_skips_the_profile_cache_but_not_within_a_minute(herd_env):
    redis, _, beckn = herd_env

    async def call(age_hours):
        _store(redis, animals_age_hours=age_hours)
        beckn.calls.clear()
        await herd.find_animals(_ctx(), name="lakshmi", refresh=True)
        await asyncio.gather(*fc._animal_refreshes)
        return list(beckn.calls)

    assert {force for _, force in asyncio.run(call(1))} == {True}
    assert asyncio.run(call(10 / 3600)) == []


def test_filters_keep_their_sense_and_plurals(herd_env):
    redis, _, _ = herd_env
    _store(redis, animals_age_hours=1)
    for farmer_animals in redis.kv.values():
        if isinstance(farmer_animals, dict) and "farmers" in farmer_animals:
            for animal, stage in zip(farmer_animals["farmers"][0]["animals"], ("non pregnant", "pregnant", "non pregnant")):
                animal["pregnancyStage"] = stage

    pregnant = json.loads(asyncio.run(herd.find_animals(_ctx(), pregnancy_stage="pregnant")))
    buffaloes = json.loads(asyncio.run(herd.find_animals(_ctx(), animal_type="buffaloes")))

    assert [m["tag"] for m in pregnant["matches"]] == [TAGS[1]]
    assert [m["tag"] for m in buffaloes["matches"]] == [TAGS[1], TAGS[2]]


def test_voice_gets_spoken_tags(herd_env):
    redis, _, _ = herd_env
    _store(redis, animals_age_hours=1, names=True)

    out = json.loads(asyncio.run(herd.find_animals_spoken(_ctx(), tag_last_digits="1111")))

    assert out["matches"][0]["tag"] == "one one one one"
    assert out["named_animals"] == [{"tag": "one one one one", "name": "lakshmi", "type": "cow"}]

    other = json.loads(asyncio.run(herd.find_animals_spoken(_ctx(), name="gauri")))
    assert other["matches"] == []
    assert "named_animals" in other["note"]


def test_the_worker_refresh_publishes_progress_and_clears_it(herd_env):
    redis, _, beckn = herd_env
    seen = []

    async def run():
        beckn.gates = {TAGS[2]: asyncio.Event()}
        refresh = asyncio.create_task(fc.refresh_farmer_data(PHONE, background=True))
        for _ in range(100):  # until the two ungated tags have landed, at most a second
            progress = await fc.read_animal_refresh(PHONE)
            if progress is not None and len(progress.fetched) == 2:
                break
            await asyncio.sleep(0.01)
        seen.append(await fc.read_animal_refresh(PHONE))
        beckn.gates[TAGS[2]].set()
        await refresh
        return await fc.read_animal_refresh(PHONE)

    after = asyncio.run(run())

    assert seen[0] is not None and seen[0].total == 3 and sorted(seen[0].fetched) == TAGS[:2]
    assert after is None
    assert redis.hashes == {}


def test_voice_context_says_how_much_of_the_herd_it_has_and_how_old(herd_env):
    redis, _, _ = herd_env
    _store(redis, animals_age_hours=29)
    envelope = FarmerDataEnvelope.model_validate(
        redis.kv[build_cache_key(fc._cache_key(PHONE), fc.FARMER_CACHE_NAMESPACE)]
    )
    envelope.farmers[0].animals = envelope.farmers[0].animals[:2]

    lines: list[str] = []
    _append_animal_records(lines, envelope)

    assert (
        "- Animal records: 2 of 3 tags loaded, fetched 29.0 hours ago, past their refresh "
        "interval, names set on 0." in lines
    )
    assert any("call find_animals" in line for line in lines)
