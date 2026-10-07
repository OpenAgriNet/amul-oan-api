"""Voice reads chat's farmer cache (agents/tools/farmer_cache.py).

With the voice route on, the worker's refresh also fetches each animal's record,
which voice answers AI/breeding history questions from. A refresh on a turn
leaves that to the worker. Every refresh keeps the animals already cached, so
chat's worker on the same queue cannot wipe them, and they are fetched again only
for new tags or once the refresh interval has passed.

Self-contained: asyncio.run, an in-memory Redis and stubbed Beckn calls.
"""
import asyncio
import copy
import json
from datetime import datetime, timedelta, timezone

import pytest

import agents.tools.farmer_cache as fc
from agents.tools.farmer import FarmerFetchOutcome
from agents.tools.models.animal import AnimalModel
from agents.tools.models.farmer_transport import AnimalRecord, FarmerDataEnvelope, FarmerRecord
from app.config import settings
from app.core.cache import build_cache_key
from app.voice.farmer import _append_animal_records, _build_ai_technician_summary

PHONE = "9876543210"
TAGS = ["102030405060", "102030405061"]


class _Redis:
    def __init__(self):
        self.kv, self.sets = {}, {}

    async def set(self, key, value, ex=None, nx=False):
        if nx and key in self.kv:
            return None
        self.kv[key] = value
        return True

    async def get(self, key):
        return self.kv.get(key)

    async def exists(self, key):
        return int(key in self.kv)

    async def delete(self, key):
        self.kv.pop(key, None)

    async def ttl(self, key):
        return 3600

    async def sadd(self, key, member):
        self.sets.setdefault(key, set()).add(member)

    async def spop(self, key, count):
        members = self.sets.get(key, set())
        return [members.pop() for _ in range(min(count, len(members)))]

    def queued(self):
        return set(self.sets.get(fc.FARMER_REFRESH_QUEUE_KEY, set()))


class _Cache:
    def __init__(self, redis):
        self.redis = redis

    async def get(self, key, namespace=None):
        return copy.deepcopy(self.redis.kv.get(build_cache_key(key, namespace)))

    async def set(self, key, value, ttl=None, namespace=None):
        self.redis.kv[build_cache_key(key, namespace)] = copy.deepcopy(value)


class _Beckn:
    """Stub Beckn calls: the farmer lookup, technicians and animal profiles."""

    def __init__(self, *, technicians_fail=False, animals=True):
        self.technicians_fail = technicians_fail
        self.animals = animals
        self.tags = list(TAGS)
        self.farmer_calls = 0
        self.animal_calls = []

    async def farmer(self, phone):
        self.farmer_calls += 1
        record = FarmerRecord.model_validate({
            "farmerName": "Ramesh", "farmerCode": "F1", "societyName": "Anand",
            "societyCode": "S1", "unionCode": "U1", "tagNo": ",".join(self.tags),
        })
        return [record], FarmerFetchOutcome.FOUND

    async def technicians(self, *, union_code, society_code, force_refresh=False):
        if self.technicians_fail:
            raise RuntimeError("bpp timeout")
        return []

    async def animal(self, tag, *, union_code=None, **kwargs):
        self.animal_calls.append((tag, union_code))
        if not self.animals:
            return None
        return AnimalModel.model_validate({
            "tagNumber": tag,
            "animalType": "Cow",
            "lastBreedingActivity": {"aiDate": "2026-09-01", "bullId": "B7"},
        })


@pytest.fixture
def shared(monkeypatch):
    redis = _Redis()
    beckn = _Beckn()
    monkeypatch.setattr(fc, "redis_client", redis)
    monkeypatch.setattr(fc, "cache", _Cache(redis))
    monkeypatch.setattr(fc, "fetch_farmer_info_with_outcome", beckn.farmer)
    monkeypatch.setattr(fc, "search_ai_technicians", beckn.technicians)
    monkeypatch.setattr(fc, "fetch_animal_profile", beckn.animal)
    monkeypatch.setattr(settings, "voice_route_enabled", True)
    return redis, beckn


def _animals(envelope):
    return [animal for farmer in envelope.farmers for animal in farmer.animals]


def test_voice_reads_a_chat_written_envelope_with_its_history_and_does_not_requeue_it(shared):
    redis, beckn = shared

    asyncio.run(fc.refresh_farmer_data(PHONE, background=True))
    envelope = asyncio.run(fc.get_farmer_data_cached_only(PHONE))

    assert [a.tagNumber for a in _animals(envelope)] == TAGS
    assert envelope.stale is False
    assert redis.queued() == set()
    lines: list[str] = []
    _append_animal_records(lines, envelope)
    assert any("last AI/breeding=" in line and "B7" in line for line in lines)
    summary = _build_ai_technician_summary(envelope)
    assert "none available for this farmer group" in summary


def test_voice_reads_chats_failed_technician_lookup_as_a_failure(shared):
    """Read as "try again later", never "no technicians", and retried once the
    backoff lapses rather than straight away."""
    redis, beckn = shared
    beckn.technicians_fail = True

    asyncio.run(fc.refresh_farmer_data(PHONE, background=True))
    envelope = asyncio.run(fc.get_farmer_data_cached_only(PHONE))

    summary = _build_ai_technician_summary(envelope)
    assert "temporarily unavailable" in summary
    assert "none available for this farmer group" not in summary
    assert envelope.staleReason == "ai_technician_lookup_failed"
    assert redis.queued() == set()


def test_cold_fetch_on_a_turn_leaves_the_animals_to_the_worker(shared, monkeypatch):
    redis, beckn = shared
    lock_key = fc._refresh_lock_key(PHONE)
    held_when_queued = []
    enqueue = fc.enqueue_farmer_refresh

    async def _enqueue(phone):
        held_when_queued.append(lock_key in redis.kv)
        await enqueue(phone)

    monkeypatch.setattr(fc, "enqueue_farmer_refresh", _enqueue)

    first = asyncio.run(fc.get_or_fetch_farmer_data(PHONE))

    assert first is not None and _animals(first) == []
    assert beckn.animal_calls == []
    # Queued once the lock was released, so the worker's refresh can take it.
    assert held_when_queued == [False]
    assert redis.queued() == {PHONE}

    assert asyncio.run(fc.drain_farmer_refresh_queue_once()) == 1
    envelope = asyncio.run(fc.get_farmer_data_cached_only(PHONE))
    assert len(_animals(envelope)) == len(TAGS)
    assert envelope.stale is False


def test_no_animal_fetch_when_the_voice_route_is_off(shared, monkeypatch):
    redis, beckn = shared
    monkeypatch.setattr(settings, "voice_route_enabled", False)

    asyncio.run(fc.refresh_farmer_data(PHONE, background=True))
    envelope = asyncio.run(fc.get_farmer_data_cached_only(PHONE))

    assert beckn.animal_calls == []
    assert envelope.stale is False
    assert redis.queued() == set()


def test_tags_with_no_animal_data_are_not_fetched_again_every_turn(shared):
    redis, beckn = shared
    beckn.animals = False

    asyncio.run(fc.refresh_farmer_data(PHONE, background=True))
    envelope = asyncio.run(fc.get_farmer_data_cached_only(PHONE))

    assert len(beckn.animal_calls) == len(TAGS)
    assert _animals(envelope) == []
    assert envelope.stale is False
    assert redis.queued() == set()


def test_record_cached_without_animals_is_queued_for_them_once_voice_is_on(shared, monkeypatch):
    """Written by a refresh that skipped them, e.g. a chat-only container."""
    redis, beckn = shared
    monkeypatch.setattr(settings, "voice_route_enabled", False)
    asyncio.run(fc.refresh_farmer_data(PHONE, background=True))
    monkeypatch.setattr(settings, "voice_route_enabled", True)

    envelope = asyncio.run(fc.get_farmer_data_cached_only(PHONE))

    assert envelope.staleReason == "missing_animals"
    assert redis.queued() == {PHONE}


def test_animal_fetch_is_passed_the_farmers_union(shared):
    redis, beckn = shared
    asyncio.run(fc.refresh_farmer_data(PHONE, background=True))
    assert {union for _, union in beckn.animal_calls} == {"U1"}


def test_animal_fetches_are_capped(shared, monkeypatch):
    redis, beckn = shared
    tags = [f"1020304050{i:02d}" for i in range(fc.FARMER_ANIMAL_FETCH_CONCURRENCY * 3)]
    running = {"now": 0, "max": 0}

    async def _farmer(phone):
        return [FarmerRecord.model_validate({"farmerCode": "F1", "tagNo": ",".join(tags)})], FarmerFetchOutcome.FOUND

    async def _animal(tag, **kwargs):
        running["now"] += 1
        running["max"] = max(running["max"], running["now"])
        await asyncio.sleep(0.01)
        running["now"] -= 1
        return None

    monkeypatch.setattr(fc, "fetch_farmer_info_with_outcome", _farmer)
    monkeypatch.setattr(fc, "fetch_animal_profile", _animal)

    asyncio.run(fc.refresh_farmer_data(PHONE, background=True))

    assert running["max"] == fc.FARMER_ANIMAL_FETCH_CONCURRENCY


def test_failed_animal_fetch_leaves_the_others(shared, monkeypatch):
    redis, beckn = shared
    animal = beckn.animal

    async def _animal(tag, **kwargs):
        if tag == TAGS[0]:
            raise RuntimeError("bpp timeout")
        return await animal(tag, **kwargs)

    monkeypatch.setattr(fc, "fetch_animal_profile", _animal)

    envelope = asyncio.run(fc.refresh_farmer_data(PHONE, background=True))

    assert [a.tagNumber for a in _animals(envelope)] == TAGS[1:]


def test_worker_refresh_is_the_background_one(monkeypatch):
    redis = _Redis()
    redis.sets[fc.FARMER_REFRESH_QUEUE_KEY] = {PHONE}
    calls = []

    async def _refresh(phone, **kwargs):
        calls.append((phone, kwargs))

    monkeypatch.setattr(fc, "redis_client", redis)
    monkeypatch.setattr(fc, "refresh_farmer_data", _refresh)

    assert asyncio.run(fc.drain_farmer_refresh_queue_once()) == 1
    assert calls == [(PHONE, {"background": True})]


def test_animals_survive_the_cache_round_trip():
    envelope = FarmerDataEnvelope.from_records([{"farmerCode": "F1", "tagNo": TAGS[0]}])
    envelope.farmers[0].animals = [
        AnimalRecord(tagNumber=TAGS[0], lastBreedingActivity={"aiDate": "2026-09-01", "bullId": "B7"})
    ]
    restored = FarmerDataEnvelope.model_validate(envelope.model_dump())
    assert restored.farmers[0].animals[0].lastBreedingActivity == {"aiDate": "2026-09-01", "bullId": "B7"}


def test_refresh_without_the_voice_route_keeps_voices_animals(shared, monkeypatch):
    """Chat's worker pops from the same queue: its refresh must not wipe them."""
    redis, beckn = shared
    asyncio.run(fc.refresh_farmer_data(PHONE, background=True))
    monkeypatch.setattr(settings, "voice_route_enabled", False)
    asyncio.run(fc.refresh_farmer_data(PHONE, background=True))
    monkeypatch.setattr(settings, "voice_route_enabled", True)

    envelope = asyncio.run(fc.get_farmer_data_cached_only(PHONE))

    assert [a.tagNumber for a in _animals(envelope)] == TAGS
    assert envelope.animalsFetchedAt is not None
    assert envelope.stale is False
    assert redis.queued() == set()


def test_refresh_on_a_turn_keeps_the_animals(shared):
    """A cold or max-serve-stale fetch rebuilds the record too."""
    redis, beckn = shared
    asyncio.run(fc.refresh_farmer_data(PHONE, background=True))

    asyncio.run(fc.refresh_farmer_data(PHONE))
    envelope = asyncio.run(fc.get_farmer_data_cached_only(PHONE))

    assert len(_animals(envelope)) == len(TAGS)
    assert redis.queued() == set()


def test_refresh_inside_the_interval_fetches_only_new_tags(shared):
    redis, beckn = shared
    asyncio.run(fc.refresh_farmer_data(PHONE, background=True))
    fetched_at = asyncio.run(fc.get_cached_farmer_data(PHONE)).animalsFetchedAt
    beckn.tags = TAGS + ["102030405062"]

    asyncio.run(fc.refresh_farmer_data(PHONE, background=True))
    envelope = asyncio.run(fc.get_cached_farmer_data(PHONE))

    assert [tag for tag, _ in beckn.animal_calls] == TAGS + ["102030405062"]
    assert [a.tagNumber for a in _animals(envelope)] == beckn.tags
    assert envelope.animalsFetchedAt == fetched_at


def test_animals_are_fetched_again_once_the_refresh_interval_has_passed(shared):
    """So a new AI date reaches voice, as with the farmer record itself."""
    redis, beckn = shared
    asyncio.run(fc.refresh_farmer_data(PHONE, background=True))
    key = build_cache_key(fc._cache_key(PHONE), namespace=fc.FARMER_CACHE_NAMESPACE)
    old = datetime.now(timezone.utc) - timedelta(seconds=fc.FARMER_REFRESH_INTERVAL + 60)
    redis.kv[key]["animalsFetchedAt"] = old.isoformat()

    asyncio.run(fc.refresh_farmer_data(PHONE, background=True))

    assert len(beckn.animal_calls) == 2 * len(TAGS)
    assert asyncio.run(fc.get_cached_farmer_data(PHONE)).animalsFetchedAt > old.isoformat()


def test_animal_of_a_tag_the_farmer_no_longer_has_is_dropped(shared, monkeypatch):
    redis, beckn = shared
    asyncio.run(fc.refresh_farmer_data(PHONE, background=True))
    beckn.tags = TAGS[:1]
    monkeypatch.setattr(settings, "voice_route_enabled", False)

    asyncio.run(fc.refresh_farmer_data(PHONE, background=True))
    envelope = asyncio.run(fc.get_cached_farmer_data(PHONE))

    assert [a.tagNumber for a in _animals(envelope)] == TAGS[:1]


def test_tags_with_no_data_wait_for_the_interval(shared):
    redis, beckn = shared
    beckn.animals = False

    asyncio.run(fc.refresh_farmer_data(PHONE, background=True))
    asyncio.run(fc.refresh_farmer_data(PHONE, background=True))

    assert len(beckn.animal_calls) == len(TAGS)


def test_failing_technician_lookup_is_not_refreshed_on_every_turn(shared):
    """During a technician outage, three voice turns cost no extra upstream calls."""
    redis, beckn = shared
    beckn.technicians_fail = True
    asyncio.run(fc.refresh_farmer_data(PHONE, background=True))
    farmer_calls, animal_calls = beckn.farmer_calls, len(beckn.animal_calls)

    for _ in range(3):
        asyncio.run(fc.get_farmer_data_cached_only(PHONE))
        asyncio.run(fc.drain_farmer_refresh_queue_once())

    assert beckn.farmer_calls == farmer_calls
    assert len(beckn.animal_calls) == animal_calls


def test_failing_technician_lookup_is_retried_once_the_backoff_lapses(shared):
    redis, beckn = shared
    beckn.technicians_fail = True
    asyncio.run(fc.refresh_farmer_data(PHONE, background=True))
    attempt_key = fc._refresh_attempt_key(PHONE)
    state = json.loads(redis.kv[attempt_key])
    state["next_retry_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    redis.kv[attempt_key] = json.dumps(state)

    asyncio.run(fc.get_farmer_data_cached_only(PHONE))
    assert redis.queued() == {PHONE}

    beckn.technicians_fail = False
    asyncio.run(fc.drain_farmer_refresh_queue_once())
    envelope = asyncio.run(fc.get_farmer_data_cached_only(PHONE))
    assert envelope.stale is False
    assert attempt_key not in redis.kv


def test_failed_animal_fetch_keeps_what_was_cached_for_that_tag(shared, monkeypatch):
    redis, beckn = shared
    asyncio.run(fc.refresh_farmer_data(PHONE, background=True))
    key = build_cache_key(fc._cache_key(PHONE), namespace=fc.FARMER_CACHE_NAMESPACE)
    old = datetime.now(timezone.utc) - timedelta(seconds=fc.FARMER_REFRESH_INTERVAL + 60)
    redis.kv[key]["animalsFetchedAt"] = old.isoformat()
    animal = beckn.animal

    async def _animal(tag, **kwargs):
        if tag == TAGS[0]:
            raise RuntimeError("bpp timeout")
        return await animal(tag, **kwargs)

    monkeypatch.setattr(fc, "fetch_animal_profile", _animal)

    envelope = asyncio.run(fc.refresh_farmer_data(PHONE, background=True))

    assert [a.tagNumber for a in _animals(envelope)] == TAGS
