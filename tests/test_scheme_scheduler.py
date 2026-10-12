from types import SimpleNamespace

import pytest

from app.services import scheme_ingestion
from app.tasks import scheme_scheduler


class MemoryRedis:
    def __init__(self):
        self.values = {}

    async def set(self, key, value, **kwargs):
        self.values[key] = value

    async def get(self, key):
        return self.values.get(key)


@pytest.mark.asyncio
async def test_startup_refresh_queues_only_missing_scheme_sources(monkeypatch):
    sources = (
        SimpleNamespace(cache_key="already-populated"),
        SimpleNamespace(cache_key="missing-source"),
    )
    queued_coroutines = []

    async def cache_exists(source_key):
        return source_key == "already-populated"

    async def refresh_source(source):
        return source.cache_key

    def create_task(coroutine):
        queued_coroutines.append(coroutine)
        coroutine.close()

    monkeypatch.setattr(scheme_scheduler, "get_scheme_sources", lambda: sources)
    monkeypatch.setattr(scheme_scheduler, "source_cache_exists", cache_exists)
    monkeypatch.setattr(scheme_scheduler, "refresh_scheme_source", refresh_source)

    queued = await scheme_scheduler.schedule_startup_scheme_refreshes(
        create_task_fn=create_task
    )

    assert queued == ["missing-source"]
    assert len(queued_coroutines) == 1


@pytest.mark.asyncio
async def test_ingestion_writes_the_shared_scheme_redis_contract(monkeypatch):
    redis = MemoryRedis()
    monkeypatch.setattr(scheme_ingestion.settings, "redis_key_prefix", "sva-cache-")
    records = [{
        "union_name": "banas",
        "scheme_title": "Cattle insurance",
        "scheme_url": "https://example.test/cattle-insurance.pdf",
    }]

    await scheme_ingestion.cache_source_records(
        scheme_ingestion.BANAS_SOURCE.cache_key,
        records,
        redis_client=redis,
    )

    expected_key = (
        "sva-cache:milk_producer_schemes:"
        "banasdairy.coop/home/inputactivities#milkproducers"
    )
    assert expected_key in redis.values
    assert await scheme_ingestion.get_cached_scheme_records_for_union(
        "banas",
        redis_client=redis,
    ) == records
