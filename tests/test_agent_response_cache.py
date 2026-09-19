import pytest

from agents.tools import response_cache


class MemoryCache:
    def __init__(self):
        self.values = {}
        self.get_calls = []
        self.set_calls = []
        self.delete_calls = []

    async def get(self, key, *, namespace):
        self.get_calls.append((key, namespace))
        return self.values.get((namespace, key))

    async def set(self, key, value, *, ttl, namespace):
        self.set_calls.append((key, value, ttl, namespace))
        self.values[(namespace, key)] = value

    async def delete(self, key, *, namespace):
        self.delete_calls.append((key, namespace))
        self.values.pop((namespace, key), None)


def _policy():
    return response_cache.ResponseCachePolicy(
        name="test-profile",
        ttl_seconds=30,
        negative_ttl_seconds=7,
    )


@pytest.mark.asyncio
async def test_successful_response_is_cached_with_hashed_identity(monkeypatch):
    memory = MemoryCache()
    monkeypatch.setattr(response_cache, "cache", memory)
    loads = 0

    async def loader():
        nonlocal loads
        loads += 1
        return {"name": "farmer"}

    arguments = dict(
        policy=_policy(),
        identity={"mobile": "9000000000"},
        loader=loader,
        encode=lambda value: value,
        decode=lambda value: value,
        is_negative=lambda value: not value,
    )
    first = await response_cache.get_or_load(**arguments)
    second = await response_cache.get_or_load(**arguments)

    assert first == second == {"name": "farmer"}
    assert loads == 1
    assert memory.set_calls[0][2] == 30
    assert "9000000000" not in memory.set_calls[0][0]


@pytest.mark.asyncio
async def test_negative_response_uses_short_ttl_and_force_refresh_replaces_it(monkeypatch):
    memory = MemoryCache()
    monkeypatch.setattr(response_cache, "cache", memory)
    values = [[], ["fresh"]]

    async def loader():
        return values.pop(0)

    arguments = dict(
        policy=_policy(),
        identity={"tag": "TAG-1"},
        loader=loader,
        encode=lambda value: value,
        decode=lambda value: value,
        is_negative=lambda value: not value,
    )
    assert await response_cache.get_or_load(**arguments) == []
    assert memory.set_calls[-1][2] == 7
    assert await response_cache.get_or_load(**arguments, force_refresh=True) == ["fresh"]
    assert memory.set_calls[-1][2] == 30
    assert len(memory.get_calls) == 1


@pytest.mark.asyncio
async def test_loader_failure_is_never_cached(monkeypatch):
    memory = MemoryCache()
    monkeypatch.setattr(response_cache, "cache", memory)

    async def loader():
        raise RuntimeError("provider failed")

    with pytest.raises(RuntimeError, match="provider failed"):
        await response_cache.get_or_load(
            policy=_policy(),
            identity={"id": "private"},
            loader=loader,
            encode=lambda value: value,
            decode=lambda value: value,
            is_negative=lambda value: False,
        )

    assert memory.set_calls == []


@pytest.mark.asyncio
async def test_invalid_cached_value_is_discarded_and_reloaded(monkeypatch):
    memory = MemoryCache()
    monkeypatch.setattr(response_cache, "cache", memory)
    key = response_cache._cache_key(_policy(), {"id": "private"})
    memory.values[(response_cache._CACHE_NAMESPACE, key)] = {
        "version": response_cache._CACHE_VERSION,
        "value": "invalid",
    }

    async def loader():
        return ["live"]

    def decode(item):
        if not isinstance(item, list):
            raise ValueError("not a list")
        return item

    value = await response_cache.get_or_load(
        policy=_policy(),
        identity={"id": "private"},
        loader=loader,
        encode=lambda item: item,
        decode=decode,
        is_negative=lambda item: not item,
    )

    assert value == ["live"]
    assert memory.delete_calls == [(key, response_cache._CACHE_NAMESPACE)]


@pytest.mark.asyncio
async def test_cache_outage_fails_open(monkeypatch):
    class UnavailableCache:
        async def get(self, *args, **kwargs):
            raise ConnectionError("redis unavailable")

        async def set(self, *args, **kwargs):
            raise ConnectionError("redis unavailable")

    monkeypatch.setattr(response_cache, "cache", UnavailableCache())

    async def loader():
        return ["live"]

    value = await response_cache.get_or_load(
        policy=_policy(),
        identity={"id": "private"},
        loader=loader,
        encode=lambda item: item,
        decode=lambda item: item,
        is_negative=lambda item: not item,
    )

    assert value == ["live"]
