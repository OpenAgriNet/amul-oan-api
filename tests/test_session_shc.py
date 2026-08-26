import pytest

from agents.tools import session_shc


class FakeCache:
    def __init__(self):
        self.values = {}

    async def set(self, key, value, *, ttl, namespace):
        self.values[(namespace, key)] = (value, ttl)

    async def get(self, key, *, namespace):
        row = self.values.get((namespace, key))
        return row[0] if row else None


@pytest.mark.asyncio
async def test_context_is_bound_to_session_and_authenticated_mobile(monkeypatch):
    fake = FakeCache()
    monkeypatch.setattr(session_shc, "cache", fake)

    await session_shc.set_session_shc_context("session-1", "+919924457046", "Nitrogen: low")

    assert await session_shc.get_session_shc_context("session-1", "9924457046") == "Nitrogen: low"
    assert await session_shc.get_session_shc_context("session-1", "+919999999999") is None
    assert await session_shc.get_session_shc_context("session-2", "+919924457046") is None
