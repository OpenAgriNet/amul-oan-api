import asyncio
import json
from types import SimpleNamespace

import pytest

import app.tasks.suggestions as sug


class FakeOwnershipRedis:
    def __init__(self):
        self.store: dict[str, str] = {}

    async def eval(self, script, numkeys, *args):
        keys = args[:numkeys]
        argv = args[numkeys:]

        if "suggestions_claim_turn" in script:
            latest_key, pending_key, result_key = keys
            turn_id = argv[0]
            self.store[latest_key] = turn_id
            self.store[pending_key] = turn_id
            self.store.pop(result_key, None)
            return 1

        if "suggestions_publish_if_latest" in script:
            latest_key, result_key, pending_key = keys
            turn_id, suggestions_json = argv[:2]
            if self.store.get(latest_key) != turn_id:
                return 0
            self.store[result_key] = suggestions_json
            if self.store.get(pending_key) == turn_id:
                self.store.pop(pending_key, None)
            self.store.pop(latest_key, None)
            return 1

        if "suggestions_clear_if_owned" in script:
            pending_key, latest_key = keys
            turn_id = argv[0]
            cleared = 0
            for key in (pending_key, latest_key):
                if self.store.get(key) == turn_id:
                    self.store.pop(key)
                    cleared += 1
            return cleared

        raise AssertionError("unexpected Lua script")


@pytest.fixture
def ownership_env(monkeypatch):
    fake_redis = FakeOwnershipRedis()

    async def fake_hist(session_id):
        return []

    monkeypatch.setattr(sug, "redis_client", fake_redis)
    monkeypatch.setattr(sug, "_get_message_history", fake_hist)
    monkeypatch.setattr(sug.settings, "fallback_enabled", False)
    monkeypatch.setattr(sug.settings, "suggestions_hybrid_enabled", False)
    monkeypatch.setattr(
        sug._llm_resolver,
        "primary_tier",
        lambda *args, **kwargs: SimpleNamespace(handle="MODEL", model_name="model"),
    )
    monkeypatch.setattr(sug, "get_langfuse_client", None)
    monkeypatch.setattr(sug, "propagate_attributes", None)
    return fake_redis


def test_older_task_finishing_last_cannot_overwrite_newer_suggestions(
    monkeypatch,
    ownership_env,
):
    async def scenario():
        old_started = asyncio.Event()
        release_old = asyncio.Event()
        run_count = 0

        async def fake_run(message, model=None):
            nonlocal run_count
            run_count += 1
            if run_count == 1:
                old_started.set()
                await release_old.wait()
                return SimpleNamespace(output=["old suggestion"])
            return SimpleNamespace(output=["new suggestion"])

        monkeypatch.setattr(sug.suggestions_agent, "run", fake_run)
        await sug.claim_suggestions_turn("session", "gu", "old-turn", 30)
        old_task = asyncio.create_task(
            sug.create_suggestions("session", "gu", request_turn_id="old-turn")
        )
        await old_started.wait()

        await sug.claim_suggestions_turn("session", "gu", "new-turn", 30)
        new_result = await sug.create_suggestions(
            "session",
            "gu",
            request_turn_id="new-turn",
        )
        release_old.set()
        old_result = await old_task

        result_key, _, _ = sug._suggestions_cache_keys("session", "gu")
        assert new_result == ["new suggestion"]
        assert old_result == []
        assert json.loads(ownership_env.store[result_key]) == ["new suggestion"]

    asyncio.run(scenario())


def test_older_task_cannot_clear_newer_pending_state(monkeypatch, ownership_env):
    async def scenario():
        old_started = asyncio.Event()
        release_old = asyncio.Event()

        async def fake_run(message, model=None):
            old_started.set()
            await release_old.wait()
            return SimpleNamespace(output=["old suggestion"])

        monkeypatch.setattr(sug.suggestions_agent, "run", fake_run)
        await sug.claim_suggestions_turn("session", "gu", "old-turn", 30)
        old_task = asyncio.create_task(
            sug.create_suggestions("session", "gu", request_turn_id="old-turn")
        )
        await old_started.wait()

        await sug.claim_suggestions_turn("session", "gu", "new-turn", 30)
        release_old.set()
        assert await old_task == []

        _, pending_key, latest_key = sug._suggestions_cache_keys("session", "gu")
        assert ownership_env.store[pending_key] == "new-turn"
        assert ownership_env.store[latest_key] == "new-turn"

    asyncio.run(scenario())


def test_latest_task_publishes_and_clears_pending(monkeypatch, ownership_env):
    async def scenario():
        async def fake_run(message, model=None):
            return SimpleNamespace(output=["current suggestion"])

        monkeypatch.setattr(sug.suggestions_agent, "run", fake_run)
        await sug.claim_suggestions_turn("session", "gu", "current-turn", 30)

        result = await sug.create_suggestions(
            "session",
            "gu",
            request_turn_id="current-turn",
        )

        result_key, pending_key, latest_key = sug._suggestions_cache_keys("session", "gu")
        assert result == ["current suggestion"]
        assert json.loads(ownership_env.store[result_key]) == ["current suggestion"]
        assert pending_key not in ownership_env.store
        assert latest_key not in ownership_env.store

    asyncio.run(scenario())


def test_legacy_task_without_turn_id_keeps_existing_cache_behavior(
    monkeypatch,
    ownership_env,
):
    async def scenario():
        writes = []
        deletes = []

        async def fake_run(message, model=None):
            return SimpleNamespace(output=["legacy suggestion"])

        async def fake_set_cache(key, value, ttl):
            writes.append((key, value, ttl))
            return True

        async def fake_delete(key):
            deletes.append(key)

        monkeypatch.setattr(sug.suggestions_agent, "run", fake_run)
        monkeypatch.setattr(sug, "set_cache", fake_set_cache)
        monkeypatch.setattr(sug.cache, "delete", fake_delete)

        result = await sug.create_suggestions("legacy-session", "gu")

        assert result == ["legacy suggestion"]
        assert writes == [
            (
                "suggestions_legacy-session_gu",
                ["legacy suggestion"],
                sug.settings.suggestions_cache_ttl,
            )
        ]
        assert deletes == ["suggestions_legacy-session_gu:pending"]

    asyncio.run(scenario())
