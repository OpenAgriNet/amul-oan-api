"""Booking tools must be idempotent per session so an agent re-run (the
OSS->managed streaming fallback re-executes tool calls) cannot double-book."""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import asyncio
from types import SimpleNamespace

import pytest

from agents.tools import ai_call as ai_mod
from agents.tools import health_call as hc_mod
from app.models.ai_call import AISpecies
from app.models.health_call import HealthCaseType


def _ctx(session_id):
    return SimpleNamespace(deps=SimpleNamespace(session_id=session_id))


def _patch_cache(monkeypatch, module):
    store = {}

    async def fake_get(key, namespace=None):
        return store.get((namespace, key))

    async def fake_set(key, value, ttl=None, namespace=None):
        store[(namespace, key)] = value

    monkeypatch.setattr(module.cache, "get", fake_get)
    monkeypatch.setattr(module.cache, "set", fake_set)
    monkeypatch.setenv("PASHUGPT_TOKEN", "tok")
    return store


def test_ai_call_idempotent_on_rerun(monkeypatch):
    _patch_cache(monkeypatch, ai_mod)
    calls = {"n": 0}

    async def fake_api(request, token):
        calls["n"] += 1
        return SimpleNamespace(ticket_number="T1", ait_name="AIT", model_dump=lambda: {"ticket_number": "T1"})

    monkeypatch.setattr(ai_mod, "create_ai_call_api", fake_api)
    species = next(iter(AISpecies))

    r1 = asyncio.run(ai_mod.create_ai_call(_ctx("s1"), "U", "S", "F", "tech1", species))
    r2 = asyncio.run(ai_mod.create_ai_call(_ctx("s1"), "U", "S", "F", "tech1", species))

    assert calls["n"] == 1                 # booking API hit exactly once across the re-run
    assert "booked successfully" in r1
    assert "already" in r2.lower()         # second call short-circuited


def test_health_call_idempotent_on_rerun(monkeypatch):
    _patch_cache(monkeypatch, hc_mod)
    calls = {"n": 0}

    async def fake_api(request, token):
        calls["n"] += 1
        return SimpleNamespace(ticket_number="H1")

    monkeypatch.setattr(hc_mod, "create_health_call_api", fake_api)
    species = next(iter(AISpecies))
    case_type = next(iter(HealthCaseType))

    r1 = asyncio.run(hc_mod.create_health_call(_ctx("s1"), "U", "S", "F", species, case_type, "remark v1"))
    # re-run: model may even produce a different remark; session key still dedupes
    r2 = asyncio.run(hc_mod.create_health_call(_ctx("s1"), "U", "S", "F", species, case_type, "remark v2"))

    assert calls["n"] == 1
    assert "booked successfully" in r1
    assert "already" in r2.lower()


def test_no_session_id_does_not_crash(monkeypatch):
    """Defensive: missing session_id (e.g. None deps) must not raise; it just
    skips the guard (no dedup, but no crash)."""
    _patch_cache(monkeypatch, ai_mod)

    async def fake_api(request, token):
        return SimpleNamespace(ticket_number="T1", ait_name="AIT", model_dump=lambda: {})

    monkeypatch.setattr(ai_mod, "create_ai_call_api", fake_api)
    species = next(iter(AISpecies))
    r = asyncio.run(ai_mod.create_ai_call(_ctx(None), "U", "S", "F", "tech1", species))
    assert "booked successfully" in r
