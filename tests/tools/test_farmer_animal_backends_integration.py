"""
Integration tests for farmer/animal backends against live APIs.

Loads env from farmer_animal_api_reference/.env when present.
Optional env: TEST_FARMER_MOBILE, TEST_ANIMAL_TAG (use your real values to get data).

Run from amul-oan-api with env from script folder:
  set -a && . ../../farmer_animal_api_reference/.env && set +a
  pytest tests/tools/test_farmer_animal_backends_integration.py -v
"""
import asyncio
import os
from pathlib import Path

import pytest

# Load env from farmer_animal_api_reference/.env when present
# Path: .../amul-api-integration/amul-oan-api/tests/tools/ -> 5x parent = OAN
_script_env = Path(__file__).resolve().parent.parent.parent.parent.parent / "farmer_animal_api_reference" / ".env"
if _script_env.exists():
    from dotenv import load_dotenv
    load_dotenv(_script_env)

from agents.tools.farmer_animal_backends import (
    fetch_animal_amulpashudhan,
    fetch_animal_herdman,
    fetch_farmer_amulpashudhan,
    fetch_farmer_herdman,
    merge_animal_data,
    merge_farmer_records,
)


def _mobile():
    return os.getenv("TEST_FARMER_MOBILE", "9876543210").strip()


def _tag():
    return os.getenv("TEST_ANIMAL_TAG", "105183817302").strip()


def _run(coro):
    return asyncio.run(coro)


class TestFarmerBackendsIntegration:
    """Live farmer API calls (skipped if no tokens)."""

    @pytest.fixture
    def token1(self):
        return os.getenv("PASHUGPT_TOKEN")

    @pytest.fixture
    def token3(self):
        return os.getenv("PASHUGPT_TOKEN_3")

    def test_fetch_farmer_amulpashudhan(self, token1):
        if not token1:
            pytest.skip("PASHUGPT_TOKEN not set")
        mobile = _mobile()
        result = _run(fetch_farmer_amulpashudhan(mobile, token1))
        assert result is None or (isinstance(result, list) and (len(result) == 0 or isinstance(result[0], dict)))

    def test_fetch_farmer_herdman(self, token3):
        if not token3:
            pytest.skip("PASHUGPT_TOKEN_3 not set")
        mobile = _mobile()
        result = _run(fetch_farmer_herdman(mobile, token3))
        assert result is None or (isinstance(result, list) and (len(result) == 0 or isinstance(result[0], dict)))

    def test_farmer_merge_and_dedupe(self, token1, token3):
        if not token1 and not token3:
            pytest.skip("Set PASHUGPT_TOKEN and/or PASHUGPT_TOKEN_3")
        mobile = _mobile()
        records = []

        async def collect():
            if token1:
                r = await fetch_farmer_amulpashudhan(mobile, token1)
                if r:
                    records.extend(r)
            if token3:
                r = await fetch_farmer_herdman(mobile, token3)
                if r:
                    records.extend(r)

        _run(collect())
        merged = merge_farmer_records(records)
        assert isinstance(merged, list)
        for rec in merged:
            assert isinstance(rec, dict)


class TestAnimalBackendsIntegration:
    """Live animal API calls (skipped if no tokens)."""

    @pytest.fixture
    def token1(self):
        return os.getenv("PASHUGPT_TOKEN")

    @pytest.fixture
    def token3(self):
        return os.getenv("PASHUGPT_TOKEN_3")

    def test_fetch_animal_amulpashudhan(self, token1):
        if not token1:
            pytest.skip("PASHUGPT_TOKEN not set")
        tag = _tag()
        result = _run(fetch_animal_amulpashudhan(tag, token1))
        assert result is None or (isinstance(result, dict) and (result.get("tagNumber") or result.get("tagNo")))

    def test_fetch_animal_herdman(self, token3):
        if not token3:
            pytest.skip("PASHUGPT_TOKEN_3 not set")
        tag = _tag()
        result = _run(fetch_animal_herdman(tag, token3))
        assert result is None or isinstance(result, dict)

    def test_animal_merge(self, token1, token3):
        if not token1 and not token3:
            pytest.skip("Set PASHUGPT_TOKEN and/or PASHUGPT_TOKEN_3")
        tag = _tag()

        async def get_both():
            primary = await fetch_animal_amulpashudhan(tag, token1) if token1 else None
            fallback = await fetch_animal_herdman(tag, token3) if token3 else None
            return merge_animal_data(primary, fallback)

        merged = _run(get_both())
        assert isinstance(merged, dict)
