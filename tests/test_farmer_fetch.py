"""Characterization tests for the farmer-fetch path (pin CURRENT behavior).

Written BEFORE the Inc 3.2 refactor (extract a shared raw-dict helper; keep
`fetch_farmer_amulpashudhan → FarmerModel` byte-identical; add
`fetch_farmer_info_raw → FarmerRecord`). These tests capture today's behavior of:
  - fetch_farmer_amulpashudhan: camelCase API → FarmerModel + normalization, caching, 204/error → None
  - fetch_farmer_herdman: "Farmer"-wrapped response → list[FarmerModel]
  - merge_farmer_data: dedup by society_name+farmer_name, union preserved
  - get_farmer_data_by_mobile: orchestration (amulpashudhan; herdman only for MEHSANA)
They MUST stay green across the refactor (the FarmerModel domain path is unchanged).
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import asyncio
import json

import httpx
import pytest

import agents.tools.farmer_animal_backends as backends
import agents.tools.farmer as farmer_mod
from app.models.farmer import FarmerModel


# ── HTTP + cache mocking ──────────────────────────────────────────────────────

class _FakeResp:
    def __init__(self, *, json_data=None, status_code=200, text=None, raise_exc=None, json_exc=None):
        self._json = json_data
        self.status_code = status_code
        self.text = text if text is not None else (json.dumps(json_data) if json_data is not None else "")
        self._raise_exc = raise_exc
        self._json_exc = json_exc

    def raise_for_status(self):
        if self._raise_exc is not None:
            raise self._raise_exc

    def json(self):
        if self._json_exc is not None:
            raise self._json_exc
        return self._json


class _FakeClient:
    def __init__(self, resp_or_exc):
        self._resp_or_exc = resp_or_exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, *a, **k):
        if isinstance(self._resp_or_exc, Exception):
            raise self._resp_or_exc
        return self._resp_or_exc


def _patch_http(monkeypatch, resp_or_exc):
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _FakeClient(resp_or_exc))


def _patch_cache_miss(monkeypatch):
    saved = {}

    async def fake_get(cache_key):
        return (False, None)

    async def fake_set(cache_key, value):
        saved["key"] = cache_key
        saved["value"] = value

    monkeypatch.setattr(backends, "get_cached_api_response", fake_get)
    monkeypatch.setattr(backends, "set_cached_api_response", fake_set)
    return saved


def _patch_cache_hit(monkeypatch, payload):
    async def fake_get(cache_key):
        return (True, payload)

    async def fake_set(cache_key, value):
        pass

    monkeypatch.setattr(backends, "get_cached_api_response", fake_get)
    monkeypatch.setattr(backends, "set_cached_api_response", fake_set)


_AMUL_ROW = {
    "unionName": "Kaira",
    "unionCode": "U1",
    "societyName": "ABC Society",
    "societyCode": "S1",
    "farmerName": "Ramesh",
    "farmerCode": "F1",
    "mobileNumber": "9999999999",
    "tagNo": "123, 456",
    "totalAnimals": 3,
}


# ── fetch_farmer_amulpashudhan ────────────────────────────────────────────────

def test_amulpashudhan_maps_camelcase_to_farmermodel_with_normalization(monkeypatch):
    _patch_cache_miss(monkeypatch)
    _patch_http(monkeypatch, _FakeResp(json_data=[dict(_AMUL_ROW)]))

    out = asyncio.run(backends.fetch_farmer_amulpashudhan("9999999999", "tok"))

    assert out is not None and len(out) == 1
    f = out[0]
    assert isinstance(f, FarmerModel)
    # validators: union_name / society_name / farmer_name lowercased + stripped
    assert f.union_name == "kaira"
    assert f.society_name == "abc society"
    assert f.farmer_name == "ramesh"
    # codes are NOT lowercased (no validator)
    assert f.farmer_code == "F1"
    assert f.union_code == "U1"
    assert f.society_code == "S1"
    assert f.mobile_number == "9999999999"
    # tagNo -> animal_tags split + stripped
    assert f.animal_tags == ["123", "456"]
    assert f.total_animals == 3


def test_amulpashudhan_cache_hit_skips_http(monkeypatch):
    _patch_cache_hit(monkeypatch, [dict(_AMUL_ROW)])
    # If HTTP is touched, the test fails (cache hit must short-circuit).
    _patch_http(monkeypatch, AssertionError("HTTP must not be called on cache hit"))

    out = asyncio.run(backends.fetch_farmer_amulpashudhan("9999999999", "tok"))
    assert out is not None and out[0].union_name == "kaira"


def test_amulpashudhan_204_returns_none(monkeypatch):
    _patch_cache_miss(monkeypatch)
    _patch_http(monkeypatch, _FakeResp(status_code=204, text=""))
    out = asyncio.run(backends.fetch_farmer_amulpashudhan("9999999999", "tok"))
    assert out is None


def test_amulpashudhan_error_returns_none(monkeypatch):
    _patch_cache_miss(monkeypatch)
    _patch_http(monkeypatch, RuntimeError("network down"))
    out = asyncio.run(backends.fetch_farmer_amulpashudhan("9999999999", "tok"))
    assert out is None


# ── fetch_farmer_herdman ──────────────────────────────────────────────────────

def test_herdman_maps_farmer_wrapper_to_farmermodel(monkeypatch):
    _patch_cache_miss(monkeypatch)
    _patch_http(monkeypatch, _FakeResp(json_data={"Farmer": [dict(_AMUL_ROW)]}))
    out = asyncio.run(backends.fetch_farmer_herdman("9999999999", "tok3"))
    assert out is not None and len(out) == 1
    assert out[0].farmer_name == "ramesh"
    assert out[0].union_name == "kaira"


# ── merge_farmer_data ─────────────────────────────────────────────────────────

def test_merge_keeps_distinct_farmers(monkeypatch):
    a = FarmerModel.model_validate({"societyName": "S-A", "farmerName": "Ravi", "unionName": "Kaira"})
    b = FarmerModel.model_validate({"societyName": "S-B", "farmerName": "Sita", "unionName": "Banas"})
    out = backends.merge_farmer_data([a, b])
    assert len(out) == 2


def test_merge_dedupes_same_society_and_name_preserving_union():
    a = FarmerModel.model_validate({"societyName": "S-A", "farmerName": "Ravi", "unionName": "Kaira"})
    b = FarmerModel.model_validate({"societyName": "S-A", "farmerName": "Ravi", "unionName": "Banas"})
    out = backends.merge_farmer_data([a, b])
    assert len(out) == 1
    # merge preserves the first record's union_name explicitly
    assert out[0].union_name == "kaira"


# ── get_farmer_data_by_mobile orchestration ───────────────────────────────────

def test_get_farmer_data_amulpashudhan_only_for_non_mehsana(monkeypatch):
    monkeypatch.setenv("PASHUGPT_TOKEN", "tok1")
    monkeypatch.setenv("PASHUGPT_TOKEN_3", "tok3")
    rec = FarmerModel.model_validate({"societyName": "S-A", "farmerName": "Ravi", "unionName": "Kaira"})
    herdman_called = {"v": False}

    async def fake_amul(mobile, token):
        return [rec]

    async def fake_herdman(mobile, token):
        herdman_called["v"] = True
        return None

    monkeypatch.setattr(farmer_mod, "fetch_farmer_amulpashudhan", fake_amul)
    monkeypatch.setattr(farmer_mod, "fetch_farmer_herdman", fake_herdman)

    out = asyncio.run(farmer_mod.get_farmer_data_by_mobile("9999999999"))
    assert out is not None and len(out) == 1 and out[0].union_name == "kaira"
    assert herdman_called["v"] is False  # herdman only consulted for MEHSANA


def test_get_farmer_data_none_when_no_records(monkeypatch):
    monkeypatch.setenv("PASHUGPT_TOKEN", "tok1")
    monkeypatch.delenv("PASHUGPT_TOKEN_3", raising=False)

    async def fake_amul(mobile, token):
        return None

    monkeypatch.setattr(farmer_mod, "fetch_farmer_amulpashudhan", fake_amul)
    out = asyncio.run(farmer_mod.get_farmer_data_by_mobile("9999999999"))
    assert out is None


# ── fetch_farmer_amulpashudhan edge paths (pin before the 3.2b extraction) ─────
# These are the branches the raw-helper extraction must preserve: HTTP decode/shape
# failures → None, and the cache-hit fall-throughs (non-list / bad-data → refetch,
# negative cache → None without HTTP).

def test_amulpashudhan_json_decode_error_returns_none(monkeypatch):
    _patch_cache_miss(monkeypatch)
    _patch_http(monkeypatch, _FakeResp(
        status_code=200, text="not-json", json_exc=json.JSONDecodeError("boom", "doc", 0),
    ))
    out = asyncio.run(backends.fetch_farmer_amulpashudhan("9999999999", "tok"))
    assert out is None


def test_amulpashudhan_non_list_response_returns_none(monkeypatch):
    _patch_cache_miss(monkeypatch)
    _patch_http(monkeypatch, _FakeResp(json_data={"not": "a list"}, text='{"not":"a list"}'))
    out = asyncio.run(backends.fetch_farmer_amulpashudhan("9999999999", "tok"))
    assert out is None


def test_amulpashudhan_negative_cache_returns_none_without_http(monkeypatch):
    # Cached None (a prior 204/empty) is served as None, never hitting HTTP.
    _patch_cache_hit(monkeypatch, None)
    _patch_http(monkeypatch, AssertionError("HTTP must not be called for negative cache"))
    out = asyncio.run(backends.fetch_farmer_amulpashudhan("9999999999", "tok"))
    assert out is None


def test_amulpashudhan_cache_non_list_refetches_from_http(monkeypatch):
    # Cache holds a non-list (corrupt) → fall through to a fresh HTTP fetch.
    _patch_cache_hit(monkeypatch, {"corrupt": "not a list"})
    _patch_http(monkeypatch, _FakeResp(json_data=[dict(_AMUL_ROW)]))
    out = asyncio.run(backends.fetch_farmer_amulpashudhan("9999999999", "tok"))
    assert out is not None and out[0].union_name == "kaira"  # came from HTTP, not cache


def test_amulpashudhan_cache_unvalidatable_list_refetches_from_http(monkeypatch):
    # Cache holds a list that fails FarmerModel validation (non-dict items) →
    # fall through to a fresh HTTP fetch. (THE branch the extraction must keep.)
    _patch_cache_hit(monkeypatch, [123, "nope"])
    _patch_http(monkeypatch, _FakeResp(json_data=[dict(_AMUL_ROW)]))
    out = asyncio.run(backends.fetch_farmer_amulpashudhan("9999999999", "tok"))
    assert out is not None and out[0].union_name == "kaira"  # refetched, not the cached junk


# ── fetch_farmer_info_raw (Inc 3.2b-ii: RAW camelCase path, Option B) ──────────
# This is the SWR/voice ingestion path. It must NOT apply FarmerModel snake_case
# normalization — the camelCase must survive byte-for-byte (the Option B contract).

def _patch_raw(monkeypatch, value):
    async def fake_raw(mobile, token, **kwargs):
        return value
    monkeypatch.setattr(farmer_mod, "_fetch_farmer_amulpashudhan_raw", fake_raw)


def test_fetch_farmer_info_raw_preserves_camelcase_unnormalized(monkeypatch):
    monkeypatch.setenv("PASHUGPT_TOKEN", "tok1")
    _patch_raw(monkeypatch, [dict(_AMUL_ROW)])

    out = asyncio.run(farmer_mod.fetch_farmer_info_raw("9999999999"))

    assert out is not None and len(out) == 1
    rec = out[0]
    # RAW: farmerName keeps its original casing ("Ramesh"), NOT lowercased like FarmerModel.
    assert rec.farmerName == "Ramesh"
    assert rec.societyName == "ABC Society"
    assert rec.totalAnimals == 3
    assert rec.tagNo == "123, 456"
    # extra camelCase fields survive untouched (extra="allow").
    dumped = rec.model_dump()
    assert dumped["unionName"] == "Kaira"
    assert dumped["mobileNumber"] == "9999999999"


def test_fetch_farmer_info_raw_drops_empty_rows(monkeypatch):
    monkeypatch.setenv("PASHUGPT_TOKEN", "tok1")
    empty = {"unionName": "Kaira"}  # no tags / no animals / no farmer|society name
    _patch_raw(monkeypatch, [empty, dict(_AMUL_ROW)])

    out = asyncio.run(farmer_mod.fetch_farmer_info_raw("9999999999"))
    assert out is not None and len(out) == 1
    assert out[0].farmerName == "Ramesh"  # the content-bearing row kept, empty dropped


def test_fetch_farmer_info_raw_none_when_no_token(monkeypatch):
    monkeypatch.delenv("PASHUGPT_TOKEN", raising=False)
    monkeypatch.delenv("PASHUGPT_TOKEN_3", raising=False)
    _patch_raw(monkeypatch, [dict(_AMUL_ROW)])
    out = asyncio.run(farmer_mod.fetch_farmer_info_raw("9999999999"))
    assert out is None


def test_fetch_farmer_info_raw_none_when_no_data(monkeypatch):
    monkeypatch.setenv("PASHUGPT_TOKEN", "tok1")
    _patch_raw(monkeypatch, None)
    out = asyncio.run(farmer_mod.fetch_farmer_info_raw("9999999999"))
    assert out is None


def test_fetch_farmer_info_raw_none_for_empty_mobile(monkeypatch):
    # normalize_phone("") -> "" (falsy) is the only invalid-mobile guard; a
    # digit-less string like "abc" falls back to the original, matching
    # get_farmer_data_by_mobile's lenient behavior.
    monkeypatch.setenv("PASHUGPT_TOKEN", "tok1")
    _patch_raw(monkeypatch, [dict(_AMUL_ROW)])
    out = asyncio.run(farmer_mod.fetch_farmer_info_raw(""))
    assert out is None


# ── fetch_farmer_info_raw herdman gating (Inc 3.2b-iii: MEHSANA-only) ──────────
# Team decision (2026-06-08): herdman is pulled ONLY for MEHSANA — chat's gating
# is canonical; voice's herdman-for-everyone was unintended.

_MEHSANA_ROW = {
    "unionName": "Mehsana", "societyName": "S-M", "farmerName": "Ramesh",
    "farmerCode": "F1", "mobileNumber": "9999999999", "tagNo": "123", "totalAnimals": 1,
}
_HERDMAN_ROW = {
    "unionName": "Mehsana", "societyName": "S-H", "farmerName": "Geeta",
    "farmerCode": "F2", "tagNo": "789", "totalAnimals": 2,
}


def _patch_herdman_raw(monkeypatch, value):
    async def fake(mobile, token, **kwargs):
        return value
    monkeypatch.setattr(farmer_mod, "_fetch_farmer_herdman_raw", fake)


def test_fetch_farmer_info_raw_calls_herdman_for_mehsana(monkeypatch):
    monkeypatch.setenv("PASHUGPT_TOKEN", "tok1")
    monkeypatch.setenv("PASHUGPT_TOKEN_3", "tok3")
    _patch_raw(monkeypatch, [dict(_MEHSANA_ROW)])
    _patch_herdman_raw(monkeypatch, [dict(_HERDMAN_ROW)])

    out = asyncio.run(farmer_mod.fetch_farmer_info_raw("9999999999"))

    assert out is not None and len(out) == 2
    names = {r.farmerName for r in out}
    assert names == {"Ramesh", "Geeta"}  # both backends merged
    # herdman row is RAW camelCase too (not lowercased).
    geeta = next(r for r in out if r.farmerName == "Geeta")
    assert geeta.model_dump()["unionName"] == "Mehsana"


def test_fetch_farmer_info_raw_skips_herdman_for_non_mehsana(monkeypatch):
    monkeypatch.setenv("PASHUGPT_TOKEN", "tok1")
    monkeypatch.setenv("PASHUGPT_TOKEN_3", "tok3")
    _patch_raw(monkeypatch, [dict(_AMUL_ROW)])  # unionName "Kaira"

    async def boom(mobile, token, **kwargs):
        raise AssertionError("herdman must not be called for non-MEHSANA")
    monkeypatch.setattr(farmer_mod, "_fetch_farmer_herdman_raw", boom)

    out = asyncio.run(farmer_mod.fetch_farmer_info_raw("9999999999"))
    assert out is not None and len(out) == 1 and out[0].farmerName == "Ramesh"


def test_fetch_farmer_info_raw_dedupes_across_backends(monkeypatch):
    monkeypatch.setenv("PASHUGPT_TOKEN", "tok1")
    monkeypatch.setenv("PASHUGPT_TOKEN_3", "tok3")
    # herdman returns the SAME society+farmerCode as amulpashudhan → one record,
    # first occurrence (amulpashudhan) wins.
    dup = {"unionName": "Mehsana", "societyName": "S-M", "farmerName": "Stale",
           "farmerCode": "F1", "tagNo": "999"}
    _patch_raw(monkeypatch, [dict(_MEHSANA_ROW)])
    _patch_herdman_raw(monkeypatch, [dup])

    out = asyncio.run(farmer_mod.fetch_farmer_info_raw("9999999999"))
    assert out is not None and len(out) == 1
    assert out[0].farmerName == "Ramesh"  # amulpashudhan kept, herdman dup dropped


def test_fetch_farmer_info_raw_herdman_skipped_when_no_token3(monkeypatch):
    # MEHSANA farmer but PASHUGPT_TOKEN_3 unset → herdman not consulted.
    monkeypatch.setenv("PASHUGPT_TOKEN", "tok1")
    monkeypatch.delenv("PASHUGPT_TOKEN_3", raising=False)
    _patch_raw(monkeypatch, [dict(_MEHSANA_ROW)])

    async def boom(mobile, token, **kwargs):
        raise AssertionError("herdman must not be called without token3")
    monkeypatch.setattr(farmer_mod, "_fetch_farmer_herdman_raw", boom)

    out = asyncio.run(farmer_mod.fetch_farmer_info_raw("9999999999"))
    assert out is not None and len(out) == 1 and out[0].farmerName == "Ramesh"


# ── _fetch_farmer_herdman_raw direct (wrapper extraction, cache compatibility) ─

def test_herdman_raw_extracts_farmer_wrapper_from_http(monkeypatch):
    _patch_cache_miss(monkeypatch)
    _patch_http(monkeypatch, _FakeResp(json_data={"Farmer": [dict(_AMUL_ROW)]}))
    out = asyncio.run(backends._fetch_farmer_herdman_raw("9999999999", "tok3"))
    assert out is not None and len(out) == 1
    assert out[0]["farmerName"] == "Ramesh"  # RAW dict, not a FarmerModel


def test_herdman_raw_cache_hit_dict_extracts_without_http(monkeypatch):
    _patch_cache_hit(monkeypatch, {"Farmer": [dict(_AMUL_ROW)]})
    _patch_http(monkeypatch, AssertionError("HTTP must not be called on cache hit"))
    out = asyncio.run(backends._fetch_farmer_herdman_raw("9999999999", "tok3"))
    assert out is not None and out[0]["farmerName"] == "Ramesh"


def test_herdman_raw_non_dict_response_returns_none(monkeypatch):
    # A non-dict response is herdman's "no info found" case (raw analogue).
    _patch_cache_miss(monkeypatch)
    _patch_http(monkeypatch, _FakeResp(json_data=[dict(_AMUL_ROW)], text='[{}]'))
    out = asyncio.run(backends._fetch_farmer_herdman_raw("9999999999", "tok3"))
    assert out is None


def test_herdman_raw_empty_text_returns_none(monkeypatch):
    _patch_cache_miss(monkeypatch)
    _patch_http(monkeypatch, _FakeResp(status_code=200, text=""))
    out = asyncio.run(backends._fetch_farmer_herdman_raw("9999999999", "tok3"))
    assert out is None


def test_herdman_raw_no_farmer_key_returns_none(monkeypatch):
    _patch_cache_miss(monkeypatch)
    _patch_http(monkeypatch, _FakeResp(json_data={"somethingElse": 1}))
    out = asyncio.run(backends._fetch_farmer_herdman_raw("9999999999", "tok3"))
    assert out is None
