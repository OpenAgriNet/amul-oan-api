"""Tests for the PII-safe API trace helpers in farmer_animal_backends (added in
Inc 3.1, previously untested): _record_api_trace, _safe_response_summary, and the
fetch_reason contextvar.

These prove we can record enough of a response (status + structure: record count,
which keys are present/null/empty) to debug inconsistent upstream returns WITHOUT
shipping farmer PII to Langfuse. Raw bodies only when FARMER_API_TRACE_BODY is on.

Also covers create_ai_call_api's trace-before-raise behavior, enabled when the
booking backends were migrated to start_observation + _record_api_trace in §13
Part B (a failed 5xx booking is now traced, not just successes).
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import asyncio
import contextlib
from unittest.mock import patch


def _capture_trace(backends, resp, reason="cold_fetch"):
    captured = {}

    class _Obs:
        def update(self, output=None, metadata=None):
            captured["output"] = output
            captured["metadata"] = metadata

    with backends.fetch_reason(reason):
        backends._record_api_trace(_Obs(), resp, provider="amulpashudhan", url="http://x")
    return captured


def test_record_api_trace_is_pii_safe_by_default():
    """By default NO raw body is shipped — only status + structure (keys/null_keys
    + record count), which still proves an inconsistent return."""
    from agents.tools import farmer_animal_backends as backends

    class _Resp:
        status_code = 200
        text = '{"farmerName": "Ramesh", "totalAnimals": null, "tagNo": "1,2"}'

    out = _capture_trace(backends, _Resp())["output"]
    assert out["status_code"] == 200
    assert out["ok"] is True
    assert out["fetch_reason"] == "cold_fetch"
    assert out["records"] == 1
    assert out["keys"] == ["farmerName", "tagNo", "totalAnimals"]
    assert out["null_keys"] == ["totalAnimals"]          # proves the shape
    assert "body" not in out                              # no PII value leaks
    assert "Ramesh" not in str(out)


def test_record_api_trace_ok_is_2xx():
    from agents.tools import farmer_animal_backends as backends

    class _R204:
        status_code = 204
        text = ""

    class _R500:
        status_code = 500
        text = "err"

    assert _capture_trace(backends, _R204())["output"]["ok"] is True   # 204 is ok
    assert _capture_trace(backends, _R500())["output"]["ok"] is False


def test_record_api_trace_body_only_when_flag_enabled(monkeypatch):
    from agents.tools import farmer_animal_backends as backends

    class _Resp:
        status_code = 200
        text = '{"totalAnimals": 5}'

    monkeypatch.setattr(backends.settings, "farmer_api_trace_body", True)
    out = _capture_trace(backends, _Resp())["output"]
    assert out["body"] == '{"totalAnimals": 5}'


def test_safe_response_summary_shapes():
    from agents.tools.farmer_animal_backends import _safe_response_summary

    full = _safe_response_summary('[{"totalAnimals": 5, "tagNo": "1", "visits": [1, 2, 3]}]')
    assert full["records"] == 1 and "totalAnimals" in full["keys"] and full["null_keys"] == []
    assert full["array_lens"] == {"visits": 3}          # array metric, no values
    missing = _safe_response_summary('[{"tagNo": "1"}]')   # totalAnimals absent
    assert "totalAnimals" not in missing["keys"]
    empty = _safe_response_summary("[]")
    assert empty["records"] == 0
    notjson = _safe_response_summary("<html>err</html>")
    assert notjson["json"] is False


def test_safe_response_summary_flags_empty_arrays_and_strings():
    from agents.tools.farmer_animal_backends import _safe_response_summary

    out = _safe_response_summary('[{"animals": [], "society": "", "tagNo": "1"}]')
    assert out["array_lens"] == {"animals": 0}     # empty array surfaced
    assert out["empty_str_keys"] == ["society"]    # empty string surfaced


def test_record_api_trace_none_observation_is_noop():
    from agents.tools import farmer_animal_backends as backends

    class _Resp:
        status_code = 500
        text = "boom"

    backends._record_api_trace(None, _Resp(), provider="x", url="y")  # must not raise


def test_fetch_reason_contextvar_default_and_scope():
    from agents.tools import farmer_animal_backends as backends

    assert backends.current_fetch_reason() == "request"
    with backends.fetch_reason("background_refresh"):
        assert backends.current_fetch_reason() == "background_refresh"
    assert backends.current_fetch_reason() == "request"


def test_trace_recorded_before_raise_for_status_on_failure():
    """A failing (5xx) booking response must still be traced — _record_api_trace
    runs BEFORE response.raise_for_status() in create_ai_call_api (§13 Part B)."""
    from agents.tools import farmer_animal_backends as backends
    from app.models.ai_call import AICallRequestModel, AISpecies

    captured = {}

    class _Obs:
        def update(self, output=None, metadata=None):
            captured["output"] = output

    @contextlib.contextmanager
    def _fake_obs(*a, **k):
        yield _Obs()

    class _Resp:
        status_code = 500
        text = '{"error": "boom"}'

        def raise_for_status(self):
            raise Exception("HTTP 500")

        def json(self):
            return {}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            return _Resp()

    req = AICallRequestModel(
        unionCode="2021", societyCode="NA4310", farmerCode="NA0002",
        userId="u1", species=AISpecies.COW,
    )
    with patch.object(backends, "start_observation", _fake_obs), \
         patch.object(backends.httpx, "AsyncClient", lambda *a, **k: _Client()):
        result = asyncio.run(backends.create_ai_call_api(req, "tok"))
    assert result is None                                    # raised -> None
    assert captured["output"]["status_code"] == 500          # but the 500 WAS traced
    assert captured["output"]["ok"] is False
