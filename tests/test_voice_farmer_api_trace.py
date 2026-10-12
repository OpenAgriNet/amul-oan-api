"""Voice's Amul API traces: status and a PII-safe response shape, the raw body
only behind FARMER_API_TRACE_BODY, and a redacted body for errors."""
import asyncio
from unittest.mock import patch

import httpx

from agents.voice.tools import farmer_animal_backends as backends


def _response(status: int, body: str = "") -> httpx.Response:
    return httpx.Response(
        status_code=status,
        text=body,
        request=httpx.Request("GET", "https://api.amulpashudhan.com/x"),
    )


def test_trace_recorded_before_raise_for_status_on_failure():
    """A failing (5xx) bonus response must still be traced — _record_api_trace
    runs BEFORE response.raise_for_status()."""
    import contextlib
    from agents.voice.tools import farmer_animal_backends as backends
    from app.voice.models.bonus import FarmerBonusAmountRequestModel

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

        async def get(self, *a, **k):
            return _Resp()

    req = FarmerBonusAmountRequestModel(
        unionCode="2021", societyCode="NA4310", farmerCode="NA0002",
    )
    with patch.object(backends, "start_observation", _fake_obs), \
         patch.object(backends.httpx, "AsyncClient", lambda *a, **k: _Client()):
        result = asyncio.run(backends.get_farmer_bonus_amount_api(req, "tok"))
    assert result is None                                    # raised -> None
    assert captured["output"]["status_code"] == 500          # but the 500 WAS traced
    assert captured["output"]["ok"] is False


def _capture_trace(backends, resp):
    captured = {}

    class _Obs:
        def update(self, output=None, metadata=None):
            captured["output"] = output
            captured["metadata"] = metadata

    backends._record_api_trace(_Obs(), resp, provider="amulpashudhan", url="http://x")
    return captured


def test_record_api_trace_is_pii_safe_by_default():
    """By default NO raw body is shipped — only status + structure (keys/null_keys
    + record count), which still proves an inconsistent return."""
    from agents.voice.tools import farmer_animal_backends as backends

    class _Resp:
        status_code = 200
        text = '{"farmerName": "Ramesh", "totalAnimals": null, "tagNo": "1,2"}'

    out = _capture_trace(backends, _Resp())["output"]
    assert out["status_code"] == 200
    assert out["ok"] is True
    assert out["records"] == 1
    assert out["keys"] == ["farmerName", "tagNo", "totalAnimals"]
    assert out["null_keys"] == ["totalAnimals"]          # proves the Turn-A shape
    assert "body" not in out                              # no PII value leaks
    assert "Ramesh" not in str(out)


def test_record_api_trace_ok_is_2xx():
    from agents.voice.tools import farmer_animal_backends as backends

    class _R204:
        status_code = 204
        text = ""

    class _R500:
        status_code = 500
        text = "err"

    assert _capture_trace(backends, _R204())["output"]["ok"] is True   # 204 is ok
    assert _capture_trace(backends, _R500())["output"]["ok"] is False


def test_record_api_trace_body_only_when_flag_enabled(monkeypatch):
    from agents.voice.tools import farmer_animal_backends as backends

    class _Resp:
        status_code = 200
        text = '{"totalAnimals": 5}'

    monkeypatch.setattr(backends.settings, "farmer_api_trace_body", True)
    out = _capture_trace(backends, _Resp())["output"]
    assert out["body"] == '{"totalAnimals": 5}'


def test_safe_response_summary_shapes():
    from agents.voice.tools.farmer_animal_backends import _safe_response_summary

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
    from agents.voice.tools.farmer_animal_backends import _safe_response_summary

    out = _safe_response_summary('[{"animals": [], "society": "", "tagNo": "1"}]')
    assert out["array_lens"] == {"animals": 0}     # empty array surfaced
    assert out["empty_str_keys"] == ["society"]    # empty string surfaced


def test_record_api_trace_none_observation_is_noop():
    from agents.voice.tools import farmer_animal_backends as backends

    class _Resp:
        status_code = 500
        text = "boom"

    backends._record_api_trace(None, _Resp(), provider="x", url="y")  # must not raise


# ── P2: the trace must be able to tell the two apart ────────────────────────

def test_error_body_is_recorded():
    """{status_code, bytes, keys} could not distinguish a negative lookup from a
    real fault, which is what made root cause B expensive to attribute."""
    captured = {}

    class _Obs:
        def update(self, **kwargs):
            captured.update(kwargs)

    backends._record_api_trace(
        _Obs(),
        _response(500, '{"Error":"Farmer Record Not Found."}'),
        provider="amulpashudhan",
        url="https://x",
    )
    assert "Farmer Record Not Found" in captured["output"]["error_body"]


def test_error_body_redacts_phone_numbers():
    """Error responses can echo the request, and the request carries the
    caller's mobile."""
    captured = {}

    class _Obs:
        def update(self, **kwargs):
            captured.update(kwargs)

    backends._record_api_trace(
        _Obs(),
        _response(500, '{"Error":"no record for 9876543210"}'),
        provider="amulpashudhan",
        url="https://x",
    )
    body = captured["output"]["error_body"]
    assert "9876543210" not in body
    assert "[redacted]" in body


def test_successful_response_records_no_error_body():
    captured = {}

    class _Obs:
        def update(self, **kwargs):
            captured.update(kwargs)

    backends._record_api_trace(
        _Obs(), _response(200, '[{"farmerCode":"5058"}]'),
        provider="amulpashudhan", url="https://x",
    )
    assert "error_body" not in captured["output"]
