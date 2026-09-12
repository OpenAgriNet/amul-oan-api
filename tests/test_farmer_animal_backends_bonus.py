import asyncio
import os
import sys

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.models.bonus import FarmerBonusAmountRequestModel
from agents.tools import farmer_animal_backends
from agents.tools.farmer_animal_backends import get_farmer_bonus_amount_api


class _FakeAsyncClient:
    calls = []
    response = httpx.Response(200, json=[], request=httpx.Request("GET", "https://example.test"))

    def __init__(self, timeout):
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url, params=None, headers=None):
        self.calls.append(
            {"url": url, "params": params, "headers": headers, "timeout": self.timeout}
        )
        return self.response


class TestFarmerAnimalBackendsBonus:
    def setup_method(self):
        _FakeAsyncClient.calls = []
        _FakeAsyncClient.response = httpx.Response(
            200,
            json=[
                {
                    "societyCode": "2004",
                    "societyName": "DEMO_2004",
                    "farmerCode": "0001",
                    "farmerName": "FARMER1",
                    "bonusAmount": 1273.1,
                    "fromDate": "2026-04-01T00:00:00",
                    "toDate": "2026-04-01T00:00:00",
                }
            ],
            request=httpx.Request("GET", "https://example.test"),
        )

    def _request(self):
        return FarmerBonusAmountRequestModel(
            unionCode="0001",
            societyCode="2004",
            farmerCode="0001",
        )

    def test_sends_expected_endpoint_query_params_and_auth_header(self, monkeypatch):
        monkeypatch.setattr(farmer_animal_backends.httpx, "AsyncClient", _FakeAsyncClient)

        result = asyncio.run(get_farmer_bonus_amount_api(self._request(), "test-token"))

        assert result is not None
        assert len(result) == 1
        assert result[0].bonus_amount == 1273.1
        assert result[0].from_date == "2026-04-01T00:00:00"
        assert _FakeAsyncClient.calls == [
            {
                "url": f"{farmer_animal_backends.BASE_AMULPASHUDHAN}/GetFarmerBonusAmount",
                "params": {
                    "unionCode": "0001",
                    "societyCode": "2004",
                    "farmerCode": "0001",
                },
                "headers": {"Authorization": "Bearer test-token"},
                "timeout": farmer_animal_backends.FARMER_BACKEND_HTTP_TIMEOUT_SECONDS,
            }
        ]

    def test_empty_list_response_returns_empty_list(self, monkeypatch):
        _FakeAsyncClient.response = httpx.Response(
            200,
            json=[],
            request=httpx.Request("GET", "https://example.test"),
        )
        monkeypatch.setattr(farmer_animal_backends.httpx, "AsyncClient", _FakeAsyncClient)

        result = asyncio.run(get_farmer_bonus_amount_api(self._request(), "test-token"))

        assert result == []

    def test_dict_envelope_response_returns_none(self, monkeypatch):
        _FakeAsyncClient.response = httpx.Response(
            200,
            json={"APIStatusCode": 0, "Data": []},
            request=httpx.Request("GET", "https://example.test"),
        )
        monkeypatch.setattr(farmer_animal_backends.httpx, "AsyncClient", _FakeAsyncClient)

        result = asyncio.run(get_farmer_bonus_amount_api(self._request(), "test-token"))

        assert result is None

    def test_http_error_returns_none(self, monkeypatch):
        _FakeAsyncClient.response = httpx.Response(
            500,
            text="server error",
            request=httpx.Request("GET", "https://example.test"),
        )
        monkeypatch.setattr(farmer_animal_backends.httpx, "AsyncClient", _FakeAsyncClient)

        result = asyncio.run(get_farmer_bonus_amount_api(self._request(), "test-token"))

        assert result is None

    def test_unsupported_union_source_returns_none(self, monkeypatch):
        _FakeAsyncClient.response = httpx.Response(
            400,
            text="Bonus amount not supported for this union’s data source.",
            request=httpx.Request("GET", "https://example.test"),
        )
        monkeypatch.setattr(farmer_animal_backends.httpx, "AsyncClient", _FakeAsyncClient)

        result = asyncio.run(get_farmer_bonus_amount_api(self._request(), "test-token"))

        assert result is None

    def test_farmer_bonus_data_not_found_returns_empty_list(self, monkeypatch):
        _FakeAsyncClient.response = httpx.Response(
            400,
            text="Farmer bonus data not found.",
            request=httpx.Request("GET", "https://example.test"),
        )
        monkeypatch.setattr(farmer_animal_backends.httpx, "AsyncClient", _FakeAsyncClient)

        result = asyncio.run(get_farmer_bonus_amount_api(self._request(), "test-token"))

        assert result == []

    def test_invalid_list_item_returns_none(self, monkeypatch):
        _FakeAsyncClient.response = httpx.Response(
            200,
            json=["not-a-dict"],
            request=httpx.Request("GET", "https://example.test"),
        )
        monkeypatch.setattr(farmer_animal_backends.httpx, "AsyncClient", _FakeAsyncClient)

        result = asyncio.run(get_farmer_bonus_amount_api(self._request(), "test-token"))

        assert result is None
