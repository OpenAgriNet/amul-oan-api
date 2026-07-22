import asyncio
import os
import sys

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.models.milk_collection import FarmerMilkCollectionRequestModel
from agents.tools import farmer_animal_backends
from agents.tools.farmer_animal_backends import get_farmer_milk_collection_details_api


class _FakeAsyncClient:
    calls = []
    response = httpx.Response(200, json={}, request=httpx.Request("GET", "https://example.test"))

    def __init__(self, timeout):
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url, params=None, headers=None):
        self.calls.append({"url": url, "params": params, "headers": headers, "timeout": self.timeout})
        return self.response


class TestFarmerAnimalBackendsMilkCollection:
    def setup_method(self):
        _FakeAsyncClient.calls = []
        _FakeAsyncClient.response = httpx.Response(
            200,
            json={
                "result": "success",
                "milk": [{"date": "2026-04-01", "qty": 5}],
                "deduction": [{"date": "2026-04-01", "accountname": "Feed", "amount": 25}],
            },
            request=httpx.Request("GET", "https://example.test"),
        )

    def _request(self):
        return FarmerMilkCollectionRequestModel(
            unionCode="0201",
            societyCode="001066",
            farmerCode="000123",
            fromdate="2026-04-01",
            todate="2026-04-01",
        )

    def test_sends_expected_endpoint_query_params_and_auth_header(self, monkeypatch):
        monkeypatch.setattr(farmer_animal_backends.httpx, "AsyncClient", _FakeAsyncClient)

        result = asyncio.run(get_farmer_milk_collection_details_api(self._request(), "test-token"))

        assert result is not None
        assert result.deduction[0].account_name == "Feed"
        assert _FakeAsyncClient.calls == [
            {
                "url": f"{farmer_animal_backends.BASE_AMULPASHUDHAN}/FarmerMilkCollectionDetails",
                "params": {
                    "unionCode": "0201",
                    "societyCode": "001066",
                    "farmerCode": "000123",
                    "fromdate": "2026-04-01",
                    "todate": "2026-04-01",
                },
                "headers": {"Authorization": "Bearer test-token"},
                "timeout": 30.0,
            }
        ]

    def test_non_dict_response_returns_none(self, monkeypatch):
        _FakeAsyncClient.response = httpx.Response(
            200,
            json=[],
            request=httpx.Request("GET", "https://example.test"),
        )
        monkeypatch.setattr(farmer_animal_backends.httpx, "AsyncClient", _FakeAsyncClient)

        result = asyncio.run(get_farmer_milk_collection_details_api(self._request(), "test-token"))

        assert result is None

    def test_http_error_returns_none(self, monkeypatch):
        _FakeAsyncClient.response = httpx.Response(
            500,
            text="server error",
            request=httpx.Request("GET", "https://example.test"),
        )
        monkeypatch.setattr(farmer_animal_backends.httpx, "AsyncClient", _FakeAsyncClient)

        result = asyncio.run(get_farmer_milk_collection_details_api(self._request(), "test-token"))

        assert result is None
