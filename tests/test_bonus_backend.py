import asyncio
import os
import sys

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.tools.models.bonus import FarmerBonusAmountRequestModel
from agents.tools import bonus_backend
from agents.tools.bonus_backend import get_farmer_bonus_amount_api
from agents.tools import farmer_ticket_summary_backend as ticket_backend
from agents.tools.farmer_ticket_summary_backend import get_farmer_ticket_summary_api
from agents.tools.models.farmer_ticket_summary import FarmerTicketSummaryRequestModel


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
        monkeypatch.setattr(bonus_backend.httpx, "AsyncClient", _FakeAsyncClient)

        result = asyncio.run(get_farmer_bonus_amount_api(self._request(), "test-token"))

        assert result is not None
        assert len(result) == 1
        assert result[0].bonus_amount == 1273.1
        assert result[0].from_date == "2026-04-01T00:00:00"
        assert _FakeAsyncClient.calls == [
            {
                "url": f"{bonus_backend.settings.amulpashudhan_base_url}/GetFarmerBonusAmount",
                "params": {
                    "unionCode": "0001",
                    "societyCode": "2004",
                    "farmerCode": "0001",
                },
                "headers": {"Authorization": "Bearer test-token"},
                "timeout": bonus_backend.settings.farmer_backend_http_timeout_seconds,
            }
        ]

    def test_empty_list_response_returns_empty_list(self, monkeypatch):
        _FakeAsyncClient.response = httpx.Response(
            200,
            json=[],
            request=httpx.Request("GET", "https://example.test"),
        )
        monkeypatch.setattr(bonus_backend.httpx, "AsyncClient", _FakeAsyncClient)

        result = asyncio.run(get_farmer_bonus_amount_api(self._request(), "test-token"))

        assert result == []

    def test_dict_envelope_response_returns_none(self, monkeypatch):
        _FakeAsyncClient.response = httpx.Response(
            200,
            json={"APIStatusCode": 0, "Data": []},
            request=httpx.Request("GET", "https://example.test"),
        )
        monkeypatch.setattr(bonus_backend.httpx, "AsyncClient", _FakeAsyncClient)

        result = asyncio.run(get_farmer_bonus_amount_api(self._request(), "test-token"))

        assert result is None

    def test_http_error_returns_none(self, monkeypatch):
        _FakeAsyncClient.response = httpx.Response(
            500,
            text="server error",
            request=httpx.Request("GET", "https://example.test"),
        )
        monkeypatch.setattr(bonus_backend.httpx, "AsyncClient", _FakeAsyncClient)

        result = asyncio.run(get_farmer_bonus_amount_api(self._request(), "test-token"))

        assert result is None

    def test_unsupported_union_source_returns_none(self, monkeypatch):
        _FakeAsyncClient.response = httpx.Response(
            400,
            text="Bonus amount not supported for this union’s data source.",
            request=httpx.Request("GET", "https://example.test"),
        )
        monkeypatch.setattr(bonus_backend.httpx, "AsyncClient", _FakeAsyncClient)

        result = asyncio.run(get_farmer_bonus_amount_api(self._request(), "test-token"))

        assert result is None

    def test_farmer_bonus_data_not_found_returns_empty_list(self, monkeypatch):
        _FakeAsyncClient.response = httpx.Response(
            400,
            text="Farmer bonus data not found.",
            request=httpx.Request("GET", "https://example.test"),
        )
        monkeypatch.setattr(bonus_backend.httpx, "AsyncClient", _FakeAsyncClient)

        result = asyncio.run(get_farmer_bonus_amount_api(self._request(), "test-token"))

        assert result == []

    def test_invalid_list_item_returns_none(self, monkeypatch):
        _FakeAsyncClient.response = httpx.Response(
            200,
            json=["not-a-dict"],
            request=httpx.Request("GET", "https://example.test"),
        )
        monkeypatch.setattr(bonus_backend.httpx, "AsyncClient", _FakeAsyncClient)

        result = asyncio.run(get_farmer_bonus_amount_api(self._request(), "test-token"))

        assert result is None

    def test_unexpected_dict_shape_returns_none(self, monkeypatch):
        _FakeAsyncClient.response = httpx.Response(
            200,
            json=[{"Message": "unexpected shape"}],
            request=httpx.Request("GET", "https://example.test"),
        )
        monkeypatch.setattr(bonus_backend.httpx, "AsyncClient", _FakeAsyncClient)

        result = asyncio.run(get_farmer_bonus_amount_api(self._request(), "test-token"))

        assert result is None


class TestFarmerTicketSummaryBackend:
    def setup_method(self):
        _FakeAsyncClient.calls = []
        _FakeAsyncClient.response = httpx.Response(
            200,
            json={
                "farmerDetails": {
                    "farmerName": "FARMER1",
                    "farmerCode": "0001",
                    "societyName": "DEMO_2004",
                    "societyCode": "2004",
                },
                "pendingTickets": [
                    {"ticketNumber": "T-1", "bookDateTime": "2026-10-20T10:00:00"}
                ],
                "lastLactationDetail": [
                    {
                        "animalTagNumber": "TAG-1",
                        "lastBreedingActivity": {
                            "lastAI": "2026-09-01T00:00:00",
                            "lastAITicketNumber": "AI-1",
                            "bullId": "BULL-7",
                            "semenType": "Sexed",
                        },
                    }
                ],
            },
            request=httpx.Request("GET", "https://example.test"),
        )

    def _request(self):
        return FarmerTicketSummaryRequestModel(
            unionCode="0001", societyCode="2004", farmerCode="0001"
        )

    def test_sends_expected_request_and_accepts_live_schema_drift(self, monkeypatch):
        monkeypatch.setattr(ticket_backend.httpx, "AsyncClient", _FakeAsyncClient)

        result = asyncio.run(get_farmer_ticket_summary_api(self._request(), "test-token"))

        assert result is not None
        assert result.pending_tickets[0].ticket_number == "T-1"
        assert result.last_lactation_detail[0].last_breeding_activity.bull_id == "BULL-7"
        assert _FakeAsyncClient.calls == [{
            "url": f"{ticket_backend.settings.amulpashudhan_base_url}/GetFarmerTicketSummary",
            "params": {"unionCode": "0001", "societyCode": "2004", "farmerCode": "0001"},
            "headers": {"accept": "application/json", "Authorization": "Bearer test-token"},
            "timeout": ticket_backend.settings.farmer_backend_http_timeout_seconds,
        }]

    def test_nullable_lists_are_normalized_to_empty(self, monkeypatch):
        _FakeAsyncClient.response = httpx.Response(
            200,
            json={"farmerDetails": None, "pendingTickets": None, "lastLactationDetail": None},
            request=httpx.Request("GET", "https://example.test"),
        )
        monkeypatch.setattr(ticket_backend.httpx, "AsyncClient", _FakeAsyncClient)
        result = asyncio.run(get_farmer_ticket_summary_api(self._request(), "test-token"))
        assert result is not None
        assert result.pending_tickets == []
        assert result.last_lactation_detail == []
        assert result.has_data() is False

    def test_record_not_found_is_an_empty_success(self, monkeypatch):
        _FakeAsyncClient.response = httpx.Response(
            500,
            json={"Error": "Union Record Not Found."},
            request=httpx.Request("GET", "https://example.test"),
        )
        monkeypatch.setattr(ticket_backend.httpx, "AsyncClient", _FakeAsyncClient)
        result = asyncio.run(get_farmer_ticket_summary_api(self._request(), "test-token"))
        assert result is not None
        assert result.has_data() is False

    def test_invalid_json_and_unexpected_shape_fail_closed(self, monkeypatch):
        monkeypatch.setattr(ticket_backend.httpx, "AsyncClient", _FakeAsyncClient)
        _FakeAsyncClient.response = httpx.Response(
            200, text="not-json", request=httpx.Request("GET", "https://example.test")
        )
        assert asyncio.run(get_farmer_ticket_summary_api(self._request(), "test-token")) is None
        _FakeAsyncClient.response = httpx.Response(
            200, json={"message": "wrong contract"}, request=httpx.Request("GET", "https://example.test")
        )
        assert asyncio.run(get_farmer_ticket_summary_api(self._request(), "test-token")) is None
