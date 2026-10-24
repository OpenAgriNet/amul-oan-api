import asyncio
from types import SimpleNamespace

from agents.tools import bonus as bonus_tool
from agents.tools import farmer_ticket_summary as ticket_tool
from agents.tools.models.bonus import FarmerBonusAmountRecordModel
from agents.tools.models.farmer import FarmerModel
from agents.tools.models.farmer_ticket_summary import FarmerTicketSummaryResponseModel


def _ctx():
    return SimpleNamespace(
        deps=SimpleNamespace(
            mobile="9000000000",
            session_id="session-1",
            farmer_unions=["kaira"],
        ),
        tool_call_id="tool-1",
    )


def _farmer(**overrides):
    data = {
        "unionCode": "0001",
        "societyCode": "2004",
        "farmerCode": "0001",
        "unionName": "Kaira",
    }
    data.update(overrides)
    return FarmerModel.model_validate(data)


def _patch_farmers(monkeypatch, farmers):
    async def lookup(mobile):
        assert mobile == "9000000000"
        return farmers

    monkeypatch.setattr(bonus_tool, "get_farmer_data_by_mobile", lookup)


def _patch_ticket_farmers(monkeypatch, farmers):
    async def lookup(mobile):
        assert mobile == "9000000000"
        return farmers

    monkeypatch.setattr(ticket_tool, "get_farmer_data_by_mobile", lookup)


def _ticket_summary(**overrides):
    data = {
        "farmerDetails": {
            "farmerName": "FARMER1", "farmerCode": "0001",
            "societyName": "DEMO_2004", "societyCode": "2004",
        },
        "pendingTickets": [{"ticketNumber": "T-1", "bookDateTime": "2026-10-20T10:00:00"}],
        "lastLactationDetail": [{
            "animalTagNumber": "TAG-1",
            "lastBreedingActivity": {
                "lastAI": "2026-09-01T00:00:00", "lastAITicketNumber": "AI-1",
                "bullId": "BULL-7", "semenType": "Sexed",
                "lastPD": "2026-10-01T00:00:00", "lastPDTicketNumber": "PD-1",
                "pdStatus": "Positive", "lastCalving": "2025-12-01T00:00:00",
                "lastCalvingTicketNumber": "C-1",
            },
        }],
    }
    data.update(overrides)
    return FarmerTicketSummaryResponseModel.model_validate(data)


class TestBonusTool:
    def test_success_uses_signed_account_and_formats_markdown(self, monkeypatch):
        _patch_farmers(monkeypatch, [_farmer()])
        monkeypatch.setenv("PASHUGPT_TOKEN", "test-token")

        async def fake_api(request, token):
            assert token == "test-token"
            assert request.to_query_params() == {
                "unionCode": "0001",
                "societyCode": "2004",
                "farmerCode": "0001",
            }
            return [
                FarmerBonusAmountRecordModel.model_validate(
                    {
                        "societyCode": "2004",
                        "societyName": "DEMO_2004",
                        "farmerCode": "0001",
                        "farmerName": "FARMER1",
                        "bonusAmount": 1273.1,
                        "fromDate": "2026-04-01T00:00:00",
                        "toDate": "2026-04-01T00:00:00",
                    }
                )
            ]

        monkeypatch.setattr(bonus_tool, "get_farmer_bonus_amount_api", fake_api)
        result = asyncio.run(bonus_tool.get_farmer_bonus_amount(_ctx()))

        assert result.startswith(
            "Farmer bonus amount details fetched successfully:\n\n"
        )
        assert "### Bonus Amount" in result
        assert (
            "| 2026-04-01 - 2026-04-01 | DEMO_2004 | FARMER1 | 1273.10 |" in result
        )

    def test_missing_authenticated_mobile_never_calls_backend(self, monkeypatch):
        called = False

        async def unexpected(*args, **kwargs):
            nonlocal called
            called = True

        monkeypatch.setattr(bonus_tool, "get_farmer_data_by_mobile", unexpected)
        ctx = _ctx()
        ctx.deps.mobile = None

        result = asyncio.run(bonus_tool.get_farmer_bonus_amount(ctx))

        assert "signed-in farmer profile" in result
        assert called is False

    def test_no_accounts_returns_clear_failure(self, monkeypatch):
        _patch_farmers(monkeypatch, [])
        monkeypatch.setenv("PASHUGPT_TOKEN", "test-token")

        result = asyncio.run(bonus_tool.get_farmer_bonus_amount(_ctx()))

        assert "No union, society, and farmer account" in result

    def test_empty_bonus_list_returns_no_records_message(self, monkeypatch):
        _patch_farmers(monkeypatch, [_farmer()])
        monkeypatch.setenv("PASHUGPT_TOKEN", "test-token")

        async def empty_api(request, token):
            return []

        monkeypatch.setattr(bonus_tool, "get_farmer_bonus_amount_api", empty_api)
        result = asyncio.run(bonus_tool.get_farmer_bonus_amount(_ctx()))

        assert "No bonus records were found" in result

    def test_not_found_empty_result_is_not_amcs_failure(self, monkeypatch):
        """Backend maps 'Farmer bonus data not found' → []; tool must not show AMCS failure."""
        _patch_farmers(monkeypatch, [_farmer()])
        monkeypatch.setenv("PASHUGPT_TOKEN", "test-token")

        async def not_found_as_empty(request, token):
            return []

        monkeypatch.setattr(bonus_tool, "get_farmer_bonus_amount_api", not_found_as_empty)
        result = asyncio.run(bonus_tool.get_farmer_bonus_amount(_ctx()))

        assert "No bonus records were found" in result
        assert "AMCS" not in result
        assert "Unable to fetch bonus amount details" not in result

    def test_all_provider_failures_return_temporary_failure(self, monkeypatch):
        _patch_farmers(monkeypatch, [_farmer()])
        monkeypatch.setenv("PASHUGPT_TOKEN", "test-token")

        async def failed(request, token):
            return None

        monkeypatch.setattr(bonus_tool, "get_farmer_bonus_amount_api", failed)
        result = asyncio.run(bonus_tool.get_farmer_bonus_amount(_ctx()))

        assert "Unable to fetch bonus amount details at the moment." in result
        assert "AMCS" in result

    def test_mixed_empty_and_failure_returns_temporary_failure(self, monkeypatch):
        _patch_farmers(
            monkeypatch,
            [
                _farmer(),
                _farmer(societyCode="2005", farmerCode="0002"),
            ],
        )
        monkeypatch.setenv("PASHUGPT_TOKEN", "test-token")

        async def mixed_api(request, token):
            if request.farmer_code == "0001":
                return []
            return None

        monkeypatch.setattr(bonus_tool, "get_farmer_bonus_amount_api", mixed_api)
        result = asyncio.run(bonus_tool.get_farmer_bonus_amount(_ctx()))

        assert "Unable to fetch bonus amount details at the moment." in result
        assert "No bonus records were found" not in result

    def test_mixed_data_and_failure_returns_temporary_failure(self, monkeypatch):
        _patch_farmers(
            monkeypatch,
            [
                _farmer(),
                _farmer(societyCode="2005", farmerCode="0002"),
            ],
        )
        monkeypatch.setenv("PASHUGPT_TOKEN", "test-token")

        async def mixed_api(request, token):
            if request.farmer_code == "0001":
                return [
                    FarmerBonusAmountRecordModel.model_validate(
                        {
                            "societyCode": "2004",
                            "societyName": "SOC_0001",
                            "farmerCode": "0001",
                            "farmerName": "F_0001",
                            "bonusAmount": 100,
                            "fromDate": "2026-04-01T00:00:00",
                            "toDate": "2026-04-01T00:00:00",
                        }
                    )
                ]
            return None

        monkeypatch.setattr(bonus_tool, "get_farmer_bonus_amount_api", mixed_api)
        result = asyncio.run(bonus_tool.get_farmer_bonus_amount(_ctx()))

        assert "Unable to fetch bonus amount details at the moment." in result
        assert "fetched successfully" not in result

    def test_missing_token_returns_provider_not_configured(self, monkeypatch):
        _patch_farmers(monkeypatch, [_farmer()])
        monkeypatch.setattr(bonus_tool, "get_config_value", lambda name: None)

        called = False

        async def unexpected(*args, **kwargs):
            nonlocal called
            called = True

        monkeypatch.setattr(bonus_tool, "get_farmer_bonus_amount_api", unexpected)
        result = asyncio.run(bonus_tool.get_farmer_bonus_amount(_ctx()))

        assert "Provider access is not configured" in result
        assert called is False

    def test_merges_records_across_accounts(self, monkeypatch):
        _patch_farmers(
            monkeypatch,
            [
                _farmer(),
                _farmer(societyCode="2005", farmerCode="0002"),
            ],
        )
        monkeypatch.setenv("PASHUGPT_TOKEN", "test-token")
        calls = []

        async def fake_api(request, token):
            calls.append(request.to_query_params())
            code = request.farmer_code
            return [
                FarmerBonusAmountRecordModel.model_validate(
                    {
                        "societyCode": request.society_code,
                        "societyName": f"SOC_{code}",
                        "farmerCode": code,
                        "farmerName": f"F_{code}",
                        "bonusAmount": 100 if code == "0001" else 200,
                        "fromDate": "2026-04-01T00:00:00",
                        "toDate": "2026-04-01T00:00:00",
                    }
                )
            ]

        monkeypatch.setattr(bonus_tool, "get_farmer_bonus_amount_api", fake_api)
        result = asyncio.run(bonus_tool.get_farmer_bonus_amount(_ctx()))

        assert len(calls) == 2
        assert "| 2026-04-01 - 2026-04-01 | SOC_0001 | F_0001 | 100.00 |" in result
        assert "| 2026-04-01 - 2026-04-01 | SOC_0002 | F_0002 | 200.00 |" in result

    def test_optional_label_fields_fallback_to_dashes(self, monkeypatch):
        _patch_farmers(monkeypatch, [_farmer()])
        monkeypatch.setenv("PASHUGPT_TOKEN", "test-token")

        async def partial_api(request, token):
            return [
                FarmerBonusAmountRecordModel(
                    society_code=None,
                    society_name=None,
                    farmer_code=None,
                    farmer_name=None,
                    bonus_amount=0,
                    from_date="2026-04-01T00:00:00",
                    to_date="2026-04-01T00:00:00",
                )
            ]

        monkeypatch.setattr(bonus_tool, "get_farmer_bonus_amount_api", partial_api)
        result = asyncio.run(bonus_tool.get_farmer_bonus_amount(_ctx()))

        assert "fetched successfully" in result
        assert "| 2026-04-01 - 2026-04-01 | - | - | 0.00 |" in result


class TestFarmerTicketSummaryTool:
    def test_success_formats_full_account_summary(self, monkeypatch):
        _patch_ticket_farmers(monkeypatch, [_farmer()])
        monkeypatch.setattr(ticket_tool, "get_config_value", lambda name: "test-token")

        async def fake_api(request, token):
            assert request.to_query_params() == {
                "unionCode": "0001", "societyCode": "2004", "farmerCode": "0001"
            }
            assert token == "test-token"
            return _ticket_summary()

        monkeypatch.setattr(ticket_tool, "get_farmer_ticket_summary_api", fake_api)
        result = asyncio.run(ticket_tool.get_farmer_ticket_summary(_ctx()))
        assert "### FARMER1 (0001)" in result
        assert "Ticket T-1 booked 20-Oct-2026" in result
        assert "Animal TAG-1" in result
        assert "Last AI 01-Sep-2026" in result
        assert "Last PD 01-Oct-2026" in result
        assert "Last calving 01-Dec-2025" in result

    def test_partial_success_is_returned_with_warning(self, monkeypatch):
        _patch_ticket_farmers(
            monkeypatch,
            [_farmer(), _farmer(societyCode="2005", farmerCode="0002")],
        )
        monkeypatch.setattr(ticket_tool, "get_config_value", lambda name: "test-token")

        async def fake_api(request, token):
            return _ticket_summary() if request.farmer_code == "0001" else None

        monkeypatch.setattr(ticket_tool, "get_farmer_ticket_summary_api", fake_api)
        result = asyncio.run(ticket_tool.get_farmer_ticket_summary(_ctx()))
        assert "Ticket T-1" in result
        assert "Some linked accounts could not be checked (1 of 2)" in result

    def test_all_failures_and_empty_success_are_distinct(self, monkeypatch):
        _patch_ticket_farmers(monkeypatch, [_farmer()])
        monkeypatch.setattr(ticket_tool, "get_config_value", lambda name: "test-token")

        async def failed(request, token):
            return None

        monkeypatch.setattr(ticket_tool, "get_farmer_ticket_summary_api", failed)
        assert "Unable to fetch ticket details" in asyncio.run(
            ticket_tool.get_farmer_ticket_summary(_ctx())
        )

        async def empty(request, token):
            return FarmerTicketSummaryResponseModel()

        monkeypatch.setattr(ticket_tool, "get_farmer_ticket_summary_api", empty)
        assert "No ticket, breeding, or calving records" in asyncio.run(
            ticket_tool.get_farmer_ticket_summary(_ctx())
        )

    def test_missing_identity_or_token_never_calls_provider(self, monkeypatch):
        called = False

        async def unexpected(*args, **kwargs):
            nonlocal called
            called = True

        monkeypatch.setattr(ticket_tool, "get_farmer_ticket_summary_api", unexpected)
        ctx = _ctx()
        ctx.deps.mobile = None
        assert "signed-in farmer profile" in asyncio.run(
            ticket_tool.get_farmer_ticket_summary(ctx)
        )
        assert called is False

        _patch_ticket_farmers(monkeypatch, [_farmer()])
        monkeypatch.setattr(ticket_tool, "get_config_value", lambda name: None)
        assert "Provider access is not configured" in asyncio.run(
            ticket_tool.get_farmer_ticket_summary(_ctx())
        )
        assert called is False
