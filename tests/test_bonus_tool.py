import asyncio
from types import SimpleNamespace

from agents.tools import bonus as bonus_tool
from app.models.bonus import FarmerBonusAmountRecordModel
from app.models.farmer import FarmerModel


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

    def test_all_provider_failures_return_temporary_failure(self, monkeypatch):
        _patch_farmers(monkeypatch, [_farmer()])
        monkeypatch.setenv("PASHUGPT_TOKEN", "test-token")

        async def failed(request, token):
            return None

        monkeypatch.setattr(bonus_tool, "get_farmer_bonus_amount_api", failed)
        result = asyncio.run(bonus_tool.get_farmer_bonus_amount(_ctx()))

        assert "Unable to fetch bonus amount details at the moment." in result
        assert "AMCS" in result

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

    def test_partial_rows_are_none_safe(self, monkeypatch):
        _patch_farmers(monkeypatch, [_farmer()])
        monkeypatch.setenv("PASHUGPT_TOKEN", "test-token")

        async def partial_api(request, token):
            return [
                FarmerBonusAmountRecordModel(
                    society_code=None,
                    society_name=None,
                    farmer_code=None,
                    farmer_name=None,
                    bonus_amount=None,
                    from_date=None,
                    to_date=None,
                )
            ]

        monkeypatch.setattr(bonus_tool, "get_farmer_bonus_amount_api", partial_api)
        result = asyncio.run(bonus_tool.get_farmer_bonus_amount(_ctx()))

        assert "fetched successfully" in result
        assert "| - - - | - | - | - |" in result
