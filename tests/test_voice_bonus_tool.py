import asyncio
import inspect
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.deps import FarmerAccount, FarmerContext
from agents.voice.tools import bonus as bonus_tool
from app.voice.models.bonus import FarmerBonusAmountRecordModel


def _ctx(accounts=None, mobile="9000000000"):
    deps = FarmerContext(
        query="what is my bonus",
        mobile=mobile,
        farmer_accounts=accounts or [],
        signed_in=bool(mobile),
    )
    return SimpleNamespace(deps=deps)


def _accounts(*specs):
    out = []
    for union, society, farmer in specs:
        out.append(
            FarmerAccount(
                union_code=union,
                society_code=society,
                farmer_code=farmer,
                society_name=f"SOC_{society}",
                farmer_name=f"F_{farmer}",
            )
        )
    return out


def _record(**overrides):
    data = {
        "societyCode": "2004",
        "societyName": "DEMO_2004",
        "farmerCode": "0001",
        "farmerName": "FARMER1",
        "bonusAmount": 1273.1,
        "fromDate": "2026-04-01T00:00:00",
        "toDate": "2026-04-01T00:00:00",
    }
    data.update(overrides)
    return FarmerBonusAmountRecordModel.model_validate(data)


class TestBonusTool:
    def test_success_uses_context_accounts_and_formats_plain_text(self, monkeypatch):
        monkeypatch.setenv("PASHUGPT_TOKEN", "test-token")

        async def fake_api(request, token):
            assert token == "test-token"
            assert request.to_query_params() == {
                "unionCode": "0001",
                "societyCode": "2004",
                "farmerCode": "0001",
            }
            return [_record()]

        monkeypatch.setattr(bonus_tool, "get_farmer_bonus_amount_api", fake_api)
        result = asyncio.run(
            bonus_tool.get_farmer_bonus_amount(
                _ctx(_accounts(("0001", "2004", "0001")))
            )
        )

        assert result.startswith("Bonus amount details fetched successfully:\n\n")
        assert "Bonus amount records (1):" in result
        assert "Period 2026-04-01 to 2026-04-01" in result
        assert "society DEMO_2004" in result
        assert "farmer FARMER1" in result
        assert "bonus amount 1273.1 rupees" in result
        assert "|" not in result  # voice: no markdown table

    def test_missing_authenticated_mobile_never_calls_backend(self, monkeypatch):
        called = False

        async def unexpected(*args, **kwargs):
            nonlocal called
            called = True

        monkeypatch.setattr(bonus_tool, "get_farmer_bonus_amount_api", unexpected)
        result = asyncio.run(bonus_tool.get_farmer_bonus_amount(_ctx(mobile=None)))

        assert "signed-in farmer profile" in result
        assert called is False

    def test_no_accounts_returns_clear_failure(self, monkeypatch):
        monkeypatch.setenv("PASHUGPT_TOKEN", "test-token")

        called = False

        async def unexpected(*args, **kwargs):
            nonlocal called
            called = True

        monkeypatch.setattr(bonus_tool, "get_farmer_bonus_amount_api", unexpected)
        result = asyncio.run(bonus_tool.get_farmer_bonus_amount(_ctx([])))

        assert "No union, society, and farmer account" in result
        assert called is False

    def test_empty_bonus_list_returns_no_records_message(self, monkeypatch):
        monkeypatch.setenv("PASHUGPT_TOKEN", "test-token")

        async def empty_api(request, token):
            return []

        monkeypatch.setattr(bonus_tool, "get_farmer_bonus_amount_api", empty_api)
        result = asyncio.run(
            bonus_tool.get_farmer_bonus_amount(
                _ctx(_accounts(("0001", "2004", "0001")))
            )
        )

        assert "No bonus records were found" in result

    def test_not_found_empty_result_is_not_amcs_failure(self, monkeypatch):
        monkeypatch.setenv("PASHUGPT_TOKEN", "test-token")

        async def not_found_as_empty(request, token):
            return []

        monkeypatch.setattr(bonus_tool, "get_farmer_bonus_amount_api", not_found_as_empty)
        result = asyncio.run(
            bonus_tool.get_farmer_bonus_amount(
                _ctx(_accounts(("0001", "2004", "0001")))
            )
        )

        assert "No bonus records were found" in result
        assert "AMCS" not in result
        assert "Unable to fetch bonus amount details" not in result

    def test_all_provider_failures_return_temporary_failure(self, monkeypatch):
        monkeypatch.setenv("PASHUGPT_TOKEN", "test-token")

        async def failed(request, token):
            return None

        monkeypatch.setattr(bonus_tool, "get_farmer_bonus_amount_api", failed)
        result = asyncio.run(
            bonus_tool.get_farmer_bonus_amount(
                _ctx(_accounts(("0001", "2004", "0001")))
            )
        )

        assert "Unable to fetch bonus amount details at the moment." in result
        assert "AMCS" in result

    def test_mixed_empty_and_failure_returns_temporary_failure(self, monkeypatch):
        monkeypatch.setenv("PASHUGPT_TOKEN", "test-token")

        async def mixed_api(request, token):
            if request.farmer_code == "0001":
                return []
            return None

        monkeypatch.setattr(bonus_tool, "get_farmer_bonus_amount_api", mixed_api)
        result = asyncio.run(
            bonus_tool.get_farmer_bonus_amount(
                _ctx(
                    _accounts(
                        ("0001", "2004", "0001"),
                        ("0001", "2005", "0002"),
                    )
                )
            )
        )

        assert "Unable to fetch bonus amount details at the moment." in result
        assert "No bonus records were found" not in result

    def test_mixed_data_and_failure_returns_temporary_failure(self, monkeypatch):
        monkeypatch.setenv("PASHUGPT_TOKEN", "test-token")

        async def mixed_api(request, token):
            if request.farmer_code == "0001":
                return [_record(bonusAmount=100, farmerCode="0001", farmerName="F_0001")]
            return None

        monkeypatch.setattr(bonus_tool, "get_farmer_bonus_amount_api", mixed_api)
        result = asyncio.run(
            bonus_tool.get_farmer_bonus_amount(
                _ctx(
                    _accounts(
                        ("0001", "2004", "0001"),
                        ("0001", "2005", "0002"),
                    )
                )
            )
        )

        assert "Unable to fetch bonus amount details at the moment." in result
        assert "fetched successfully" not in result

    def test_exception_during_fan_out_counts_as_failure(self, monkeypatch):
        monkeypatch.setenv("PASHUGPT_TOKEN", "test-token")

        async def boom(request, token):
            raise RuntimeError("upstream down")

        monkeypatch.setattr(bonus_tool, "get_farmer_bonus_amount_api", boom)
        result = asyncio.run(
            bonus_tool.get_farmer_bonus_amount(
                _ctx(_accounts(("0001", "2004", "0001")))
            )
        )

        assert "Unable to fetch bonus amount details at the moment." in result

    def test_missing_token_returns_not_configured(self, monkeypatch):
        monkeypatch.delenv("PASHUGPT_TOKEN", raising=False)

        called = False

        async def unexpected(*args, **kwargs):
            nonlocal called
            called = True

        monkeypatch.setattr(bonus_tool, "get_farmer_bonus_amount_api", unexpected)
        result = asyncio.run(
            bonus_tool.get_farmer_bonus_amount(
                _ctx(_accounts(("0001", "2004", "0001")))
            )
        )

        assert "Service is not configured" in result
        assert called is False

    def test_merges_records_across_accounts(self, monkeypatch):
        monkeypatch.setenv("PASHUGPT_TOKEN", "test-token")
        calls = []

        async def fake_api(request, token):
            calls.append(request.to_query_params())
            code = request.farmer_code
            return [
                _record(
                    societyCode=request.society_code,
                    societyName=f"SOC_{code}",
                    farmerCode=code,
                    farmerName=f"F_{code}",
                    bonusAmount=100 if code == "0001" else 200,
                )
            ]

        monkeypatch.setattr(bonus_tool, "get_farmer_bonus_amount_api", fake_api)
        result = asyncio.run(
            bonus_tool.get_farmer_bonus_amount(
                _ctx(
                    _accounts(
                        ("0001", "2004", "0001"),
                        ("0001", "2005", "0002"),
                    )
                )
            )
        )

        assert len(calls) == 2
        assert "Bonus amount records (2):" in result
        assert "farmer F_0001, bonus amount 100 rupees" in result
        assert "farmer F_0002, bonus amount 200 rupees" in result

    def test_optional_label_fields_fallback_to_unknown(self, monkeypatch):
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
        result = asyncio.run(
            bonus_tool.get_farmer_bonus_amount(
                _ctx(_accounts(("0001", "2004", "0001")))
            )
        )

        assert "fetched successfully" in result
        assert "society unknown society" in result
        assert "farmer unknown farmer" in result
        assert "bonus amount 0 rupees" in result


class TestBonusPrepareGuard:
    def test_prepare_hides_tool_without_authenticated_mobile(self):
        sentinel = object()
        assert (
            asyncio.run(
                bonus_tool.prepare_get_farmer_bonus_amount(
                    _ctx(mobile=None), sentinel
                )
            )
            is None
        )

    def test_prepare_shows_tool_for_authenticated_mobile_even_without_accounts(self):
        sentinel = object()
        assert (
            asyncio.run(bonus_tool.prepare_get_farmer_bonus_amount(_ctx([]), sentinel))
            is sentinel
        )

    def test_prepare_shows_tool_even_when_union_names_absent(self):
        """Regression for chat PR review: do not gate on farmer_unions/unionName."""
        sentinel = object()
        ctx = _ctx(_accounts(("0001", "2004", "0001")))
        ctx.deps.farmer_unions = []
        assert (
            asyncio.run(bonus_tool.prepare_get_farmer_bonus_amount(ctx, sentinel))
            is sentinel
        )

    def test_model_facing_signature_has_only_ctx(self):
        parameters = inspect.signature(bonus_tool.get_farmer_bonus_amount).parameters
        assert list(parameters) == ["ctx"]
        assert "union_code" not in parameters
        assert "society_code" not in parameters
        assert "farmer_code" not in parameters
