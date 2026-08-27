from types import SimpleNamespace

import pytest

from agents.services import beckn_amul as adapter
from app.services.beckn_operations import BecknActionResult, OperationState


def _result(payload, state=OperationState.SUCCEEDED):
    return BecknActionResult(SimpleNamespace(state=state), payload)


def _tag(code, value=None, entries=None):
    result = {"descriptor": {"code": code}}
    if value is not None:
        result["value"] = value
    if entries is not None:
        result["list"] = entries
    return result


def _farmer_payload(accounts, global_tags=None):
    tags = [_tag("farmer_accounts", entries=account) for account in accounts]
    if global_tags:
        tags.append(_tag("animal_tags", entries=[_tag("tag_id", value=value) for value in global_tags]))
    return {
        "message": {
            "order": {
                "state": "COMPLETED",
                "fulfillments": [{"customer": {"person": {"tags": tags}}}],
            }
        }
    }


def test_farmer_mapper_preserves_account_level_owned_tags():
    payload = _farmer_payload([
        [
            _tag("union_name", "Banas"),
            _tag("union_code", "U1"),
            _tag("society_code", "S1"),
            _tag("farmer_code", "F1"),
            _tag("tag_id", "TAG-A"),
            _tag("tag_id", "TAG-B"),
        ],
        [
            _tag("union_name", "Kaira"),
            _tag("union_code", "U2"),
            _tag("society_code", "S2"),
            _tag("farmer_code", "F2"),
            _tag("tag_id", "TAG-C"),
        ],
    ], global_tags=["TAG-A", "TAG-B", "TAG-C"])

    farmers = adapter._farmer_models_from_payload(payload, authenticated_mobile="9000000000")

    assert farmers[0].animal_tags == ["TAG-A", "TAG-B"]
    assert farmers[1].animal_tags == ["TAG-C"]
    assert farmers[0].mobile_number == "9000000000"


def test_farmer_mapper_refuses_to_guess_global_tag_ownership_for_multiple_accounts():
    payload = _farmer_payload([
        [_tag("union_code", "U1"), _tag("society_code", "S1"), _tag("farmer_code", "F1")],
        [_tag("union_code", "U2"), _tag("society_code", "S2"), _tag("farmer_code", "F2")],
    ], global_tags=["TAG-A"])

    farmers = adapter._farmer_models_from_payload(payload, authenticated_mobile="9000000000")

    assert farmers[0].animal_tags is None
    assert farmers[1].animal_tags is None


@pytest.mark.asyncio
async def test_ai_technician_catalog_mapper(monkeypatch):
    payload = {
        "message": {"catalog": {"providers": [{"items": [{
            "id": "ait:TECH-1",
            "descriptor": {"name": "Technician One (AI technician)"},
            "tags": [_tag("technician-details", entries=[
                _tag("mobile", "9000000000"),
                _tag("technician_id", "TECH-1"),
            ])],
        }]}]}}
    }

    class Client:
        async def search_ai_technicians(self, **kwargs):
            assert kwargs["union_code"] == "U1"
            assert kwargs["society_code"] == "S1"
            return _result(payload)

    monkeypatch.setattr(adapter, "get_beckn_operation_client", lambda: Client())
    technicians = await adapter.search_ai_technicians(union_code="U1", society_code="S1")

    assert [record.model_dump() for record in technicians] == [{
        "userId": "TECH-1",
        "fullName": "Technician One",
        "mobileNumber": "9000000000",
    }]


@pytest.mark.asyncio
async def test_milk_callback_mapper(monkeypatch):
    payload = {"message": {"order": {"items": [{"tags": [
        _tag("query-period", entries=[_tag("result", "success")]),
        _tag("milk-record", entries=[
            _tag("date", "2026-08-01"), _tag("qty", "8.5"), _tag("fat", "6.2"),
        ]),
        _tag("deduction-record", entries=[
            _tag("date", "2026-08-01"), _tag("accountname", "Feed"), _tag("amount", "25"),
        ]),
    ]}]}}}

    class Client:
        async def init_milk_collection(self, **kwargs):
            return _result(payload)

    monkeypatch.setattr(adapter, "get_beckn_operation_client", lambda: Client())
    account = adapter.AuthenticatedFarmerAccount("U1", "S1", "F1")
    result = await adapter.fetch_milk_collection(
        account,
        fromdate="2026-08-01",
        todate="2026-08-01",
        session_id="session",
        tool_call_id="tool",
    )

    assert result.result == "success"
    assert result.milk[0].qty == 8.5
    assert result.deduction[0].account_name == "Feed"


def test_business_failure_is_not_reported_as_empty_data():
    with pytest.raises(adapter.BecknProviderUnavailable, match="provider unavailable"):
        adapter._completed_payload(
            _result(
                {"error": {"code": "50000", "message": "provider unavailable"}},
                OperationState.BUSINESS_FAILED,
            ),
            "farmer profile",
        )


@pytest.mark.asyncio
async def test_cvcc_children_are_associated_to_their_treatment(monkeypatch):
    payload = {"message": {"order": {"items": [{"tags": [
        _tag("treatment", entries=[_tag("treatment_index", "0"), _tag("treatment", "First")]),
        _tag("treatment", entries=[_tag("treatment_index", "1"), _tag("treatment", "Second")]),
        _tag("treatment_medicine", entries=[
            _tag("treatment_index", "1"), _tag("medicine_name", "Medicine B"),
        ]),
        _tag("fodder_detail", entries=[
            _tag("treatment_index", "0"), _tag("fodder_name", "Fodder A"),
        ]),
    ]}]}}}

    class Client:
        async def init_animal_profile(self, **kwargs):
            return _result(payload)

    monkeypatch.setattr(adapter, "get_beckn_operation_client", lambda: Client())
    result = await adapter.fetch_cvcc_health("TAG", union_code="U1")

    assert result is not None and result.data is not None
    assert result.data.treatment[0].fodder_detail[0].fodder_name == "Fodder A"
    assert result.data.treatment[0].medicine == []
    assert result.data.treatment[1].medicine[0].medicine_name == "Medicine B"


@pytest.mark.asyncio
async def test_banas_children_are_associated_to_their_visit(monkeypatch):
    payload = {"message": {"order": {"items": [{"tags": [
        _tag("operated_visit", entries=[_tag("visit_index", "0"), _tag("visit_code", "V-A")]),
        _tag("operated_visit", entries=[_tag("visit_index", "1"), _tag("visit_code", "V-B")]),
        _tag("visit_medicine", entries=[
            _tag("visit_index", "1"), _tag("medicine_name", "Medicine B"),
        ]),
        _tag("lab_report", entries=[
            _tag("visit_index", "0"), _tag("sample_name", "Sample A"),
        ]),
    ]}]}}}

    class Client:
        async def init_animal_profile(self, **kwargs):
            return _result(payload)

    monkeypatch.setattr(adapter, "get_beckn_operation_client", lambda: Client())
    visits = await adapter.fetch_banas_visits("TAG", union_code="U1")

    assert visits[0].visit_code == "V-A"
    assert visits[0].medicines == []
    assert visits[0].lab_reports[0].sample_name == "Sample A"
    assert visits[1].medicines[0].medicine_name == "Medicine B"
    assert visits[1].lab_reports == []
