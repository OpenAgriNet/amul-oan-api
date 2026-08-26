import base64
import inspect
from types import SimpleNamespace

import pytest
from pydantic_ai import Tool

from agents.deps import FarmerContext
from agents.tools import vistaar_shc as shc
from app.chat_artifacts import CHAT_ARTIFACTS_END, CHAT_ARTIFACTS_START, encode_chat_artifacts
from app.services.beckn_operations import (
    BecknActionResult,
    BecknOperation,
    OperationState,
)


def _operation(state: OperationState = OperationState.SUCCEEDED) -> BecknOperation:
    return BecknOperation(
        operation_id="op-1",
        transaction_id="11111111-1111-4111-8111-111111111111",
        message_id="22222222-2222-4222-8222-222222222222",
        action="init",
        expected_callback="on_init",
        domain="schemes:vistaar",
        bap_id="bap.amul-net.internal",
        bpp_id="provider-network-vistaar.da.gov.in",
        session_id="session-1",
        tool_call_id="tool-1",
        request_hash="hash",
        idempotency_key="idem",
        state=state,
    )


def _callback(html: str) -> dict:
    encoded = base64.b64encode(html.encode()).decode()
    return {
        "message": {
            "order": {
                "providers": [
                    {
                        "id": "shc-discovery",
                        "items": [
                            {
                                "id": "soil-health-card",
                                "media": [
                                    {
                                        "mimetype": "text/html",
                                        "url": f"data:text/html;base64,{encoded}",
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
        }
    }


class FakeClient:
    def __init__(self, result):
        self.result = result
        self.kwargs = None

    async def init_soil_health_card(self, **kwargs):
        self.kwargs = kwargs
        return self.result


def _ctx(mobile="9924457046"):
    deps = FarmerContext(
        query="show my card",
        mobile=mobile,
        session_id="session-1",
        supports_rich_artifacts=True,
    )
    return SimpleNamespace(deps=deps, tool_call_id="tool-1")


@pytest.mark.asyncio
async def test_tool_uses_independent_shc_feature_gate(monkeypatch):
    tool_def = SimpleNamespace(name="get_vistaar_soil_health_card")
    monkeypatch.setattr(shc.settings, "beckn_callback_transactions_enabled", False)
    monkeypatch.setattr(shc.settings, "vistaar_shc_enabled", True)
    assert await shc.prepare_get_vistaar_soil_health_card(_ctx(), tool_def) is tool_def

    monkeypatch.setattr(shc.settings, "vistaar_shc_enabled", False)
    assert await shc.prepare_get_vistaar_soil_health_card(_ctx(), tool_def) is None


def test_decodes_html_from_callback_and_sync_wrapper():
    html = "<!doctype html><html><body><table><tr><td>pH 7</td></tr></table></body></html>"
    callback = _callback(html)
    assert shc._decode_html_media(callback) == html
    assert shc._decode_html_media({"responses": [callback]}) == html


def test_rejects_invalid_cycle_and_phone_is_not_a_model_argument():
    assert shc._normalize_cycle("2024-25") == "2024-25"
    assert shc._normalize_cycle("2024-26") is None
    assert shc._registered_mobile("+91 99244 57046") == "+919924457046"
    Tool(
        shc.get_vistaar_soil_health_card,
        takes_ctx=True,
        docstring_format="auto",
        require_parameter_descriptions=True,
    )
    model_arguments = list(inspect.signature(shc.get_vistaar_soil_health_card).parameters)[1:]
    assert model_arguments == ["cycle"]


@pytest.mark.asyncio
async def test_success_attaches_html_outside_the_tool_text(monkeypatch):
    html = "<html><body><h1>Soil Health Card</h1><p>Private report</p></body></html>"
    client = FakeClient(BecknActionResult(_operation(), _callback(html)))
    monkeypatch.setattr(shc, "get_beckn_operation_client", lambda: client)
    ctx = _ctx()

    result = await shc.get_vistaar_soil_health_card(ctx, "2024-25")

    assert "attached" in result
    assert "Private report" not in result
    assert client.kwargs["mobile"] == "+919924457046"
    artifact = ctx.deps.take_chat_artifacts()[0]
    assert artifact["kind"] == "soil_health_card"
    assert artifact["content"] == html
    assert artifact["cycle"] == "2024-25"


@pytest.mark.asyncio
async def test_missing_profile_never_asks_for_a_phone_or_calls_network(monkeypatch):
    monkeypatch.setattr(
        shc,
        "get_beckn_operation_client",
        lambda: (_ for _ in ()).throw(AssertionError("network must not be called")),
    )
    result = await shc.get_vistaar_soil_health_card(_ctx(mobile=None), "2024-25")
    assert "signed-in account" in result
    assert "Do not ask them to type" in result


def test_artifact_frame_base64_is_delimiter_safe():
    artifact = {"id": "a", "content": f"inside {CHAT_ARTIFACTS_END}"}
    frame = encode_chat_artifacts([artifact])
    assert frame.startswith(CHAT_ARTIFACTS_START)
    assert frame.count(CHAT_ARTIFACTS_END) == 1


def test_farmer_context_artifacts_are_deduplicated_and_consumed_once():
    deps = FarmerContext(query="q")
    deps.add_chat_artifact({"id": "same", "content": "old"})
    deps.add_chat_artifact({"id": "same", "content": "new"})
    assert deps.take_chat_artifacts() == [{"id": "same", "content": "new"}]
    assert deps.take_chat_artifacts() == []
