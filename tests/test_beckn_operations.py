import asyncio
import json
from typing import Any

import httpx
import pytest

from app.services.beckn_operations import (
    BecknOperationClient,
    BecknOperationStore,
    CallbackRecordResult,
    OperationState,
)


class MemoryRedis:
    def __init__(self):
        self.values: dict[str, str] = {}
        self.hashes: dict[str, dict[str, str]] = {}

    async def set(self, key, value, ex=None, nx=False):
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    async def get(self, key):
        return self.values.get(key)

    async def delete(self, key):
        self.values.pop(key, None)
        self.hashes.pop(key, None)

    async def hset(self, key, mapping=None):
        self.hashes.setdefault(key, {}).update({k: str(v) for k, v in (mapping or {}).items()})
        return len(mapping or {})

    async def hsetnx(self, key, field, value):
        row = self.hashes.setdefault(key, {})
        if field in row:
            return 0
        row[field] = value
        return 1

    async def hget(self, key, field):
        return self.hashes.get(key, {}).get(field)

    async def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    async def expire(self, key, ttl):
        return True


def _create(store: BecknOperationStore, *, idem="tool-call-1"):
    return store.create(
        operation_id="op-1",
        transaction_id="11111111-1111-4111-8111-111111111111",
        message_id="22222222-2222-4222-8222-222222222222",
        action="confirm",
        expected_callback="on_confirm",
        domain="services:amul-vet-booking",
        bap_id="bap.amul-net.internal",
        bpp_id="bpp-booking.amul-net.internal",
        request_payload={"context": {"action": "confirm"}, "message": {}},
        idempotency_key=idem,
        session_id="session-1",
        tool_call_id="tool-call-1",
    )


def _callback(ticket="T-1") -> dict[str, Any]:
    return {
        "context": {
            "domain": "services:amul-vet-booking",
            "action": "on_confirm",
            "version": "1.1.0",
            "bap_id": "bap.amul-net.internal",
            "bpp_id": "bpp-booking.amul-net.internal",
            "transaction_id": "11111111-1111-4111-8111-111111111111",
            "message_id": "22222222-2222-4222-8222-222222222222",
        },
        "message": {"order": {"id": ticket, "status": "ACTIVE"}},
    }


@pytest.mark.asyncio
async def test_operation_idempotency_reuses_the_existing_transaction():
    store = BecknOperationStore(MemoryRedis(), ttl_seconds=3600)
    first = await _create(store)
    second = await _create(store)

    assert first.created is True
    assert second.created is False
    assert second.operation.transaction_id == first.operation.transaction_id
    assert second.operation.message_id == first.operation.message_id


@pytest.mark.asyncio
async def test_callback_before_ack_remains_terminal_and_duplicates_are_idempotent():
    store = BecknOperationStore(MemoryRedis(), ttl_seconds=3600)
    created = await _create(store)

    accepted = await store.record_callback(_callback())
    await store.mark_ack(created.operation, {"message": {"ack": {"status": "ACK"}}})
    duplicate = await store.record_callback(_callback())

    operation = await store.get(created.operation.transaction_id, created.operation.message_id)
    assert accepted.accepted and not accepted.duplicate
    assert duplicate.accepted and duplicate.duplicate
    assert operation is not None
    assert operation.state is OperationState.SUCCEEDED
    assert operation.callback["message"]["order"]["id"] == "T-1"


@pytest.mark.asyncio
async def test_different_second_callback_is_rejected_without_overwrite():
    store = BecknOperationStore(MemoryRedis(), ttl_seconds=3600)
    created = await _create(store)
    assert (await store.record_callback(_callback("T-1"))).accepted

    conflict = await store.record_callback(_callback("T-2"))
    operation = await store.get(created.operation.transaction_id, created.operation.message_id)

    assert not conflict.accepted
    assert conflict.code == "CALLBACK_CONFLICT"
    assert operation.callback["message"]["order"]["id"] == "T-1"


@pytest.mark.asyncio
async def test_timeout_is_pending_and_late_callback_is_retained():
    store = BecknOperationStore(MemoryRedis(), ttl_seconds=3600)
    created = await _create(store)
    await store.mark_ack(created.operation, {"message": {"ack": {"status": "ACK"}}})

    timed_out = await store.wait(created.operation, timeout_seconds=0)
    assert timed_out.state is OperationState.TIMED_OUT_PENDING

    assert (await store.record_callback(_callback())).accepted
    recovered = await store.get(created.operation.transaction_id, created.operation.message_id)
    assert recovered.state is OperationState.SUCCEEDED


@pytest.mark.asyncio
async def test_callback_correlation_checks_action_domain_and_participants():
    store = BecknOperationStore(MemoryRedis(), ttl_seconds=3600)
    await _create(store)
    payload = _callback()
    payload["context"]["bpp_id"] = "attacker.example"

    result = await store.record_callback(payload)
    assert not result.accepted
    assert result.code == "CORRELATION_MISMATCH"


@pytest.mark.asyncio
async def test_unknown_shc_callback_orphan_redacts_private_report(monkeypatch):
    from app.services import beckn_operations as module

    monkeypatch.setattr(module.settings, "shc_artifact_ttl_seconds", 600)
    redis = MemoryRedis()
    store = BecknOperationStore(redis, ttl_seconds=3600)
    payload = {
        "context": {
            "domain": "schemes:vistaar",
            "action": "on_init",
            "transaction_id": "unknown-txn",
            "message_id": "unknown-msg",
        },
        "message": {"order": {"private_html": "farmer report"}},
    }

    result = await store.record_callback(payload)

    assert result.code == "UNKNOWN_TRANSACTION"
    orphan = json.loads(redis.values[store._orphan_key("unknown-txn", "unknown-msg")])
    assert "payload" not in orphan
    assert "payload_hash" in orphan
    assert "farmer report" not in json.dumps(orphan)


class CallbackDuringPostClient:
    def __init__(self, store: BecknOperationStore):
        self.store = store
        self.payload = None

    async def post(self, url, json=None):
        self.payload = json
        ctx = json["context"]
        callback = {
            "context": {
                **ctx,
                "action": "on_confirm",
            },
            "message": {"order": {"id": "BOOK-42", "status": "ACTIVE"}},
        }
        assert (await self.store.record_callback(callback)).accepted
        return httpx.Response(
            200,
            json={"message": {"ack": {"status": "ACK"}}},
            request=httpx.Request("POST", url),
        )


class ShcCallbackDuringPostClient:
    def __init__(self, store: BecknOperationStore):
        self.store = store
        self.payload = None
        self.url = None

    async def post(self, url, json=None):
        self.url = url
        self.payload = json
        ctx = json["context"]
        callback = {
            "context": {**ctx, "action": "on_init"},
            "message": {"order": {"providers": []}},
        }
        assert (await self.store.record_callback(callback)).accepted
        return httpx.Response(
            200,
            json={"message": {"ack": {"status": "ACK"}}},
            request=httpx.Request("POST", url),
        )


@pytest.mark.asyncio
async def test_confirm_builds_directed_core_order_and_correlates_fast_callback(monkeypatch):
    from app.services import beckn_operations as module

    monkeypatch.setattr(module.settings, "beckn_bap_caller_url", "http://onix/bap/caller")
    monkeypatch.setattr(module.settings, "beckn_bap_uri", "https://bap.example/bap/receiver")
    monkeypatch.setattr(module.settings, "beckn_booking_bpp_uri", "https://bpp.example/bpp/receiver")
    monkeypatch.setattr(module.settings, "beckn_callback_wait_seconds", 0.2)

    store = BecknOperationStore(MemoryRedis(), ttl_seconds=3600)
    http_client = CallbackDuringPostClient(store)
    client = BecknOperationClient(store, http_client=http_client)
    result = await client.confirm_booking(
        service="ai-call",
        union_code="U",
        society_code="S",
        farmer_code="F",
        species="cow",
        session_id="session-1",
        tool_call_id="tool-call-1",
        technician_id="TECH-1",
    )

    assert result.ok
    assert result.payload["message"]["order"]["id"] == "BOOK-42"
    sent = http_client.payload
    assert sent["context"]["action"] == "confirm"
    assert sent["context"]["bpp_id"] == "bpp-booking.amul-net.internal"
    assert "bpp_uri" in sent["context"]
    assert "fulfillments" in sent["message"]["order"]
    assert "fulfillment" not in sent["message"]["order"]
    assert json.loads(json.dumps(sent))["message"]["order"]["items"][0]["id"] == "ait:TECH-1"


@pytest.mark.asyncio
async def test_health_confirm_uses_health_provider_and_veterinary_fulfillment(monkeypatch):
    from app.services import beckn_operations as module

    monkeypatch.setattr(module.settings, "beckn_bap_caller_url", "http://onix/bap/caller")
    monkeypatch.setattr(module.settings, "beckn_bap_uri", "https://bap.example/bap/receiver")
    monkeypatch.setattr(module.settings, "beckn_booking_bpp_uri", "https://bpp.example/bpp/receiver")
    monkeypatch.setattr(module.settings, "beckn_callback_wait_seconds", 0.2)

    store = BecknOperationStore(MemoryRedis(), ttl_seconds=3600)
    http_client = CallbackDuringPostClient(store)
    client = BecknOperationClient(store, http_client=http_client)
    result = await client.confirm_booking(
        service="health-call",
        union_code="U",
        society_code="S",
        farmer_code="F",
        species="buffalo",
        session_id="session-1",
        tool_call_id="tool-call-2",
        case_type="emergency",
        remark="not eating",
    )

    assert result.ok
    order = http_client.payload["message"]["order"]
    assert order["provider"]["id"] == "amul-animal-health-service"
    assert order["items"] == [{"id": "health-call"}]
    assert order["fulfillments"][0]["type"] == "VETERINARY_VISIT"
    assert order["fulfillments"][0]["tags"][0]["descriptor"]["code"] == "health-call-details"


@pytest.mark.asyncio
async def test_shc_init_is_directed_and_waits_for_on_init(monkeypatch):
    from app.services import beckn_operations as module

    monkeypatch.setattr(module.settings, "beckn_bap_caller_url", "http://onix/bap/caller")
    monkeypatch.setattr(module.settings, "beckn_bap_uri", "https://bap.example/bap/receiver")
    monkeypatch.setattr(module.settings, "vistaar_bpp_id", "provider-network-vistaar.da.gov.in")
    monkeypatch.setattr(module.settings, "vistaar_bpp_uri", "https://provider-network-vistaar.da.gov.in")
    monkeypatch.setattr(module.settings, "beckn_callback_wait_seconds", 0.2)
    monkeypatch.setattr(module.settings, "shc_artifact_ttl_seconds", 600)

    store = BecknOperationStore(MemoryRedis(), ttl_seconds=3600)
    http_client = ShcCallbackDuringPostClient(store)
    client = BecknOperationClient(store, http_client=http_client)
    result = await client.init_soil_health_card(
        mobile="+919924457046",
        cycle="2024-25",
        session_id="session-1",
        tool_call_id="tool-shc",
    )

    assert result.ok
    assert http_client.url == "http://onix/bap/caller/init"
    sent = http_client.payload
    assert sent["context"]["action"] == "init"
    assert sent["context"]["domain"] == "schemes:vistaar"
    assert sent["context"]["bpp_id"] == "provider-network-vistaar.da.gov.in"
    order = sent["message"]["order"]
    assert order["provider"] == {"id": "shc-discovery"}
    assert order["items"] == [{"id": "soil-health-card"}]
    assert order["fulfillments"][0]["customer"]["contact"]["phone"] == "+919924457046"
    assert order["fulfillments"][0]["customer"]["person"]["tags"][0]["value"] == "2024-25"


@pytest.mark.asyncio
async def test_callback_router_is_default_off_and_acks_only_after_store(monkeypatch):
    from app.routers import beckn as router_module

    monkeypatch.setattr(router_module.settings, "beckn_callback_transactions_enabled", False)
    monkeypatch.setattr(router_module.settings, "vistaar_shc_enabled", False)
    disabled = await router_module.receive_callback("on_confirm", _callback(), None)
    assert disabled.status_code == 503
    assert json.loads(disabled.body)["message"]["ack"]["status"] == "NACK"

    calls = {"n": 0}

    class Store:
        async def record_callback(self, payload):
            calls["n"] += 1
            return CallbackRecordResult(True)

    monkeypatch.setattr(router_module.settings, "beckn_callback_transactions_enabled", True)
    monkeypatch.setattr(router_module.settings, "beckn_callback_token", "secret")
    monkeypatch.setattr(router_module, "get_beckn_operation_store", lambda: Store())
    accepted = await router_module.receive_callback("on_confirm", _callback(), "secret")

    assert accepted.status_code == 200
    assert json.loads(accepted.body)["message"]["ack"]["status"] == "ACK"
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_shc_callback_gate_does_not_enable_booking_callbacks(monkeypatch):
    from app.routers import beckn as router_module

    calls = {"n": 0}

    class Store:
        async def record_callback(self, payload):
            calls["n"] += 1
            return CallbackRecordResult(True)

    monkeypatch.setattr(router_module.settings, "beckn_callback_transactions_enabled", False)
    monkeypatch.setattr(router_module.settings, "vistaar_shc_enabled", True)
    monkeypatch.setattr(router_module.settings, "beckn_callback_token", None)
    monkeypatch.setattr(router_module, "get_beckn_operation_store", lambda: Store())

    booking = await router_module.receive_callback("on_confirm", _callback(), None)
    assert booking.status_code == 503

    payload = _callback()
    payload["context"].update({"domain": "schemes:vistaar", "action": "on_init"})
    shc_callback = await router_module.receive_callback("on_init", payload, None)
    assert shc_callback.status_code == 200
    assert json.loads(shc_callback.body)["message"]["ack"]["status"] == "ACK"
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_callback_router_returns_protocol_nack_without_transport_failure(monkeypatch):
    from app.routers import beckn as router_module

    class Store:
        async def record_callback(self, payload):
            return CallbackRecordResult(
                False,
                code="UNKNOWN_TRANSACTION",
                message="No matching Beckn operation",
            )

    monkeypatch.setattr(router_module.settings, "beckn_callback_transactions_enabled", True)
    monkeypatch.setattr(router_module.settings, "beckn_callback_token", None)
    monkeypatch.setattr(router_module, "get_beckn_operation_store", lambda: Store())
    rejected = await router_module.receive_callback("on_confirm", _callback(), None)

    assert rejected.status_code == 200
    body = json.loads(rejected.body)
    assert body["message"]["ack"]["status"] == "NACK"
    assert body["error"]["code"] == "UNKNOWN_TRANSACTION"
