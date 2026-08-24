"""Durable Beckn action/callback correlation for business tools.

The Beckn wire is asynchronous: a forward action receives only an ACK/NACK
and the business result arrives later on the paired ``on_*`` callback.  This
module keeps that protocol state in Redis so callbacks survive worker changes,
HTTP timeouts, and process restarts.

The store is intentionally append-like.  ACK, NACK, timeout, and callback
records occupy separate hash fields and ``state`` is derived with terminal
events taking precedence.  A callback that races the initial HTTP ACK can
therefore never be overwritten by a later ``mark_ack`` call.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional

import httpx

from app.config import settings
from app.core.cache import redis_client
from helpers.utils import get_logger

logger = get_logger(__name__)

_TERMINAL_STATES = {"SUCCEEDED", "BUSINESS_FAILED", "NACKED"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


class OperationState(str, Enum):
    CREATED = "CREATED"
    SENT = "SENT"
    ACKED_WAITING_CALLBACK = "ACKED_WAITING_CALLBACK"
    SUCCEEDED = "SUCCEEDED"
    BUSINESS_FAILED = "BUSINESS_FAILED"
    NACKED = "NACKED"
    TIMED_OUT_PENDING = "TIMED_OUT_PENDING"


@dataclass(frozen=True)
class BecknOperation:
    operation_id: str
    transaction_id: str
    message_id: str
    action: str
    expected_callback: str
    domain: str
    bap_id: str
    bpp_id: str
    session_id: Optional[str]
    tool_call_id: Optional[str]
    request_hash: str
    idempotency_key: str
    state: OperationState
    callback: Optional[dict[str, Any]] = None
    nack: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class CreateOperationResult:
    operation: BecknOperation
    created: bool


@dataclass(frozen=True)
class CallbackRecordResult:
    accepted: bool
    duplicate: bool = False
    code: Optional[str] = None
    message: Optional[str] = None


@dataclass(frozen=True)
class BecknActionResult:
    operation: BecknOperation
    payload: Optional[dict[str, Any]]
    authoritative_rejection: bool = False

    @property
    def pending(self) -> bool:
        return self.operation.state is OperationState.TIMED_OUT_PENDING

    @property
    def ok(self) -> bool:
        return self.operation.state is OperationState.SUCCEEDED


class BecknOperationStore:
    """Redis-backed operation inbox with idempotent callback delivery."""

    def __init__(self, redis: Any = redis_client, *, ttl_seconds: Optional[int] = None):
        self._redis = redis
        self._ttl = ttl_seconds or settings.beckn_operation_ttl_seconds
        self._prefix = f"{settings.redis_key_prefix}beckn-op"

    def _key(self, transaction_id: str, message_id: str) -> str:
        return f"{self._prefix}:{transaction_id}:{message_id}"

    def _idempotency_key(self, domain: str, idempotency_key: str) -> str:
        digest = hashlib.sha256(f"{domain}:{idempotency_key}".encode("utf-8")).hexdigest()
        return f"{self._prefix}:idem:{digest}"

    def _orphan_key(self, transaction_id: str, message_id: str) -> str:
        return f"{self._prefix}:orphan:{transaction_id}:{message_id}"

    async def create(
        self,
        *,
        operation_id: str,
        transaction_id: str,
        message_id: str,
        action: str,
        expected_callback: str,
        domain: str,
        bap_id: str,
        bpp_id: str,
        request_payload: Mapping[str, Any],
        idempotency_key: str,
        session_id: Optional[str] = None,
        tool_call_id: Optional[str] = None,
    ) -> CreateOperationResult:
        key = self._key(transaction_id, message_id)
        idem_key = self._idempotency_key(domain, idempotency_key)

        # Claim the business idempotency key before any forward action is sent.
        # Redis failure is deliberately not swallowed: a side effect must not be
        # submitted when its durable correlation/idempotency record is absent.
        claimed = await self._redis.set(idem_key, key, ex=self._ttl, nx=True)
        if not claimed:
            existing_key = await self._redis.get(idem_key)
            if existing_key:
                existing = await self._get_by_key(existing_key)
                if existing is not None:
                    return CreateOperationResult(existing, created=False)
            raise RuntimeError("Beckn idempotency record exists without an operation")

        created_at = _now()
        mapping = {
            "operation_id": operation_id,
            "transaction_id": transaction_id,
            "message_id": message_id,
            "action": action,
            "expected_callback": expected_callback,
            "domain": domain,
            "bap_id": bap_id,
            "bpp_id": bpp_id,
            "session_id": session_id or "",
            "tool_call_id": tool_call_id or "",
            "request_hash": _hash(request_payload),
            "idempotency_key": idempotency_key,
            "created_at": created_at,
            "updated_at": created_at,
        }
        try:
            await self._redis.hset(key, mapping=mapping)
            await self._redis.expire(key, self._ttl)
        except Exception:
            # The action has not been sent yet, so releasing our incomplete
            # idempotency claim is safe and lets an operator retry after Redis
            # recovers.
            await self._redis.delete(idem_key)
            raise
        operation = await self._get_by_key(key)
        if operation is None:  # pragma: no cover - defensive Redis inconsistency
            raise RuntimeError("Failed to read newly-created Beckn operation")
        return CreateOperationResult(operation, created=True)

    async def mark_sent(self, operation: BecknOperation) -> None:
        await self._redis.hset(
            self._key(operation.transaction_id, operation.message_id),
            mapping={"sent_at": _now(), "updated_at": _now()},
        )

    async def mark_ack(self, operation: BecknOperation, ack: Mapping[str, Any]) -> None:
        await self._redis.hset(
            self._key(operation.transaction_id, operation.message_id),
            mapping={"ack_record": _json(ack), "ack_received_at": _now(), "updated_at": _now()},
        )

    async def mark_nack(self, operation: BecknOperation, nack: Mapping[str, Any]) -> None:
        await self._redis.hset(
            self._key(operation.transaction_id, operation.message_id),
            mapping={"nack_record": _json(nack), "nack_received_at": _now(), "updated_at": _now()},
        )

    async def mark_transport_error(self, operation: BecknOperation, error: str) -> None:
        await self._redis.hset(
            self._key(operation.transaction_id, operation.message_id),
            mapping={"transport_error": error[:1000], "updated_at": _now()},
        )

    async def mark_timeout(self, operation: BecknOperation) -> None:
        await self._redis.hset(
            self._key(operation.transaction_id, operation.message_id),
            mapping={"timed_out_at": _now(), "updated_at": _now()},
        )

    async def get(self, transaction_id: str, message_id: str) -> Optional[BecknOperation]:
        return await self._get_by_key(self._key(transaction_id, message_id))

    async def _get_by_key(self, key: str) -> Optional[BecknOperation]:
        row = await self._redis.hgetall(key)
        if not row:
            return None
        callback_record = json.loads(row["callback_record"]) if row.get("callback_record") else None
        callback = callback_record.get("payload") if callback_record else None
        nack = json.loads(row["nack_record"]) if row.get("nack_record") else None

        if callback is not None:
            state = OperationState.BUSINESS_FAILED if callback.get("error") else OperationState.SUCCEEDED
        elif nack is not None:
            state = OperationState.NACKED
        elif row.get("timed_out_at"):
            state = OperationState.TIMED_OUT_PENDING
        elif row.get("ack_record"):
            state = OperationState.ACKED_WAITING_CALLBACK
        elif row.get("sent_at"):
            state = OperationState.SENT
        else:
            state = OperationState.CREATED

        return BecknOperation(
            operation_id=row["operation_id"],
            transaction_id=row["transaction_id"],
            message_id=row["message_id"],
            action=row["action"],
            expected_callback=row["expected_callback"],
            domain=row["domain"],
            bap_id=row["bap_id"],
            bpp_id=row["bpp_id"],
            session_id=row.get("session_id") or None,
            tool_call_id=row.get("tool_call_id") or None,
            request_hash=row["request_hash"],
            idempotency_key=row["idempotency_key"],
            state=state,
            callback=callback,
            nack=nack,
        )

    async def record_callback(self, payload: Mapping[str, Any]) -> CallbackRecordResult:
        context = payload.get("context") if isinstance(payload, Mapping) else None
        if not isinstance(context, Mapping):
            return CallbackRecordResult(False, code="INVALID_CONTEXT", message="context is required")
        transaction_id = context.get("transaction_id")
        message_id = context.get("message_id")
        action = context.get("action")
        if not all(isinstance(v, str) and v for v in (transaction_id, message_id, action)):
            return CallbackRecordResult(
                False,
                code="INVALID_CONTEXT",
                message="transaction_id, message_id and action are required",
            )

        key = self._key(transaction_id, message_id)
        row = await self._redis.hgetall(key)
        if not row:
            orphan = {"received_at": _now(), "payload": payload}
            await self._redis.set(self._orphan_key(transaction_id, message_id), _json(orphan), ex=self._ttl, nx=True)
            return CallbackRecordResult(False, code="UNKNOWN_TRANSACTION", message="No matching Beckn operation")

        checks = {
            "action": (row["expected_callback"], action),
            "domain": (row["domain"], context.get("domain")),
            "bap_id": (row["bap_id"], context.get("bap_id")),
            "bpp_id": (row["bpp_id"], context.get("bpp_id")),
        }
        for field, (expected, actual) in checks.items():
            if expected and expected != actual:
                return CallbackRecordResult(False, code="CORRELATION_MISMATCH", message=f"Unexpected {field}")

        callback_record = {"hash": _hash(payload), "received_at": _now(), "payload": payload}
        inserted = await self._redis.hsetnx(key, "callback_record", _json(callback_record))
        if inserted:
            await self._redis.hset(key, mapping={"updated_at": _now()})
            await self._redis.expire(key, self._ttl)
            await self._redis.expire(
                self._idempotency_key(row["domain"], row["idempotency_key"]), self._ttl
            )
            return CallbackRecordResult(True)

        existing = await self._redis.hget(key, "callback_record")
        existing_hash = (json.loads(existing) if existing else {}).get("hash")
        if existing_hash == callback_record["hash"]:
            return CallbackRecordResult(True, duplicate=True)
        return CallbackRecordResult(False, code="CALLBACK_CONFLICT", message="A different callback is already stored")

    async def wait(self, operation: BecknOperation, timeout_seconds: float) -> BecknOperation:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, timeout_seconds)
        while True:
            current = await self.get(operation.transaction_id, operation.message_id)
            if current is None:  # pragma: no cover - operation TTL misconfiguration
                raise RuntimeError("Beckn operation expired while awaiting callback")
            if current.state.value in _TERMINAL_STATES:
                return current
            remaining = deadline - loop.time()
            if remaining <= 0:
                await self.mark_timeout(current)
                return (await self.get(current.transaction_id, current.message_id)) or current
            await asyncio.sleep(min(settings.beckn_callback_poll_interval_seconds, remaining))


class BecknOperationClient:
    """BAP-side facade for directed actions through ONIX."""

    def __init__(self, store: BecknOperationStore, http_client: Optional[httpx.AsyncClient] = None):
        self.store = store
        self._http_client = http_client

    async def confirm_booking(
        self,
        *,
        service: str,
        union_code: str,
        society_code: str,
        farmer_code: str,
        species: str,
        session_id: Optional[str],
        tool_call_id: Optional[str],
        technician_id: Optional[str] = None,
        case_type: Optional[str] = None,
        remark: Optional[str] = None,
    ) -> BecknActionResult:
        self._validate_configuration()
        if service not in {"ai-call", "health-call"}:
            raise ValueError(f"Unsupported booking service: {service}")
        operation_id = str(uuid.uuid4())
        transaction_id = str(uuid.uuid4())
        message_id = str(uuid.uuid4())
        # Pydantic AI preserves tool_call_id when retrying/replaying the same tool
        # invocation.  A missing id deliberately gets a fresh nonce so two
        # legitimate identical bookings are not collapsed.
        invocation_id = tool_call_id or secrets.token_urlsafe(18)
        idempotency_key = f"{session_id or 'no-session'}:{invocation_id}:{service}"
        is_ai_call = service == "ai-call"
        item_id = f"ait:{technician_id}" if is_ai_call else "health-call"
        provider_id = "amul-ai-service" if is_ai_call else "amul-animal-health-service"
        fulfillment_type = "TECHNICIAN_VISIT" if is_ai_call else "VETERINARY_VISIT"
        tag_group_code = "booking-details" if is_ai_call else "health-call-details"

        private_tags = [
            {"descriptor": {"code": "farmer_code"}, "value": farmer_code},
            {"descriptor": {"code": "union_code"}, "value": union_code},
            {"descriptor": {"code": "species"}, "value": species},
        ]
        if case_type:
            private_tags.append({"descriptor": {"code": "case_type"}, "value": case_type})
        if remark:
            private_tags.append({"descriptor": {"code": "remark"}, "value": remark})

        order = {
            "provider": {"id": provider_id},
            "items": [{"id": item_id}],
            "fulfillments": [
                {
                    "id": "fulfillment-1",
                    "type": fulfillment_type,
                    "customer": {"person": {"id": f"farmer:{farmer_code}"}},
                    "stops": [{"location": {"descriptor": {"code": f"society:{society_code}"}}}],
                    "tags": [
                        {
                            "descriptor": {"code": tag_group_code},
                            "list": private_tags,
                        }
                    ],
                }
            ],
            "tags": [
                {
                    "descriptor": {"code": "client-reference"},
                    "list": [{"descriptor": {"code": "idempotency_key"}, "value": operation_id}],
                }
            ],
        }
        context = self._context(
            action="confirm", transaction_id=transaction_id, message_id=message_id
        )
        payload = {"context": context, "message": {"order": order}}
        created = await self.store.create(
            operation_id=operation_id,
            transaction_id=transaction_id,
            message_id=message_id,
            action="confirm",
            expected_callback="on_confirm",
            domain=settings.beckn_booking_domain,
            bap_id=settings.beckn_bap_id,
            bpp_id=settings.beckn_booking_bpp_id,
            request_payload=payload,
            idempotency_key=idempotency_key,
            session_id=session_id,
            tool_call_id=tool_call_id,
        )
        operation = created.operation
        if created.created:
            await self._send(operation, payload)
        return await self._finish(operation)

    async def request_status(self, *, previous: BecknOperation, provider_order_id: str) -> BecknActionResult:
        """Recover the state of a known provider order using status/on_status.

        A timed-out confirm without a provider order id remains pending; inventing
        a second confirm or a non-standard status key would risk a duplicate.
        """
        self._validate_configuration()
        message_id = str(uuid.uuid4())
        payload = {
            "context": self._context(
                action="status", transaction_id=previous.transaction_id, message_id=message_id
            ),
            "message": {"order_id": provider_order_id},
        }
        created = await self.store.create(
            operation_id=str(uuid.uuid4()),
            transaction_id=previous.transaction_id,
            message_id=message_id,
            action="status",
            expected_callback="on_status",
            domain=previous.domain,
            bap_id=previous.bap_id,
            bpp_id=previous.bpp_id,
            request_payload=payload,
            idempotency_key=f"status:{previous.transaction_id}:{provider_order_id}",
            session_id=previous.session_id,
            tool_call_id=previous.tool_call_id,
        )
        if created.created:
            await self._send(created.operation, payload)
        return await self._finish(created.operation)

    def _context(self, *, action: str, transaction_id: str, message_id: str) -> dict[str, Any]:
        return {
            "domain": settings.beckn_booking_domain,
            "location": {
                "country": {"code": settings.beckn_country_code},
                "city": {"code": settings.beckn_city_code},
            },
            "action": action,
            "version": "1.1.0",
            "bap_id": settings.beckn_bap_id,
            "bap_uri": settings.beckn_bap_uri,
            "bpp_id": settings.beckn_booking_bpp_id,
            "bpp_uri": settings.beckn_booking_bpp_uri,
            "transaction_id": transaction_id,
            "message_id": message_id,
            "timestamp": _now(),
            "ttl": settings.beckn_message_ttl,
        }

    async def _send(self, operation: BecknOperation, payload: Mapping[str, Any]) -> None:
        await self.store.mark_sent(operation)
        url = f"{settings.beckn_bap_caller_url.rstrip('/')}/{operation.action}/"
        owns_client = self._http_client is None
        client = self._http_client or httpx.AsyncClient(timeout=settings.amul_network_timeout_s)
        try:
            for attempt in range(1, settings.beckn_forward_connect_attempts + 1):
                try:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    body = response.json()
                    break
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    if attempt >= settings.beckn_forward_connect_attempts:
                        raise
                    # Same payload, transaction_id and message_id: this is a
                    # transport retry, not a new Beckn interaction.
                    await asyncio.sleep(settings.beckn_forward_retry_delay_seconds)
        except Exception as exc:
            await self.store.mark_transport_error(operation, repr(exc))
            # A callback may still arrive after read timeout/5xx.  Preserve the
            # original exception classification for the booking guard while the
            # durable operation remains available for late reconciliation.
            raise
        finally:
            if owns_client:
                await client.aclose()

        ack_status = (((body.get("message") or {}).get("ack") or {}).get("status")) if isinstance(body, dict) else None
        if ack_status == "NACK":
            await self.store.mark_nack(operation, body)
            return
        if ack_status != "ACK":
            await self.store.mark_transport_error(operation, "Invalid Beckn ACK response")
            raise RuntimeError("Booking network did not return a Beckn ACK/NACK")
        await self.store.mark_ack(operation, body)

    async def _finish(self, operation: BecknOperation) -> BecknActionResult:
        current = await self.store.get(operation.transaction_id, operation.message_id)
        if current is None:  # pragma: no cover - defensive
            raise RuntimeError("Beckn operation disappeared")
        if current.state is OperationState.NACKED:
            return BecknActionResult(current, current.nack, authoritative_rejection=True)
        if current.state in (OperationState.SUCCEEDED, OperationState.BUSINESS_FAILED):
            return BecknActionResult(current, current.callback)
        completed = await self.store.wait(current, settings.beckn_callback_wait_seconds)
        return BecknActionResult(
            completed,
            completed.callback or completed.nack,
            authoritative_rejection=completed.state is OperationState.NACKED,
        )

    @staticmethod
    def _validate_configuration() -> None:
        required = {
            "BECKN_BAP_CALLER_URL": settings.beckn_bap_caller_url,
            "BECKN_BAP_ID": settings.beckn_bap_id,
            "BECKN_BAP_URI": settings.beckn_bap_uri,
            "BECKN_BOOKING_BPP_ID": settings.beckn_booking_bpp_id,
            "BECKN_BOOKING_BPP_URI": settings.beckn_booking_bpp_uri,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise RuntimeError("Beckn callback transactions are missing configuration: " + ", ".join(missing))


_operation_store = BecknOperationStore()
_operation_client = BecknOperationClient(_operation_store)


def get_beckn_operation_store() -> BecknOperationStore:
    return _operation_store


def get_beckn_operation_client() -> BecknOperationClient:
    return _operation_client
