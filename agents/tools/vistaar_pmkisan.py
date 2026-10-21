"""Private two-turn PM-KISAN status lookup over Bharat Vistaar."""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition

from agents.deps import FarmerContext
from agents.tools.beckn.operations import (
    OperationState,
    get_beckn_operation_client,
    get_beckn_operation_store,
)
from agents.tools.session_pmkisan import (
    allow_otp_request,
    clear_pending_otp,
    clear_pm_kisan_session,
    count_otp_attempt,
    get_pending_otp,
    get_pm_kisan_session,
    otp_attempts,
    resolve_identifier,
    set_pm_kisan_session,
)
from app.config import settings
from helpers.utils import get_logger, to_ascii_digits

logger = get_logger(__name__)
_MAX_RESULT_CHARS = 6_000


def _redact_tool_span(outcome: str) -> None:
    """Replace sensitive tool arguments with stable placeholders in Langfuse."""
    try:
        from langfuse import get_client

        get_client().update_current_span(
            input={"identifier": "[REDACTED]", "otp": "[REDACTED]"},
            output={"outcome": outcome},
            metadata={"private_data": True, "tool": "pmkisan"},
        )
    except Exception as exc:  # noqa: BLE001 - tracing must never affect farmer flow
        logger.debug("Unable to redact PM-KISAN tracing span: %s", type(exc).__name__)


def _normalize_identifier(value: str | None) -> str:
    return re.sub(r"\s+", "", to_ascii_digits(value or "")).upper()


def _orders(payload: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    message = payload.get("message")
    if isinstance(message, Mapping) and isinstance(message.get("order"), Mapping):
        yield message["order"]
    responses = payload.get("responses")
    if isinstance(responses, list):
        for response in responses:
            nested = response.get("message") if isinstance(response, Mapping) else None
            if isinstance(nested, Mapping) and isinstance(nested.get("order"), Mapping):
                yield nested["order"]


def _tag_codes(tags: object) -> set[str]:
    codes: set[str] = set()
    if not isinstance(tags, list):
        return codes
    for tag in tags:
        if not isinstance(tag, Mapping):
            continue
        descriptor = tag.get("descriptor")
        code = descriptor.get("code") if isinstance(descriptor, Mapping) else None
        if isinstance(code, str) and code:
            codes.add(code)
    return codes


def _init_outcome(payload: Mapping[str, Any]) -> str | None:
    for order in _orders(payload):
        items = order.get("items")
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, Mapping):
                continue
            tags = item.get("tags")
            if "otp_status" in _tag_codes(tags):
                return "otp_sent"
            for tag in tags if isinstance(tags, list) else []:
                descriptor = tag.get("descriptor") if isinstance(tag, Mapping) else None
                if isinstance(descriptor, Mapping) and descriptor.get("code") == "otp_error":
                    # BV also answers "Try again" when its own service fails.
                    short_desc = str(descriptor.get("short_desc") or "")
                    return "no_record" if "no record found" in short_desc.lower() else "service_error"
    return None


def _status_order(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    return next(iter(_orders(payload)), None)


def _is_invalid_otp(order: Mapping[str, Any]) -> bool:
    return order.get("id") == "error" and "invalid_otp" in _tag_codes(order.get("tags"))


def _mask_phone(value: object) -> str:
    digits = re.sub(r"\D", "", to_ascii_digits(value or ""))
    return f"******{digits[-4:]}" if digits else ""


def _mask_keep_last4(value: str) -> str:
    return "*" * max(len(value) - 4, 0) + value[-4:]


def _mask_registration(text: str, identifier: str) -> str:
    text = re.sub(
        r"(Registration (?:Number|No\.?)\s*[-:]\s*)([A-Za-z0-9]+)",
        lambda match: match.group(1) + _mask_keep_last4(match.group(2)),
        text,
        flags=re.IGNORECASE,
    )
    return text.replace(identifier, _mask_keep_last4(identifier)) if len(identifier) > 4 else text


def _status_summary(order: Mapping[str, Any], identifier: str) -> str:
    lines: list[str] = []
    state = order.get("state")
    if state:
        lines.append(f"Status: {state}")
    fulfillments = order.get("fulfillments")
    for fulfillment in fulfillments if isinstance(fulfillments, list) else []:
        if not isinstance(fulfillment, Mapping):
            continue
        customer = fulfillment.get("customer")
        if isinstance(customer, Mapping):
            person = customer.get("person")
            contact = customer.get("contact")
            if isinstance(person, Mapping) and person.get("name"):
                lines.append(f"Beneficiary: {person['name']}")
            if isinstance(contact, Mapping) and contact.get("phone"):
                masked_phone = _mask_phone(contact["phone"])
                if masked_phone:
                    lines.append(f"Registered phone: {masked_phone}")
        fulfillment_state = fulfillment.get("state")
        if not isinstance(fulfillment_state, Mapping):
            continue
        descriptor = fulfillment_state.get("descriptor")
        if isinstance(descriptor, Mapping):
            for key in ("name", "short_desc", "long_desc"):
                value = descriptor.get(key)
                if isinstance(value, str) and value.strip():
                    lines.append(value.strip())
        updated_at = fulfillment_state.get("updated_at")
        if isinstance(updated_at, str) and updated_at:
            lines.append(f"Last updated: {updated_at}")
    return _mask_registration("\n".join(lines), identifier)[:_MAX_RESULT_CHARS]


async def prepare_pm_kisan_tool(
    ctx: RunContext[FarmerContext], tool_def: ToolDefinition
) -> ToolDefinition | None:
    del ctx
    return tool_def if settings.vistaar_pmkisan_enabled else None


async def initiate_pm_kisan_status_check(
    ctx: RunContext[FarmerContext], identifier: str = "", use_account_mobile: bool = False
) -> str:
    """Send a PM-KISAN OTP for a farmer-provided identifier.

    Use this only when the farmer asks to check THEIR PM-KISAN beneficiary or
    installment status. Never invent an identifier.

    Args:
        ctx: Current farmer and session context.
        identifier: PM-KISAN registration number, beneficiary ID, mobile, or the
            [AADHAAR_1234] placeholder exactly as it appears in the farmer's message.
        use_account_mobile: True only after the farmer confirmed that the
            registered mobile of this account should be used.
    """
    _redact_tool_span("started")
    session_id = ctx.deps.session_id
    if not session_id:
        _redact_tool_span("missing_session")
        return "A valid chat session is required. Ask the farmer to start again."
    if identifier:
        supplied = await resolve_identifier(identifier, session_id, ctx.deps.mobile)
        if supplied is None:
            _redact_tool_span("identifier_expired")
            return "Ask the farmer to type their Aadhaar number again."
    elif ctx.deps.mobile and use_account_mobile:
        supplied = ctx.deps.mobile
    elif ctx.deps.mobile:
        _redact_tool_span("confirm_mobile")
        return (
            "CONFIRM_MOBILE. Ask the farmer whether to check PM-KISAN with their registered "
            f"mobile {_mask_phone(ctx.deps.mobile)}, or to type their PM-KISAN registration "
            "number or Aadhaar instead."
        )
    else:
        supplied = ""
    normalized = _normalize_identifier(supplied)
    if not normalized:
        _redact_tool_span("missing_identifier")
        return (
            "Ask the farmer for their PM-KISAN registration number, beneficiary ID, "
            "Aadhaar, or registered mobile number. Never guess it."
        )
    try:
        allowed = await allow_otp_request(session_id, normalized)
    except Exception:
        logger.exception("PM-KISAN OTP request limit could not be checked")
        _redact_tool_span("state_unavailable")
        return "The PM-KISAN status service is temporarily unavailable. Please try again later."
    if not allowed:
        _redact_tool_span("rate_limited")
        return "TOO_MANY_OTP_REQUESTS. Tell the farmer to try again after an hour."

    transaction_id = str(uuid.uuid4())
    message_id = str(uuid.uuid4())
    state = {
        "transaction_id": transaction_id,
        "init_message_id": message_id,
        "last_status_message_id": "",
        "identifier": normalized,
    }
    try:
        # A new check starts clean. Store first: never request an OTP that
        # cannot be correlated next turn.
        await clear_pm_kisan_session(session_id, ctx.deps.mobile)
        await set_pm_kisan_session(session_id, ctx.deps.mobile, state)
    except Exception:
        logger.exception("PM-KISAN session state could not be created")
        _redact_tool_span("state_unavailable")
        return "The PM-KISAN status service is temporarily unavailable. Please try again later."

    try:
        result = await get_beckn_operation_client().init_pm_kisan_status(
            identifier=normalized,
            transaction_id=transaction_id,
            message_id=message_id,
            session_id=session_id,
            tool_call_id=getattr(ctx, "tool_call_id", None),
        )
    except Exception:
        logger.exception("PM-KISAN init transaction failed")
        _redact_tool_span("network_error")
        return "The PM-KISAN request is pending or unavailable. Please try again shortly."

    if result.operation.state is OperationState.NACKED:
        await clear_pm_kisan_session(session_id, ctx.deps.mobile)
        _redact_tool_span("rejected")
        return "The PM-KISAN request was rejected. Please try again later."
    if result.operation.state is OperationState.TIMED_OUT_PENDING:
        _redact_tool_span("pending")
        return "The PM-KISAN OTP request is still processing. If an OTP arrives, enter it here."
    if not result.ok or not isinstance(result.payload, Mapping):
        await clear_pm_kisan_session(session_id, ctx.deps.mobile)
        _redact_tool_span("invalid_response")
        return "The PM-KISAN service returned an unreadable response. Please try again later."

    outcome = _init_outcome(result.payload)
    if outcome == "otp_sent":
        _redact_tool_span("otp_sent")
        return "OTP_SENT. Ask the farmer for the 4-digit OTP sent to their registered mobile."
    await clear_pm_kisan_session(session_id, ctx.deps.mobile)
    if outcome == "no_record":
        _redact_tool_span("no_record")
        return "NO_RECORD. Ask the farmer to check the identifier and start again."
    if outcome == "service_error":
        _redact_tool_span("service_error")
        return "SERVICE_ERROR. The PM-KISAN service is not responding. Ask the farmer to try again later."
    _redact_tool_span("invalid_response")
    return "The PM-KISAN service returned an unreadable response. Please try again later."


async def _handle_status_result(
    ctx: RunContext[FarmerContext], state: dict[str, Any], payload: Mapping[str, Any]
) -> str:
    order = _status_order(payload)
    if order is None:
        state["last_status_message_id"] = ""
        await set_pm_kisan_session(ctx.deps.session_id, ctx.deps.mobile, state)
        _redact_tool_span("invalid_response")
        return "The PM-KISAN service returned an unreadable response. Please try again later."
    if _is_invalid_otp(order):
        attempts = await otp_attempts(ctx.deps.session_id, ctx.deps.mobile)
        if attempts >= settings.vistaar_pmkisan_max_otp_attempts:
            await clear_pm_kisan_session(ctx.deps.session_id, ctx.deps.mobile)
            _redact_tool_span("invalid_otp_limit")
            return "OTP_RETRY_LIMIT. Ask the farmer to restart and request a new OTP."
        state["last_status_message_id"] = ""
        await set_pm_kisan_session(ctx.deps.session_id, ctx.deps.mobile, state)
        _redact_tool_span("invalid_otp")
        return "INVALID_OTP. Ask the farmer to re-enter the 4-digit OTP."

    summary = _status_summary(order, str(state.get("identifier") or ""))
    if not summary:
        state["last_status_message_id"] = ""
        await set_pm_kisan_session(ctx.deps.session_id, ctx.deps.mobile, state)
        _redact_tool_span("invalid_response")
        return "The PM-KISAN service returned no status details. Please try again later."
    await clear_pm_kisan_session(ctx.deps.session_id, ctx.deps.mobile)
    _redact_tool_span("success")
    return "PM-KISAN STATUS FOUND. Present these exact details only to this farmer:\n" + summary


async def check_pm_kisan_status_with_otp(ctx: RunContext[FarmerContext]) -> str:
    """Verify the OTP the farmer just sent and fetch their PM-KISAN status.

    The farmer's OTP appears as [OTP] in their message; the tool reads the real
    digits itself, so never pass or repeat them.

    Args:
        ctx: Current farmer and session context.
    """
    _redact_tool_span("started")
    session_id, mobile = ctx.deps.session_id, ctx.deps.mobile
    try:
        state = await get_pm_kisan_session(session_id, mobile)
        otp = await get_pending_otp(session_id, mobile)
    except Exception:
        logger.exception("PM-KISAN session state could not be read")
        _redact_tool_span("state_unavailable")
        return "The PM-KISAN status service is temporarily unavailable. Please try again later."
    if not state:
        _redact_tool_span("missing_state")
        return "NO_PENDING_CHECK. Ask the farmer to start the PM-KISAN status check again."

    previous_message_id = str(state.get("last_status_message_id") or "")
    if previous_message_id:
        previous = await get_beckn_operation_store().get(
            str(state["transaction_id"]), previous_message_id
        )
        if (
            previous
            and previous.state in {OperationState.SUCCEEDED, OperationState.BUSINESS_FAILED}
            and isinstance(previous.callback, Mapping)
        ):
            return await _handle_status_result(ctx, state, previous.callback)
        # A newly sent OTP goes out even when the last verification never answered.
        if previous and previous.state is not OperationState.NACKED and not otp:
            _redact_tool_span("pending")
            return "The previous OTP verification is still processing. Please wait and try again shortly."
        state["last_status_message_id"] = ""

    if not otp:
        _redact_tool_span("invalid_otp_format")
        return "Ask the farmer to send only the 4-digit OTP exactly as received."
    message_id = str(uuid.uuid4())
    state["last_status_message_id"] = message_id
    try:
        # One submission per OTP the farmer sent, counted before it goes out so
        # parallel or abandoned submissions cannot exceed the attempt limit.
        await clear_pending_otp(session_id, mobile)
        if await count_otp_attempt(session_id, mobile) > settings.vistaar_pmkisan_max_otp_attempts:
            await clear_pm_kisan_session(session_id, mobile)
            _redact_tool_span("invalid_otp_limit")
            return "OTP_RETRY_LIMIT. Ask the farmer to restart and request a new OTP."
        await set_pm_kisan_session(session_id, mobile, state)
        result = await get_beckn_operation_client().submit_pm_kisan_otp(
            identifier=str(state["identifier"]),
            otp=otp,
            transaction_id=str(state["transaction_id"]),
            message_id=message_id,
            session_id=str(session_id),
            tool_call_id=getattr(ctx, "tool_call_id", None),
        )
    except Exception:
        logger.exception("PM-KISAN status transaction failed")
        state["last_status_message_id"] = ""
        try:
            await set_pm_kisan_session(session_id, mobile, state)
        except Exception:
            logger.exception("PM-KISAN session state could not be saved")
        _redact_tool_span("network_error")
        return "The OTP could not be verified right now. Ask the farmer to send the same OTP again shortly."

    if result.operation.state is OperationState.NACKED:
        await clear_pm_kisan_session(session_id, mobile)
        _redact_tool_span("rejected")
        return "The PM-KISAN OTP request was rejected. Ask the farmer to start again."
    if result.operation.state is OperationState.TIMED_OUT_PENDING:
        _redact_tool_span("pending")
        return "The OTP verification is still processing. Please wait and try again shortly."
    if not result.ok or not isinstance(result.payload, Mapping):
        _redact_tool_span("invalid_response")
        return "The PM-KISAN service returned an unreadable response. Please try again later."
    return await _handle_status_result(ctx, state, result.payload)
