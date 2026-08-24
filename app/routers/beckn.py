"""Internal callback ingress for ONIX-validated Beckn callbacks."""

from __future__ import annotations

import hmac
import json
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import JSONResponse

from app.config import settings
from app.services.beckn_operations import get_beckn_operation_store

router = APIRouter(prefix="/beckn", tags=["beckn-internal"])

_SUPPORTED_CALLBACKS = {"on_confirm", "on_status"}


def _ack() -> dict[str, Any]:
    return {"message": {"ack": {"status": "ACK"}}}


def _nack(code: str, message: str) -> dict[str, Any]:
    return {
        "message": {"ack": {"status": "NACK"}},
        "error": {"code": code, "message": message},
    }


@router.post("/{callback_action}")
async def receive_callback(
    callback_action: str,
    payload: dict[str, Any],
    x_beckn_callback_token: str | None = Header(default=None),
) -> JSONResponse:
    """Persist a callback quickly and return a Beckn ACK/NACK.

    This route is intended to be the *internal* target of ONIX's BAP receiver;
    ONIX remains responsible for subscriber lookup and signature validation.
    The default-off feature gate prevents an accidentally exposed application
    route from becoming an unsigned callback bypass during rollout.
    """
    if not settings.beckn_callback_transactions_enabled:
        return JSONResponse(
            status_code=503,
            content=_nack("CALLBACK_INGRESS_DISABLED", "Beckn callback transactions are disabled"),
        )
    if callback_action not in _SUPPORTED_CALLBACKS:
        raise HTTPException(status_code=404, detail="Unsupported Beckn callback")

    configured_token = settings.beckn_callback_token
    if configured_token and not hmac.compare_digest(
        x_beckn_callback_token or "", configured_token
    ):
        return JSONResponse(
            status_code=401,
            content=_nack("UNAUTHORIZED_CALLBACK", "Invalid callback ingress token"),
        )

    encoded_size = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    if encoded_size > settings.beckn_callback_max_body_bytes:
        return JSONResponse(
            status_code=413,
            content=_nack("CALLBACK_TOO_LARGE", "Callback exceeds configured size limit"),
        )
    if (payload.get("context") or {}).get("action") != callback_action:
        return JSONResponse(
            # The HTTP exchange succeeded; the Beckn NACK carries the protocol
            # rejection and prevents a transport layer from classifying this
            # as a retryable delivery failure.
            status_code=200,
            content=_nack("ACTION_MISMATCH", "URL action does not match context.action"),
        )

    result = await get_beckn_operation_store().record_callback(payload)
    if result.accepted:
        return JSONResponse(status_code=200, content=_ack())
    return JSONResponse(
        status_code=200,
        content=_nack(result.code or "CALLBACK_REJECTED", result.message or "Callback rejected"),
    )
