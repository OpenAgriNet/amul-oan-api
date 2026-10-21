"""Shared-secret auth for inbound partner webhooks.

Not JWT — partners send the static secret as ``Authorization: Bearer <token>``.
Comparison uses ``hmac.compare_digest`` to avoid timing leaks. When
``WEBHOOK_SHARED_TOKEN`` is unset or empty, every request is rejected (fail closed).
"""
from __future__ import annotations

import hmac

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings
from helpers.utils import get_logger

logger = get_logger(__name__)

_bearer = HTTPBearer(auto_error=False, description="Shared secret configured as WEBHOOK_SHARED_TOKEN")


async def require_webhook_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    """FastAPI dependency: require ``Authorization: Bearer`` matching the shared secret."""
    expected = (settings.webhook_shared_token or "").strip()
    if not expected:
        logger.error("WEBHOOK_SHARED_TOKEN is not configured; rejecting webhook")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Webhook auth is not configured",
        )

    provided = (credentials.credentials if credentials else "").strip()
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing webhook token",
            headers={"WWW-Authenticate": "Bearer"},
        )
