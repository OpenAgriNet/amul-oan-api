"""
Simple authentication router that generates JWT tokens for the frontend.
This closes the auth loop - FE can call this endpoint to get a valid token.
"""
import os
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import re

import jwt
from cryptography.hazmat.primitives import serialization
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from app.auth.fcm_auth import require_fcm_token
from app.config import settings
from helpers.utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


class PhoneTokenRequest(BaseModel):
    """Request body for phone-based token generation."""
    phone: str


class TokenResponse(BaseModel):
    """JWT token response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds until expiration


def load_private_key():
    """Load the private key for signing JWT tokens. Uses JWT_PRIVATE_KEY value if set, else file path."""
    if settings.jwt_private_key and settings.jwt_private_key.strip():
        try:
            key_bytes = (
                settings.jwt_private_key.strip().encode()
                if isinstance(settings.jwt_private_key, str)
                else settings.jwt_private_key
            )
            return serialization.load_pem_private_key(key_bytes, password=None)
        except Exception as e:
            logger.warning(f"Failed to load JWT private key from value: {e}")
    private_key_path = settings.base_dir / (settings.jwt_private_key_path or "jwt_private_key.pem")
    if not private_key_path.exists():
        raise FileNotFoundError(f"Private key not found at {private_key_path}")
    with open(private_key_path, 'rb') as key_file:
        return serialization.load_pem_private_key(key_file.read(), password=None)


def create_jwt_for_phone(phone: str, expires_days: int = 365) -> str:
    """
    Create a JWT token containing the phone number, signed with jwt_private_key.pem.
    Used for webview URL so the FE can identify the user.
    """
    try:
        private_key = load_private_key()
        exp = datetime.utcnow() + timedelta(days=expires_days)
        iat = datetime.utcnow()
        payload = {
            "phone": phone,
            "sub": phone,
            "iat": int(iat.timestamp()),
            "exp": int(exp.timestamp()),
            "aud": "oan-ui-service",
            "iss": "mh-oan-api",
        }
        return jwt.encode(
            payload,
            private_key,
            algorithm=settings.jwt_algorithm,
        )
    except Exception as e:
        logger.error(f"Error creating JWT for phone: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate token: {str(e)}",
        )


def create_jwt_for_phone_with_data(
    phone: str,
    farmer_data: dict | None = None,
    expires_days: int = 365,
) -> str:
    """
    Create a JWT containing phone number and pre-fetched farmer data.
    Used by the demo-ui integration so the chat UI gets farmer context immediately.
    """
    try:
        private_key = load_private_key()
        exp = datetime.utcnow() + timedelta(days=expires_days)
        iat = datetime.utcnow()
        payload = {
            "phone": phone,
            "sub": phone,
            "iat": int(iat.timestamp()),
            "exp": int(exp.timestamp()),
            "aud": "oan-ui-service",
            "iss": "mh-oan-api",
        }
        if farmer_data:
            payload["data"] = farmer_data
        return jwt.encode(
            payload,
            private_key,
            algorithm=settings.jwt_algorithm,
        )
    except Exception as e:
        logger.error(f"Error creating JWT for phone with data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate token: {str(e)}",
        )


def create_anonymous_jwt_token(expires_days: float = 1.0) -> str:
    """
    Create a 1-day signed JWT for anonymous usage.
    sub is a randomly generated anonymous user ID (UUID4).
    """
    try:
        private_key = load_private_key()
        anonymous_id = f"anon-{uuid.uuid4().hex}"
        exp = datetime.utcnow() + timedelta(days=expires_days)
        iat = datetime.utcnow()
        payload = {
            "sub": anonymous_id,
            "iat": int(iat.timestamp()),
            "exp": int(exp.timestamp()),
            "aud": "oan-ui-service",
            "iss": "mh-oan-api",
            "anonymous": True,
        }
        return jwt.encode(
            payload,
            private_key,
            algorithm=settings.jwt_algorithm,
        )
    except Exception as e:
        logger.error(f"Error creating anonymous JWT: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate anonymous token: {str(e)}",
        )


@router.post("/anonymous", response_model=TokenResponse)
async def anonymous_token():
    """
    Issue a 1-day signed JWT for anonymous usage.
    No credentials required. sub is a randomly generated anonymous user ID.
    """
    expires_days = 1.0
    token = create_anonymous_jwt_token(expires_days=expires_days)
    expires_in = int(expires_days * 24 * 60 * 60)
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=expires_in,
    )


def _build_webview_url(base_url: str, jwt_token: str) -> str:
    """Parse base_url and append token as query param (merge with existing query if any)."""
    parsed = urlparse(base_url.strip())
    query = parsed.query
    params = dict()
    if query:
        for k, v in parse_qs(query).items():
            params[k] = v[0] if len(v) == 1 else v
    params["token"] = jwt_token
    new_query = urlencode(params, doseq=True)
    return urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.params,
        new_query,
        parsed.fragment,
    ))


_PHONE_DIGITS_RE = re.compile(r"^\+?\d{7,15}$")


def _validate_phone(raw: str) -> str:
    """Strip whitespace and validate that phone contains 7-15 digits (with optional leading +)."""
    phone = raw.strip()
    if not phone:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="phone parameter is required and must not be empty",
        )
    if not _PHONE_DIGITS_RE.match(phone):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid phone number: must be 7-15 digits (optional leading +), got '{phone}'",
        )
    return phone


@router.get(
    "/webview-url",
    summary="Get app FE URL for webview (FCM auth)",
    response_model=None,
    responses={
        200: {
            "description": "Success. Returns URL with token query param.",
            "content": {"application/json": {"example": {"url": "https://app.example.com?token=eyJ..."}}},
        },
        400: {"description": "Missing or invalid phone number"},
        401: {"description": "Missing or invalid FCM token"},
        503: {"description": "FCM/Firebase not configured"},
    },
)
async def get_webview_url(
    phone: str = Query(..., description="User phone number (included in issued JWT)"),
    _fcm_token: str = Depends(require_fcm_token),
) -> Any:
    """
    **Endpoint:** `GET /api/auth/webview-url`

    **Headers** (one required for authentication):
    - `Authorization: Bearer <fcm_token>` — FCM device token
    - `X-FCM-Token: <fcm_token>` — alternative header

    **Params:**
    - `phone` (query, required) — User phone number; issued JWT will contain this.

    **Response:** JSON with `url` (string): FE base URL from `APP_FE_URL` with `token={jwt}` added as query param.
    The JWT is signed with `jwt_private_key.pem` and contains `phone` (and standard claims).
    """
    phone = _validate_phone(phone)
    if not settings.app_fe_url or not settings.app_fe_url.strip():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="APP_FE_URL not configured",
        )
    token = create_jwt_for_phone(phone)
    url = _build_webview_url(settings.app_fe_url, token)
    return {"url": url}


def _require_api_key(
    api_key: str = Query(..., alias="api_key", description="Server-side API key for demo-ui backends"),
) -> str:
    """Validate the demo-ui API key."""
    expected = os.getenv("DEMO_UI_API_KEY", "").strip()
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DEMO_UI_API_KEY not configured on server",
        )
    if api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return api_key


@router.post(
    "/token-for-phone",
    summary="Generate JWT with farmer data for a phone number (API-key auth)",
    responses={
        200: {
            "description": "JWT with embedded farmer data + webview URL",
            "content": {"application/json": {"example": {
                "url": "https://app.example.com?token=eyJ...",
                "access_token": "eyJ...",
                "token_type": "bearer",
                "expires_in": 86400,
                "farmer_records_count": 2,
            }}},
        },
        400: {"description": "Invalid phone number"},
        401: {"description": "Invalid API key"},
    },
)
async def token_for_phone(
    body: PhoneTokenRequest,
    _key: str = Depends(_require_api_key),
):
    """
    Fetch farmer data from PashuGPT APIs for the given phone number,
    embed it in a signed JWT, and return the token + webview URL.

    Intended for trusted server-side callers (e.g. demo-ui backend).
    Requires `api_key` query parameter matching `DEMO_UI_API_KEY` env var.
    """
    phone = _validate_phone(body.phone)

    # Fetch farmer data from PashuGPT backends
    farmer_records = None
    try:
        from agents.tools.farmer import get_farmer_data_by_mobile
        farmer_records = await get_farmer_data_by_mobile(phone)
    except Exception as e:
        logger.warning(f"Failed to fetch farmer data for {phone}: {e}")

    farmer_data = {"farmers": farmer_records} if farmer_records else None
    records_count = len(farmer_records) if farmer_records else 0

    token = create_jwt_for_phone_with_data(phone, farmer_data=farmer_data)

    # Build webview URL if configured
    url = None
    if settings.app_fe_url and settings.app_fe_url.strip():
        url = _build_webview_url(settings.app_fe_url, token)

    return {
        "url": url,
        "access_token": token,
        "token_type": "bearer",
        "expires_in": 365 * 24 * 60 * 60,
        "farmer_records_count": records_count,
    }

