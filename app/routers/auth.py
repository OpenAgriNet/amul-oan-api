"""
Simple authentication router that generates JWT tokens for the frontend.
This closes the auth loop - FE can call this endpoint to get a valid token.
"""
from datetime import datetime, timedelta
from typing import Any, Optional
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import jwt
from cryptography.hazmat.primitives import serialization
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from app.auth.fcm_auth import require_fcm_token
from app.config import settings
from helpers.utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    """Simple login request - can be extended later"""
    username: str = "demo-user"
    password: str = ""  # Optional for demo


class TokenResponse(BaseModel):
    """JWT token response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds until expiration


def load_private_key():
    """Load the private key for signing JWT tokens"""
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


def create_jwt_token(username: str, user_id: str = None, email: str = None) -> str:
    """
    Create a JWT token signed with the private key.
    Returns a token that can be validated with the public key.
    """
    try:
        private_key = load_private_key()
        
        # Token expiration (1 year for demo, adjust as needed)
        exp = datetime.utcnow() + timedelta(days=365)
        iat = datetime.utcnow()
        
        # JWT payload
        payload = {
            "sub": user_id or username,  # Subject (user identifier)
            "name": username,
            "email": email or f"{username}@example.com",
            "iat": int(iat.timestamp()),  # Issued at
            "exp": int(exp.timestamp()),  # Expiration
            "aud": "oan-ui-service",  # Audience
            "iss": "mh-oan-api",  # Issuer
        }
        
        # Sign and encode the token
        token = jwt.encode(
            payload,
            private_key,
            algorithm=settings.jwt_algorithm
        )
        
        logger.info(f"Generated JWT token for user: {username}")
        return token
        
    except Exception as e:
        logger.error(f"Error creating JWT token: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate token: {str(e)}"
        )


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """
    Simple login endpoint that generates a JWT token.
    No actual authentication - just generates a valid token for demo purposes.
    In production, you would validate credentials here.
    """
    try:
        # Generate token (no actual password check for demo)
        token = create_jwt_token(
            username=request.username,
            user_id=request.username,
            email=f"{request.username}@example.com"
        )
        
        # Calculate expiration in seconds (1 year)
        expires_in = 365 * 24 * 60 * 60
        
        return TokenResponse(
            access_token=token,
            token_type="bearer",
            expires_in=expires_in
        )
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
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


@router.get(
    "/webview-url",
    summary="Get app FE URL for webview (FCM auth)",
    response_model=None,
    responses={
        200: {
            "description": "Success. Returns URL with token query param.",
            "content": {"application/json": {"example": {"url": "https://app.example.com?token=eyJ..."}}},
        },
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
    if not settings.app_fe_url or not settings.app_fe_url.strip():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="APP_FE_URL not configured",
        )
    token = create_jwt_for_phone(phone)
    url = _build_webview_url(settings.app_fe_url, token)
    return {"url": url}


@router.get("/demo")
async def demo_login():
    """
    Demo endpoint that returns a token without any credentials.
    Useful for testing and development.
    """
    try:
        token = create_jwt_token(
            username="demo-user",
            user_id="demo-user-123",
            email="demo@example.com"
        )
        
        expires_in = 365 * 24 * 60 * 60
        
        return TokenResponse(
            access_token=token,
            token_type="bearer",
            expires_in=expires_in
        )
        
    except Exception as e:
        logger.error(f"Demo login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Demo login failed: {str(e)}"
        )

