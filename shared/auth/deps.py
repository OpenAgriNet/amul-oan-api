"""JWT + optional chat API-key auth dependencies (shared by chat and voice domains)."""

from __future__ import annotations

import hmac
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from fastapi.security.utils import get_authorization_scheme_param

from shared.auth.jwt import decode_jwt, load_public_key

_PHONE_DIGITS_RE = re.compile(r"\D+")


@runtime_checkable
class JwtAuthSettings(Protocol):
    """Settings required to build JWT and chat-style API-key auth."""

    environment: str
    jwt_algorithm: str
    jwt_public_key: Optional[str]
    jwt_public_key_path: str
    base_dir: Path
    chat_api_key: Optional[str]


@dataclass(frozen=True)
class JwtAuthBundle:
    oauth2_scheme: Any
    chat_oauth2_scheme: OAuth2PasswordBearer
    public_key: Any
    get_current_user: Callable[..., Any]
    get_chat_user: Callable[..., Any]


def _normalize_phone(raw: str) -> str:
    digits = _PHONE_DIGITS_RE.sub("", raw or "")
    if digits.startswith("91") and len(digits) > 10:
        digits = digits[2:].lstrip("0") or digits
    return digits.lstrip("0")


def _jwt_public_key_base_dir(settings: JwtAuthSettings) -> Path:
    """Directory used to resolve JWT_PUBLIC_KEY_PATH files.

    Voice package ``base_dir`` is ``.../voice`` (for assets, prompts); chat uses the monorepo
    root. Resolving PEM paths from ``voice/`` would miss keys next to chat's ``app/`` unless
    callers set ``jwt_public_key_base_dir`` to the shared repo root.
    """
    explicit = getattr(settings, "jwt_public_key_base_dir", None)
    if isinstance(explicit, Path) and explicit is not None:
        return explicit
    return settings.base_dir


def _validate_chat_api_key(
    request: Request,
    chat_api_key: Optional[str],
) -> dict | None:
    expected = (chat_api_key or "").strip()
    if not expected:
        return None

    provided = (request.headers.get("X-API-Key") or "").strip()
    if not provided:
        return None

    if not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    raw_phone = (request.headers.get("X-User-Phone") or "").strip()
    phone = _normalize_phone(raw_phone)
    if not phone:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-User-Phone header is required with X-API-Key auth",
        )

    return {
        "phone": phone,
        "sub": phone,
        "auth_type": "api_key",
    }


def build_jwt_auth(settings: JwtAuthSettings, logger: Any) -> JwtAuthBundle:
    """Construct OAuth2 schemes and FastAPI dependencies for one domain."""
    class OptionalOAuth2PasswordBearer(OAuth2PasswordBearer):
        """OAuth2 scheme that's optional in development."""

        async def __call__(self, request: Request) -> str | None:
            if settings.environment == "development":
                authorization = request.headers.get("Authorization")
                if not authorization:
                    return None
                scheme, param = get_authorization_scheme_param(authorization)
                if scheme.lower() != "bearer":
                    return None
                return param
            return await super().__call__(request)

    oauth2_scheme = OptionalOAuth2PasswordBearer(tokenUrl="token")
    chat_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

    public_key = load_public_key(
        logger=logger,
        inline_public_key=settings.jwt_public_key,
        base_dir=_jwt_public_key_base_dir(settings),
        public_key_path=settings.jwt_public_key_path,
    )
    if public_key is None:
        logger.warning(
            "JWT Public Key not loaded (no JWT_PUBLIC_KEY value and path not found or invalid)"
        )

    async def get_current_user(token: str | None = Depends(oauth2_scheme)):
        decoded_token = await decode_jwt(
            token=token,
            public_key=public_key,
            algorithm=settings.jwt_algorithm,
            logger=logger,
        )
        logger.info("Decoded token: %s", decoded_token)
        return decoded_token

    async def get_chat_user(
        request: Request, token: str | None = Depends(chat_oauth2_scheme)
    ):
        """
        Unified auth: Bearer JWT or X-API-Key + X-User-Phone (same as chat).
        """
        if token:
            return await get_current_user(token)

        api_key_identity = _validate_chat_api_key(request, settings.chat_api_key)
        if api_key_identity:
            return api_key_identity

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return JwtAuthBundle(
        oauth2_scheme=oauth2_scheme,
        chat_oauth2_scheme=chat_oauth2_scheme,
        public_key=public_key,
        get_current_user=get_current_user,
        get_chat_user=get_chat_user,
    )
