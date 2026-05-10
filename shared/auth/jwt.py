import asyncio
from pathlib import Path
from typing import Any

import jwt
from cryptography.hazmat.primitives import serialization
from fastapi import HTTPException, status


def load_public_key(
    *,
    logger: Any,
    inline_public_key: str | bytes | None,
    base_dir: Path,
    public_key_path: str,
):
    """Load a PEM public key from inline value or base-dir relative file."""
    if inline_public_key and str(inline_public_key).strip():
        try:
            key_bytes = (
                inline_public_key.strip().encode()
                if isinstance(inline_public_key, str)
                else inline_public_key
            )
            return serialization.load_pem_public_key(key_bytes)
        except Exception as exc:
            logger.warning("Failed to load JWT public key from value: %s", exc)

    key_path = base_dir / public_key_path
    if key_path.exists():
        try:
            with open(key_path, "rb") as key_file:
                return serialization.load_pem_public_key(key_file.read())
        except Exception as exc:
            logger.warning("Failed to load JWT public key from %s: %s", key_path, exc)

    return None


def build_credentials_exception() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def decode_jwt(
    *,
    token: str | None,
    public_key,
    algorithm: str,
    logger: Any,
) -> dict[str, Any]:
    """Decode and validate JWT in a threadpool."""
    if public_key is None:
        logger.error("JWT public key is not loaded, cannot verify tokens.")
        raise build_credentials_exception()
    if token is None:
        raise build_credentials_exception()

    try:
        return await asyncio.to_thread(
            jwt.decode,
            token,
            public_key,
            algorithms=[algorithm],
            options={
                "verify_signature": True,
                "verify_aud": False,
                "verify_iss": False,
            },
        )
    except jwt.ExpiredSignatureError as exc:
        logger.warning("Token has expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except jwt.InvalidTokenError as exc:
        logger.warning("Invalid token error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive branch
        logger.error("Unexpected error during token verification: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token verification failed",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

