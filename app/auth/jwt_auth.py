import asyncio
import jwt
import os
from dotenv import load_dotenv
from cryptography.hazmat.primitives import serialization
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from fastapi.security.utils import get_authorization_scheme_param
from helpers.utils import get_logger
from app.config import settings # Import the application settings

load_dotenv()

logger = get_logger(__name__)

class OptionalOAuth2PasswordBearer(OAuth2PasswordBearer):
    """OAuth2 scheme that's optional in development"""
    async def __call__(self, request: Request) -> str | None:
        if settings.environment == "development":
            # In development, don't require the token
            authorization = request.headers.get("Authorization")
            if not authorization:
                return None
            scheme, param = get_authorization_scheme_param(authorization)
            if scheme.lower() != "bearer":
                return None
            return param
        # In production, use normal OAuth2 behavior
        return await super().__call__(request)

# OAuth2 scheme for FastAPI - optional in development
oauth2_scheme = OptionalOAuth2PasswordBearer(tokenUrl="token")


def _load_public_key():
    """Load JWT public key from inline value or file path (value takes precedence)."""
    if settings.jwt_public_key and settings.jwt_public_key.strip():
        try:
            key_bytes = settings.jwt_public_key.strip().encode() if isinstance(settings.jwt_public_key, str) else settings.jwt_public_key
            return serialization.load_pem_public_key(key_bytes)
        except Exception as e:
            logger.warning(f"Failed to load JWT public key from value: {e}")
    key_path = settings.base_dir / settings.jwt_public_key_path
    if key_path.exists():
        try:
            with open(key_path, 'rb') as key_file:
                key = serialization.load_pem_public_key(key_file.read())
            logger.info(f"Successfully loaded JWT Public Key from: {key_path}")
            return key
        except Exception as e:
            logger.warning(f"Failed to load JWT public key from {key_path}: {e}")
    return None


public_key = _load_public_key()
if public_key is None:
    logger.warning("JWT Public Key not loaded (no JWT_PUBLIC_KEY value and path not found or invalid)")

async def get_current_user(token: str | None = Depends(oauth2_scheme)):
    """
    FastAPI dependency to get current authenticated user from JWT token.
    This replaces the Django middleware approach.
    Bypasses authentication in development environment.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if public_key is None:
        logger.error("JWT Public Key is not loaded, cannot verify tokens.")
        raise credentials_exception
        
    try:
        # jwt.decode is CPU-bound (crypto); run in thread pool to avoid blocking the event loop
        decoded_token = await asyncio.to_thread(
            jwt.decode,
            token,
            public_key,
            algorithms=[settings.jwt_algorithm],
            options={
                "verify_signature": True,
                "verify_aud": False,
                "verify_iss": False,
            },
        )

        logger.info(f"Decoded token: {decoded_token}")
        
        return decoded_token
        
    except jwt.ExpiredSignatureError:
        logger.warning("Token has expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    except Exception as e:
        logger.error(f"Unexpected error during token verification: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token verification failed",
            headers={"WWW-Authenticate": "Bearer"},
        )