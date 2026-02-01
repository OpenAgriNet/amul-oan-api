"""
FCM token authentication for app/webview endpoints.
Accepts token via header: Authorization: Bearer <fcm_token> or X-FCM-Token: <fcm_token>.
Verifies token using Firebase Admin (dry_run send).
"""
from typing import Optional

from fastapi import HTTPException, Request, status
from helpers.utils import get_logger

from app.config import settings

logger = get_logger(__name__)

_firebase_initialized = False


def _ensure_firebase():
    global _firebase_initialized
    if _firebase_initialized:
        return
    try:
        import firebase_admin
        from firebase_admin import credentials

        path = settings.base_dir / (settings.firebase_service_account_path or "service-account.json")
        if not path.exists():
            raise FileNotFoundError(f"Firebase service account not found: {path}")
        cred = credentials.Certificate(str(path))
        firebase_admin.initialize_app(cred)
        _firebase_initialized = True
        logger.info("Firebase Admin initialized for FCM verification")
    except Exception as e:
        logger.error(f"Firebase init failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="FCM authentication unavailable (Firebase not configured)",
        )


def verify_fcm_token(fcm_token: str) -> bool:
    """Verify FCM token via Firebase dry_run send. Returns True if valid."""
    _ensure_firebase()
    from firebase_admin import messaging, exceptions as fcm_exceptions

    message = messaging.Message(token=fcm_token)
    try:
        messaging.send(message, dry_run=True)
        return True
    except fcm_exceptions.FirebaseError as e:
        logger.warning(f"FCM verification failed: {e.code} - {e}")
        return False
    except Exception as e:
        logger.warning(f"FCM verification error: {e}")
        return False


def get_fcm_token_from_request(request: Request) -> Optional[str]:
    """
    Extract FCM token from request.
    Accepts: Authorization: Bearer <token> or X-FCM-Token: <token>.
    """
    auth = request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("X-FCM-Token", "").strip() or None


async def require_fcm_token(request: Request) -> str:
    """
    FastAPI dependency: require valid FCM token from headers.
    Headers (either accepted):
      - Authorization: Bearer <fcm_token>
      - X-FCM-Token: <fcm_token>
    """
    token = get_fcm_token_from_request(request)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing FCM token. Provide Authorization: Bearer <fcm_token> or X-FCM-Token: <fcm_token>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not verify_fcm_token(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired FCM token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token
