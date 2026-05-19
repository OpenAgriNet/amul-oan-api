"""
FCM token authentication for app/webview endpoints.
Accepts token via header: Authorization: Bearer <fcm_token> or X-FCM-Token: <fcm_token>.
Verifies token using Firebase Admin (dry_run send). Supports up to three Firebase
projects (primary, secondary, tertiary); if any project accepts the token, authorization
is allowed.
Service account can be provided as inline JSON
(FIREBASE_SERVICE_ACCOUNT / FIREBASE_SERVICE_ACCOUNT_2 / FIREBASE_SERVICE_ACCOUNT_3)
or as file paths
(FIREBASE_SERVICE_ACCOUNT_PATH / FIREBASE_SERVICE_ACCOUNT_PATH_2 / FIREBASE_SERVICE_ACCOUNT_PATH_3);
value takes precedence.
"""
import asyncio
import json
from typing import Optional, Dict, Union, Tuple, List

from fastapi import HTTPException, Request, status
from helpers.utils import get_logger

from app.config import settings

logger = get_logger(__name__)

_firebase_initialized = False
_firebase_apps: Dict[str, object] = {}


def _get_primary_credential() -> Optional[Union[str, dict]]:
    """Resolve primary Firebase credential: inline JSON value or file path."""
    if settings.firebase_service_account and settings.firebase_service_account.strip():
        try:
            return json.loads(settings.firebase_service_account.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Invalid FIREBASE_SERVICE_ACCOUNT JSON: {e}")
            return None
    path = settings.base_dir / (settings.firebase_service_account_path or "service-account.json")
    if path.exists():
        return str(path)
    return None


def _get_secondary_credential() -> Optional[Union[str, dict]]:
    """Resolve secondary Firebase credential: inline JSON value or file path."""
    if settings.firebase_service_account_2 and settings.firebase_service_account_2.strip():
        try:
            return json.loads(settings.firebase_service_account_2.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Invalid FIREBASE_SERVICE_ACCOUNT_2 JSON: {e}")
            return None
    if settings.firebase_service_account_path_2:
        path = settings.base_dir / settings.firebase_service_account_path_2
        if path.exists():
            return str(path)
    return None


def _get_tertiary_credential() -> Optional[Union[str, dict]]:
    """Resolve tertiary Firebase credential: inline JSON value or file path."""
    if settings.firebase_service_account_3 and settings.firebase_service_account_3.strip():
        try:
            return json.loads(settings.firebase_service_account_3.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Invalid FIREBASE_SERVICE_ACCOUNT_3 JSON: {e}")
            return None
    if settings.firebase_service_account_path_3:
        path = settings.base_dir / settings.firebase_service_account_path_3
        if path.exists():
            return str(path)
    return None


def _ensure_firebase():
    """
    Lazily initialize one or more Firebase apps for FCM verification.
    Supports a primary, optional secondary, and optional tertiary service account.
    Credentials from inline values (FIREBASE_SERVICE_ACCOUNT / _2 / _3) take precedence over paths.
    """
    global _firebase_initialized, _firebase_apps
    if _firebase_initialized:
        return
    try:
        import firebase_admin
        from firebase_admin import credentials

        firebase_configs: List[Tuple[str, Union[str, dict]]] = []

        primary = _get_primary_credential()
        if primary is not None:
            firebase_configs.append(("default", primary))
        else:
            logger.error("Primary Firebase service account not configured (no value and path not found)")

        secondary = _get_secondary_credential()
        if secondary is not None:
            firebase_configs.append(("secondary", secondary))

        tertiary = _get_tertiary_credential()
        if tertiary is not None:
            firebase_configs.append(("tertiary", tertiary))

        if not firebase_configs:
            raise FileNotFoundError("No Firebase service accounts configured for FCM verification")

        for name, cred_source in firebase_configs:
            cred = credentials.Certificate(cred_source)
            if name == "default":
                app = firebase_admin.initialize_app(cred)
            else:
                app = firebase_admin.initialize_app(cred, name=name)
            _firebase_apps[name] = app

        _firebase_initialized = True
        logger.info(
            "Firebase Admin initialized for FCM verification "
            f"with apps: {', '.join(_firebase_apps.keys())}"
        )
    except Exception as e:
        logger.error(f"Firebase init failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="FCM authentication unavailable (Firebase not configured)",
        )


def _verify_against_app_sync(fcm_token: str, app_name: str, app: object) -> bool:
    """Single-app verification (sync). Returns True if Firebase accepts the token."""
    from firebase_admin import messaging, exceptions as fcm_exceptions

    message = messaging.Message(token=fcm_token)
    try:
        messaging.send(message, dry_run=True, app=app)
        logger.debug(f"FCM token valid for app: {app_name}")
        return True
    except fcm_exceptions.FirebaseError as e:
        logger.debug(f"FCM verification failed for app {app_name}: {e.code} - {e}")
    except Exception as e:
        logger.debug(f"FCM verification error for app {app_name}: {e}")
    return False


def verify_fcm_token(fcm_token: str) -> bool:
    """
    Verify FCM token via Firebase dry_run send (sequential).

    Retained for backwards compatibility with synchronous callers. New
    async code paths should prefer :func:`verify_fcm_token_async`, which
    validates against all configured projects concurrently and returns as
    soon as any one accepts the token.
    """
    _ensure_firebase()
    for app_name, app in _firebase_apps.items():
        if _verify_against_app_sync(fcm_token, app_name, app):
            return True
    logger.warning("FCM token invalid for all configured Firebase apps")
    return False


async def verify_fcm_token_async(fcm_token: str) -> bool:
    """
    Verify FCM token by checking all configured Firebase apps concurrently.

    Returns ``True`` as soon as any project accepts the token; outstanding
    checks are then cancelled. With N configured projects this turns the
    verification latency from O(N · T) (sequential dry-run sends, where T
    is the per-call Firebase round-trip) into O(T) in the common case
    where the user's token belongs to one of the configured projects.

    The Firebase Admin SDK exposes only a synchronous ``messaging.send``,
    so each per-app check is offloaded to a worker thread via
    :func:`asyncio.to_thread`. The async wrapper here is the
    coordination layer that makes them race.
    """
    _ensure_firebase()
    if not _firebase_apps:
        logger.warning("FCM token rejected: no Firebase apps initialized")
        return False

    tasks = [
        asyncio.create_task(
            asyncio.to_thread(_verify_against_app_sync, fcm_token, name, app),
            name=f"fcm-verify[{name}]",
        )
        for name, app in _firebase_apps.items()
    ]

    try:
        for finished in asyncio.as_completed(tasks):
            try:
                if await finished:
                    return True
            except Exception as e:  # noqa: BLE001
                # One task failing unexpectedly must never abort the race:
                # a different project may still accept the token. Log and
                # keep waiting on the remaining tasks ("any success wins").
                # (CancelledError is BaseException, so a real cancel of this
                # coroutine still propagates.)
                logger.debug(f"FCM verification task errored, ignoring: {e}")
                continue
        logger.warning("FCM token invalid for all configured Firebase apps")
        return False
    finally:
        # Cancel any still-pending verifications so we don't keep threads
        # blocked on Firebase round-trips after we already have an answer.
        for task in tasks:
            if not task.done():
                task.cancel()


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
    # Concurrent verification across all configured Firebase projects;
    # returns on first success without blocking the event loop.
    if not await verify_fcm_token_async(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired FCM token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token
