"""
FCM token authentication for app/webview endpoints.
Accepts token via header: Authorization: Bearer <fcm_token> or X-FCM-Token: <fcm_token>.
Verifies token using Firebase Admin (dry_run send). Supports any number of Firebase
projects; if any project accepts the token, authorization is allowed.

The primary service account uses FIREBASE_SERVICE_ACCOUNT or
FIREBASE_SERVICE_ACCOUNT_PATH. Additional accounts use matching numbered variables,
for example FIREBASE_SERVICE_ACCOUNT_2 or FIREBASE_SERVICE_ACCOUNT_PATH_2. Numbered
accounts are discovered from the environment, so adding _4, _5, and so on does not
require a code change. For each account, the inline JSON value takes precedence over
the file path.
"""
import asyncio
import json
import os
import re
from typing import Dict, List, Optional, Tuple, Union

from fastapi import HTTPException, Request, status
from helpers.utils import get_logger

from app.config import settings

logger = get_logger(__name__)

_firebase_initialized = False
_firebase_apps: Dict[str, object] = {}


CredentialSource = Union[str, dict]
FirebaseConfig = Tuple[str, CredentialSource]

_NUMBERED_CREDENTIAL_ENV_RE = re.compile(
    r"^FIREBASE_SERVICE_ACCOUNT(?:_PATH)?_([1-9]\d*)$"
)


def _env_names_for_account(index: int) -> Tuple[str, str]:
    """Return the inline-value and file-path env names for a 1-based account index."""
    if index < 1:
        raise ValueError("Firebase account index must be positive")
    suffix = "" if index == 1 else f"_{index}"
    return f"FIREBASE_SERVICE_ACCOUNT{suffix}", f"FIREBASE_SERVICE_ACCOUNT_PATH{suffix}"


def _setting_name(env_name: str) -> str:
    return env_name.lower()


def _configured_account_indices() -> List[int]:
    """Discover all configured credential slots, including arbitrary numbered slots."""
    indices = {1}

    for env_name in os.environ:
        match = _NUMBERED_CREDENTIAL_ENV_RE.match(env_name)
        if match and int(match.group(1)) > 1:
            indices.add(int(match.group(1)))

    # Settings retains explicit fields for the legacy slots. Include them when set
    # programmatically even if there is no corresponding process environment value.
    for index in (2, 3):
        value_name, path_name = _env_names_for_account(index)
        if getattr(settings, _setting_name(value_name), None) or getattr(
            settings, _setting_name(path_name), None
        ):
            indices.add(index)

    return sorted(indices)


def _configured_value(env_name: str) -> Optional[str]:
    """Read a config value, allowing environment-discovered slots beyond Settings."""
    value = os.getenv(env_name)
    if value is None:
        value = getattr(settings, _setting_name(env_name), None)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _get_credential(index: int) -> Optional[CredentialSource]:
    """Resolve one Firebase credential; inline JSON takes precedence over its path."""
    value_env, path_env = _env_names_for_account(index)
    inline_value = _configured_value(value_env)
    if inline_value:
        try:
            return json.loads(inline_value)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid {value_env} JSON: {e}")
            return None

    configured_path = _configured_value(path_env)
    if index == 1 and not configured_path:
        configured_path = "service-account.json"
    if not configured_path:
        return None

    path = settings.base_dir / configured_path
    if path.exists():
        return str(path)
    logger.warning(f"Firebase service account file for slot {index} not found: {path}")
    return None


def _firebase_app_name(index: int) -> str:
    """Return stable Firebase Admin app names, preserving legacy names for slots 1-3."""
    legacy_names = {1: "default", 2: "secondary", 3: "tertiary"}
    return legacy_names.get(index, f"firebase-{index}")


def _get_firebase_configs() -> List[FirebaseConfig]:
    """Build Firebase app configs for every discovered, valid credential slot."""
    configs: List[FirebaseConfig] = []
    for index in _configured_account_indices():
        credential = _get_credential(index)
        if credential is not None:
            configs.append((_firebase_app_name(index), credential))
        elif index == 1:
            logger.error(
                "Primary Firebase service account not configured "
                "(no inline value and path not found)"
            )
    return configs


def _get_primary_credential() -> Optional[Union[str, dict]]:
    """Resolve the primary Firebase credential (backwards-compatible helper)."""
    return _get_credential(1)


def _get_secondary_credential() -> Optional[Union[str, dict]]:
    """Resolve the secondary Firebase credential (backwards-compatible helper)."""
    return _get_credential(2)


def _get_tertiary_credential() -> Optional[Union[str, dict]]:
    """Resolve the tertiary Firebase credential (backwards-compatible helper)."""
    return _get_credential(3)


def _ensure_firebase():
    """
    Lazily initialize one or more Firebase apps for FCM verification.
    Supports a primary account and any number of optional numbered accounts.
    """
    global _firebase_initialized, _firebase_apps
    if _firebase_initialized:
        return
    try:
        import firebase_admin
        from firebase_admin import credentials

        firebase_configs = _get_firebase_configs()

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
