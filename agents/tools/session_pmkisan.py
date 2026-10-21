"""Short-lived, session-owned state for the two-turn PM-KISAN OTP flow."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

from app.config import settings
from app.core.cache import build_cache_key, cache, redis_client
from helpers.utils import get_logger, to_ascii_digits

logger = get_logger(__name__)

PMKISAN_SESSION_NAMESPACE = "pmkisan-session"
PMKISAN_RATE_NAMESPACE = "pmkisan-otp-requests"
OTP_PLACEHOLDER = "[OTP]"
# OTP SMS requests allowed per chat session and per identifier in one window.
MAX_OTP_REQUESTS = 3
OTP_REQUEST_WINDOW_SECONDS = 60 * 60

_SEP = r"[\s.\-]"
# 12 digits in any script, with up to two spaces, dots or hyphens between digits.
_AADHAAR_RE = re.compile(
    rf"(?<!\d)(?<!\d{_SEP})(?<!\d{_SEP}{_SEP})\d(?:{_SEP}{{0,2}}\d){{11}}(?!{_SEP}{{0,2}}\d)"
)
_AADHAAR_PLACEHOLDER_RE = re.compile(r"AADHAAR_(\d{4})", re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(r"\[(?:AADHAAR_\d{4}|OTP)\]")
_OTP_RE = re.compile(r"\d(?:[\s\-]{0,2}\d){3}")

# Verhoeff tables: UIDAI's check digit, so cattle ear tags rarely look like Aadhaar.
_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6), (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8), (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2), (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4), (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2), (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0), (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5), (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


def _is_aadhaar(digits: str) -> bool:
    if not re.fullmatch(r"[2-9]\d{11}", digits):
        return False
    check = 0
    for position, digit in enumerate(reversed(digits)):
        check = _VERHOEFF_D[check][_VERHOEFF_P[position % 8][int(digit)]]
    return check == 0


def owner_mobile(user_info: Mapping[str, Any] | None, persona: str) -> str | None:
    """The mobile a chat turn's FarmerContext gets: the JWT phone, farmer persona only."""
    from agents.tools.farmer import normalize_phone_to_mobile

    if persona != "farmer" or not user_info or not user_info.get("phone"):
        return None
    return normalize_phone_to_mobile(user_info["phone"])


def _owner_token(mobile: str | None) -> str:
    digits = re.sub(r"\D", "", mobile or "")[-10:]
    owner = digits or "anonymous"
    return hashlib.sha256(owner.encode("utf-8")).hexdigest()


def _cache_key(session_id: str, mobile: str | None) -> str:
    return f"{session_id}:{_owner_token(mobile)}"


def _ids_key(session_id: str, mobile: str | None) -> str:
    return f"{_cache_key(session_id, mobile)}:ids"


def _otp_key(session_id: str, mobile: str | None) -> str:
    return f"{_cache_key(session_id, mobile)}:otp"


async def get_pm_kisan_session(
    session_id: str | None, mobile: str | None
) -> dict[str, Any] | None:
    if not session_id:
        return None
    value = await cache.get(
        _cache_key(session_id, mobile), namespace=PMKISAN_SESSION_NAMESPACE
    )
    return dict(value) if isinstance(value, Mapping) else None


async def set_pm_kisan_session(
    session_id: str | None,
    mobile: str | None,
    state: Mapping[str, Any],
) -> None:
    if not session_id:
        raise ValueError("session_id is required for PM-KISAN status checks")
    await cache.set(
        _cache_key(session_id, mobile),
        dict(state),
        ttl=settings.vistaar_pmkisan_session_ttl_seconds,
        namespace=PMKISAN_SESSION_NAMESPACE,
    )


async def clear_pm_kisan_session(
    session_id: str | None, mobile: str | None
) -> None:
    if not session_id:
        return
    await cache.delete(
        _cache_key(session_id, mobile), namespace=PMKISAN_SESSION_NAMESPACE
    )


def _mask_aadhaar(text: str) -> tuple[str, dict[str, str]]:
    found: dict[str, str] = {}

    def _replace(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", to_ascii_digits(match.group(0)))
        if not _is_aadhaar(digits):
            return match.group(0)
        found[digits[-4:]] = digits
        return f"[AADHAAR_{digits[-4:]}]"

    return _AADHAAR_RE.sub(_replace, text), found


def _otp_match(text: str) -> re.Match[str] | None:
    """The message's only digits when there are exactly four, written together or spaced.

    A date, an amount next to another number, or any longer number is not an OTP.
    """
    bare = _PLACEHOLDER_RE.sub(lambda match: "#" * len(match.group(0)), text)
    if len(re.findall(r"\d", bare)) != 4:
        return None
    return _OTP_RE.search(bare)


def _otp_digits(match: re.Match[str]) -> str:
    return re.sub(r"\D", "", to_ascii_digits(match.group(0)))


def _replace_span(text: str, match: re.Match[str]) -> str:
    return text[: match.start()] + OTP_PLACEHOLDER + text[match.end() :]


async def mask_private_numbers(
    query: str, session_id: str | None, mobile: str | None
) -> str:
    """Hide Aadhaar numbers, and the OTP of a pending check, in a farmer message.

    Runs before the message is logged, translated, traced or stored. The real
    values stay in this session's short-lived state, where only the PM-KISAN
    tools read them, so the model and every record see placeholders.
    """
    if not settings.vistaar_pmkisan_enabled or not session_id or not query:
        return query
    masked, found = _mask_aadhaar(query)
    otp = _otp_match(masked)
    try:
        pending = otp is not None and await get_pm_kisan_session(session_id, mobile) is not None
        if found or pending:
            ids_key = _ids_key(session_id, mobile)
            known = await cache.get(ids_key, namespace=PMKISAN_SESSION_NAMESPACE)
            known = dict(known) if isinstance(known, Mapping) else {}
            known["aadhaar"] = {**known.get("aadhaar", {}), **found}
            if pending:
                known["otps"] = [*known.get("otps", []), _otp_digits(otp)][-5:]
                await cache.set(
                    _otp_key(session_id, mobile),
                    _otp_digits(otp),
                    ttl=settings.vistaar_pmkisan_session_ttl_seconds,
                    namespace=PMKISAN_SESSION_NAMESPACE,
                )
            await cache.set(
                ids_key,
                known,
                ttl=settings.vistaar_pmkisan_session_ttl_seconds,
                namespace=PMKISAN_SESSION_NAMESPACE,
            )
    except Exception as exc:  # noqa: BLE001 - masking must never block a chat turn
        logger.warning("PM-KISAN private number state unavailable: %s", type(exc).__name__)
        # Fail closed: without state, any lone 4-digit number may be an OTP.
        pending = otp is not None
    return _replace_span(masked, otp) if pending else masked


async def mask_telemetry_text(
    text: str, session_id: str | None, mobile: str | None
) -> str:
    """A farmer question as UI telemetry reports it, masked like the chat query.

    Only an OTP this session already sent to a check is hidden, so ordinary
    numbers in questions stay readable.
    """
    if not settings.vistaar_pmkisan_enabled or not text:
        return text
    masked, _ = _mask_aadhaar(text)
    otp = _otp_match(masked)
    if otp is None:
        return masked
    try:
        known = (
            await cache.get(_ids_key(session_id, mobile), namespace=PMKISAN_SESSION_NAMESPACE)
            if session_id
            else None
        )
        sent = isinstance(known, Mapping) and _otp_digits(otp) in known.get("otps", [])
    except Exception as exc:  # noqa: BLE001 - telemetry must never fail on masking
        logger.warning("PM-KISAN private number state unavailable: %s", type(exc).__name__)
        sent = True
    return _replace_span(masked, otp) if sent else masked


async def get_pending_otp(session_id: str | None, mobile: str | None) -> str:
    if not session_id:
        return ""
    value = await cache.get(_otp_key(session_id, mobile), namespace=PMKISAN_SESSION_NAMESPACE)
    return value if isinstance(value, str) else ""


async def clear_pending_otp(session_id: str | None, mobile: str | None) -> None:
    if session_id:
        await cache.delete(_otp_key(session_id, mobile), namespace=PMKISAN_SESSION_NAMESPACE)


async def resolve_identifier(
    identifier: str, session_id: str | None, mobile: str | None
) -> str | None:
    """The real value behind an [AADHAAR_1234] placeholder, else the identifier.

    None when the placeholder's number is no longer held for this session.
    """
    match = _AADHAAR_PLACEHOLDER_RE.search(identifier or "")
    if match is None:
        return identifier
    if not session_id:
        return None
    known = await cache.get(
        _ids_key(session_id, mobile), namespace=PMKISAN_SESSION_NAMESPACE
    )
    aadhaar = known.get("aadhaar") if isinstance(known, Mapping) else None
    return aadhaar.get(match.group(1)) if isinstance(aadhaar, Mapping) else None


# INCR and the window's expiry in one step, so a key can never lose its expiry.
_COUNT_SCRIPT = """
local count = redis.call('INCR', KEYS[1])
if redis.call('TTL', KEYS[1]) < 0 then
    redis.call('EXPIRE', KEYS[1], tonumber(ARGV[1]))
end
return count
"""


async def allow_otp_request(session_id: str, identifier: str) -> bool:
    """Count one OTP request against the session and the identifier."""
    identifier_hash = hashlib.sha256(identifier.encode("utf-8")).hexdigest()
    allowed = True
    for key in (f"session:{session_id}", f"id:{identifier_hash}"):
        full_key = build_cache_key(key, namespace=PMKISAN_RATE_NAMESPACE)
        count = await redis_client.eval(_COUNT_SCRIPT, 1, full_key, str(OTP_REQUEST_WINDOW_SECONDS))
        if int(count) > MAX_OTP_REQUESTS:
            allowed = False
    return allowed
