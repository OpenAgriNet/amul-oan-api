"""Session request ownership locks (ported from voice backend)."""

from dataclasses import dataclass

from app.redis.cache import redis_client
from app.redis.config import SESSION_OWNER_TTL_SECONDS
from app.redis.config import key as redis_key

SESSION_OWNER_NAMESPACE = "session_owner"
SESSION_EPOCH_NAMESPACE = "session_epoch"


@dataclass(frozen=True)
class SessionRequestOwner:
    session_id: str
    request_token: str
    epoch: int


def _session_owner_key(session_id: str) -> str:
    return redis_key(SESSION_OWNER_NAMESPACE, session_id)


def _session_epoch_key(session_id: str) -> str:
    return redis_key(SESSION_EPOCH_NAMESPACE, session_id)


async def claim_session_request_ownership(session_id: str) -> SessionRequestOwner:
    epoch = await redis_client.incr(_session_epoch_key(session_id))
    request_token = f"{epoch}:{session_id}"
    await redis_client.set(
        _session_owner_key(session_id),
        request_token,
        ex=SESSION_OWNER_TTL_SECONDS,
    )
    return SessionRequestOwner(
        session_id=session_id,
        request_token=request_token,
        epoch=int(epoch),
    )


async def is_session_request_owner(owner: SessionRequestOwner | None) -> bool:
    if owner is None:
        return False
    current = await redis_client.get(_session_owner_key(owner.session_id))
    return current == owner.request_token


async def refresh_session_request_ownership(owner: SessionRequestOwner | None) -> bool:
    if owner is None:
        return False
    refreshed = await redis_client.eval(
        """
        if redis.call('get', KEYS[1]) == ARGV[1] then
            return redis.call('expire', KEYS[1], tonumber(ARGV[2]))
        end
        return 0
        """,
        1,
        _session_owner_key(owner.session_id),
        owner.request_token,
        str(SESSION_OWNER_TTL_SECONDS),
    )
    return bool(refreshed)


async def release_session_request_ownership(owner: SessionRequestOwner | None) -> bool:
    if owner is None:
        return False
    deleted = await redis_client.eval(
        """
        if redis.call('get', KEYS[1]) == ARGV[1] then
            return redis.call('del', KEYS[1])
        end
        return 0
        """,
        1,
        _session_owner_key(owner.session_id),
        owner.request_token,
    )
    return bool(deleted)

