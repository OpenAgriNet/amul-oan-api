"""The dashboard's numbers from the telemetry database, read-only.

For a server (the dashboard service), not a browser: send the key as X-API-Key.
Days are UTC and both ends are included. With no ``from`` the range starts at
the first turn on record, and with no ``to`` it ends today.
"""
import hmac
from datetime import date, datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status

from app.config import settings
from app.services import telemetry_query
from helpers.utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/telemetry", tags=["telemetry"])

# ClickHouse's first Date: "from the start".
_FIRST_DAY = date(1970, 1, 1)


def _require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    expected = (settings.telemetry_query_api_key or "").strip()
    if not expected or not settings.telemetry_dashboard_password:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Telemetry queries are not configured",
        )
    if not hmac.compare_digest((x_api_key or "").strip(), expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


def _day_range(first: Optional[date], last: Optional[date]) -> tuple[date, date]:
    first_day = first or _FIRST_DAY
    last_day = last or datetime.now(timezone.utc).date()
    if first_day > last_day:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="'from' is after 'to'")
    return first_day, last_day


def _run(queries, **options):
    client = telemetry_query.dashboard_client()
    try:
        return [query(client, **options) for query in queries]
    finally:
        client.close()


def _answer(first: Optional[date], last: Optional[date], *queries, **options):
    first_day, last_day = _day_range(first, last)
    try:
        results = _run(queries, first_day=first_day, last_day=last_day, **options)
    except Exception as e:
        logger.error("telemetry query failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Telemetry database unavailable",
        ) from e
    return {"from": first_day.isoformat(), "to": last_day.isoformat()}, *results


@router.get("/stats", dependencies=[Depends(_require_api_key)])
def get_stats(
    first: Optional[date] = Query(default=None, alias="from"),
    last: Optional[date] = Query(default=None, alias="to"),
):
    """Questions, sessions, users and new users per channel, and both together.
    A farmer who used voice and chat is a user in each, since their ids can't
    be matched. A new user's first turn on record is in the range."""
    body, counts, new = _answer(first, last, telemetry_query.totals, telemetry_query.new_users)
    for channel in ("voice", "chat"):
        body[channel] = {**counts[channel].as_dict(), "new_users": new[channel]}
    body["total"] = {
        **(counts["voice"] + counts["chat"]).as_dict(),
        "new_users": new["voice"] + new["chat"],
    }
    return body


@router.get("/graph", dependencies=[Depends(_require_api_key)])
def get_graph(
    first: Optional[date] = Query(default=None, alias="from"),
    last: Optional[date] = Query(default=None, alias="to"),
    granularity: Literal["hour", "day", "week", "month"] = "day",
):
    """Questions, sessions, users and failed turns per hour, day, week (from
    Monday) or month, for each channel. ``start`` is when the bucket begins, in
    UTC. Buckets without turns are left out."""
    body, series = _answer(first, last, telemetry_query.graph, bucket=granularity)
    return {**body, "granularity": granularity, **series}


@router.get("/sessions", dependencies=[Depends(_require_api_key)])
def get_sessions(
    first: Optional[date] = Query(default=None, alias="from"),
    last: Optional[date] = Query(default=None, alias="to"),
):
    """Average questions per session, and seconds from its first turn to its
    last, per channel. Only sessions with a question count; null when there
    were none."""
    body, averages = _answer(first, last, telemetry_query.sessions)
    return {**body, **averages}


@router.get("/outcomes", dependencies=[Depends(_require_api_key)])
def get_outcomes(
    first: Optional[date] = Query(default=None, alias="from"),
    last: Optional[date] = Query(default=None, alias="to"),
):
    """Turns, sessions and users per outcome class, per channel: delivered,
    failed, and for voice also non_question and refused_or_blocked. Turns with
    no outcome recorded are not_recorded; a rate should leave them out."""
    body, classes = _answer(first, last, telemetry_query.outcomes)
    return {**body, **classes}
