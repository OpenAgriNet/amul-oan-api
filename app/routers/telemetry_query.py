"""The dashboard's numbers from the telemetry database, read-only.

For a server (the dashboard service), not a browser: send the key as X-API-Key.
Days are UTC and both ends are included. With no ``from`` the range starts at
the first turn on record, and with no ``to`` it ends today.
"""
import hmac
from datetime import date, datetime, timezone
from typing import Optional

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


def _run(query, **days):
    client = telemetry_query.dashboard_client()
    try:
        return query(client, **days)
    finally:
        client.close()


def _answer(query, first: Optional[date], last: Optional[date]):
    first_day, last_day = _day_range(first, last)
    try:
        return first_day, last_day, _run(query, first_day=first_day, last_day=last_day)
    except Exception as e:
        logger.error("telemetry query failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Telemetry database unavailable",
        ) from e


@router.get("/stats", dependencies=[Depends(_require_api_key)])
def get_stats(
    first: Optional[date] = Query(default=None, alias="from"),
    last: Optional[date] = Query(default=None, alias="to"),
):
    """Questions, sessions and users per channel, and both together. A farmer
    who used voice and chat is a user in each, since their ids can't be matched."""
    first_day, last_day, counts = _answer(telemetry_query.totals, first, last)
    return {
        "from": first_day.isoformat(),
        "to": last_day.isoformat(),
        "voice": counts["voice"].as_dict(),
        "chat": counts["chat"].as_dict(),
        "total": (counts["voice"] + counts["chat"]).as_dict(),
    }


@router.get("/daily", dependencies=[Depends(_require_api_key)])
def get_daily(
    first: Optional[date] = Query(default=None, alias="from"),
    last: Optional[date] = Query(default=None, alias="to"),
):
    """The same counts per day and channel, for graphs. Days without turns are left out."""
    first_day, last_day, series = _answer(telemetry_query.daily, first, last)
    return {
        "from": first_day.isoformat(),
        "to": last_day.isoformat(),
        **{
            channel: [{"day": day.isoformat(), **counts.as_dict()} for day, counts in days]
            for channel, days in series.items()
        },
    }
