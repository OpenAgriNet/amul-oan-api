"""The dashboard's numbers from the telemetry database, read-only.

For a server (the dashboard service), not a browser: send the key as X-API-Key.
Every query needs ``from`` and ``to``, in UTC days, both ends included.
"""
import hmac
from datetime import date
from typing import Literal, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.config import settings
from app.services import telemetry_query
from helpers.utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/telemetry", tags=["telemetry"])

def _require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    expected = (settings.telemetry_query_api_key or "").strip()
    if not expected or not settings.telemetry_dashboard_password:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Telemetry queries are not configured",
        )
    if not hmac.compare_digest((x_api_key or "").strip(), expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


def _day_range(first: date, last: date) -> tuple[date, date]:
    if first > last:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="'from' is after 'to'")
    return first, last


def _run(queries, **options):
    client = telemetry_query.dashboard_client()
    try:
        return [query(client, **options) for query in queries]
    finally:
        client.close()


def _answer(first: date, last: date, *queries, **options):
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
    first: date = Query(alias="from"),
    last: date = Query(alias="to"),
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
    first: date = Query(alias="from"),
    last: date = Query(alias="to"),
    granularity: Literal["hour", "day", "week", "month"] = "day",
):
    """Questions, sessions, users and failed turns per hour, day, week (from
    Monday) or month, for each channel. ``start`` is when the bucket begins, in
    UTC. Buckets without turns are left out."""
    body, series = _answer(first, last, telemetry_query.graph, bucket=granularity)
    return {**body, "granularity": granularity, **series}


@router.get("/sessions", dependencies=[Depends(_require_api_key)])
def get_sessions(
    first: date = Query(alias="from"),
    last: date = Query(alias="to"),
):
    """Average questions per session, and seconds from its first turn to its
    last, per channel. Only sessions with a question count; null when there
    were none."""
    body, averages = _answer(first, last, telemetry_query.sessions)
    return {**body, **averages}


@router.get("/outcomes", dependencies=[Depends(_require_api_key)])
def get_outcomes(
    first: date = Query(alias="from"),
    last: date = Query(alias="to"),
):
    """Turns, sessions and users per outcome class, per channel: delivered,
    failed, and for voice also non_question and refused_or_blocked. Turns with
    no outcome recorded are not_recorded; a rate should leave them out."""
    body, classes = _answer(first, last, telemetry_query.outcomes)
    return {**body, **classes}


# ── for the dashboard service ───────────────────────────────────────────────

Dimension = Literal[
    "environment", "schema_version", "source_era", "pipeline_profile", "source_lang", "target_lang",
    "outcome", "outcome_class", "route", "call_type", "provider", "signed_in", "chat_channel",
    "pipeline", "persona", "served_tier",
]


class TurnFilters(BaseModel):
    """Which turns count. A filter on something a channel doesn't record (a
    voice-only one on chat, say) leaves that channel out: it answers null."""

    environment: Optional[str] = Field(
        None, description="One environment, or 'all'. By default each channel's production one."
    )
    schema_version: Optional[str] = None
    source_era: Optional[str] = None
    pipeline_profile: Optional[str] = None
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None
    outcome: Optional[str] = None
    outcome_class: Optional[str] = None
    route: Optional[str] = Field(None, description="Voice only.")
    call_type: Optional[str] = Field(None, description="Voice only: inbound or outbound.")
    provider: Optional[str] = Field(None, description="Voice only.")
    signed_in: Optional[Literal["true", "false"]] = Field(None, description="Voice only.")
    chat_channel: Optional[str] = Field(None, description="Chat only: web or whatsapp.")
    pipeline: Optional[str] = Field(None, description="Chat only.")
    persona: Optional[str] = Field(None, description="Chat only.")
    served_tier: Optional[str] = Field(None, description="Chat only.")

    def match(self) -> dict[str, str]:
        return {name: value for name, value in self.model_dump(exclude={"environment"}).items() if value is not None}


@router.get("/overview", dependencies=[Depends(_require_api_key)])
def get_overview(
    first: date = Query(alias="from"),
    last: date = Query(alias="to"),
    filters: TurnFilters = Depends(),
):
    """Per channel, for the range and for each day in it: turns, questions,
    delivered, failed, refused_or_blocked and non_question turns, turns with no
    outcome recorded, the delivered rate (of questions with an outcome),
    sessions, known users, anonymous turns and sessions, and p50/p95 full-turn
    latency in ms."""
    body, result = _answer(
        first, last, telemetry_query.overview, environment=filters.environment, match=filters.match()
    )
    return {**body, **result}


@router.get("/breakdown", dependencies=[Depends(_require_api_key)])
def get_breakdown(
    by: Dimension,
    first: date = Query(alias="from"),
    last: date = Query(alias="to"),
    filters: TurnFilters = Depends(),
):
    """The overview's numbers per value of ``by``, most turns first, each with
    its share of the channel's turns. Null for a channel that doesn't record
    ``by``. ``by=outcome`` or ``outcome_class`` gives outcome counts and rates."""
    body, result = _answer(
        first, last, telemetry_query.breakdown, by=by, environment=filters.environment, match=filters.match()
    )
    return {**body, "by": by, **result}


@router.get("/latency", dependencies=[Depends(_require_api_key)])
def get_latency(
    first: date = Query(alias="from"),
    last: date = Query(alias="to"),
    by: Optional[Dimension] = None,
    filters: TurnFilters = Depends(),
):
    """p50/p95/p99 full-turn latency in ms per channel, also per value of
    ``by`` when given, and voice's per stage. Chat doesn't time its stages, so
    its stages are null."""
    body, result = _answer(
        first, last, telemetry_query.latency, by=by, environment=filters.environment, match=filters.match()
    )
    return {**body, **result}


@router.get("/tools", dependencies=[Depends(_require_api_key)])
def get_tools(
    first: date = Query(alias="from"),
    last: date = Query(alias="to"),
    filters: TurnFilters = Depends(),
):
    """Turns that used each tool, and turns by how many tool calls they made,
    with their share of all turns. Voice keeps no tool names yet, so it is null."""
    body, result = _answer(
        first, last, telemetry_query.tools, environment=filters.environment, match=filters.match()
    )
    return {**body, **result}


@router.get("/health", dependencies=[Depends(_require_api_key)])
def get_health(
    first: date = Query(alias="from"),
    last: date = Query(alias="to"),
    environment: Optional[str] = Query(
        None, description="One environment, or 'all'. By default each channel's production one."
    ),
):
    """Per channel: each day's import (traces read, turns, rejected), when each
    environment was last imported, how many root traces became turns, were
    rejected, were known activity or weren't recognised, the reasons for the
    last two, and the schema versions and eras the turns came in."""
    body, result = _answer(first, last, telemetry_query.health, environment=environment)
    return {**body, **result}


@router.get("/turns", dependencies=[Depends(_require_api_key)])
def get_turns(
    channel: Literal["voice", "chat"],
    first: date = Query(alias="from"),
    last: date = Query(alias="to"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0, le=100_000),
    filters: TurnFilters = Depends(),
):
    """One page of a channel's turns, newest first: what the turn was and how
    it went, never its text, the hash of its text or the caller's id."""
    match = filters.match()
    unknown = sorted(set(match) - set(telemetry_query.DIMENSIONS[channel]))
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"{channel} turns don't record {', '.join(unknown)}"
        )
    body, page = _answer(
        first, last, telemetry_query.turn_page,
        channel=channel, limit=limit, offset=offset, environment=filters.environment, match=match,
    )
    return {**body, "channel": channel, "limit": limit, "offset": offset, **page}
