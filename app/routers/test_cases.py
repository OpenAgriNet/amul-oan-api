"""Dev-only test case intake for the amul-ui test case screen (#487).

Mounted only when TEST_CASES_ROUTE_ENABLED is on (main.py). One case per request:
the screen's bulk upload sends one request per case. The body is checked against
the v1 contract (app/models/test_cases.py) and stored exactly as it came, under an
id made from the body. Labelling (#488) reads the stored submissions. Turn text
never goes to the logs. GET /options/ gives the screen its dropdown lists.
"""
import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.requests import ClientDisconnect

from app.auth.jwt_auth import get_chat_user
from app.config import settings
from app.models.case_options import CASE_OPTIONS, CaseOptions
from app.models.test_cases import CASE_SOURCES, CaseSubmission, SubmitTestCaseResponse
from app.services.test_case_store import CaseStore, get_case_store
from helpers.utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/test-cases", tags=["test-cases"])

# The id comes from the body, so the same case sent again (a retry after a timeout,
# a double click, the same sheet twice) gets the same id and the store keeps one
# copy. The dataset name is part of it because Langfuse item ids are unique across
# a project.
_CASE_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "amul-oan-api/test-cases")

# A body of unknown keys can fail tens of thousands of times; the screen needs a few.
_MAX_REPORTED = 20
_MAX_TEXT = 200


def _error(status_code: int, detail: Any, **extra: Any) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"status": "error", "detail": detail, **extra})


def _failed_rules(exc: ValidationError) -> list[dict[str, Any]]:
    """Each failed rule as a field path and a message the screen can show."""
    rules = []
    for err in exc.errors(include_url=False, include_context=False, include_input=False)[:_MAX_REPORTED]:
        loc = list(err["loc"])
        if err["type"] in ("union_tag_not_found", "union_tag_invalid"):
            loc = ["source"]
        elif loc and loc[0] in CASE_SOURCES:
            loc = loc[1:]  # the source the case was checked as, not a field
        loc = [part[:_MAX_TEXT] if isinstance(part, str) else part for part in loc]
        rules.append({"loc": loc, "msg": err["msg"].removeprefix("Value error, ")[:_MAX_TEXT]})
    return rules


def _case_id(canonical_body: bytes) -> str:
    dataset = (settings.test_cases_raw_dataset or "").strip()
    digest = hashlib.sha256(canonical_body).hexdigest()
    return str(uuid.uuid5(_CASE_ID_NAMESPACE, f"{dataset}:{digest}"))


@router.post(
    "/",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=SubmitTestCaseResponse,
)
async def submit_test_case(
    request: Request,
    _user: dict = Depends(get_chat_user),
    store: Optional[CaseStore] = Depends(get_case_store),
):
    """Check one test case against the v1 contract and keep it as sent.

    400 bad JSON or a cut-off body, 413 too big, 422 a broken rule (up to 20 in
    ``errors``, the first in ``detail``), 503 no store or the store failed. The body
    is checked before the store, so a client can see its 422s even while no store
    is configured.
    """
    limit = settings.test_cases_max_body_bytes
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > limit:
                logger.warning("submit_test_case rejected=body_too_large content_length=%s", content_length)
                return _error(status.HTTP_413_CONTENT_TOO_LARGE, "Payload too large")
        except ValueError:
            return _error(status.HTTP_400_BAD_REQUEST, "Invalid Content-Length header")

    # Read with the cap, so a chunked body without Content-Length can't fill memory.
    raw_body = bytearray()
    try:
        async for chunk in request.stream():
            raw_body += chunk
            if len(raw_body) > limit:
                logger.warning("submit_test_case rejected=body_too_large bytes>%s", limit)
                return _error(status.HTTP_413_CONTENT_TOO_LARGE, "Payload too large")
    except ClientDisconnect:
        # The tester closed the tab or lost the network mid-upload.
        logger.info("submit_test_case rejected=client_disconnected")
        return _error(status.HTTP_400_BAD_REQUEST, "Request body incomplete")

    try:
        payload = json.loads(raw_body)
    except (ValueError, RecursionError):
        return _error(status.HTTP_400_BAD_REQUEST, "Invalid JSON payload")
    if not isinstance(payload, dict):
        return _error(status.HTTP_400_BAD_REQUEST, "Payload must be a JSON object")

    try:
        # Off the event loop: a big body of bad keys takes a moment to check.
        case = await run_in_threadpool(CaseSubmission.validate_python, payload)
    except ValidationError as exc:
        rules = _failed_rules(exc)
        first = rules[0]
        detail = ".".join(map(str, first["loc"])) + ": " + first["msg"] if first["loc"] else first["msg"]
        logger.info("submit_test_case rejected=invalid errors=%s", len(rules))
        return _error(status.HTTP_422_UNPROCESSABLE_CONTENT, detail, errors=rules)

    try:
        # One byte form per case, whatever the key order, for the id.
        canonical_body = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except UnicodeEncodeError:
        # A lone surrogate from a "\ud83d" escape: valid JSON, but not text.
        return _error(status.HTTP_400_BAD_REQUEST, "Invalid JSON payload")

    if store is None:
        logger.warning("submit_test_case rejected=no_store")
        return _error(status.HTTP_503_SERVICE_UNAVAILABLE, "Test case store is not configured")

    case_id = _case_id(canonical_body)
    received_at = datetime.now(timezone.utc).isoformat()
    try:
        await run_in_threadpool(store.save, case_id=case_id, raw=payload, received_at=received_at)
    except Exception as exc:
        logger.error("submit_test_case store_failed id=%s error=%s", case_id, type(exc).__name__)
        return _error(status.HTTP_503_SERVICE_UNAVAILABLE, "Test case store is unavailable")

    logger.info("submit_test_case stored id=%s source=%s turns=%s", case_id, case.source, len(case.turns))
    return SubmitTestCaseResponse(id=case_id)


@router.get("/options/", response_model=CaseOptions)
async def case_options(_user: dict = Depends(get_chat_user)) -> CaseOptions:
    """The screen's dropdowns: testers, test accounts, case types and tags."""
    return CASE_OPTIONS
