"""Dev-only test case API for the amul-ui test case screen (#487, #517, #488).

Mounted only when TEST_CASES_ROUTE_ENABLED is on (main.py). POST / keeps one case as
sent. GET /options/ gives the screen its lists. GET / lists the cases as Langfuse keeps
them, with phone and Aadhaar numbers cut. Both queue cases with no label for the
labeller. Turn text never goes to the logs.
"""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.auth.jwt_auth import get_chat_user
from app.services import test_cases as cases
from helpers.utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/test-cases", tags=["test-cases"], dependencies=[Depends(get_chat_user)])


def _error(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"status": "error", "detail": detail})


@router.post("/", status_code=202)
async def submit_test_case(request: Request):
    """Keep one case as sent. 400 not a JSON object, 422 no turns, 503 no store."""
    try:
        body = await request.json()
    except (ValueError, RecursionError):
        body = None
    if not isinstance(body, dict):
        return _error(400, "Payload must be a JSON object")
    turns = body.get("turns")
    if not isinstance(turns, list) or not turns:
        return _error(422, "Add at least one turn.")
    store = cases.store()
    if store is None:
        return _error(503, "Test case store is not configured")
    try:
        item_id = cases.case_id(body)
    except (ValueError, RecursionError):  # NaN, Infinity or a lone surrogate
        return _error(400, "Invalid JSON payload")
    try:
        # The id comes from the body, so a case sent again is stored already: it keeps its
        # label, time and status.
        if await run_in_threadpool(cases.get_item, store[0], item_id) is None:
            metadata = {"received_at": datetime.now(timezone.utc).isoformat()}
            await run_in_threadpool(cases.save_case, *store, item_id, input=body, metadata=metadata)
    except Exception as exc:
        logger.error("submit_test_case store_failed id=%s error=%s", item_id, type(exc).__name__)
        return _error(503, "Test case store is unavailable")
    source = body.get("source") if body.get("source") in ("chat", "manual", "bulk") else "other"
    logger.info("submit_test_case stored id=%s source=%s turns=%s", item_id, source, len(turns))
    cases.label_later(*store, item_id)
    return {"id": item_id, "status": "received"}


@router.get("/options/")
async def case_options():
    """Testers, case types and tags for the screen, plus the labeller's groups and answer types."""
    return cases.OPTIONS


@router.get("/")
async def list_test_cases():
    """Every case that isn't archived, newest first, as Langfuse keeps it: {id, input,
    metadata}. metadata.received_at is when it came in, and metadata.label its group and
    answer type once it is labelled."""
    store = cases.store()
    if store is None:
        return _error(503, "Test case store is not configured")
    try:
        items = await run_in_threadpool(cases.list_items, *store)
    except Exception as exc:
        logger.error("list_test_cases store_failed error=%s", type(exc).__name__)
        return _error(503, "Test case store is unavailable")
    for item in items:
        if cases.label_of(item) is None:  # sent while the labeller was off, or its labelling failed
            cases.label_later(*store, item.id)

    def shown() -> list:
        return [
            {"id": item.id, "input": cases.mask(item.input), "metadata": cases.mask(item.metadata)} for item in items
        ]

    # Masking a big dataset takes a while, so it runs off the event loop. JSONResponse uses
    # the C encoder, so a deeply nested case can't break the list.
    return JSONResponse({"items": await run_in_threadpool(shown)})
