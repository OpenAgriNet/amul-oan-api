import asyncio
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from app.models.requests import SuggestionsRequest
from app.utils import get_cache
from app.auth.jwt_auth import get_current_user

router = APIRouter(prefix="/suggest", tags=["suggest"])
SUGGESTIONS_PENDING_TTL = 30
SUGGESTIONS_WAIT_TIMEOUT_SECONDS = 8.0
SUGGESTIONS_WAIT_INTERVAL_SECONDS = 0.2

@router.get("/")
async def suggest(request: SuggestionsRequest = Depends(), user_info: dict = Depends(get_current_user)):
    """
    Get suggestions for a conversation session.
    Read suggestions from cache only. Generation happens in background from chat flow.
    On cache miss while background generation is pending, wait briefly for cache to fill.
    """
    cache_key = f"suggestions_{request.session_id}_{request.target_lang}"
    status_key = f"{cache_key}:pending"
    suggestions = await get_cache(cache_key)
    if suggestions is None:
        pending = await get_cache(status_key)
        if pending:
            deadline = asyncio.get_running_loop().time() + SUGGESTIONS_WAIT_TIMEOUT_SECONDS
            while asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(SUGGESTIONS_WAIT_INTERVAL_SECONDS)
                suggestions = await get_cache(cache_key)
                if suggestions is not None:
                    break
                pending = await get_cache(status_key)
                if not pending:
                    break
    suggestions = suggestions or []
    return JSONResponse(suggestions)
