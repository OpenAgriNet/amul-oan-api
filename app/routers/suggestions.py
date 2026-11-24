import asyncio
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from app.models.requests import SuggestionsRequest
from app.utils import get_cache
from app.auth.jwt_auth import get_current_user
from app.agui.session import storage_session_id
from app.personas import resolve_chat_persona

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
    session_id = request.session_id
    if request.scope == "agui-v1":
        from agents.tools.farmer import normalize_phone_to_mobile

        owner = str(user_info.get("sub") or "")
        persona = resolve_chat_persona(user_info)
        mobile = normalize_phone_to_mobile(str(user_info.get("phone") or ""))
        if not owner or not mobile or user_info.get("anonymous") or persona != "farmer":
            raise HTTPException(status_code=403, detail="Sign in as a farmer to view AG-UI suggestions.")
        session_id = storage_session_id(owner, persona, request.session_id)
    cache_key = f"suggestions_{session_id}_{request.target_lang}"
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
