from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from app.models.requests import SuggestionsRequest
from app.redis.cache import get_cache
from app.redis.config import key as redis_key
from app.tasks.suggestions import create_suggestions
from app.auth.jwt_auth import get_current_user

router = APIRouter(prefix="/suggest", tags=["suggest"])

@router.get("/")
async def suggest(request: SuggestionsRequest = Depends(), user_info: dict = Depends(get_current_user)):
    """
    Get suggestions for a conversation session.
    On cache miss, generate suggestions and return them in the same response.
    """
    cache_key = redis_key("suggestions", f"{request.session_id}:{request.target_lang}")
    suggestions = await get_cache(cache_key)
    if suggestions is None:
        suggestions = await create_suggestions(request.session_id, request.target_lang)
    suggestions = suggestions or []
    return JSONResponse(suggestions)
