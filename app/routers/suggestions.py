from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from app.auth.jwt_auth import get_current_user
from app.models.requests import SuggestionsRequest
from app.redis.cache import get_cache, set_cache
from app.redis.config import SUGGESTIONS_TTL_SECONDS, key as redis_key
from app.tasks.suggestions import create_suggestions
from helpers.utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/suggest", tags=["suggest"])


@router.get("/")
async def suggest(
    request: SuggestionsRequest = Depends(),
    user_info: dict = Depends(get_current_user),
):
    """
    Cache-aside: read suggestions from Redis; on miss, generate, persist, then return.
    """
    cache_key = redis_key("suggestions", f"{request.session_id}:{request.target_lang}")

    try:
        suggestions = await get_cache(cache_key)
    except Exception as e:
        logger.warning(
            "suggestions cache read failed for key=%s: %s",
            cache_key,
            e,
            exc_info=True,
        )
        suggestions = None

    if suggestions is not None:
        return JSONResponse(suggestions)

    try:
        suggestions = await create_suggestions(request.session_id, request.target_lang)
    except Exception as e:
        logger.error(
            "create_suggestions failed for session=%s: %s",
            request.session_id,
            e,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to generate suggestions",
        ) from e

    try:
        await set_cache(cache_key, suggestions, ttl=SUGGESTIONS_TTL_SECONDS)
    except Exception as e:
        logger.error(
            "suggestions cache write failed for key=%s: %s",
            cache_key,
            e,
            exc_info=True,
        )
        raise HTTPException(
            status_code=503,
            detail="Failed to persist suggestions",
        ) from e

    return JSONResponse(suggestions)
