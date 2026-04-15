from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from app.auth.jwt_auth import get_current_user
from app.models.requests import SuggestionsRequest
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
    Generate suggestions from current message history on every request.

    No Redis caching: each call runs the suggestions agent against the latest history.
    """
    _ = user_info  # JWT validated; reserved for future audit/rate limits

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

    return JSONResponse(suggestions)
