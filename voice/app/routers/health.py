from fastapi import APIRouter, HTTPException, status
from voice.app.utils import cache
from voice.app.config import settings
import time
from typing import Dict, Any

router = APIRouter(prefix="/health", tags=["health"])

# Track when the application started
START_TIME = time.time()

# Canary sentence — short enough to be cheap, long enough to exercise the
# full prompt template + tokenization + decode path on TranslateGemma.
_CANARY_EN = "Your cow produced 12 liters of milk today."
_CANARY_GU_SUBSTR = "દૂધ"  # "milk" in Gujarati — must appear in a valid translation


async def check_cache_connection() -> Dict[str, Any]:
    """Check Redis cache connection"""
    try:
        test_key = "health_check_test"
        test_value = "test"
        await cache.set(test_key, test_value, ttl=5)
        cached_value = await cache.get(test_key)
        return {
            "status": "healthy" if cached_value == test_value else "unhealthy",
            "latency_ms": 0  # TODO: Add actual latency measurement
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def check_translation_service() -> Dict[str, Any]:
    """Translate a canary sentence end-to-end through TranslateGemma."""
    from voice.app.services.translation import translate_text
    try:
        t0 = time.monotonic()
        result = await translate_text(
            text=_CANARY_EN,
            source_lang="english",
            target_lang="gujarati",
        )
        latency_ms = round((time.monotonic() - t0) * 1000)
        if not result or _CANARY_GU_SUBSTR not in result:
            return {
                "status": "unhealthy",
                "error": f"Unexpected translation output: {result[:100] if result else '(empty)'}",
                "latency_ms": latency_ms,
            }
        return {"status": "healthy", "latency_ms": latency_ms}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


async def check_llm_service() -> Dict[str, Any]:
    """Check that the LLM provider (OpenAI) is reachable with a lightweight models list call."""
    from openai import AsyncOpenAI
    try:
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        t0 = time.monotonic()
        await client.models.list()
        latency_ms = round((time.monotonic() - t0) * 1000)
        return {"status": "healthy", "latency_ms": latency_ms}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@router.get("/live", status_code=status.HTTP_200_OK)
async def liveness():
    """
    Liveness probe - simple check to see if the application is running
    Used by Kubernetes to know when to restart the pod
    """
    return {"status": "alive"}


@router.get("/ready", status_code=status.HTTP_200_OK)
async def readiness():
    """
    Readiness probe - checks if the application is ready to handle traffic.
    Runs actual translation + LLM reachability checks.
    Used by Kubernetes to know when to send traffic to the pod.
    """
    cache_health = await check_cache_connection()
    translation_health = await check_translation_service()
    llm_health = await check_llm_service()

    checks = {
        "cache": cache_health,
        "translation": translation_health,
        "llm": llm_health,
    }

    unhealthy = [k for k, v in checks.items() if v["status"] != "healthy"]
    if unhealthy:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not ready", **checks}
        )

    return {"status": "ready", **checks}


@router.get("/", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Full health check with uptime and all dependency statuses.
    """
    cache_health = await check_cache_connection()
    translation_health = await check_translation_service()
    llm_health = await check_llm_service()
    uptime_seconds = int(time.time() - START_TIME)

    health_status = {
        "app": {
            "name": settings.app_name,
            "environment": settings.environment,
            "uptime_seconds": uptime_seconds
        },
        "dependencies": {
            "cache": cache_health,
            "translation": translation_health,
            "llm": llm_health,
        }
    }

    unhealthy = [
        k for k, v in health_status["dependencies"].items()
        if v["status"] != "healthy"
    ]
    if unhealthy:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=health_status
        )

    return health_status
