from fastapi import APIRouter, HTTPException, status
from app.utils import cache
from app.config import settings
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
    """Translate a canary sentence end-to-end through TranslateGemma.

    Deep dependency check for the rich /health/ endpoint only — NOT wired into
    the readiness gate (see merge decision #14): a translation blip must not pull
    pods from rotation, and request-time OSS→managed fallback keeps serving.
    """
    from app.services.translation import translate_text
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
    """Check the LLM provider (OpenAI) is reachable via a lightweight models list.

    Deep dependency check for /health/ only — not part of the readiness gate (#14).
    """
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
    Used by Kubernetes to know when to send traffic to the pod.

    Deliberately CHEAP (merge decision #14): cache only. Translation/LLM are NOT
    checked here — a third-party/model blip would otherwise fail readiness on
    every pod at once and pull the whole fleet from rotation. Those deep checks
    live on /health/ for monitoring; request-time OSS→managed fallback covers
    model degradation.
    """
    cache_health = await check_cache_connection()
    
    if cache_health["status"] != "healthy":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not ready", "cache": cache_health}
        )
    
    return {"status": "ready", "cache": cache_health}

@router.get("/", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Rich health report (for monitoring/alerting, NOT a k8s probe — decision #14):
    - Application metadata (name, environment, uptime)
    - Hard dependency: Redis cache
    - Deep dependencies: translation (TranslateGemma canary) + LLM (OpenAI)

    503 policy: only a HARD dependency (cache) failing returns 503 — that means
    the pod genuinely can't function. Translation/LLM are surfaced in the body
    (status + latency_ms) but are NON-FATAL here, so a model blip shows on
    dashboards without making this endpoint flap.
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

    # Only a hard dependency (cache) failing makes /health/ return 503.
    # translation/llm are reported but non-fatal (see docstring).
    if cache_health["status"] != "healthy":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=health_status
        )

    return health_status