"""The telemetry query API on its own, for a deployment next to ClickHouse.

Serves only /api/telemetry/* and a liveness check, so it needs none of the
chat app's config (Beckn, LLM, Redis). Run it with
``uvicorn app.telemetry_api:app``; the settings it reads are the ones listed
under "From the API" in docs/TELEMETRY_PIPELINE.md.
"""
from fastapi import FastAPI

from app.config import settings
from app.routers import telemetry_query

app = FastAPI(title="Amul telemetry API", docs_url=None, redoc_url=None, openapi_url=None)
app.include_router(telemetry_query.router, prefix=settings.api_prefix)


@app.get(f"{settings.api_prefix}/health/live")
async def liveness():
    return {"status": "alive"}
