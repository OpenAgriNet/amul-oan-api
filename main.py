from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from contextlib import asynccontextmanager
from app.tasks.scheme_scheduler import start_scheme_scheduler, stop_scheme_scheduler
from app.tasks.telemetry_queue import start_telemetry_worker, stop_telemetry_worker
# Imported before the routers so the cold import order matches the cycle-safety
# regression test (worker module is side-effect-free; see farmer_refresh_worker).
from app.tasks.farmer_refresh_worker import start_farmer_refresh_worker, stop_farmer_refresh_worker
# P2 health poller: active LB /health probe feeding the per-endpoint breaker.
# start_/stop_ are no-ops unless HEALTH_POLLER_ENABLED (flag-off boot is untouched).
from app.tasks.health_poller import start_health_poller, stop_health_poller

load_dotenv()

# Configure observability (Langfuse + pydantic-ai instrumentation) before router
# imports that pull in agents, tools, and voice/chat pipelines.
import app.observability  # noqa: F401, E402

# Import all routers
from app.routers import chat, transcribe, suggestions, tts, health, auth, user, telemetry, voice

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for startup and shutdown"""
    # Startup
    print(f"🚀 {settings.app_name} starting up...")
    print(f"📍 Environment: {settings.environment}")
    print(f"🔧 Debug mode: {settings.debug}")
    print(f"🌐 CORS origins: {settings.allowed_origins}")
    # Load prompt templates into memory (no disk I/O at request time)
    from helpers.utils import load_prompt_templates
    load_prompt_templates(settings.base_dir / "assets" / "prompts")
    # Unified LLM pipeline (the only model-selection path): synthesize/validate the
    # config and run the resolvability self-check (logs the resolved per-step
    # provider/model/endpoint; non-fatal). Best-effort — a configure/self-check
    # edge case must never block startup.
    try:
        from app.llm_core import runtime as _llm_runtime
        _llm_runtime.configure()
    except Exception as _llm_exc:  # pragma: no cover - defensive
        print(f"⚠️  llm_core configure skipped: {_llm_exc}")
    await start_telemetry_worker()
    await start_scheme_scheduler()
    await start_farmer_refresh_worker()
    await start_health_poller()
    yield
    # Shutdown
    await stop_health_poller()
    await stop_farmer_refresh_worker()
    await stop_scheme_scheduler()
    await stop_telemetry_worker()
    print(f"🛑 {settings.app_name} shutting down...")

# Disable API docs in production to avoid exposing full API surface
_docs_enabled = settings.environment != "production"

# Create FastAPI app with settings
app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    description="AI-powered Voice Assistant API for Agricultural Support",
    lifespan=lifespan,
    docs_url="/docs" if _docs_enabled else None,
    redoc_url="/redoc" if _docs_enabled else None,
    openapi_url="/openapi.json" if _docs_enabled else None,
)

# Add CORS middleware with enhanced settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=settings.allowed_credentials,
    allow_methods=settings.allowed_methods,
    allow_headers=settings.allowed_headers,
)


@app.get("/")
async def root():
    """Root endpoint with app information"""
    return {
        "app": settings.app_name,
        "environment": settings.environment,
        "debug": settings.debug,
        "api_prefix": settings.api_prefix
    }

# Include all routers with API prefix from settings
app.include_router(auth.router, prefix=settings.api_prefix)  # Auth router (no auth required)
app.include_router(chat.router, prefix=settings.api_prefix)
app.include_router(transcribe.router, prefix=settings.api_prefix)
app.include_router(suggestions.router, prefix=settings.api_prefix)
app.include_router(tts.router, prefix=settings.api_prefix)
app.include_router(user.router, prefix=settings.api_prefix)
app.include_router(voice.router, prefix=settings.api_prefix)
app.include_router(health.router, prefix=settings.api_prefix)
# Keep telemetry path compatible with existing frontend calls:
# /observability-service/action/data/v3/telemetry
app.include_router(telemetry.router)
