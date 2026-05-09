from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings as chat_settings
from app.tasks.scheme_scheduler import (
    start_scheme_scheduler as start_chat_scheme_scheduler,
    stop_scheme_scheduler as stop_chat_scheme_scheduler,
)
from voice.app.config import settings as voice_settings
from voice.app.tasks.scheme_scheduler import (
    start_scheme_scheduler as start_voice_scheme_scheduler,
    stop_scheme_scheduler as stop_voice_scheme_scheduler,
)
from voice.helpers.utils import load_prompt_templates

load_dotenv()

# Configure voice observability before importing voice routers that may emit traces.
import voice.app.observability  # noqa: E402,F401

# Import all routers
from app.routers import auth, chat, health, suggestions, transcribe, tts, user
from voice.app.routers import health as voice_health
from voice.app.routers import voice as voice_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for startup and shutdown"""
    # Startup
    print(f"🚀 {chat_settings.app_name} starting up...")
    print(f"📍 Environment: {chat_settings.environment}")
    print(f"🔧 Debug mode: {chat_settings.debug}")
    print(f"🌐 CORS origins: {chat_settings.allowed_origins}")

    # Voice prompts are loaded once at startup to avoid runtime disk I/O.
    load_prompt_templates(voice_settings.base_dir / "assets" / "prompts")

    await start_chat_scheme_scheduler()
    await start_voice_scheme_scheduler()
    yield
    # Shutdown
    await stop_voice_scheme_scheduler()
    await stop_chat_scheme_scheduler()
    print(f"🛑 {chat_settings.app_name} shutting down...")

# Disable API docs in production to avoid exposing full API surface
_docs_enabled = chat_settings.environment != "production"

# Create FastAPI app with settings
app = FastAPI(
    title=chat_settings.app_name,
    debug=chat_settings.debug,
    description="Unified backend for chat and voice AI services",
    lifespan=lifespan,
    docs_url="/docs" if _docs_enabled else None,
    redoc_url="/redoc" if _docs_enabled else None,
    openapi_url="/openapi.json" if _docs_enabled else None,
)

# Add CORS middleware with enhanced settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=chat_settings.allowed_origins,
    allow_credentials=chat_settings.allowed_credentials,
    allow_methods=chat_settings.allowed_methods,
    allow_headers=chat_settings.allowed_headers,
)


@app.get("/")
async def root():
    """Root endpoint with app information"""
    return {
        "app": chat_settings.app_name,
        "environment": chat_settings.environment,
        "debug": chat_settings.debug,
        "api_prefix": chat_settings.api_prefix,
        "domains": ["chat", "voice"],
    }

# Chat compatibility routes (existing clients keep working)
app.include_router(auth.router, prefix=chat_settings.api_prefix)
app.include_router(chat.router, prefix=chat_settings.api_prefix)
app.include_router(transcribe.router, prefix=chat_settings.api_prefix)
app.include_router(suggestions.router, prefix=chat_settings.api_prefix)
app.include_router(tts.router, prefix=chat_settings.api_prefix)
app.include_router(user.router, prefix=chat_settings.api_prefix)
app.include_router(health.router, prefix=chat_settings.api_prefix)

# Voice compatibility route for existing voice clients
app.include_router(voice_router.router, prefix=voice_settings.api_prefix)

# Service namespaced chat API
chat_api = APIRouter(prefix="/chat", tags=["chat-service"])
chat_api.include_router(auth.router, prefix=chat_settings.api_prefix)
chat_api.include_router(chat.router, prefix=chat_settings.api_prefix)
chat_api.include_router(transcribe.router, prefix=chat_settings.api_prefix)
chat_api.include_router(suggestions.router, prefix=chat_settings.api_prefix)
chat_api.include_router(tts.router, prefix=chat_settings.api_prefix)
chat_api.include_router(user.router, prefix=chat_settings.api_prefix)
chat_api.include_router(health.router, prefix=chat_settings.api_prefix)
app.include_router(chat_api)

# Service namespaced voice API
voice_api = APIRouter(prefix="/voice", tags=["voice-service"])
voice_api.include_router(voice_router.router, prefix=voice_settings.api_prefix)
voice_api.include_router(voice_health.router, prefix=voice_settings.api_prefix)
app.include_router(voice_api)
