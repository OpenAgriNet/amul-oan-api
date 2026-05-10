import os
from pathlib import Path
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def _get_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

class Settings(BaseSettings):
    # Core Application Settings
    app_name: str = "Amul Voice AI API"
    environment: str = os.getenv("ENVIRONMENT", "production")
    debug: bool = False
    base_dir: Path = Path(__file__).resolve().parent.parent
    # Repo root for JWT PEM files (same directory chat uses). ``base_dir`` above is ``.../voice``.
    jwt_public_key_base_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent
    )
    secret_key: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    timezone: str = "Asia/Kolkata"

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8003
    api_prefix: str = "/api"
    rate_limit_requests_per_minute: int = 1000

    # Security Settings
    allowed_origins: List[str] = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    allowed_credentials: bool = True
    allowed_methods: List[str] = ["*"]
    allowed_headers: List[str] = ["*"]
    # Same trusted-integration key as chat (WhatsApp / server-to-server)
    chat_api_key: Optional[str] = os.getenv("CHAT_API_KEY")

    # JWT Configuration
    # Keys can be provided as raw PEM values (JWT_PUBLIC_KEY / JWT_PRIVATE_KEY) or as file paths (JWT_PUBLIC_KEY_PATH / JWT_PRIVATE_KEY_PATH). Values take precedence over paths.
    jwt_algorithm: str = "RS256"
    jwt_public_key: Optional[str] = os.getenv("JWT_PUBLIC_KEY")
    jwt_public_key_path: str = os.getenv("JWT_PUBLIC_KEY_PATH", "jwt_public_key.pem")
    jwt_private_key: Optional[str] = os.getenv("JWT_PRIVATE_KEY")
    jwt_private_key_path: Optional[str] = os.getenv("JWT_PRIVATE_KEY_PATH")

    # Worker Settings
    uvicorn_workers: int = os.cpu_count() or 1

    # Redis Settings
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = 0
    redis_password: Optional[str] = os.getenv("REDIS_PASSWORD")
    redis_key_prefix: str = "sva-cache-"
    redis_socket_connect_timeout: int = 10
    redis_socket_timeout: int = 10
    redis_max_connections: int = 100
    redis_retry_on_timeout: bool = True

    # Cache Configuration
    default_cache_ttl: int = 60 * 60 * 24  # 24 hours
    session_owner_ttl_seconds: int = int(os.getenv("SESSION_OWNER_TTL_SECONDS", "120"))
    session_owner_refresh_interval_seconds: int = int(os.getenv("SESSION_OWNER_REFRESH_INTERVAL_SECONDS", "15"))

    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # External Service URLs
    telemetry_api_url: str = "https://vistaar.kenpath.ai/observability-service/action/data/v3/telemetry"
    nudge_api_url: str = os.getenv("NUDGE_API_URL", "https://vistaar.getraya.app/api/nudge-user")
    nudge_timeout_seconds: float = float(os.getenv("NUDGE_TIMEOUT_SECONDS", "3.0"))
    enable_voice_nudges: bool = _get_bool_env("ENABLE_VOICE_NUDGES", default=True)
    stt_signal_retry_ceiling: int = int(os.getenv("STT_SIGNAL_RETRY_CEILING", "3"))
    openai_pretranslation_timeout_seconds: float = float(os.getenv("OPENAI_PRETRANSLATION_TIMEOUT_SECONDS", "10.0"))

    # Voice pipeline behavioral flags
    # RETRIEVAL_AUDIT_LOG: log intent/retrieval_called/query per turn for replay analysis
    retrieval_audit_log: bool = _get_bool_env("RETRIEVAL_AUDIT_LOG", default=False)
    # AMBIGUITY_MATCH_THRESHOLD: fuzzy-match cutoff for ambiguity_terms.json (0.0–1.0)
    ambiguity_match_threshold: float = float(os.getenv("AMBIGUITY_MATCH_THRESHOLD", "0.80"))
    ollama_endpoint_url: Optional[str] = None
    marqo_endpoint_url: Optional[str] = None
    inference_endpoint_url: Optional[str] = None

    # External Service API Keys
    openai_api_key: Optional[str] = None
    sarvam_api_key: Optional[str] = None
    meity_api_key_value: Optional[str] = None
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_base_url: Optional[str] = None
    # Langfuse labels: release = app/service name for grouping; environment = deployment env
    langfuse_release: Optional[str] = None  # e.g. "voice-oan-api" – shown in Langfuse for filtering
    langfuse_environment: Optional[str] = None  # e.g. "production" – defaults to ENVIRONMENT
    inference_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    mapbox_api_token: Optional[str] = None

    # AWS Configuration
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: Optional[str] = None
    aws_s3_bucket: Optional[str] = None

    # LLM Configuration
    llm_provider: Optional[str] = None
    llm_model_name: Optional[str] = None
    marqo_index_name: Optional[str] = None

    class Config:
        env_file = ".env"
        extra = 'ignore'  # Ignore extra fields from .env

settings = Settings() 
