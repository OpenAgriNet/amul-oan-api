import json
import os
import logging
import math
from pathlib import Path
from typing import Any, ClassVar, List, Optional
from pydantic import AliasChoices, Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
_config_logger = logging.getLogger(__name__)
_scheme_ocr_page_batch_size_deprecation_logged = False


def _get_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _safe_bool_env_value(value: object, *, env_name: str, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    _config_logger.warning("Invalid bool for %s=%r; using default=%s", env_name, value, default)
    return default


def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        _config_logger.warning("Invalid int for %s=%r; using default=%s", name, value, default)
        return default


def _get_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        _config_logger.warning("Invalid float for %s=%r; using default=%s", name, value, default)
        return default


def _get_str_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _load_vault_secrets() -> dict[str, Any]:
    """Load the app's secret-only Vault document as Settings values."""
    if not _get_bool_env("HASHICORP_VAULT_ENABLED"):
        return {}

    address = _get_str_env("VAULT_ADDR", "http://127.0.0.1:8200")
    mount = _get_str_env("VAULT_MOUNT_POINT", "secret")
    path = _get_str_env("VAULT_SECRET_PATH", "amul-oan-api")
    try:
        import hvac

        client = hvac.Client(
            url=address,
            token=_get_str_env("VAULT_TOKEN"),
            namespace=_get_str_env("VAULT_NAMESPACE"),
            verify=_get_str_env("VAULT_CACERT") or _get_bool_env("VAULT_TLS_VERIFY", True),
            timeout=_get_float_env("VAULT_TIMEOUT_SECONDS", 5.0),
        )
        auth_method = (_get_str_env("VAULT_AUTH_METHOD", "token") or "token").lower()
        if auth_method == "kubernetes":
            role = _get_str_env("VAULT_KUBERNETES_ROLE")
            if not role:
                raise ValueError("VAULT_KUBERNETES_ROLE is required")
            jwt_path = _get_str_env(
                "VAULT_KUBERNETES_JWT_PATH",
                "/var/run/secrets/kubernetes.io/serviceaccount/token",
            )
            client.auth.kubernetes.login(
                role=role,
                jwt=Path(jwt_path).read_text(encoding="utf-8").strip(),
                mount_point=_get_str_env("VAULT_KUBERNETES_MOUNT_POINT", "kubernetes"),
            )
        elif auth_method != "token":
            raise ValueError("VAULT_AUTH_METHOD must be 'token' or 'kubernetes'")
        if not client.is_authenticated():
            raise RuntimeError("Vault authentication failed")

        if _get_int_env("VAULT_KV_VERSION", 2) == 1:
            data = client.secrets.kv.v1.read_secret(path=path, mount_point=mount).get("data", {})
        else:
            response = client.secrets.kv.v2.read_secret_version(path=path, mount_point=mount)
            data = response.get("data", {}).get("data", {})
        if not isinstance(data, dict):
            raise RuntimeError("Vault response did not contain a secret mapping")
        return {
            key.lower(): json.dumps(value, separators=(",", ":"))
            if isinstance(value, (dict, list))
            else value
            for key, value in data.items()
            if value not in (None, "")
        }
    except Exception as exc:
        _config_logger.warning(
            "Vault load failed for mount=%s path=%s; using environment values: %s",
            mount,
            path,
            exc,
        )
        return {}


_VAULT_SECRETS = _load_vault_secrets()


def get_config_value(env_name: str, default: Any = None) -> Any:
    """Central compatibility accessor for config that must be read dynamically.

    Most application code should use ``settings``.  The LLM config synthesizer
    and a few tests intentionally re-read values after process start; routing
    those reads through this function keeps all environment access in this file.
    """

    value = _VAULT_SECRETS.get(env_name.lower())
    if value in (None, ""):
        value = os.getenv(env_name)
    return default if value in (None, "") else value


def _safe_int_env_value(
    value: object,
    *,
    env_name: str,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        _config_logger.warning("Invalid int for %s=%r; using default=%s", env_name, value, default)
        return default

    if minimum is not None and parsed < minimum:
        _config_logger.warning("Out-of-range int for %s=%r; clamping to min=%s", env_name, parsed, minimum)
        parsed = minimum
    if maximum is not None and parsed > maximum:
        _config_logger.warning("Out-of-range int for %s=%r; clamping to max=%s", env_name, parsed, maximum)
        parsed = maximum
    return parsed


def _safe_float_env_value(
    value: object,
    *,
    env_name: str,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        _config_logger.warning("Invalid float for %s=%r; using default=%s", env_name, value, default)
        return default
    if not math.isfinite(parsed):
        _config_logger.warning("Non-finite float for %s=%r; using default=%s", env_name, value, default)
        return default

    if minimum is not None and parsed < minimum:
        _config_logger.warning("Out-of-range float for %s=%r; clamping to min=%s", env_name, parsed, minimum)
        parsed = minimum
    if maximum is not None and parsed > maximum:
        _config_logger.warning("Out-of-range float for %s=%r; clamping to max=%s", env_name, parsed, maximum)
        parsed = maximum
    return parsed


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    def __init__(self, **values: Any):
        # Explicit values win, then Vault; missing Vault keys fall back to env/.env.
        super().__init__(**{**_VAULT_SECRETS, **values})

    # HashiCorp Vault bootstrap/configuration (always environment driven).
    hashicorp_vault_enabled: bool = False
    vault_addr: str = "http://127.0.0.1:8200"
    vault_secret_path: str = "amul-oan-api"
    vault_mount_point: str = "secret"
    vault_kv_version: int = 2
    vault_namespace: Optional[str] = None
    vault_auth_method: str = "token"
    vault_token: Optional[str] = Field(default=None, exclude=True)
    vault_kubernetes_role: Optional[str] = None
    vault_kubernetes_jwt_path: str = "/var/run/secrets/kubernetes.io/serviceaccount/token"
    vault_kubernetes_mount_point: str = "kubernetes"
    vault_cacert: Optional[str] = None
    vault_tls_verify: bool = True
    vault_timeout_seconds: float = 5.0

    # Core Application Settings
    app_name: str = "Amul AI API"
    environment: str = os.getenv("ENVIRONMENT", "production")
    debug: bool = False
    base_dir: Path = Path(__file__).resolve().parent.parent
    secret_key: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    timezone: str = "Asia/Kolkata"

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    api_prefix: str = "/api"
    rate_limit_requests_per_minute: int = 1000
    prometheus_multiproc_dir: Optional[str] = None
    model_boundary_capture_enabled: bool = False
    model_boundary_capture_dir: str = "/tmp/voice_model_boundary"

    # Security Settings
    allowed_origins: List[str] = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    allowed_credentials: bool = True
    allowed_methods: List[str] = ["*"]
    allowed_headers: List[str] = ["*"]
    chat_api_key: Optional[str] = os.getenv("CHAT_API_KEY")

    # JWT Configuration
    # Inline PEM values take precedence; if not set, keys are loaded from paths.
    jwt_algorithm: str = "RS256"
    jwt_public_key: Optional[str] = os.getenv("JWT_PUBLIC_KEY")  # PEM string; overrides path if set
    jwt_public_key_path: str = os.getenv("JWT_PUBLIC_KEY_PATH", "jwt_public_key.pem")
    jwt_private_key: Optional[str] = os.getenv("JWT_PRIVATE_KEY")  # PEM string; overrides path if set
    jwt_private_key_path: Optional[str] = os.getenv("JWT_PRIVATE_KEY_PATH")

    # Webview / App FE URL (served behind FCM auth; JWT token appended for FE)
    app_fe_url: Optional[str] = os.getenv("APP_FE_URL")

    # Firebase / FCM (for webview endpoint auth)
    # Inline JSON values take precedence; if not set, credentials are loaded from paths.
    firebase_service_account: Optional[str] = os.getenv("FIREBASE_SERVICE_ACCOUNT")  # JSON string; overrides path if set
    firebase_service_account_path: Optional[str] = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "service-account.json")
    firebase_service_account_2: Optional[str] = os.getenv("FIREBASE_SERVICE_ACCOUNT_2")  # JSON string; overrides path if set
    firebase_service_account_path_2: Optional[str] = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH_2")
    firebase_service_account_3: Optional[str] = os.getenv("FIREBASE_SERVICE_ACCOUNT_3")  # JSON string; overrides path if set
    firebase_service_account_path_3: Optional[str] = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH_3")

    # Worker Settings
    uvicorn_workers: int = Field(
        default=os.cpu_count() or 1,
        validation_alias=AliasChoices("AMUL_CHAT_BE_UVICORN_WORKERS", "UVICORN_WORKERS"),
    )

    # Redis Settings (set REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD, etc. via env)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_key_prefix: str = "sva-cache-"
    redis_socket_connect_timeout: int = 10
    redis_socket_timeout: int = 10
    redis_max_connections: int = 100
    redis_retry_on_timeout: bool = True

    # Cache Configuration
    default_cache_ttl: int = 60 * 60 * 24  # 24 hours
    # Conversation-history retention in Redis (app/utils.py DEFAULT_CACHE_TTL).
    # Rolling inactivity window: update_message_history rewrites the key with this
    # TTL every turn, so history expires this long after the LAST turn. session_id
    # is client-supplied and the backend enforces no session/call length, so the
    # only requirement is TTL > the gap between turns — exact value is
    # non-load-bearing (2h is generous slack; voice's old 24h was incidental).
    history_cache_ttl_seconds: int = int(os.getenv("HISTORY_CACHE_TTL_SECONDS", str(60 * 60 * 2)))
    suggestions_cache_ttl: int = 60 * 30    # 30 minutes
    farmer_animal_api_cache_ttl: int = 60 * 60 * 24 * 17  # 17 days
    # AI technician list cache: keyed by (union_code, society_code), not per farmer.
    # Shared across farmer SWR refresh and prompt context; API called only on miss.
    ai_technician_cache_ttl_seconds: int = int(
        os.getenv("AI_TECHNICIAN_CACHE_TTL_SECONDS", str(60 * 60 * 24))
    )
    # Session-ownership locking (voice call concurrency) — consumed by app/utils.py
    # once the voice surface folds in; inert on the chat path.
    session_owner_ttl_seconds: int = int(os.getenv("SESSION_OWNER_TTL_SECONDS", "120"))
    session_owner_refresh_interval_seconds: int = int(os.getenv("SESSION_OWNER_REFRESH_INTERVAL_SECONDS", "15"))
    # Farmer cache policy: beyond this age a cached record is too stale to serve —
    # the read blocks on a bounded API call instead of serving it (falls back to
    # the stale record only if the API also fails). Backstop above the 12h/2h
    # soft-refresh; the 7d hard Redis TTL still deletes records entirely.
    # (Consumed by the farmer SWR cache layer — bucket A Layer 2.)
    farmer_max_serve_stale_seconds: int = int(os.getenv("FARMER_MAX_SERVE_STALE_SECONDS", str(60 * 60 * 24)))
    # Farmer SWR cache timers (Inc 4) — all env-tunable. Soft-refresh: a cached
    # record older than its interval is served stale and refreshed in the
    # background. "found" data changes slowly (12h); a cached "not_found" is
    # re-checked sooner (2h) because a farmer may newly register.
    # KNOWN LIMITATION: a register-then-immediately-call flow can keep seeing
    # not_found for up to the not_found interval; the proper fix is active
    # cache-invalidation on registration (cross-service, out of scope — follow-up).
    farmer_refresh_interval_seconds: int = int(os.getenv("FARMER_REFRESH_INTERVAL_SECONDS", str(60 * 60 * 12)))
    farmer_negative_refresh_interval_seconds: int = int(os.getenv("FARMER_NEGATIVE_REFRESH_INTERVAL_SECONDS", str(60 * 60 * 2)))
    # Hard retention: Redis deletes a farmer record after this idle period.
    farmer_cache_retention_seconds: int = int(os.getenv("FARMER_CACHE_RETENTION_SECONDS", str(60 * 60 * 24 * 7)))
    # Farmer/animal API tracing records a PII-SAFE structure summary by default
    # (status, record count, which keys are present/null). Raw response bodies are
    # only captured when FARMER_API_TRACE_BODY is explicitly enabled (temporary
    # deep-debug), capped at FARMER_API_TRACE_BODY_CHARS.
    farmer_api_trace_body: bool = _get_bool_env("FARMER_API_TRACE_BODY", default=False)
    farmer_api_trace_body_chars: int = int(os.getenv("FARMER_API_TRACE_BODY_CHARS", "8000"))
    # Farmer cache consolidation rollout (Layer 2-first for farmer-by-mobile).
    # Defaults preserve current behavior until later steps wire these flags in:
    #   chat stays on Layer 1; fallback is armed for Phase A; Layer 1 mobile
    #   cache remains active until Phase C bypass.
    farmer_layer2_chat_context_enabled: bool = Field(
        default=False,
        validation_alias="FARMER_LAYER2_CHAT_CONTEXT_ENABLED",
    )
    farmer_layer2_fallback_to_legacy_enabled: bool = Field(
        default=True,
        validation_alias="FARMER_LAYER2_FALLBACK_TO_LEGACY_ENABLED",
    )
    farmer_layer1_mobile_cache_bypass_enabled: bool = Field(
        default=False,
        validation_alias="FARMER_LAYER1_MOBILE_CACHE_BYPASS_ENABLED",
    )

    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Telemetry Queue Configuration
    telemetry_queue_max_size: int = 1000
    telemetry_queue_max_retries: int = 3
    telemetry_queue_retry_base_delay_ms: int = 250
    telemetry_queue_retry_max_delay_ms: int = 4000
    telemetry_dead_letter_max: int = 200
    telemetry_ingest_max_body_bytes: int = 256 * 1024
    telemetry_ingest_max_string_len_default: int = 1000
    telemetry_ingest_max_question_text_len: int = 2000
    telemetry_ingest_max_answer_text_len: int = 12000
    telemetry_ingest_max_feedback_text_len: int = 4000
    telemetry_ingest_max_error_text_len: int = 2000

    # External Service URLs
    telemetry_api_url: str = "https://vistaar.kenpath.ai/observability-service/action/data/v3/telemetry"
    bhashini_api_url: str = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"
    ollama_endpoint_url: Optional[str] = None
    marqo_endpoint_url: Optional[str] = None
    inference_endpoint_url: Optional[str] = None
    raya_tts_url: Optional[str] = None
    tts_timeout_seconds: float = 60.0
    transcribe_timeout_seconds: float = 30.0

    # Nudge settings — inert on the chat path; consumed by the voice surface when
    # it folds in (voice is served by voice-oan-api today).
    nudge_api_url: str = os.getenv("NUDGE_API_URL", "https://vistaar.getraya.app/api/nudge-user")
    nudge_timeout_seconds: float = float(os.getenv("NUDGE_TIMEOUT_SECONDS", "3.0"))
    # Kill switch for Hindi chat. Default ON: hi/hindi requests use the full
    # src->en->agent->hi translation pipeline. Set HINDI_CHAT_ENABLED=false to
    # disable Hindi independently (hi/hindi then bypass the pipeline and are
    # served like an unsupported language — English passthrough) without
    # touching Gujarati.
    hindi_chat_enabled: bool = _get_bool_env("HINDI_CHAT_ENABLED", default=True)
    # Master kill switch for the doctor persona. When false, every chat request
    # is routed through the farmer persona regardless of JWT claims or request
    # overrides. Keep default-off until the doctor experience is approved for
    # the target environment.
    doctor_persona_enabled: bool = _get_bool_env("DOCTOR_PERSONA_ENABLED", default=False)
    # Allows the feature-flagged chat UI to exercise a persona different from
    # the signed JWT, but only while the master doctor-persona gate is enabled.
    # Keep off outside controlled testing; JWT user_type remains authoritative
    # when the override is disabled.
    chat_persona_override_enabled: bool = _get_bool_env("CHAT_PERSONA_OVERRIDE_ENABLED", default=False)
    openai_pretranslation_timeout_seconds: float = float(os.getenv("OPENAI_PRETRANSLATION_TIMEOUT_SECONDS", "10.0"))
    # RETRIEVAL_AUDIT_LOG: log intent/retrieval_called/query per turn for replay analysis
    retrieval_audit_log: bool = _get_bool_env("RETRIEVAL_AUDIT_LOG", default=False)

    # External Service API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    azure_openai_api_key: Optional[str] = None
    sarvam_api_key: Optional[str] = None
    meity_api_key_value: Optional[str] = None
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_base_url: Optional[str] = None
    langfuse_host: Optional[str] = None  # backward-compatible alias used by old deployments
    langfuse_release: Optional[str] = None  # LANGFUSE_RELEASE: app version for metrics (git sha, semver)
    langfuse_tracing_environment: Optional[str] = None  # LANGFUSE_TRACING_ENVIRONMENT: production/staging/development
    bhashini_api_key: str = ""
    inference_api_key: Optional[str] = None
    oss_inference_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    mapbox_api_token: Optional[str] = None
    banas_mobile_api_key: Optional[str] = os.getenv("BANAS_MOBILE_API_KEY")
    pashugpt_token: Optional[str] = None
    pashugpt_token_2: Optional[str] = None
    pashugpt_token_3: Optional[str] = None
    raya_tts_api_key: Optional[str] = None
    demo_ui_api_key: Optional[str] = None

    # AWS Configuration
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: Optional[str] = None
    aws_s3_bucket: Optional[str] = None
    aws_endpoint_url: Optional[str] = None

    # LLM Configuration
    llm_provider: Optional[str] = None
    llm_model_name: Optional[str] = None
    oss_inference_endpoint_url: Optional[str] = None
    oss_llm_model_name: str = "gemma-4-31b-it"
    oss_pretranslation_model: Optional[str] = None
    pretranslation_provider: Optional[str] = None
    pretranslation_model: Optional[str] = None
    anthropic_pretranslation_model: str = "claude-haiku-4-5"
    openai_pretranslation_model: str = "gpt-4.1-mini"
    azure_openai_endpoint: Optional[str] = None
    azure_openai_api_version: Optional[str] = None
    azure_openai_deployment_name: Optional[str] = None
    translategemma_27b_base_endpoint: str = "http://localhost:18002/v1"
    translategemma_27b_base_endpoints: Optional[str] = None  # deprecated; warning compatibility only
    translategemma_27b_base_model: str = "translategemma-27b-base"
    pipeline_config_path: Optional[str] = None
    pipeline_config_redis_enabled: bool = False
    pipeline_channel: str = "chat"
    pipeline_config_refresh_s: float = 10.0
    pipeline_config_redis_timeout_s: float = 0.5
    require_overflow_armed: bool = False
    oss_pipeline_pct: int = 0
    oss_variant_ttl: int = 60 * 60 * 24 * 7
    fallback_post_translation_tg_ttft_ms: int = 5000
    fallback_post_translation_llm_timeout_ms: int = 30000
    managed_max_concurrency: int = 64
    chat_history_max_tokens: Optional[int] = None
    chat_history_max_tokens_vllm_gemma: int = 10000
    agrinet_max_tokens: Optional[int] = None
    agrinet_max_tokens_vllm_gemma: int = 2048
    doctor_max_tokens: Optional[int] = None
    doctor_request_limit: int = 10
    marqo_index_name: Optional[str] = None
    # Tool retrieval config (search_documents): keep env names/defaults unchanged.
    marqo_use_e5_query_prefix: bool = Field(default=True, validation_alias="MARQO_USE_E5_QUERY_PREFIX")
    marqo_exclude_reference: bool = Field(default=True, validation_alias="MARQO_EXCLUDE_REFERENCE")
    marqo_query_expansion_profile: str = os.getenv("MARQO_QUERY_EXPANSION_PROFILE", "gu-v1")
    marqo_default_final_chunks: int = Field(default=8, validation_alias="MARQO_DEFAULT_FINAL_CHUNKS")
    marqo_max_final_chunks: int = Field(default=20, validation_alias="MARQO_MAX_FINAL_CHUNKS")
    marqo_max_chunks_per_doc: int = Field(default=2, validation_alias="MARQO_MAX_CHUNKS_PER_DOC")
    marqo_candidate_multiplier: int = Field(default=10, validation_alias="MARQO_CANDIDATE_MULTIPLIER")
    marqo_candidate_cap: int = Field(default=120, validation_alias="MARQO_CANDIDATE_CAP")
    marqo_hybrid_alpha: float = Field(default=0.6, validation_alias="MARQO_HYBRID_ALPHA")
    marqo_hybrid_rrfk: int = Field(default=60, validation_alias="MARQO_HYBRID_RRFK")
    marqo_search_mode: str = os.getenv("MARQO_SEARCH_MODE", "hybrid")
    marqo_rerank_mode: str = os.getenv("MARQO_RERANK_MODE", "bm25lite")

    # OSS pipeline %-split, sticky TTL and OSS model/endpoint are no longer read
    # via `settings`: they map to llm_core's weighted-profile config, synthesized
    # from the raw env (OSS_PIPELINE_PCT / OSS_VARIANT_TTL / OSS_INFERENCE_* /
    # OSS_LLM_MODEL_NAME) by app/llm_core/legacy_shim.py. The env vars stay; the
    # duplicate settings attributes + the pipeline_router that read them are gone.

    # The unified LLM pipeline (app/llm_core) is now the ONLY model-selection path
    # — the LLM_CORE_ENABLED / PROFILES_ENABLED kill-switches (P0/P1 identity gates)
    # were removed at P4. The weighted-profile split + config-driven fallback chain
    # are always live. Health keeps explicit operational toggles; concurrency is
    # enabled by the presence of a configured ConcurrencyGate.
    # Health filter — pre-flight chain FILTER (llm_core P2). Two independent
    # kill-switches, both default OFF (zero behaviour change when off):
    #   * HEALTH_BREAKER_ENABLED — the passive circuit-breaker, fed by the
    #     fallback failure/success path (per-endpoint consecutive-failure trip).
    #   * HEALTH_POLLER_ENABLED  — the active LB `/health` poller (a lifespan
    #     background task) that updates breaker state with hysteresis failback.
    # The health prune is active when EITHER is on; it only ever DROPS tiers whose
    # endpoint is currently `open` from an already-resolved chain, never reorders,
    # and never returns an empty chain (degrade-safe). Orthogonal to the sticky
    # split: a session's profile assignment is unchanged; only its chain is pruned
    # while an endpoint is down.
    health_breaker_enabled: bool = os.getenv("HEALTH_BREAKER_ENABLED", "true").strip().lower() in {
        "1", "true", "yes", "on"
    }
    health_poller_enabled: bool = os.getenv("HEALTH_POLLER_ENABLED", "true").strip().lower() in {
        "1", "true", "yes", "on"
    }
    # Consecutive FALLBACKABLE failures on an endpoint before the breaker trips.
    health_breaker_fail_threshold: int = int(os.getenv("HEALTH_BREAKER_FAIL_THRESHOLD", "5"))
    # Cooldown before an `open` endpoint is allowed a single half-open probe.
    health_breaker_cooldown_ms: int = int(os.getenv("HEALTH_BREAKER_COOLDOWN_MS", "30000"))
    # Active poller cadence and per-probe HTTP timeout.
    health_poller_interval_ms: int = int(os.getenv("HEALTH_POLLER_INTERVAL_MS", "10000"))
    health_poller_timeout_ms: int = int(os.getenv("HEALTH_POLLER_TIMEOUT_MS", "2000"))
    # Hysteresis: consecutive healthy polls required to fail an `open` endpoint
    # back to `closed` (guards against the H200 crash-and-half-boot flap).
    health_poller_healthy_polls: int = int(os.getenv("HEALTH_POLLER_HEALTHY_POLLS", "3"))
    health_fail_rate_window: int = 20
    health_fail_rate_threshold: float = 0.5
    health_probe_max_s: float = 30.0
    # Concurrency-gauge trigger — pre-flight REORDER filter (llm_core P3). A step
    # carrying an explicit ConcurrencyGate (metrics_url + max_concurrency) has its vLLM tier
    # DEPRIORITIZED behind the managed tier while that box's in-flight
    # (running+waiting) requests are at/above max_concurrency — so managed is tried
    # first under load. Unreadable metrics FAIL OPEN (order unchanged), never a
    # forced flip to managed. Never drops a tier and never empties the chain;
    # orthogonal to the sticky split and composes AFTER the health prune.
    # Short TTL (seconds) for the shared Redis cache of a vLLM engine's in-flight
    # count (mirrors bh's ~2s), and the per-probe metrics HTTP timeout.
    concurrency_metrics_cache_ttl_s: int = int(os.getenv("CONCURRENCY_METRICS_CACHE_TTL_S", "2"))
    concurrency_metrics_timeout_ms: int = int(os.getenv("CONCURRENCY_METRICS_TIMEOUT_MS", "2000"))
    concurrency_shed_start_frac: float = 0.7
    concurrency_prometheus_url: Optional[str] = None
    concurrency_prometheus_query: Optional[str] = None
    # Scheme tool union scoping:
    # true  -> require authenticated farmer union to match a supported scheme union
    # false -> testing mode; allow any farmer union and fall back to supported unions
    scheme_require_union_auth: bool = os.getenv("SCHEME_REQUIRE_UNION_AUTH", "true").strip().lower() in {
        "1", "true", "yes", "on"
    }
    # Union scheme ingestion source URLs.
    banas_scheme_site_origin: str = Field(
        default="https://www.banasdairy.coop",
        validation_alias="BANAS_SCHEME_SITE_ORIGIN",
    )
    banas_scheme_documents_api_url: str = Field(
        default="https://www.banasdairy.coop/api/documents",
        validation_alias="BANAS_SCHEME_DOCUMENTS_API_URL",
    )
    sarhad_scheme_source_url: str = Field(
        default="https://sarhaddairy.coop/for-our-milk-producers/",
        validation_alias="SARHAD_SCHEME_SOURCE_URL",
    )
    sumul_scheme_source_url: str = Field(
        default="https://www.sumul.com/farmer-section.html",
        validation_alias="SUMUL_SCHEME_SOURCE_URL",
    )
    sursagar_scheme_source_url: str = Field(
        default="https://sursagardairy.com/Farmer/MilkProducers",
        validation_alias="SURSAGAR_SCHEME_SOURCE_URL",
    )
    # Banas/Sumul/Sursagar/Sabar scheme PDF ingestion via stock Chandra vLLM
    # (OpenAI-compatible /v1/chat/completions; see scheme_ingestion.py).
    # Base URL e.g. http://10.185.25.197:8011 (optional trailing /v1 is stripped).
    scheme_ocr_endpoint_url: Optional[str] = os.getenv("SCHEME_OCR_ENDPOINT_URL")
    # Per-page OCR timeout budget. Each OCR POST is one page via stock Chandra
    # /v1/chat/completions; HTTP timeout equals this value.
    scheme_ocr_timeout_seconds: float = float(os.getenv("SCHEME_OCR_TIMEOUT_SECONDS", "120"))
    scheme_pdf_render_dpi: int = int(os.getenv("SCHEME_PDF_RENDER_DPI", "200"))
    # Scheme ingestion operational tunables.
    scheme_lock_ttl_seconds: int = Field(default=60 * 60, validation_alias="SCHEME_LOCK_TTL_SECONDS")
    scheme_http_timeout_seconds: float = Field(default=30.0, validation_alias="SCHEME_HTTP_TIMEOUT_SECONDS")
    scheme_pdf_max_render_pages: int = Field(default=30, validation_alias="SCHEME_PDF_MAX_RENDER_PAGES")
    scheme_ocr_prompt_type: str = os.getenv("SCHEME_OCR_PROMPT_TYPE", "ocr_layout")
    scheme_ocr_max_output_tokens: int = Field(default=12284, validation_alias="SCHEME_OCR_MAX_OUTPUT_TOKENS")
    # Max in-flight one-page OCR chat-completions requests per PDF (latency knob).
    # Prefer SCHEME_OCR_CONCURRENCY. When unset, falls back to SCHEME_OCR_PAGE_BATCH_SIZE.
    scheme_ocr_concurrency: int = Field(default=4, validation_alias="SCHEME_OCR_CONCURRENCY")
    # Deprecated alias for SCHEME_OCR_CONCURRENCY. No longer means "images per HTTP
    # request" (stock Chandra is always one image per /v1/chat/completions call).
    scheme_ocr_page_batch_size: int = Field(default=4, validation_alias="SCHEME_OCR_PAGE_BATCH_SIZE")
    scheme_ocr_max_failed_page_ratio: float = Field(default=0.15, validation_alias="SCHEME_OCR_MAX_FAILED_PAGE_RATIO")
    scheme_banas_min_record_coverage_ratio: float = Field(default=0.85, validation_alias="SCHEME_BANAS_MIN_RECORD_COVERAGE_RATIO")

    # Ambiguity-term fuzzy-match cutoff (0-1) for get_ambiguity_hints_for_query.
    # Overridable via env; defaults to 0.80 (prior hard-coded behaviour).
    ambiguity_match_threshold: float = float(os.getenv("AMBIGUITY_MATCH_THRESHOLD", "0.80"))

    # ── Micro-loan eligibility feature ───────────────────────────────────────
    # Master switch. When false the loan tool is hidden and evaluate_and_issue
    # short-circuits, so the flow is fully inert unless explicitly enabled.
    loan_feature_enabled: bool = os.getenv("LOAN_FEATURE_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
    # Per-session guard on AI-call booking (try_reserve / SET-NX).
    #
    # OFF (default) matches the product decision documented in agents/tools/ai_call.py:
    # a farmer must be able to book multiple AI visits in one session — two cows in
    # heat is a real case — so we do not dedupe on session/species/technician. The
    # accepted cost is that an OSS->managed fallback re-run can re-fire the tool and
    # create a duplicate booking, with the upstream CreateAICall API as the backstop.
    #
    # ON trades that the other way: no duplicate bookings (and no duplicate SMS to a
    # farmer), at the cost of blocking a legitimate second booking within the TTL.
    # amul-prod has historically run WITH the guard.
    ai_call_booking_guard_enabled: bool = os.getenv("AI_CALL_BOOKING_GUARD_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
    # How long a session stays reserved after a booking. Only consulted when the
    # guard is on. Shorter = a farmer can legitimately re-book sooner; longer =
    # wider protection against a delayed fallback re-fire.
    ai_call_cooldown_ttl_seconds: int = int(os.getenv("AI_CALL_COOLDOWN_TTL_SECONDS", str(60 * 30)))
    # Health-call booking idempotency TTL.
    health_call_cooldown_ttl_seconds: int = Field(default=60 * 30, validation_alias="HEALTH_CALL_COOLDOWN_TTL_SECONDS")
    # Per-check toggles. A disabled check is BYPASSED (treated as pass) so product
    # can test the end-to-end flow without real Amul submissions / bank-list rows.
    loan_check_bank_list_enabled: bool = os.getenv("LOAN_CHECK_BANK_LIST_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    loan_check_milk_enabled: bool = os.getenv("LOAN_CHECK_MILK_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    # When true, an eligible number can be issued MULTIPLE codes (a fresh code each
    # request). When false, an existing active code is RE-SHARED instead of minting a
    # new one (asking for the loan / the code again returns the same code). Flippable.
    loan_allow_multiple_codes: bool = os.getenv("LOAN_ALLOW_MULTIPLE_CODES", "false").strip().lower() in {"1", "true", "yes", "on"}
    # When true, (re)send the approval SMS on EVERY code request — including when an
    # existing code is re-shared (i.e. every time the farmer asks for their OTP). When
    # false, the SMS is only sent when a new code is first issued. Flippable.
    loan_resend_sms_on_request: bool = os.getenv("LOAN_RESEND_SMS_ON_REQUEST", "false").strip().lower() in {"1", "true", "yes", "on"}
    # Loan parameters (script: "up to ₹5,000 if last-month milk ≥ ₹3,000").
    loan_max_amount: float = float(os.getenv("LOAN_MAX_AMOUNT", "5000"))
    loan_interest_rate_pct: float = float(os.getenv("LOAN_INTEREST_RATE_PCT", "7"))
    loan_milk_threshold: float = float(os.getenv("LOAN_MILK_THRESHOLD", "3000"))
    loan_milk_lookback_days: int = int(os.getenv("LOAN_MILK_LOOKBACK_DAYS", "30"))
    loan_code_length: int = int(os.getenv("LOAN_CODE_LENGTH", "6"))
    loan_code_expiry_days: int = int(os.getenv("LOAN_CODE_EXPIRY_DAYS", "0"))  # 0 = no expiry
    # Postgres connection for the loan tables (SQLAlchemy async URL, asyncpg driver),
    # e.g. postgresql+asyncpg://user:pass@host:5432/db. Secret — env only.
    loan_db_url: Optional[str] = os.getenv("LOAN_DB_URL")
    loan_db_pool_size: int = int(os.getenv("LOAN_DB_POOL_SIZE", "5"))

    # ── Onex-Aura / OneXtel SMS gateway (DLT-approved KDCC micro-loan template) ─
    # When false, SMS is a dry-run: nothing is sent, the code is still issued and
    # stored, and sms_status is recorded as 'dry_run'. Keep OFF while testing.
    loan_sms_enabled: bool = os.getenv("LOAN_SMS_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
    onex_sms_base_url: str = os.getenv("ONEX_SMS_BASE_URL", "https://sapi.onex-aura.com/api/sms")
    onex_sms_key: Optional[str] = os.getenv("ONEX_SMS_KEY")           # secret
    onex_sms_from: str = os.getenv("ONEX_SMS_FROM", "AMULHO")         # DLT sender header
    onex_sms_entity_id: Optional[str] = os.getenv("ONEX_SMS_ENTITY_ID")     # secret (DLT)
    onex_sms_template_id: Optional[str] = os.getenv("ONEX_SMS_TEMPLATE_ID")  # secret (DLT)
    onex_sms_timeout_secs: float = float(os.getenv("ONEX_SMS_TIMEOUT_SECS", "15"))
    # DLT-approved Gujarati body. Placeholders: {name}, {amount}, {code}. The amount
    # is rendered as an integer with a thousands separator (e.g. 5,000).
    onex_sms_body_template: str = os.getenv(
        "ONEX_SMS_BODY_TEMPLATE",
        "{name}, અભિનંદન! આપની વિનંતી મુજબ ₹{amount} ની માઈક્રો લોન મંજૂર કરવામાં આવી છે. "
        "પેમેન્ટ મેળવવા માટે આપની KDCC બેંક શાખામાં આ કોડ રજૂ કરો:{code} .",
    )

    # Beckn Network Feature Flag
    # When enable_network is true, agent tools that have a Beckn equivalent
    # (vet-KB search, union schemes, AI-call booking) route through the Amul
    # Beckn network instead of the direct integrations (Marqo / Redis /
    # PashuGPT). Default false → existing behaviour is unchanged.
    enable_network: bool = os.getenv("ENABLE_NETWORK", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
    # Seeker BAP base URL for network discovery (vet KB, union schemes).
    amul_network_url: str = os.getenv("AMUL_NETWORK_URL", "http://amul-bap-seeker:3000")
    # Booking BPP base URL for network AI-call booking.
    amul_booking_bpp_url: str = os.getenv("AMUL_BOOKING_BPP_URL", "http://amul-net-bpp-booking:6002")
    # Timeout (seconds) for network calls from the agent tools.
    amul_network_timeout_s: float = float(os.getenv("AMUL_NETWORK_TIMEOUT_S", "35"))
    # Durable callback-mode transactions. This is a separate, default-off gate:
    # ENABLE_NETWORK continues to preserve the deployed synchronous booking
    # adapter until ONIX has directed confirm/on_confirm routes and the booking
    # BPP has been upgraded to ACK + callback semantics.
    beckn_callback_transactions_enabled: bool = _get_bool_env(
        "BECKN_CALLBACK_TRANSACTIONS_ENABLED", default=False
    )
    # SHC has its own rollout gate because enabling its init/on_init flow must
    # not switch unrelated booking tools away from their deployed synchronous
    # adapters before those BPPs emit durable callbacks.
    vistaar_shc_enabled: bool = _get_bool_env("VISTAAR_SHC_ENABLED", default=False)
    # ONIX BAP caller base, e.g. http://amul-onix:3001/bap/caller. The client
    # appends /confirm/ or /status/.
    beckn_bap_caller_url: str = os.getenv("BECKN_BAP_CALLER_URL", "").rstrip("/")
    # Bearer credential for a private transaction bridge. Direct, in-network
    # ONIX callers (including the dev SHC path) do not require this credential.
    beckn_transaction_bridge_token: Optional[str] = os.getenv("BECKN_TRANSACTION_BRIDGE_TOKEN")
    beckn_bap_id: str = os.getenv("BECKN_BAP_ID", "bap.amul-net.internal")
    # Public ONIX callback receiver base; ONIX validates/signature-routes the
    # callback to this application's /api/beckn/on_* ingress.
    beckn_bap_uri: str = os.getenv("BECKN_BAP_URI", "")
    # Every Amul-owned domain is exposed by one registered BPP participant.
    # The public receiver URI is injected per environment; raw upstream hosts
    # remain private to the application-BPP adapters.
    beckn_amul_bpp_id: str = os.getenv("BECKN_AMUL_BPP_ID", "bpp-amul.amul-net.internal")
    beckn_amul_bpp_uri: str = os.getenv("BECKN_AMUL_BPP_URI", "")
    beckn_booking_domain: str = os.getenv("BECKN_BOOKING_DOMAIN", "services:amul-vet-booking")
    beckn_milk_domain: str = "services:amul-milk-collection"
    beckn_farmer_domain: str = "data:amul-farmer-profile"
    beckn_animal_domain: str = "data:amul-animal-profile"
    beckn_country_code: str = os.getenv("BECKN_COUNTRY_CODE", "IND")
    beckn_city_code: str = os.getenv("BECKN_CITY_CODE", "std:079")
    beckn_message_ttl: str = os.getenv("BECKN_MESSAGE_TTL", "PT30S")
    # The HTTP/voice turn may stop waiting after this budget, while the durable
    # Redis record remains alive for late on_confirm/on_status recovery.
    beckn_callback_wait_seconds: float = Field(default=30.0, validation_alias="BECKN_CALLBACK_WAIT_SECONDS")
    beckn_callback_poll_interval_seconds: float = Field(
        default=0.1, validation_alias="BECKN_CALLBACK_POLL_INTERVAL_SECONDS"
    )
    # Only connection-establishment failures are retried, with the exact same
    # transaction/message IDs and payload. Read/write/HTTP failures are
    # ambiguous and are never blindly replayed.
    beckn_forward_connect_attempts: int = Field(
        default=2, validation_alias="BECKN_FORWARD_CONNECT_ATTEMPTS"
    )
    beckn_forward_retry_delay_seconds: float = Field(
        default=0.2, validation_alias="BECKN_FORWARD_RETRY_DELAY_SECONDS"
    )
    beckn_operation_ttl_seconds: int = Field(
        default=60 * 60 * 24, validation_alias="BECKN_OPERATION_TTL_SECONDS"
    )
    beckn_callback_max_body_bytes: int = Field(
        default=2 * 1024 * 1024, validation_alias="BECKN_CALLBACK_MAX_BODY_BYTES"
    )
    # SHC on_init carries a private base64 HTML report. Bound both its decoded
    # size and its durable Redis retention independently of ordinary operations.
    shc_html_max_bytes: int = Field(default=1024 * 1024, validation_alias="SHC_HTML_MAX_BYTES")
    shc_artifact_ttl_seconds: int = Field(default=10 * 60, validation_alias="SHC_ARTIFACT_TTL_SECONDS")
    # Optional defense in depth for the internal app ingress. Signature
    # verification remains ONIX's responsibility; when set, its reverse proxy
    # must inject X-Beckn-Callback-Token.
    beckn_callback_token: Optional[str] = os.getenv("BECKN_CALLBACK_TOKEN")
    # Bharat Vistaar network settings used by tool layer.
    vistaar_bap_url: str = os.getenv("VISTAAR_BAP_URL", "https://bap-client-playground-sandbox-vistaar.da.gov.in").rstrip("/")
    vistaar_default_lat: float = Field(default=22.55, validation_alias="VISTAAR_DEFAULT_LAT")
    vistaar_default_lon: float = Field(default=72.93, validation_alias="VISTAAR_DEFAULT_LON")
    vistaar_seeker_url: str = os.getenv("VISTAAR_SEEKER_URL", "http://amul-bap-seeker:3000").rstrip("/")
    vistaar_leg: str = os.getenv("VISTAAR_LEG", "vistaar")
    vistaar_bap_id: str = os.getenv("VISTAAR_BAP_ID", "amul-dev")
    vistaar_bap_uri: str = os.getenv("VISTAAR_BAP_URI", "https://bap-network-playground-sandbox-vistaar.da.gov.in")
    vistaar_bpp_id: str = os.getenv("VISTAAR_BPP_ID", "bpp-network-playground-sandbox-vistaar.da.gov.in")
    vistaar_bpp_uri: str = os.getenv("VISTAAR_BPP_URI", "https://bpp-network-playground-sandbox-vistaar.da.gov.in")
    vistaar_max_items: int = Field(default=20, validation_alias="VISTAAR_MAX_ITEMS")
    mandi_max_candidates: int = 3
    mandi_location_ttl_s: int = 3600
    # Farmer/animal tool backend URLs and timeout (non-secret, previously hardcoded).
    amulpashudhan_base_url: str = Field(
        default="https://api.amulpashudhan.com/configman/v1/PashuGPT",
        validation_alias="AMULPASHUDHAN_BASE_URL",
    )
    herdman_base_url: str = Field(
        default="https://herdman.live/apis/api",
        validation_alias="HERDMAN_BASE_URL",
    )
    banas_mobile_base_url: str = Field(
        default="https://banasmobileapi.amnex.com/api/FarmerVisitAPIKOS",
        validation_alias="BANAS_MOBILE_BASE_URL",
    )
    cvcc_base_url: str = Field(
        default="https://api.amuldairy.com/ai_cattle_dtl.php",
        validation_alias="CVCC_BASE_URL",
    )
    farmer_backend_http_timeout_seconds: float = Field(default=30.0, validation_alias="FARMER_BACKEND_HTTP_TIMEOUT_SECONDS")
    # Farmer cache SWR worker tunables (tool-adjacent path used by loan checks).
    farmer_refresh_lock_ttl_seconds: int = Field(default=60 * 5, validation_alias="FARMER_REFRESH_LOCK_TTL_SECONDS")
    farmer_cold_fetch_timeout_seconds: float = Field(default=4.0, validation_alias="FARMER_COLD_FETCH_TIMEOUT_SECONDS")
    farmer_refresh_queue_batch_size: int = Field(default=20, validation_alias="FARMER_REFRESH_QUEUE_BATCH_SIZE")
    farmer_refresh_retry_base_seconds: int = Field(
        default=60,
        validation_alias="FARMER_REFRESH_RETRY_BASE_SECONDS",
    )
    farmer_refresh_retry_max_seconds: int = Field(
        default=60 * 60,
        validation_alias="FARMER_REFRESH_RETRY_MAX_SECONDS",
    )

    # Config hardening policy: malformed numeric env values warn and fall back to
    # defaults; parseable but out-of-range values are clamped to safe bounds.
    _SAFE_INT_FIELDS: ClassVar[dict[str, tuple[str, int, int | None, int | None]]] = {
        "marqo_default_final_chunks": ("MARQO_DEFAULT_FINAL_CHUNKS", 8, 1, None),
        "marqo_max_final_chunks": ("MARQO_MAX_FINAL_CHUNKS", 20, 1, None),
        "marqo_max_chunks_per_doc": ("MARQO_MAX_CHUNKS_PER_DOC", 2, 1, None),
        "marqo_candidate_multiplier": ("MARQO_CANDIDATE_MULTIPLIER", 10, 1, None),
        "marqo_candidate_cap": ("MARQO_CANDIDATE_CAP", 120, 1, None),
        "marqo_hybrid_rrfk": ("MARQO_HYBRID_RRFK", 60, 1, None),
        "scheme_lock_ttl_seconds": ("SCHEME_LOCK_TTL_SECONDS", 60 * 60, 1, None),
        "scheme_pdf_max_render_pages": ("SCHEME_PDF_MAX_RENDER_PAGES", 30, 1, None),
        "scheme_ocr_max_output_tokens": ("SCHEME_OCR_MAX_OUTPUT_TOKENS", 12284, 1, None),
        "scheme_ocr_concurrency": ("SCHEME_OCR_CONCURRENCY", 4, 1, 8),
        "scheme_ocr_page_batch_size": ("SCHEME_OCR_PAGE_BATCH_SIZE", 4, 1, 8),
        "health_call_cooldown_ttl_seconds": ("HEALTH_CALL_COOLDOWN_TTL_SECONDS", 60 * 30, 1, None),
        "vistaar_max_items": ("VISTAAR_MAX_ITEMS", 20, 1, None),
        "farmer_refresh_lock_ttl_seconds": ("FARMER_REFRESH_LOCK_TTL_SECONDS", 60 * 5, 1, None),
        "farmer_refresh_queue_batch_size": ("FARMER_REFRESH_QUEUE_BATCH_SIZE", 20, 1, None),
        "farmer_refresh_retry_base_seconds": ("FARMER_REFRESH_RETRY_BASE_SECONDS", 60, 1, None),
        "farmer_refresh_retry_max_seconds": ("FARMER_REFRESH_RETRY_MAX_SECONDS", 60 * 60, 1, None),
        "beckn_operation_ttl_seconds": ("BECKN_OPERATION_TTL_SECONDS", 60 * 60 * 24, 60, None),
        "beckn_callback_max_body_bytes": ("BECKN_CALLBACK_MAX_BODY_BYTES", 2 * 1024 * 1024, 1024, 5 * 1024 * 1024),
        "shc_html_max_bytes": ("SHC_HTML_MAX_BYTES", 1024 * 1024, 1024, 2 * 1024 * 1024),
        "shc_artifact_ttl_seconds": ("SHC_ARTIFACT_TTL_SECONDS", 10 * 60, 60, 60 * 60),
        "beckn_forward_connect_attempts": ("BECKN_FORWARD_CONNECT_ATTEMPTS", 2, 1, 5),
    }
    _SAFE_FLOAT_FIELDS: ClassVar[dict[str, tuple[str, float, float | None, float | None]]] = {
        "marqo_hybrid_alpha": ("MARQO_HYBRID_ALPHA", 0.6, 0.0, 1.0),
        "scheme_http_timeout_seconds": ("SCHEME_HTTP_TIMEOUT_SECONDS", 30.0, 0.001, None),
        "scheme_ocr_max_failed_page_ratio": ("SCHEME_OCR_MAX_FAILED_PAGE_RATIO", 0.15, 0.0, 1.0),
        "scheme_banas_min_record_coverage_ratio": ("SCHEME_BANAS_MIN_RECORD_COVERAGE_RATIO", 0.85, 0.0, 1.0),
        "vistaar_default_lat": ("VISTAAR_DEFAULT_LAT", 22.55, -90.0, 90.0),
        "vistaar_default_lon": ("VISTAAR_DEFAULT_LON", 72.93, -180.0, 180.0),
        "farmer_backend_http_timeout_seconds": ("FARMER_BACKEND_HTTP_TIMEOUT_SECONDS", 30.0, 0.001, None),
        "farmer_cold_fetch_timeout_seconds": ("FARMER_COLD_FETCH_TIMEOUT_SECONDS", 4.0, 0.001, None),
        "beckn_callback_wait_seconds": ("BECKN_CALLBACK_WAIT_SECONDS", 30.0, 0.1, 300.0),
        "beckn_callback_poll_interval_seconds": ("BECKN_CALLBACK_POLL_INTERVAL_SECONDS", 0.1, 0.01, 5.0),
        "beckn_forward_retry_delay_seconds": ("BECKN_FORWARD_RETRY_DELAY_SECONDS", 0.2, 0.0, 5.0),
    }
    _SAFE_BOOL_FIELDS: ClassVar[dict[str, tuple[str, bool]]] = {
        "marqo_use_e5_query_prefix": ("MARQO_USE_E5_QUERY_PREFIX", True),
        "marqo_exclude_reference": ("MARQO_EXCLUDE_REFERENCE", True),
        "farmer_layer2_chat_context_enabled": ("FARMER_LAYER2_CHAT_CONTEXT_ENABLED", False),
        "farmer_layer2_fallback_to_legacy_enabled": ("FARMER_LAYER2_FALLBACK_TO_LEGACY_ENABLED", True),
        "farmer_layer1_mobile_cache_bypass_enabled": ("FARMER_LAYER1_MOBILE_CACHE_BYPASS_ENABLED", False),
    }

    @field_validator(
        "marqo_use_e5_query_prefix",
        "marqo_exclude_reference",
        "farmer_layer2_chat_context_enabled",
        "farmer_layer2_fallback_to_legacy_enabled",
        "farmer_layer1_mobile_cache_bypass_enabled",
        mode="before",
    )
    @classmethod
    def _validate_safe_bool_fields(cls, value: object, info: ValidationInfo) -> bool:
        env_name, default = cls._SAFE_BOOL_FIELDS[info.field_name]
        return _safe_bool_env_value(value, env_name=env_name, default=default)

    @field_validator(
        "marqo_default_final_chunks",
        "marqo_max_final_chunks",
        "marqo_max_chunks_per_doc",
        "marqo_candidate_multiplier",
        "marqo_candidate_cap",
        "marqo_hybrid_rrfk",
        "scheme_lock_ttl_seconds",
        "scheme_pdf_max_render_pages",
        "scheme_ocr_max_output_tokens",
        "scheme_ocr_concurrency",
        "scheme_ocr_page_batch_size",
        "health_call_cooldown_ttl_seconds",
        "vistaar_max_items",
        "farmer_refresh_lock_ttl_seconds",
        "farmer_refresh_queue_batch_size",
        "farmer_refresh_retry_base_seconds",
        "farmer_refresh_retry_max_seconds",
        "beckn_operation_ttl_seconds",
        "beckn_callback_max_body_bytes",
        "shc_html_max_bytes",
        "shc_artifact_ttl_seconds",
        "beckn_forward_connect_attempts",
        mode="before",
    )
    @classmethod
    def _validate_safe_int_fields(cls, value: object, info: ValidationInfo) -> int:
        env_name, default, minimum, maximum = cls._SAFE_INT_FIELDS[info.field_name]
        return _safe_int_env_value(
            value,
            env_name=env_name,
            default=default,
            minimum=minimum,
            maximum=maximum,
        )

    @field_validator(
        "marqo_hybrid_alpha",
        "scheme_http_timeout_seconds",
        "scheme_ocr_max_failed_page_ratio",
        "scheme_banas_min_record_coverage_ratio",
        "vistaar_default_lat",
        "vistaar_default_lon",
        "farmer_backend_http_timeout_seconds",
        "farmer_cold_fetch_timeout_seconds",
        "beckn_callback_wait_seconds",
        "beckn_callback_poll_interval_seconds",
        "beckn_forward_retry_delay_seconds",
        mode="before",
    )
    @classmethod
    def _validate_safe_float_fields(cls, value: object, info: ValidationInfo) -> float:
        env_name, default, minimum, maximum = cls._SAFE_FLOAT_FIELDS[info.field_name]
        return _safe_float_env_value(
            value,
            env_name=env_name,
            default=default,
            minimum=minimum,
            maximum=maximum,
        )

    @field_validator(
        "amulpashudhan_base_url",
        "herdman_base_url",
        "banas_mobile_base_url",
        "cvcc_base_url",
        "banas_scheme_site_origin",
        "banas_scheme_documents_api_url",
        "sarhad_scheme_source_url",
        "sumul_scheme_source_url",
        "sursagar_scheme_source_url",
        mode="before",
    )
    @classmethod
    def _normalize_backend_base_urls(cls, value: object) -> str:
        if value is None:
            return ""
        return str(value).rstrip("/")

    @model_validator(mode="after")
    def _resolve_scheme_ocr_concurrency(self):
        """Prefer SCHEME_OCR_CONCURRENCY; else alias from deprecated PAGE_BATCH_SIZE."""
        global _scheme_ocr_page_batch_size_deprecation_logged
        concurrency_raw = os.getenv("SCHEME_OCR_CONCURRENCY")
        concurrency_explicit = concurrency_raw is not None and str(concurrency_raw).strip() != ""
        if concurrency_explicit:
            return self

        # Unset concurrency → use (already clamped) page_batch_size as concurrency.
        self.scheme_ocr_concurrency = self.scheme_ocr_page_batch_size
        batch_raw = os.getenv("SCHEME_OCR_PAGE_BATCH_SIZE")
        batch_explicit = batch_raw is not None and str(batch_raw).strip() != ""
        if batch_explicit and not _scheme_ocr_page_batch_size_deprecation_logged:
            _config_logger.warning(
                "SCHEME_OCR_PAGE_BATCH_SIZE is deprecated for OCR transport batching; "
                "use SCHEME_OCR_CONCURRENCY=%s instead (aliased for now)",
                self.scheme_ocr_concurrency,
            )
            _scheme_ocr_page_batch_size_deprecation_logged = True
        return self

    @model_validator(mode="after")
    def _require_beckn_transport_credentials_when_enabled(self):
        if not (self.beckn_callback_transactions_enabled or self.vistaar_shc_enabled):
            return self
        required = {
            "BECKN_BAP_CALLER_URL": self.beckn_bap_caller_url,
            "BECKN_BAP_URI": self.beckn_bap_uri,
        }
        if self.beckn_callback_transactions_enabled:
            required.update(
                {
                    "BECKN_TRANSACTION_BRIDGE_TOKEN": self.beckn_transaction_bridge_token,
                    "BECKN_CALLBACK_TOKEN": self.beckn_callback_token,
                    "BECKN_AMUL_BPP_ID": self.beckn_amul_bpp_id,
                    "BECKN_AMUL_BPP_URI": self.beckn_amul_bpp_uri,
                }
            )
        missing = [name for name, value in required.items() if not str(value or "").strip()]
        if missing:
            raise ValueError(
                "Beckn callback transactions require non-empty transport settings: "
                + ", ".join(missing)
            )
        return self

settings = Settings()
