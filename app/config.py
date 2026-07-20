import os
from pathlib import Path
from typing import List, Optional
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
    uvicorn_workers: int = os.cpu_count() or 1

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
    bhashini_api_url: str = ""
    ollama_endpoint_url: Optional[str] = None
    marqo_endpoint_url: Optional[str] = None
    inference_endpoint_url: Optional[str] = None

    # Voice service settings (nudge, STT signals, voice tracing, pretranslation
    # timeout) — inert on the chat path; consumed by the voice surface once it
    # folds in. langfuse_environment (voice's name for chat's
    # langfuse_tracing_environment) is intentionally left out pending the
    # observability reconciliation (bucket C).
    nudge_api_url: str = os.getenv("NUDGE_API_URL", "https://vistaar.getraya.app/api/nudge-user")
    nudge_timeout_seconds: float = float(os.getenv("NUDGE_TIMEOUT_SECONDS", "3.0"))
    enable_voice_nudges: bool = _get_bool_env("ENABLE_VOICE_NUDGES", default=True)
    stt_signal_retry_ceiling: int = int(os.getenv("STT_SIGNAL_RETRY_CEILING", "3"))
    openai_pretranslation_timeout_seconds: float = float(os.getenv("OPENAI_PRETRANSLATION_TIMEOUT_SECONDS", "10.0"))
    voice_non_meaningful_timeout_seconds: float = float(os.getenv("VOICE_NON_MEANINGFUL_TIMEOUT_SECONDS", "0.60"))
    voice_non_meaningful_gate_timeout_seconds: float = float(os.getenv("VOICE_NON_MEANINGFUL_GATE_TIMEOUT_SECONDS", "0.50"))
    enable_voice_tracing: bool = _get_bool_env("ENABLE_VOICE_TRACING", default=True)
    voice_trace_text_mode: str = os.getenv("VOICE_TRACE_TEXT_MODE", "preview_hash")
    voice_trace_preview_chars: int = int(os.getenv("VOICE_TRACE_PREVIEW_CHARS", "120"))
    voice_trace_log_summary: bool = _get_bool_env("VOICE_TRACE_LOG_SUMMARY", default=True)
    # RETRIEVAL_AUDIT_LOG: log intent/retrieval_called/query per turn for replay analysis
    retrieval_audit_log: bool = _get_bool_env("RETRIEVAL_AUDIT_LOG", default=False)

    # External Service API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    sarvam_api_key: Optional[str] = None
    meity_api_key_value: Optional[str] = None
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_base_url: Optional[str] = None
    langfuse_release: Optional[str] = None  # LANGFUSE_RELEASE: app version for metrics (git sha, semver)
    langfuse_tracing_environment: Optional[str] = None  # LANGFUSE_TRACING_ENVIRONMENT: production/staging/development
    bhashini_api_key: str = ""
    inference_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    mapbox_api_token: Optional[str] = None
    banas_mobile_api_key: Optional[str] = os.getenv("BANAS_MOBILE_API_KEY")

    # AWS Configuration
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: Optional[str] = None
    aws_s3_bucket: Optional[str] = None

    # LLM Configuration
    llm_provider: Optional[str] = None
    llm_model_name: Optional[str] = None
    marqo_index_name: Optional[str] = None

    # OSS pipeline %-split, sticky TTL and OSS model/endpoint are no longer read
    # via `settings`: they map to llm_core's weighted-profile config, synthesized
    # from the raw env (OSS_PIPELINE_PCT / OSS_VARIANT_TTL / OSS_INFERENCE_* /
    # OSS_LLM_MODEL_NAME) by app/llm_core/legacy_shim.py. The env vars stay; the
    # duplicate settings attributes + the pipeline_router that read them are gone.

    # Standard OSS -> managed fallback (see docs/oss-fallback-design.md).
    # Kill-switch defaults OFF: when false, pipelines keep today's behaviour.
    fallback_enabled: bool = os.getenv("FALLBACK_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }
    # Per-pipeline OSS time-to-respond budgets before falling back to managed.
    fallback_chat_oss_timeout_ms: int = int(os.getenv("FALLBACK_CHAT_OSS_TIMEOUT_MS", "8000"))
    fallback_moderation_oss_timeout_ms: int = int(os.getenv("FALLBACK_MODERATION_OSS_TIMEOUT_MS", "5000"))
    fallback_pretranslation_oss_timeout_ms: int = int(os.getenv("FALLBACK_PRETRANSLATION_OSS_TIMEOUT_MS", "10000"))
    fallback_suggestions_oss_timeout_ms: int = int(os.getenv("FALLBACK_SUGGESTIONS_OSS_TIMEOUT_MS", "6000"))
    # Deadline for the managed (fallback) tier.
    fallback_managed_timeout_ms: int = int(os.getenv("FALLBACK_MANAGED_TIMEOUT_MS", "20000"))

    # The unified LLM pipeline (app/llm_core) is now the ONLY model-selection path
    # — the LLM_CORE_ENABLED / PROFILES_ENABLED kill-switches (P0/P1 identity gates)
    # were removed at P4. The weighted-profile split + config-driven fallback chain
    # are always live. Operational trigger flags (HEALTH_* / CONCURRENCY_GAUGE_*)
    # below remain as real toggles.
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
    # Concurrency-gauge trigger — pre-flight REORDER filter (llm_core P3). Default
    # OFF (zero behaviour change when off). When on, a step carrying an explicit
    # ConcurrencyGate (metrics_url + max_concurrency) has its vLLM tier
    # DEPRIORITIZED behind the managed tier while that box's in-flight
    # (running+waiting) requests are at/above max_concurrency — so managed is tried
    # first under load. Unreadable metrics FAIL OPEN (order unchanged), never a
    # forced flip to managed. Never drops a tier and never empties the chain;
    # orthogonal to the sticky split and composes AFTER the health prune.
    concurrency_gauge_enabled: bool = os.getenv("CONCURRENCY_GAUGE_ENABLED", "true").strip().lower() in {
        "1", "true", "yes", "on"
    }
    # Explicit vLLM Prometheus /metrics URL that arms the AGENT-step concurrency
    # gate (P3). When set, synthesize_from_env() attaches a ConcurrencyGate to the
    # OSS agent step so the vLLM tier is deprioritized under load. When unset, the
    # gauge is a harmless no-op even with CONCURRENCY_GAUGE_ENABLED on (no gate =>
    # nothing to reorder). NOT derived by stripping /v1 off the inference endpoint
    # (that was bh's fragile derivation); this is given explicitly.
    agent_concurrency_metrics_url: Optional[str] = os.getenv("AGENT_CONCURRENCY_METRICS_URL")
    # In-flight (running+waiting) threshold at/above which the gate deprioritizes
    # the vLLM tier. Shared by the shim when it builds the gate.
    concurrency_max: int = int(os.getenv("CONCURRENCY_MAX", "10"))
    # Short TTL (seconds) for the shared Redis cache of a vLLM engine's in-flight
    # count (mirrors bh's ~2s), and the per-probe metrics HTTP timeout.
    concurrency_metrics_cache_ttl_s: int = int(os.getenv("CONCURRENCY_METRICS_CACHE_TTL_S", "2"))
    concurrency_metrics_timeout_ms: int = int(os.getenv("CONCURRENCY_METRICS_TIMEOUT_MS", "2000"))
    # Scheme tool union scoping:
    # true  -> require authenticated farmer union to match a supported scheme union
    # false -> testing mode; allow any farmer union and fall back to supported unions
    scheme_require_union_auth: bool = os.getenv("SCHEME_REQUIRE_UNION_AUTH", "true").strip().lower() in {
        "1", "true", "yes", "on"
    }
    # Banas scheme PDF ingestion via Chandra OCR (see scheme_ingestion.py).
    # Page cap and HTTP fetch timeout stay as module constants, not env vars.
    scheme_ocr_endpoint_url: Optional[str] = os.getenv("SCHEME_OCR_ENDPOINT_URL")
    scheme_ocr_timeout_seconds: float = float(os.getenv("SCHEME_OCR_TIMEOUT_SECONDS", "120"))
    scheme_pdf_render_dpi: int = int(os.getenv("SCHEME_PDF_RENDER_DPI", "200"))

    # Ambiguity-term fuzzy-match cutoff (0-1) for get_ambiguity_hints_for_query.
    # Overridable via env; defaults to 0.80 (prior hard-coded behaviour).
    ambiguity_match_threshold: float = float(os.getenv("AMBIGUITY_MATCH_THRESHOLD", "0.80"))

    # ── Micro-loan eligibility feature ───────────────────────────────────────
    # Master switch. When false the loan tool is hidden and evaluate_and_issue
    # short-circuits, so the flow is fully inert unless explicitly enabled.
    loan_feature_enabled: bool = os.getenv("LOAN_FEATURE_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
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

    class Config:
        env_file = ".env"
        extra = 'ignore'  # Ignore extra fields from .env

settings = Settings()
