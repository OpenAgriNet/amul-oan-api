"""
Offline pipeline evaluation / debugging runner.

Calls each chat pipeline stage directly (no HTTP) and captures structured
artifacts for E2E evaluation, batch runs, and future CSV / Langfuse hooks.

Usage (from repo root):
    python evals/pipeline_eval_runner.py "તમારો પ્રશ્ન" --mobile 9876543210
    python evals/pipeline_eval_runner.py "What is the weather?" --save eval_artifact.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# Ensure repo root is on sys.path when invoked as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

load_dotenv(_REPO_ROOT / ".env")

from agents.agrinet import agrinet_agent
from agents.deps import FarmerContext
from agents.farmer_context import (
    SUPPORTED_SCHEME_CONTEXT_UNIONS,
    get_farmer_context_bundle_by_mobile,
)
from agents.moderation import QueryModerationResult, moderation_agent
from agents.models import (
    LLM_MODEL_NAME,
    OSS_LLM_MODEL_NAME,
    get_model_for_variant,
    provider_for_variant,
)
from app.services.scheme_ingestion import (
    SchemeCacheError,
    SchemeDependencyError,
    get_cached_scheme_records_for_union,
)
from app.services.translation import (
    INDIAN_LANGUAGES,
    PRETRANSLATION_MODEL,
    PRETRANSLATION_PROVIDER,
    translate_text,
    translate_to_english_pretranslation,
)
from app.utils import format_message_pairs, trim_history
from helpers.utils import get_logger

logger = get_logger(__name__)

WHATSAPP_RESPONSE_MAX_CHARS = 1600
GENERIC_UNAVAILABLE_MESSAGE_EN = (
    "I am unable to process your request right now. Please try again later."
)


@dataclass
class StageArtifact:
    """Structured capture for a single pipeline stage."""

    input: Any = None
    output: Any = None
    latency_ms: float = 0.0
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    skipped: bool = False
    skip_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineEvalConfig:
    query: str
    mobile: Optional[str] = None
    source_lang: str = "gu"
    target_lang: str = "gu"
    use_translation_pipeline: bool = True
    pipeline_variant: str = "legacy"
    channel: str = "web"
    history: list = field(default_factory=list)
    history_pair_limit: int = 3
    trim_history_max_tokens: int = 80_000


@dataclass
class PipelineEvalArtifact:
    query: str
    config: dict[str, Any]
    started_at: str
    pretranslation: StageArtifact = field(default_factory=StageArtifact)
    moderation: StageArtifact = field(default_factory=StageArtifact)
    farmer_context: StageArtifact = field(default_factory=StageArtifact)
    scheme_summary: StageArtifact = field(default_factory=StageArtifact)
    agent_response: StageArtifact = field(default_factory=StageArtifact)
    output_translation: StageArtifact = field(default_factory=StageArtifact)
    final_response: str = ""
    total_latency_ms: float = 0.0
    pipeline_status: str = "pending"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "config": self.config,
            "started_at": self.started_at,
            "pretranslation": self.pretranslation.to_dict(),
            "moderation": self.moderation.to_dict(),
            "farmer_context": self.farmer_context.to_dict(),
            "scheme_summary": self.scheme_summary.to_dict(),
            "agent_response": self.agent_response.to_dict(),
            "output_translation": self.output_translation.to_dict(),
            "final_response": self.final_response,
            "total_latency_ms": self.total_latency_ms,
            "pipeline_status": self.pipeline_status,
            "metadata": self.metadata,
        }


def _response_max_chars_for_channel(channel: str | None) -> int | None:
    if (channel or "").lower() == "whatsapp":
        return WHATSAPP_RESPONSE_MAX_CHARS
    return None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate_preview(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [{len(text) - limit} more chars]"


class PipelineEvalRunner:
    """Runs pipeline stages sequentially and collects eval artifacts."""

    def __init__(self, config: PipelineEvalConfig):
        self.config = config
        self.is_oss = config.pipeline_variant == "oss"
        self.use_translation_pipeline = bool(config.use_translation_pipeline) or self.is_oss
        self.request_model = get_model_for_variant(config.pipeline_variant)
        self.request_provider = provider_for_variant(config.pipeline_variant)
        self.request_model_name = OSS_LLM_MODEL_NAME if self.is_oss else LLM_MODEL_NAME
        self.processing_query = config.query
        self.processing_lang = config.target_lang
        self.farmer_data = ""
        self.farmer_unions: list[str] = []
        self.deps: Optional[FarmerContext] = None
        self.moderation_result: Optional[QueryModerationResult] = None
        self.agent_response_en = ""
        self.needs_output_translation = (
            self.use_translation_pipeline
            and config.target_lang.lower() in INDIAN_LANGUAGES
        )

    async def run(self) -> PipelineEvalArtifact:
        pipeline_start = time.perf_counter()
        artifact = PipelineEvalArtifact(
            query=self.config.query,
            config={
                "mobile": self.config.mobile,
                "source_lang": self.config.source_lang,
                "target_lang": self.config.target_lang,
                "use_translation_pipeline": self.use_translation_pipeline,
                "pipeline_variant": self.config.pipeline_variant,
                "channel": self.config.channel,
                "history_message_count": len(self.config.history),
            },
            started_at=_utc_now_iso(),
            metadata={
                "request_model_name": self.request_model_name,
                "request_provider": self.request_provider,
                "pretranslation_provider": PRETRANSLATION_PROVIDER,
                "pretranslation_model": PRETRANSLATION_MODEL,
            },
        )

        logger.info("Pipeline eval started query_preview=%s", _truncate_preview(self.config.query, 120))

        artifact.pretranslation = await self._run_pretranslation()
        artifact.farmer_context = await self._run_farmer_context()
        self.farmer_data = artifact.farmer_context.output.get("farmer_data_markdown", "")
        self.farmer_unions = artifact.farmer_context.output.get("farmer_unions", [])
        artifact.scheme_summary = await self._run_scheme_summary()
        self._build_deps()
        artifact.moderation = await self._run_moderation()

        moderation_output = artifact.moderation.output or {}
        category = moderation_output.get("category")
        if artifact.moderation.error or category != "valid_agricultural":
            decline = moderation_output.get("action") or GENERIC_UNAVAILABLE_MESSAGE_EN
            artifact.final_response = await self._localize_system_text(decline)
            artifact.pipeline_status = "blocked_at_moderation"
            artifact.total_latency_ms = (time.perf_counter() - pipeline_start) * 1000
            logger.info(
                "Pipeline eval finished status=%s total_latency_ms=%.1f",
                artifact.pipeline_status,
                artifact.total_latency_ms,
            )
            return artifact

        artifact.agent_response = await self._run_agent_response()
        self.agent_response_en = artifact.agent_response.output.get("response_en", "")

        if self.needs_output_translation:
            artifact.output_translation = await self._run_output_translation()
            artifact.final_response = artifact.output_translation.output.get("translated_text", "")
        else:
            artifact.output_translation = StageArtifact(
                skipped=True,
                skip_reason="Output translation not required for this config",
                metadata={"needs_output_translation": False},
            )
            artifact.final_response = self.agent_response_en

        artifact.pipeline_status = "completed"
        artifact.total_latency_ms = (time.perf_counter() - pipeline_start) * 1000
        logger.info(
            "Pipeline eval finished status=%s total_latency_ms=%.1f final_preview=%s",
            artifact.pipeline_status,
            artifact.total_latency_ms,
            _truncate_preview(artifact.final_response, 160),
        )
        return artifact

    async def _run_pretranslation(self) -> StageArtifact:
        stage = StageArtifact(
            input={
                "text": self.config.query,
                "source_lang": self.config.source_lang,
            },
            metadata={
                "use_translation_pipeline": self.use_translation_pipeline,
                "pipeline_variant": self.config.pipeline_variant,
            },
        )

        source = self.config.source_lang.lower()
        if not self.use_translation_pipeline or source not in {"gu", "gujarati"}:
            stage.skipped = True
            stage.skip_reason = "Pretranslation only runs for gu source with translation pipeline enabled"
            self.processing_query = self.config.query
            self.processing_lang = (
                "en" if self.use_translation_pipeline and self.needs_output_translation else self.config.target_lang
            )
            stage.output = {
                "processing_query": self.processing_query,
                "processing_lang": self.processing_lang,
            }
            logger.info("Stage pretranslation skipped: %s", stage.skip_reason)
            return stage

        logger.info("Stage pretranslation started")
        start = time.perf_counter()
        pretrans_provider = "vllm" if self.is_oss else None
        stage.metadata["pretrans_provider"] = pretrans_provider or PRETRANSLATION_PROVIDER

        try:
            translated = await translate_to_english_pretranslation(
                text=self.config.query,
                source_lang=self.config.source_lang,
                provider=pretrans_provider,
            )
            self.processing_query = translated
            self.processing_lang = "en"
            stage.output = {
                "processing_query": translated,
                "processing_lang": "en",
                "method": "translate_to_english_pretranslation",
            }
        except Exception as primary_error:
            logger.warning("Pretranslation primary path failed: %s", primary_error)
            try:
                translated = await translate_text(
                    text=self.config.query,
                    source_lang=self.config.source_lang,
                    target_lang="english",
                )
                self.processing_query = translated
                self.processing_lang = "en"
                stage.output = {
                    "processing_query": translated,
                    "processing_lang": "en",
                    "method": "translate_text_fallback",
                }
                stage.metadata["primary_error"] = str(primary_error)
            except Exception as fallback_error:
                self.processing_query = self.config.query
                self.processing_lang = self.config.target_lang
                stage.error = str(fallback_error)
                stage.metadata["primary_error"] = str(primary_error)
                stage.metadata["traceback"] = traceback.format_exc()
                stage.output = {
                    "processing_query": self.config.query,
                    "processing_lang": self.config.target_lang,
                    "method": "passthrough_on_failure",
                }
                logger.error("Stage pretranslation failed: %s", fallback_error)

        if self.use_translation_pipeline and self.needs_output_translation:
            self.processing_lang = "en"

        stage.latency_ms = (time.perf_counter() - start) * 1000
        logger.info("Stage pretranslation finished latency_ms=%.1f", stage.latency_ms)
        return stage

    async def _run_farmer_context(self) -> StageArtifact:
        stage = StageArtifact(
            input={"mobile": self.config.mobile},
            metadata={"note": "Union scheme titles are also embedded in farmer markdown"},
        )

        if not self.config.mobile:
            stage.skipped = True
            stage.skip_reason = "No mobile number provided"
            stage.output = {"farmer_data_markdown": "", "farmer_unions": []}
            logger.info("Stage farmer_context skipped: %s", stage.skip_reason)
            return stage

        logger.info("Stage farmer_context started mobile=%s", self.config.mobile)
        start = time.perf_counter()
        try:
            farmer_data, farmer_unions = await get_farmer_context_bundle_by_mobile(self.config.mobile)
            stage.output = {
                "farmer_data_markdown": farmer_data,
                "farmer_unions": farmer_unions,
                "farmer_data_length": len(farmer_data),
                "farmer_data_preview": _truncate_preview(farmer_data),
            }
        except Exception as exc:
            stage.error = str(exc)
            stage.metadata["traceback"] = traceback.format_exc()
            stage.output = {"farmer_data_markdown": "", "farmer_unions": []}
            logger.error("Stage farmer_context failed: %s", exc)

        stage.latency_ms = (time.perf_counter() - start) * 1000
        logger.info("Stage farmer_context finished latency_ms=%.1f", stage.latency_ms)
        return stage

    async def _run_scheme_summary(self) -> StageArtifact:
        scheme_unions = [
            union_name
            for union_name in self.farmer_unions
            if union_name in SUPPORTED_SCHEME_CONTEXT_UNIONS
        ]
        stage = StageArtifact(
            input={"farmer_unions": self.farmer_unions, "scheme_unions": scheme_unions},
        )

        if not scheme_unions:
            stage.skipped = True
            stage.skip_reason = "No supported scheme unions in farmer context"
            stage.output = {"records_by_union": {}}
            logger.info("Stage scheme_summary skipped: %s", stage.skip_reason)
            return stage

        logger.info("Stage scheme_summary started unions=%s", scheme_unions)
        start = time.perf_counter()
        records_by_union: dict[str, Any] = {}
        errors_by_union: dict[str, str] = {}

        for union_name in scheme_unions:
            try:
                records = await get_cached_scheme_records_for_union(union_name)
                records_by_union[union_name] = records
            except SchemeDependencyError as exc:
                errors_by_union[union_name] = f"dependency_unavailable: {exc}"
            except SchemeCacheError as exc:
                errors_by_union[union_name] = f"cache_error: {exc}"
            except Exception as exc:
                errors_by_union[union_name] = str(exc)

        stage.output = {
            "records_by_union": records_by_union,
            "scheme_counts": {union: len(records) for union, records in records_by_union.items()},
            "summaries": {
                union: [
                    {
                        "scheme_title": record.get("scheme_title"),
                        "scheme_url": record.get("scheme_url"),
                    }
                    for record in records[:10]
                ]
                for union, records in records_by_union.items()
            },
        }
        if errors_by_union:
            stage.metadata["errors_by_union"] = errors_by_union
            stage.error = "; ".join(f"{union}: {msg}" for union, msg in errors_by_union.items())

        stage.latency_ms = (time.perf_counter() - start) * 1000
        logger.info("Stage scheme_summary finished latency_ms=%.1f", stage.latency_ms)
        return stage

    def _build_deps(self) -> None:
        self.deps = FarmerContext(
            query=self.processing_query,
            lang_code=self.processing_lang,
            farmer_info=self.farmer_data,
            farmer_unions=self.farmer_unions,
            use_translation_pipeline=self.use_translation_pipeline,
            response_max_chars=_response_max_chars_for_channel(self.config.channel),
        )

    def _build_moderation_user_message(self) -> str:
        message_pairs = "\n\n".join(
            format_message_pairs(self.config.history, self.config.history_pair_limit)
        )
        if message_pairs:
            last_response = f"**Conversation**\n\n{message_pairs}\n\n---\n\n"
        else:
            last_response = ""
        assert self.deps is not None
        return f"{last_response}{self.deps.get_user_message()}"

    async def _run_moderation(self) -> StageArtifact:
        user_message = self._build_moderation_user_message()
        stage = StageArtifact(
            input={"user_message": user_message},
            metadata={"model_name": self.request_model_name},
        )

        logger.info("Stage moderation started")
        start = time.perf_counter()
        try:
            moderation_run = await moderation_agent.run(user_message, model=self.request_model)
            self.moderation_result = moderation_run.output
            assert self.deps is not None
            self.deps.update_moderation_str(str(self.moderation_result))
            stage.output = {
                "category": self.moderation_result.category,
                "action": self.moderation_result.action,
                "moderation_str": str(self.moderation_result),
            }
        except Exception as exc:
            stage.error = str(exc)
            stage.metadata["traceback"] = traceback.format_exc()
            stage.output = {"category": None, "action": GENERIC_UNAVAILABLE_MESSAGE_EN}
            logger.error("Stage moderation failed: %s", exc)

        stage.latency_ms = (time.perf_counter() - start) * 1000
        logger.info("Stage moderation finished latency_ms=%.1f", stage.latency_ms)
        return stage

    async def _run_agent_response(self) -> StageArtifact:
        assert self.deps is not None
        user_message = self.deps.get_user_message()
        trimmed_history = trim_history(
            self.config.history,
            max_tokens=self.config.trim_history_max_tokens,
            include_system_prompts=False,
            include_tool_calls=True,
        )
        stage = StageArtifact(
            input={
                "user_message": user_message,
                "trimmed_history_count": len(trimmed_history),
            },
            metadata={
                "model_name": self.request_model_name,
                "provider": self.request_provider,
            },
        )

        logger.info("Stage agent_response started")
        start = time.perf_counter()
        try:
            result = await agrinet_agent.run(
                user_prompt=user_message,
                message_history=trimmed_history,
                deps=self.deps,
                model=self.request_model,
            )
            response_en = result.output
            stage.output = {
                "response_en": response_en,
                "response_preview": _truncate_preview(response_en),
                "new_message_count": len(result.new_messages()),
            }
            # TODO: Persist result.new_messages() for multi-turn eval replay.
            # TODO: Capture tool calls / retrieval spans for Langfuse or judge evals.
        except Exception as exc:
            stage.error = str(exc)
            stage.metadata["traceback"] = traceback.format_exc()
            stage.output = {"response_en": "", "response_preview": ""}
            logger.error("Stage agent_response failed: %s", exc)

        stage.latency_ms = (time.perf_counter() - start) * 1000
        logger.info("Stage agent_response finished latency_ms=%.1f", stage.latency_ms)
        return stage

    async def _run_output_translation(self) -> StageArtifact:
        stage = StageArtifact(
            input={
                "text": self.agent_response_en,
                "source_lang": "english",
                "target_lang": self.config.target_lang,
            },
            metadata={
                "method": "translate_text",
                "note": "Production uses translate_text_stream_fast with batching; eval uses single-shot translate_text",
            },
        )

        if not self.agent_response_en:
            stage.skipped = True
            stage.skip_reason = "No agent response to translate"
            stage.output = {"translated_text": ""}
            return stage

        logger.info("Stage output_translation started target_lang=%s", self.config.target_lang)
        start = time.perf_counter()
        try:
            translated = await translate_text(
                text=self.agent_response_en,
                source_lang="english",
                target_lang=self.config.target_lang,
                max_output_chars=self.deps.get_response_max_chars() if self.deps else None,
            )
            stage.output = {
                "translated_text": translated,
                "translated_preview": _truncate_preview(translated),
            }
            # TODO: Add side-by-side EN/target pairs for LLM-as-a-judge translation evals.
        except Exception as exc:
            stage.error = str(exc)
            stage.metadata["traceback"] = traceback.format_exc()
            stage.output = {
                "translated_text": self.agent_response_en,
                "fallback": "english_agent_response",
            }
            logger.error("Stage output_translation failed: %s", exc)

        stage.latency_ms = (time.perf_counter() - start) * 1000
        logger.info("Stage output_translation finished latency_ms=%.1f", stage.latency_ms)
        return stage

    async def _localize_system_text(self, text_en: str) -> str:
        if not text_en or not self.config.target_lang:
            return text_en

        lang = self.config.target_lang.lower()
        if lang in {"english", "en"}:
            return text_en

        if lang in INDIAN_LANGUAGES:
            try:
                return await translate_text(
                    text=text_en,
                    source_lang="english",
                    target_lang=self.config.target_lang,
                    max_output_chars=_response_max_chars_for_channel(self.config.channel),
                )
            except Exception as exc:
                logger.warning("System text localization failed: %s", exc)
        return text_en


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _print_artifact(artifact: PipelineEvalArtifact) -> None:
    payload = artifact.to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))


def _save_artifact(artifact: PipelineEvalArtifact, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = artifact.to_dict()
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    logger.info("Saved eval artifact to %s", output_path.resolve())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Amul OAN chat pipeline stage-by-stage for offline eval/debug.",
    )
    parser.add_argument("query", help="User query to evaluate")
    parser.add_argument("--mobile", help="Farmer mobile number for context retrieval")
    parser.add_argument("--source-lang", default="gu", help="Source language code (default: gu)")
    parser.add_argument("--target-lang", default="gu", help="Target response language (default: gu)")
    parser.add_argument(
        "--pipeline-variant",
        default="legacy",
        choices=["legacy", "oss"],
        help="Pipeline variant (default: legacy)",
    )
    parser.add_argument(
        "--no-translation-pipeline",
        action="store_true",
        help="Disable gu->en pretranslation and en->target post-translation",
    )
    parser.add_argument("--channel", default="web", help="Channel hint for response limits (web/whatsapp)")
    parser.add_argument(
        "--save",
        nargs="?",
        const="eval_artifact.json",
        help="Save artifact JSON to disk (default filename: eval_artifact.json)",
    )
    return parser


async def _async_main(args: argparse.Namespace) -> int:
    config = PipelineEvalConfig(
        query=args.query,
        mobile=args.mobile,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        use_translation_pipeline=not args.no_translation_pipeline,
        pipeline_variant=args.pipeline_variant,
        channel=args.channel,
    )
    runner = PipelineEvalRunner(config)
    artifact = await runner.run()
    _print_artifact(artifact)

    if args.save:
        save_path = Path(args.save)
        if not save_path.is_absolute():
            save_path = _REPO_ROOT / save_path
        _save_artifact(artifact, save_path)

    # TODO: Emit CSV row for batch dataset evaluation.
    # TODO: Push artifact stages to Langfuse as nested observations.
    # TODO: Run LLM-as-a-judge scoring hooks against final_response.

    return 0 if artifact.pipeline_status == "completed" else 1


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_async_main(args)))


if __name__ == "__main__":
    main()
