"""
Shared helpers for Langfuse-based offline evaluation scripts.

Uses Langfuse Python SDK 4.x public REST API clients exposed on Langfuse().api:
  - api.health.health()
  - api.trace.list(...)
  - api.trace.get(trace_id, fields=...)
  - api.sessions.list(...)
  - api.sessions.get(session_id)
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_OUTPUT_DIR = _REPO_ROOT / "eval_outputs"
RAW_JSON_DIR = DEFAULT_OUTPUT_DIR / "langfuse_raw_json"
TRACE_FIELDS_FULL = "core,io,observations,scores,metrics"

# Observation names emitted by this application (see app/services/chat.py, translation.py, suggestions.py).
STAGE_NAME_PATTERNS: dict[str, tuple[str, ...]] = {
    "query_pretranslation": ("query_pretranslation",),
    "moderation": ("moderation", "moderation agent"),
    "farmer_background": ("farmer_background", "farmer context", "farmer_background"),
    "agent": ("amul ai agent", "amul ai agent run"),
    "suggestions": ("suggestions",),
    "text_translation": ("text_translation",),
    "stream_translation": ("stream_translation",),
}


class LangfuseEvalError(Exception):
    """Raised when Langfuse is misconfigured or an eval API call fails."""


def load_repo_env() -> None:
    load_dotenv(_REPO_ROOT / ".env")


def get_langfuse_settings() -> dict[str, Optional[str]]:
    """Read the same env vars used by agents/__init__.py and Langfuse SDK."""
    load_repo_env()
    return {
        "public_key": os.getenv("LANGFUSE_PUBLIC_KEY"),
        "secret_key": os.getenv("LANGFUSE_SECRET_KEY"),
        "base_url": os.getenv("LANGFUSE_BASE_URL")
        or os.getenv("LANGFUSE_HOST")
        or "https://cloud.langfuse.com",
        "environment": os.getenv("LANGFUSE_TRACING_ENVIRONMENT") or "chat-development",
        "release": os.getenv("LANGFUSE_RELEASE"),
    }


def get_langfuse_client():
    """
    Construct a Langfuse SDK client using repo env configuration.

    Mirrors agents/__init__.py defaults (environment fallback: chat-development).
    """
    from langfuse import Langfuse

    settings = get_langfuse_settings()
    if not settings["public_key"] or not settings["secret_key"]:
        raise LangfuseEvalError(
            "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set in the environment or .env file."
        )

    kwargs: dict[str, Any] = {
        "public_key": settings["public_key"],
        "secret_key": settings["secret_key"],
        "base_url": settings["base_url"],
        "environment": settings["environment"],
    }
    if settings["release"]:
        kwargs["release"] = settings["release"]
    return Langfuse(**kwargs)


def get_langfuse_api():
    """Return the Fern-generated LangfuseAPI client (client.api)."""
    client = get_langfuse_client()
    api = getattr(client, "api", None)
    if api is None:
        raise LangfuseEvalError(
            "Langfuse client is disabled. Verify LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY."
        )
    return api


def verify_langfuse_connection(api) -> dict[str, Any]:
    """Ping Langfuse API health endpoint."""
    health = api.health.health()
    if hasattr(health, "model_dump"):
        return health.model_dump(by_alias=True, mode="json")
    return {"status": str(health)}


def json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return value.model_dump(by_alias=True, mode="json")
    return str(value)


def to_jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=json_default, ensure_ascii=False))


def summarize_trace(trace: Any) -> dict[str, Any]:
    return {
        "trace_id": getattr(trace, "id", None),
        "trace_name": getattr(trace, "name", None),
        "session_id": getattr(trace, "session_id", None),
        "user_id": getattr(trace, "user_id", None),
        "timestamp": getattr(trace, "timestamp", None),
        "environment": getattr(trace, "environment", None),
        "tags": getattr(trace, "tags", None),
    }


def list_latest_traces(api, *, limit: int = 10, order_by: str = "timestamp.desc") -> list[Any]:
    response = api.trace.list(limit=limit, order_by=order_by)
    return list(response.data or [])


def fetch_trace(api, trace_id: str, *, fields: str = TRACE_FIELDS_FULL) -> Any:
    return api.trace.get(trace_id, fields=fields)


def list_session_traces(api, session_id: str, *, limit: int = 100) -> list[Any]:
    response = api.trace.list(session_id=session_id, limit=limit, order_by="timestamp.asc")
    return list(response.data or [])


def _trace_names(traces: list[Any]) -> set[str]:
    return {_normalize_name(getattr(trace, "name", None)) for trace in traces}


def _moderation_outputs_from_trace_detail(detail: Any) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for obs in getattr(detail, "observations", None) or []:
        if _matches_stage(getattr(obs, "name", None), STAGE_NAME_PATTERNS["moderation"]):
            out = getattr(obs, "output", None) or {}
            if isinstance(out, dict):
                outputs.append(out)
    trace_out = getattr(detail, "output", None) or {}
    if isinstance(trace_out, dict) and trace_out:
        outputs.append(trace_out)
    return outputs


def _moderation_result_from_traces(api, traces: list[Any]) -> Optional[bool]:
    """
    Return True when moderation allows the agent, False when it blocks, None if unknown.
    """
    for trace in traces:
        if "moderation" not in _normalize_name(getattr(trace, "name", None)):
            continue
        detail = fetch_trace(api, trace.id)
        for output in _moderation_outputs_from_trace_detail(detail):
            category = str(output.get("category") or "").strip().lower()
            if category == "valid_agricultural":
                return True
            if category:
                return False
    return None


def wait_for_session_traces(
    api,
    session_id: str,
    *,
    min_traces: int = 2,
    timeout_s: float = 180,
    poll_s: float = 5,
) -> list[Any]:
    """Poll Langfuse until the chat session has enough traces to reconstruct."""
    deadline = time.time() + timeout_s
    last_count = 0
    while time.time() < deadline:
        traces = list_session_traces(api, session_id, limit=50)
        last_count = len(traces)
        names = _trace_names(traces)
        has_agent = any("amul ai agent" in name for name in names)
        if len(traces) >= min_traces and has_agent:
            return traces

        moderation_result = _moderation_result_from_traces(api, traces)
        if moderation_result is False:
            # Moderation declined the query — agent trace will never appear.
            return traces

        time.sleep(poll_s)
    raise LangfuseEvalError(
        f"Timed out waiting for Langfuse traces for session_id={session_id!r} "
        f"(last_trace_count={last_count}, timeout_s={timeout_s})"
    )


def list_latest_sessions(api, *, limit: int = 10) -> list[Any]:
    response = api.sessions.list(limit=limit)
    return list(response.data or [])


def _normalize_name(name: Optional[str]) -> str:
    return (name or "").strip().lower()


def _matches_stage(name: Optional[str], patterns: tuple[str, ...]) -> bool:
    normalized = _normalize_name(name)
    return any(pattern in normalized for pattern in patterns)


def _find_observations(observations: list[Any], stage_key: str) -> list[Any]:
    patterns = STAGE_NAME_PATTERNS[stage_key]
    return [obs for obs in observations if _matches_stage(getattr(obs, "name", None), patterns)]


def _first_observation(observations: list[Any], stage_key: str) -> Optional[Any]:
    matches = _find_observations(observations, stage_key)
    return matches[0] if matches else None


def _observation_latency_ms(observation: Any) -> Optional[float]:
    latency = getattr(observation, "latency", None)
    if latency is None:
        start = getattr(observation, "start_time", None)
        end = getattr(observation, "end_time", None)
        if start and end:
            latency = (end - start).total_seconds()
    if latency is None:
        return None
    return round(float(latency) * 1000, 2)


def _serialize_observation(observation: Any) -> dict[str, Any]:
    payload = to_jsonable(observation)
    if isinstance(payload, dict):
        payload["latency_ms"] = _observation_latency_ms(observation)
    return payload


def _extract_response_en(
    all_observations: list[Any],
    *,
    translation_obs: Optional[Any] = None,
    text_translation_obs: Optional[Any] = None,
    agent_obs: Optional[Any] = None,
    fallback_final: str = "",
) -> str:
    """
    English agent answer before post-translation.

    Production stores it on stream_translation / text_translation observation input.text,
    not on Amul AI Agent output (which is overwritten with the translated response).
    """
    translation_observations: list[Any] = []
    for obs in all_observations:
        name = getattr(obs, "name", None)
        if _matches_stage(name, STAGE_NAME_PATTERNS["stream_translation"]) or _matches_stage(
            name, STAGE_NAME_PATTERNS["text_translation"]
        ):
            translation_observations.append(obs)

    candidates: list[str] = []
    for obs in sorted(
        translation_observations,
        key=lambda item: getattr(item, "start_time", None) or "",
    ):
        inp = getattr(obs, "input", None) or {}
        if isinstance(inp, dict):
            text = str(inp.get("text") or "").strip()
            if text:
                candidates.append(text)

    if candidates:
        merged = candidates[0]
        for text in candidates[1:]:
            if text in merged or merged in text:
                if len(text) > len(merged):
                    merged = text
                continue
            merged = f"{merged.rstrip()} {text.lstrip()}"
        return merged

    for obs in (translation_obs, text_translation_obs):
        if obs is None:
            continue
        inp = getattr(obs, "input", None) or {}
        if isinstance(inp, dict):
            text = str(inp.get("text") or "").strip()
            if text:
                return text

    agent_output = getattr(agent_obs, "output", None) if agent_obs else None
    if isinstance(agent_output, dict):
        en = str(agent_output.get("response_en") or "").strip()
        if en:
            return en

    # No post-translation stage: agent/final output is already English.
    if not (translation_obs or text_translation_obs) and fallback_final:
        return fallback_final
    return ""


def _extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("translated_text", "text", "query", "response", "answer", "output"):
            if key in value and value[key]:
                return str(value[key])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _extract_moderation(observation: Optional[Any]) -> dict[str, Any]:
    if observation is None:
        return {}
    output = getattr(observation, "output", None) or {}
    if isinstance(output, dict):
        return {
            "category": output.get("category"),
            "action": output.get("action"),
            "raw_output": output,
        }
    return {"raw_output": output}


def _extract_tool_calls(observations: list[Any]) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    for obs in observations:
        obs_type = _normalize_name(getattr(obs, "type", None))
        name = getattr(obs, "name", None) or ""
        normalized_name = _normalize_name(name)

        is_tool_type = obs_type in {"tool", "tool_call", "tool-call"}
        is_known_tool = any(
            token in normalized_name
            for token in (
                "get_union_scheme_data",
                "search_documents",
                "create_ai_call",
                "create_health_call",
                "get_farmer_milk_collection_details",
                "tool",
            )
        )
        if not is_tool_type and not is_known_tool:
            continue

        tool_calls.append(
            {
                "name": name,
                "type": getattr(obs, "type", None),
                "input": to_jsonable(getattr(obs, "input", None)),
                "output": to_jsonable(getattr(obs, "output", None)),
                "latency_ms": _observation_latency_ms(obs),
                "status_message": getattr(obs, "status_message", None),
            }
        )
    return tool_calls


def _pick_primary_trace(traces: list[Any], trace_details: list[Any]) -> Optional[Any]:
    for detail in reversed(trace_details):
        observations = getattr(detail, "observations", None) or []
        if _first_observation(observations, "agent"):
            return detail
    return trace_details[-1] if trace_details else None


def reconstruct_session(api, session_id: str) -> dict[str, Any]:
    """
    Fetch all traces for a session and build one eval artifact-shaped JSON object.
    """
    traces = list_session_traces(api, session_id)
    if not traces:
        raise LangfuseEvalError(f"No traces found for session_id={session_id!r}")

    trace_details = [fetch_trace(api, trace.id) for trace in traces]
    traces_by_name: dict[str, list[dict[str, Any]]] = {}
    all_observations: list[Any] = []

    for summary, detail in zip(traces, trace_details):
        trace_name = summary.name or "unnamed"
        traces_by_name.setdefault(trace_name, []).append(
            {
                "trace_id": summary.id,
                "timestamp": to_jsonable(summary.timestamp),
                "latency_ms": round((getattr(detail, "latency", 0) or 0) * 1000, 2),
            }
        )
        all_observations.extend(getattr(detail, "observations", None) or [])

    primary = _pick_primary_trace(traces, trace_details) or trace_details[-1]
    primary_observations = getattr(primary, "observations", None) or []

    pre_obs = _first_observation(all_observations, "query_pretranslation")
    mod_obs = _first_observation(all_observations, "moderation")
    farmer_obs = _first_observation(all_observations, "farmer_background")
    agent_obs = _first_observation(primary_observations, "agent") or _first_observation(
        all_observations, "agent"
    )
    suggestions_obs = _first_observation(all_observations, "suggestions")
    text_trans_obs = _first_observation(all_observations, "text_translation")
    stream_trans_obs = _first_observation(all_observations, "stream_translation")

    trace_input = getattr(primary, "input", None) or {}
    question_gu = ""
    question_en = ""

    if isinstance(trace_input, dict):
        question_gu = str(trace_input.get("query") or "")
        if trace_input.get("source_lang", "").lower() in {"gu", "gujarati"}:
            question_gu = str(trace_input.get("query") or question_gu)

    if pre_obs is not None:
        pre_input = getattr(pre_obs, "input", None) or {}
        pre_output = getattr(pre_obs, "output", None)
        if isinstance(pre_input, dict) and pre_input.get("text"):
            question_gu = question_gu or str(pre_input["text"])
        question_en = _extract_text(pre_output) or question_en

    if not question_en and isinstance(trace_input, dict):
        question_en = str(trace_input.get("query") or "")

    agent_output = getattr(agent_obs, "output", None) if agent_obs else None
    final_answer = ""
    if isinstance(agent_output, dict):
        final_answer = str(agent_output.get("response") or agent_output.get("response_en") or "")
    else:
        final_answer = _extract_text(agent_output)
    if not final_answer:
        final_answer = _extract_text(getattr(primary, "output", None))

    translated_answer = ""
    translation_obs = stream_trans_obs or text_trans_obs
    if translation_obs is not None:
        translated_answer = _extract_text(getattr(translation_obs, "output", None))
    if not translated_answer and question_gu and final_answer != question_gu:
        translated_answer = _extract_text(getattr(primary, "output", None))

    response_en = _extract_response_en(
        all_observations,
        translation_obs=translation_obs,
        text_translation_obs=text_trans_obs,
        agent_obs=agent_obs,
        fallback_final=final_answer,
    )

    latencies = {
        "trace_ms": round((getattr(primary, "latency", 0) or 0) * 1000, 2),
        "query_pretranslation_ms": _observation_latency_ms(pre_obs) if pre_obs else None,
        "moderation_ms": _observation_latency_ms(mod_obs) if mod_obs else None,
        "farmer_background_ms": _observation_latency_ms(farmer_obs) if farmer_obs else None,
        "agent_ms": _observation_latency_ms(agent_obs) if agent_obs else None,
        "text_translation_ms": _observation_latency_ms(text_trans_obs) if text_trans_obs else None,
        "stream_translation_ms": _observation_latency_ms(stream_trans_obs) if stream_trans_obs else None,
        "suggestions_ms": _observation_latency_ms(suggestions_obs) if suggestions_obs else None,
    }

    tool_calls = _extract_tool_calls(all_observations)

    return {
        "session_id": session_id,
        "question_gu": question_gu,
        "question_en": question_en,
        "moderation": _extract_moderation(mod_obs),
        "farmer_context": {
            "observation": _serialize_observation(farmer_obs) if farmer_obs else None,
            "note": (
                "No dedicated farmer_background span found; farmer markdown is usually embedded in the agent system prompt."
                if farmer_obs is None
                else None
            ),
        },
        "suggestions": _serialize_observation(suggestions_obs) if suggestions_obs else None,
        "tool_calls": tool_calls,
        "response_en": response_en,
        "final_answer": final_answer,
        "translated_answer": translated_answer or final_answer,
        "latencies": latencies,
        "trace_ids": [trace.id for trace in traces],
        "traces_by_name": traces_by_name,
        "primary_trace_id": getattr(primary, "id", None),
    }


def flatten_json_artifact_for_team_csv(data: dict[str, Any], query_index: int) -> dict[str, Any]:
    """Flatten a saved query_XXXX.json artifact into a team-friendly CSV row."""
    moderation = data.get("moderation") or {}
    latencies = data.get("latencies") or {}
    tools = data.get("tool_calls") or []
    suggestions = data.get("suggestions") or {}
    suggestion_texts: list[str] = []
    if isinstance(suggestions, dict):
        suggestion_output = suggestions.get("output") or {}
        if isinstance(suggestion_output, dict):
            suggestion_texts = [
                str(item) for item in (suggestion_output.get("suggestions") or [])
            ]

    error_text = str(data.get("error") or data.get("langfuse_error") or "").strip()
    if error_text:
        error_text = error_text.splitlines()[0][:500]

    has_answer = bool(
        data.get("response_en")
        or data.get("final_answer")
        or data.get("chat_response")
    )
    status = "failed" if error_text and not has_answer else "success"
    if data.get("langfuse_error") and has_answer:
        status = "partial"
    elif error_text and has_answer:
        status = "partial"

    return {
        "query_index": query_index,
        "row_id": data.get("row_id") or "",
        "category": data.get("category") or "",
        "status": status,
        "question_gu": data.get("question_gu") or "",
        "question_en": data.get("question_en") or "",
        "response_en": data.get("response_en") or "",
        "moderation_category": moderation.get("category") or "",
        "moderation_action": moderation.get("action") or "",
        "final_answer_gu": data.get("final_answer") or data.get("chat_response") or "",
        "translated_answer": data.get("translated_answer")
        or data.get("chat_response")
        or data.get("final_answer")
        or "",
        "tool_names": "|".join(str(item.get("name") or "") for item in tools),
        "tool_count": len(tools),
        "session_id": data.get("session_id") or "",
        "primary_trace_id": data.get("primary_trace_id") or "",
        "trace_ids": "|".join(data.get("trace_ids") or []),
        "total_latency_ms": latencies.get("trace_ms") or "",
        "pretranslation_ms": latencies.get("query_pretranslation_ms") or "",
        "moderation_ms": latencies.get("moderation_ms") or "",
        "agent_ms": latencies.get("agent_ms") or "",
        "stream_translation_ms": latencies.get("stream_translation_ms") or "",
        "suggestions_ms": latencies.get("suggestions_ms") or "",
        "suggestions_gu": "|".join(suggestion_texts),
        "error": error_text,
        "raw_json_file": f"eval_outputs/langfuse_raw_json/query_{query_index:04d}.json",
    }


TEAM_SHARE_CSV_COLUMNS = [
    "query_index",
    "status",
    "question_gu",
    "question_en",
    "response_en",
    "moderation_category",
    "moderation_action",
    "final_answer_gu",
    "translated_answer",
    "tool_names",
    "tool_count",
    "session_id",
    "primary_trace_id",
    "trace_ids",
    "total_latency_ms",
    "pretranslation_ms",
    "moderation_ms",
    "agent_ms",
    "stream_translation_ms",
    "suggestions_ms",
    "suggestions_gu",
    "error",
    "raw_json_file",
]

# Golden-set eval export: golden metadata + query/answer fields only (no trace/latency cols).
GOLDEN_SET_EVAL_CSV_COLUMNS = [
    "query_index",
    "row_id",
    "category",
    "status",
    "question_gu",
    "question_en",
    "response_en",
    "moderation_category",
    "moderation_action",
    "final_answer_gu",
    "error",
]

GOLDEN_FULL_CSV_COLUMNS = ["row_id", "category", *TEAM_SHARE_CSV_COLUMNS]


def write_team_shareable_csv(
    raw_json_dir: Path,
    output_path: Path,
    *,
    max_queries: Optional[int] = None,
    fieldnames: Optional[list[str]] = None,
) -> int:
    """Build a UTF-8 CSV (Excel-friendly) from all query_XXXX.json files in raw_json_dir."""
    import csv

    raw_json_dir = Path(raw_json_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    upper = max_queries if max_queries is not None else 10_000
    rows: list[dict[str, Any]] = []
    for index in range(1, upper + 1):
        json_path = raw_json_dir / f"query_{index:04d}.json"
        if not json_path.exists():
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        rows.append(flatten_json_artifact_for_team_csv(data, index))

    columns = fieldnames or TEAM_SHARE_CSV_COLUMNS
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def artifact_needs_retry(data: dict[str, Any]) -> bool:
    """True when a saved artifact should be re-run (hard fail or Langfuse-only partial)."""
    if data.get("langfuse_error"):
        return True
    if data.get("tool_calls") is not None and data.get("trace_ids"):
        return False
    error_text = str(data.get("error") or "").strip()
    has_answer = bool(
        data.get("response_en")
        or data.get("final_answer")
        or data.get("chat_response")
    )
    return bool(error_text) or not has_answer


def collect_retry_indices(
    raw_json_dir: Path,
    *,
    team_csv_path: Optional[Path] = None,
    max_index: int = 200,
) -> list[int]:
    """Indices to re-run: non-success team rows and/or raw JSON artifacts that need retry."""
    raw_json_dir = Path(raw_json_dir)
    indices: set[int] = set()

    if team_csv_path and team_csv_path.exists():
        import csv

        with team_csv_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                status = (row.get("status") or "").strip().lower()
                if status in {"failed", "partial"}:
                    try:
                        indices.add(int(row["query_index"]))
                    except (KeyError, ValueError):
                        pass

    for index in range(1, max_index + 1):
        json_path = raw_json_dir / f"query_{index:04d}.json"
        if not json_path.exists():
            indices.add(index)
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if artifact_needs_retry(data):
            indices.add(index)

    return sorted(indices)


def artifact_to_detail_row(data: dict[str, Any]) -> dict[str, Any]:
    """Map a saved query_XXXX.json artifact to langfuse_queries_eval.csv columns."""
    if data.get("tool_calls") is not None and data.get("trace_ids"):
        flat = flatten_session_for_csv(data)
        preview = str(data.get("chat_response_preview") or data.get("final_answer") or "")
        if len(preview) > 300:
            preview = preview[:300] + f"... [{len(preview) - 300} more chars]"
        flat["chat_response_preview"] = preview
        flat["error"] = str(data.get("error") or "")
        return flat

    chat_response = str(data.get("chat_response") or "")
    langfuse_error = str(data.get("langfuse_error") or "")
    error_text = str(data.get("error") or "").strip()

    if langfuse_error:
        return {
            "question_gu": data.get("question_gu") or "",
            "session_id": data.get("session_id") or "",
            "final_answer": chat_response,
            "translated_answer": chat_response,
            "chat_response_preview": chat_response[:300],
            "error": (
                f"{langfuse_error} "
                "(Chat succeeded; Langfuse traces missing — "
                "restart API with LANGFUSE_* in .env or use dev API host.)"
            ),
        }

    return {
        "question_gu": data.get("question_gu") or "",
        "session_id": data.get("session_id") or "",
        "error": error_text,
    }


def rebuild_detail_csv_from_raw(
    raw_json_dir: Path,
    output_path: Path,
    *,
    max_index: int = 200,
) -> int:
    """Rewrite detail eval CSV from all query_XXXX.json files (1..max_index)."""
    import csv

    raw_json_dir = Path(raw_json_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for index in range(1, max_index + 1):
        json_path = raw_json_dir / f"query_{index:04d}.json"
        if not json_path.exists():
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        rows.append(artifact_to_detail_row(data))

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "question_gu",
                "question_en",
                "response_en",
                "moderation_category",
                "moderation_action",
                "final_answer",
                "translated_answer",
                "tool_names",
                "tool_count",
                "session_id",
                "trace_ids",
                "latency",
                "chat_response_preview",
                "error",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def flatten_session_for_csv(reconstructed: dict[str, Any]) -> dict[str, Any]:
    moderation = reconstructed.get("moderation") or {}
    tool_calls = reconstructed.get("tool_calls") or []
    latencies = reconstructed.get("latencies") or {}
    return {
        "question_gu": reconstructed.get("question_gu") or "",
        "question_en": reconstructed.get("question_en") or "",
        "response_en": reconstructed.get("response_en") or "",
        "moderation_category": moderation.get("category") or "",
        "moderation_action": moderation.get("action") or "",
        "final_answer": reconstructed.get("final_answer") or "",
        "translated_answer": reconstructed.get("translated_answer") or "",
        "tool_names": "|".join(
            str(item.get("name") or "") for item in tool_calls if item.get("name")
        ),
        "tool_count": len(tool_calls),
        "session_id": reconstructed.get("session_id") or "",
        "trace_ids": "|".join(reconstructed.get("trace_ids") or []),
        "latency": latencies.get("trace_ms") or "",
    }
