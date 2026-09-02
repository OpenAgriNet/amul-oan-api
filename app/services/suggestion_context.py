"""Helpers for grounding suggestion generation on current-turn tool outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.models.union import UnionName, any_union_banned_from_ai_calls, resolve_supported_unions
from helpers.utils import get_logger

_BANKS_PATH = Path(__file__).resolve().parents[2] / "assets" / "suggestion_banks.json"
_SCHEME_TOOLS = {"get_union_scheme_data", "get_vistaar_scheme_info"}
_CONTEXTUAL_TOOLS = {
    "get_farmer_milk_collection_details",
    "get_vistaar_weather",
    "get_vistaar_mandi_prices",
    "check_loan_eligibility",
    "create_ai_call",
    "create_health_call",
}
_SUPPORTED_SCHEME_CACHE_UNIONS = {
    UnionName.BANAS.value,
    UnionName.KUTCH.value,
    UnionName.SUMUL.value,
    UnionName.SURENDRANAGAR.value,
}
_MAX_SCHEME_CATALOG_ENTRIES = 20
_LOAN_OUTCOME_MARKERS: list[tuple[str, str]] = [
    ("ELIGIBLE — OFFER ONLY", "eligible_offer"),
    ("APPROVED.", "approved"),
    ("ALREADY ISSUED.", "already_issued"),
    ("NOT ELIGIBLE.", "not_eligible"),
    ("NO PROFILE.", "no_profile"),
]

logger = get_logger(__name__)


def load_suggestion_banks(path: Path | None = None) -> dict[str, Any]:
    """Load the curated suggestion banks JSON."""
    target = path or _BANKS_PATH
    return json.loads(target.read_text(encoding="utf-8"))


def tools_called_this_turn(raw_history: list[Any]) -> list[str]:
    """Return ordered unique tool names for the latest user turn."""
    turn = _latest_turn_messages(raw_history)
    call_name_by_id: dict[str, str] = {}
    ordered: list[str] = []
    seen: set[str] = set()

    for message in turn:
        for part in getattr(message, "parts", []):
            part_kind = getattr(part, "part_kind", "")
            tool_name = None
            if part_kind == "tool-call":
                tool_name = getattr(part, "tool_name", None)
                call_id = getattr(part, "tool_call_id", None)
                if call_id and tool_name:
                    call_name_by_id[call_id] = tool_name
            elif part_kind == "tool-return":
                tool_name = getattr(part, "tool_name", None)
                if not tool_name:
                    call_id = getattr(part, "tool_call_id", None)
                    tool_name = call_name_by_id.get(call_id)

            if tool_name and tool_name not in seen:
                seen.add(tool_name)
                ordered.append(tool_name)

    return ordered


def open_bank_domains(
    tools_called: list[str],
    farmer_unions: list[str] | None,
    *,
    enable_network: bool,
    loan_feature_enabled: bool,
    banks: dict[str, Any] | None = None,
) -> list[str]:
    """Open only domains triggered by called tools and allowed by capability gates."""
    bank_data = banks or load_suggestion_banks()
    domains = bank_data.get("domains", {})
    called = set(tools_called or [])
    unions = [u for u in (farmer_unions or []) if (u or "").strip()]

    opened: list[str] = []
    for domain_name, domain_meta in domains.items():
        opens_on_tools = set(domain_meta.get("opens_on_tools", []))
        if not opens_on_tools or not (called & opens_on_tools):
            continue
        if domain_name == "milk_quantity" and not unions:
            continue
        if domain_name == "ai_call" and any_union_banned_from_ai_calls(unions):
            continue
        if domain_name == "vistaar" and not enable_network:
            continue
        if domain_name == "loan_eligibility" and not loan_feature_enabled:
            continue
        opened.append(domain_name)

    return opened


def capability_allowlist(
    farmer_unions: list[str] | None,
    *,
    enable_network: bool,
    loan_feature_enabled: bool,
) -> list[str]:
    """Capabilities currently answerable for this farmer/session."""
    unions = [u for u in (farmer_unions or []) if (u or "").strip()]
    allowed = ["animal_health"]
    if unions:
        allowed.append("milk_quantity")
    if not any_union_banned_from_ai_calls(unions):
        allowed.append("ai_call")
    if enable_network:
        allowed.append("vistaar")
    if loan_feature_enabled:
        allowed.append("loan_eligibility")
    return allowed


def pick_candidates(
    open_domains: list[str],
    banks: dict[str, Any],
    *,
    tools_called: list[str],
    tool_outcomes: dict[str, str] | None = None,
    max_candidates: int = 10,
) -> list[dict[str, Any]]:
    """Pick candidate questions only from opened domains."""
    selected: list[dict[str, Any]] = []
    called = set(tools_called or [])
    domains = banks.get("domains", {})
    outcomes = tool_outcomes or {}

    for domain_name in open_domains:
        domain_meta = domains.get(domain_name)
        if not domain_meta:
            continue
        opens_on_tools = list(domain_meta.get("opens_on_tools", []))
        domain_tool = next((tool for tool in opens_on_tools if tool in called), None)
        domain_outcome = outcomes.get(domain_tool or "", "unknown")
        questions = domain_meta.get("questions", [])
        if domain_name == "vistaar":
            allow_weather = "get_vistaar_weather" in called
            allow_mandi = "get_vistaar_mandi_prices" in called
            if allow_weather or allow_mandi:
                filtered = []
                for question in questions:
                    tag = question.get("tag")
                    if tag == "weather" and allow_weather:
                        filtered.append(question)
                    elif tag == "mandi" and allow_mandi:
                        filtered.append(question)
                questions = filtered
            else:
                questions = []
        for question in questions:
            required_outcomes = question.get("requires_outcomes")
            if required_outcomes:
                if domain_outcome not in set(required_outcomes):
                    continue
            selected.append(
                {
                    "domain": domain_name,
                    "id": question.get("id"),
                    "en": question.get("en", ""),
                    "gu": question.get("gu", ""),
                    "hi": question.get("hi", ""),
                    "tag": question.get("tag"),
                }
            )
            if len(selected) >= max_candidates:
                return selected

    return selected


def extract_returned_docs(
    raw_history: list[Any],
    *,
    max_search_chunks: int = 2,
    max_chars: int = 1200,
) -> dict[str, Any]:
    """Extract current-turn tool-return payloads for suggestions grounding."""
    turn = _latest_turn_messages(raw_history)
    call_name_by_id: dict[str, str] = {}
    search_chunks: list[str] = []
    scheme_tool_returns: list[dict[str, str]] = []
    contextual_tool_returns: list[dict[str, str]] = []
    tool_outcomes: dict[str, str] = {}

    for message in turn:
        for part in getattr(message, "parts", []):
            if getattr(part, "part_kind", "") == "tool-call":
                call_id = getattr(part, "tool_call_id", None)
                tool_name = getattr(part, "tool_name", None)
                if call_id and tool_name:
                    call_name_by_id[call_id] = tool_name

    for message in turn:
        for part in getattr(message, "parts", []):
            if getattr(part, "part_kind", "") != "tool-return":
                continue
            tool_name = getattr(part, "tool_name", None)
            call_id = getattr(part, "tool_call_id", None)
            if not tool_name and call_id:
                tool_name = call_name_by_id.get(call_id)
            if not tool_name:
                continue

            raw_content = str(getattr(part, "content", "") or "")
            if tool_name == "search_documents":
                # Split the full tool return first, then truncate each selected
                # chunk. Truncating the whole payload first can erase later hits
                # when the first document alone exceeds max_chars.
                # Accumulate across multiple search_documents returns in this turn
                # and cap the final list at max_search_chunks.
                if len(search_chunks) < max_search_chunks:
                    remaining = max_search_chunks - len(search_chunks)
                    search_chunks.extend(
                        _extract_search_chunks(
                            raw_content,
                            max_chunks=remaining,
                            max_chars=max_chars,
                        )
                    )
                continue
            content = _truncate(raw_content, max_chars=max_chars)
            outcome = _classify_tool_outcome(tool_name, raw_content)
            if outcome:
                tool_outcomes[tool_name] = outcome
            if tool_name in _SCHEME_TOOLS:
                scheme_tool_returns.append({"tool_name": tool_name, "content": content})
            if tool_name in _CONTEXTUAL_TOOLS:
                contextual_tool_returns.append({"tool_name": tool_name, "content": content})

    return {
        "search_chunks": search_chunks,
        "scheme_tool_returns": scheme_tool_returns,
        "contextual_tool_returns": contextual_tool_returns,
        "tool_outcomes": tool_outcomes,
    }


async def load_union_scheme_catalog(
    tools_called: list[str],
    farmer_unions: list[str] | None,
    *,
    max_entries: int = _MAX_SCHEME_CATALOG_ENTRIES,
) -> list[str]:
    """Load cached union scheme titles only when a scheme tool ran this turn.

    Never injects the Redis catalog on unrelated turns (milk/health/AI/etc.).
    Cache/read failures degrade to [] so suggestions still run.
    """
    called = set(tools_called or [])
    if not (called & _SCHEME_TOOLS):
        return []

    scheme_unions = resolve_supported_unions(farmer_unions, _SUPPORTED_SCHEME_CACHE_UNIONS)
    if not scheme_unions:
        return []

    # Lazy import keeps suggestion_context importable when Redis scheme deps
    # are unavailable in lightweight unit tests.
    from app.services.scheme_ingestion import (
        SchemeCacheError,
        SchemeDependencyError,
        get_cached_scheme_records_for_union,
    )

    lines: list[str] = []
    seen: set[tuple[str, str]] = set()
    for union_name in scheme_unions:
        try:
            records = await get_cached_scheme_records_for_union(union_name)
        except (SchemeDependencyError, SchemeCacheError) as exc:
            logger.warning(
                "Suggestion scheme catalog skipped because cache unavailable union=%s error=%s",
                union_name,
                exc,
            )
            continue
        except Exception as exc:
            logger.warning(
                "Suggestion scheme catalog skipped because of unexpected error union=%s error=%s",
                union_name,
                exc,
            )
            continue

        for record in records or []:
            title = str(record.get("scheme_title") or "").strip()
            link = str(record.get("scheme_url") or "").strip()
            if not title:
                continue
            dedupe_key = (title.casefold(), link)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            if link:
                lines.append(f"{union_name.title()}: {title} — {link}")
            else:
                lines.append(f"{union_name.title()}: {title}")
            if len(lines) >= max_entries:
                return lines

    return lines


def _latest_turn_messages(raw_history: list[Any]) -> list[Any]:
    if not raw_history:
        return []
    last_user_idx = -1
    for idx in range(len(raw_history) - 1, -1, -1):
        if any(getattr(part, "part_kind", "") == "user-prompt" for part in getattr(raw_history[idx], "parts", [])):
            last_user_idx = idx
            break
    if last_user_idx == -1:
        return []
    return raw_history[last_user_idx:]


def _extract_search_chunks(content: str, *, max_chunks: int, max_chars: int) -> list[str]:
    if not content:
        return []
    if "\n\n----\n\n" in content:
        chunks = [chunk.strip() for chunk in content.split("\n\n----\n\n") if chunk.strip()]
    else:
        chunks = [content.strip()]
    return [_truncate(chunk, max_chars=max_chars) for chunk in chunks[:max_chunks]]


def _truncate(value: str, *, max_chars: int) -> str:
    text = (value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def _classify_tool_outcome(tool_name: str, raw_content: str) -> str:
    """Infer a coarse outcome label from a tool-return payload."""
    text = (raw_content or "").strip()
    lower = text.lower()
    if not text:
        return "unknown"

    if tool_name == "check_loan_eligibility":
        for marker, outcome in _LOAN_OUTCOME_MARKERS:
            if text.startswith(marker):
                return outcome
        if "temporarily unavailable" in lower:
            return "unavailable"
        return "unknown"

    if tool_name == "create_health_call":
        if "booked successfully" in lower:
            return "success"
        if "already has an active health call booking" in lower:
            return "already_booked"
        if "failed" in lower or "unable to create" in lower:
            return "failed"
        if "only handles dairy farming" in lower:
            return "blocked"
        return "unknown"

    if tool_name == "create_ai_call":
        if "ticket number" in lower or "booked successfully" in lower:
            return "success"
        if "already has an active artificial insemination booking" in lower:
            return "already_booked"
        if "booking could not be confirmed" in lower:
            return "unconfirmed"
        if "booking failed" in lower:
            return "failed"
        if "only handles dairy farming" in lower:
            return "blocked"
        if "milk society" in lower:
            return "blocked"
        return "unknown"

    return "unknown"
