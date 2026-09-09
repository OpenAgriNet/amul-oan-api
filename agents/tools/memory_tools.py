"""Search passages, list episodes, and read selected chunks with bounded output."""
from __future__ import annotations

import asyncio
from functools import wraps
import os
import uuid

import httpx
from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition

from agents.deps import FarmerContext
from app.services.memory import MEMORY_ENABLED
from app.services.memory_store import MemoryStore
from helpers.utils import get_logger

logger = get_logger(__name__)
TIMEOUT = float(os.getenv("MEMORY_TOOL_TIMEOUT_SECONDS", "4.0"))
TURN_MAX_CHARS = int(os.getenv("MEMORY_TOOL_TURN_MAX_CHARS", "12000"))
SEARCH_LIMIT = min(10, max(1, int(os.getenv("MEMORY_RECALL_LIMIT", "3"))))
LIST_LIMIT = min(15, max(1, int(os.getenv("MEMORY_LIST_LIMIT", "8"))))
MAX_CALLS = int(os.getenv("MEMORY_TOOL_MAX_CALLS", "6"))
FilterValue = str | int | float | bool
UNAVAILABLE = "Memory could not be checked. Do not infer that nothing is on record."
DISABLED = "Memory is disabled or unavailable for this farmer. Do not infer absence."
CONTEXT = ("Records from earlier chats may be incomplete or out of date. Confirm current circumstances. "
           "Keep episode references and chunk numbers internal.\n")


def _one_at_a_time(fn):
    """Parallel model calls share one per-turn output budget; check it in order."""
    @wraps(fn)
    async def run(ctx, *args, **kwargs):
        lock = getattr(ctx.deps, "_memory_tool_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            ctx.deps._memory_tool_lock = lock
        async with lock:
            return await fn(ctx, *args, **kwargs)
    return run


BUDGET_REACHED = "Memory lookup budget reached for this turn. No additional records were shown; say when the answer is partial."


def _budget(ctx):
    return max(0, TURN_MAX_CHARS - getattr(ctx.deps, "memory_tool_chars", 0))


def _finish(ctx, text):
    # Never shorten a chunk to fit. The program stops additional evidence instead.
    if len(text) > _budget(ctx):
        return BUDGET_REACHED
    ctx.deps.memory_tool_chars = getattr(ctx.deps, "memory_tool_chars", 0) + len(text)
    return text


def _reference(value):
    try:
        return str(uuid.UUID(value))
    except (ValueError, TypeError, AttributeError):
        return None


async def _request(ctx, operation, **kwargs):
    farmer = (ctx.deps.mobile or "").strip()
    if not MEMORY_ENABLED or not farmer or getattr(ctx.deps, "persona", "farmer") != "farmer":
        return None, DISABLED
    used = getattr(ctx.deps, "memory_tool_calls", 0)
    if used >= MAX_CALLS:
        return None, "Memory lookup budget reached for this turn. Use existing evidence; do not invent missing detail."
    ctx.deps.memory_tool_calls = used + 1

    async def fetch():
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            store = MemoryStore(client, farmer)
            body = await getattr(store, operation)(**kwargs)
            if not isinstance(body, dict):
                raise ValueError("invalid memory response")
            return body
    try:
        body = await asyncio.wait_for(fetch(), timeout=TIMEOUT)
    except PermissionError:
        return None, DISABLED
    except Exception as exc:
        logger.warning("memory %s failed: %s", operation, exc)
        return None, UNAVAILABLE
    return (None, DISABLED) if body.get("memory_enabled") is False else (body, None)


def _rows(body, name):
    rows = body.get(name)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        return None
    return rows


def _format(rows, *, passages=False, prefix="", footer=""):
    """Render complete summaries or complete passages; no display text slicing."""
    output = CONTEXT + prefix
    for row in rows:
        ref = f"\n- [memory_ref: {row.get('id', 'unknown')}]"
        if passages and row.get("chunk_number"):
            ref += f" [chunk {row['chunk_number']}]"
        facts = f" [state: {row.get('status', 'n/a')}; last mentioned: {row.get('source_ts') or 'date not recorded'}]"
        if row.get("first_source_ts") and row.get("first_source_ts") != row.get("source_ts"):
            facts += f" [first: {row['first_source_ts']}]"
        if row.get("expires_at"):
            facts += f" [end date: {row['expires_at']}; not proof of resolution]"
        text = (row.get("passage") if passages else row.get("headline")) or row.get("headline") or ""
        if not isinstance(text, str):
            return UNAVAILABLE
        if row.get("chunk_count"):
            facts += f" [{row['chunk_count']} chunks]"
        output += ref + facts + " " + text
    return output + footer


@_one_at_a_time
async def search_memories(
    ctx: RunContext[FarmerContext], query: str, memory_ref: str | None = None,
) -> str:
    """Find complete matching chunks using ordinary words or specific keywords.

    Rephrase/broaden if useful. Use list_memories for lists or counts of records.
    The program controls how many results are returned; chunks are never shortened.

    Args:
        query: What to find in the remembered detail.
        memory_ref: Optional episode reference to search within; omit to search this farmer's episodes.
    """
    if _budget(ctx) < 512:
        return BUDGET_REACHED
    if not query.strip():
        return "Provide words to search for, or use list_memories to browse."
    if memory_ref is not None:
        memory_ref = _reference(memory_ref)
        if memory_ref is None:
            return "Use a valid memory_ref from a result."
    body, error = await _request(ctx, "search", query=query, using="expanded",
                                 limit=SEARCH_LIMIT, memory_ref=memory_ref)
    if error:
        return error
    rows = _rows(body, "hits")
    if rows is None:
        return UNAVAILABLE
    if not rows:
        return _finish(ctx, CONTEXT + "No search matches returned. Try different wording if useful.")
    footer = "\nPartial search results; broaden or retry if needed." if body.get("partial") else ""
    if any(row.get("chunk_indexed") is False for row in rows):
        footer += "\nSome old episodes lack a chunk index; direct reading is still available."
    return _finish(ctx, _format(rows, passages=True,
                               prefix="Likely passages, not an exhaustive list.", footer=footer))


@_one_at_a_time
async def list_memories(
    ctx: RunContext[FarmerContext], state: str | None = None,
    filters: dict[str, FilterValue] | None = None, cursor: str | None = None,
) -> str:
    """List recorded episodes by optional state and metadata, with pagination.

    Call this for requests to list or count remembered matters. Automatic search
    matches are only a sample and cannot answer such requests completely.

    Args:
        state: Optional open, pending, resolved, n/a, or unresolved (open OR pending). Omit for all states.
        filters: Optional farmer-specific metadata matches (AND); untagged memories may be missed.
        cursor: Next-page reference from a previous listing with the same filters.
    """
    if _budget(ctx) < 512:
        return BUDGET_REACHED
    statuses = None
    if state == "unresolved":
        statuses = ["open", "pending"]
    elif state in {"open", "pending", "resolved", "n/a"}:
        statuses = [state]
    elif state not in (None, ""):
        return "Unknown state. Use open, pending, resolved, n/a, unresolved, or omit it."
    body, error = await _request(ctx, "list_entries", status=statuses, filters=filters,
                                 cursor=cursor, limit=LIST_LIMIT)
    if error:
        return error
    rows = _rows(body, "entries")
    if rows is None:
        return UNAVAILABLE
    scope = f"Recorded state: {state or 'all'}; matching the requested detail filters.\n"
    if not rows:
        return _finish(ctx, CONTEXT + scope + "No matching entries returned. Missing tags can affect this result.")
    footer = ""
    if body.get("next_cursor"):
        footer = f"\nMore records exist. Continue with the same filters and cursor={body['next_cursor']}."
    elif body.get("truncated"):
        footer = "\nOnly part of the matching records was returned; do not infer totals."
    return _finish(ctx, _format(rows, prefix=scope, footer=footer))


def _render_read(body):
    entry = body["entry"]
    output = CONTEXT + f"Selected memory_ref: {entry['id']}. State: {entry.get('status', 'n/a')}.\n"
    if entry.get("source_ts"):
        output += f"Last mentioned: {entry['source_ts']}.\n"
    if entry.get("expires_at"):
        output += f"End date: {entry['expires_at']}; not proof of resolution.\n"
    if body.get("current") is False:
        output += "This is an older version.\n"
    for row in body["chunks"]:
        output += f"\nChunk {row['number']}:\n{row['text']}"
    next_ids = (body.get("continuation") or {}).get("chunk_ids")
    if next_ids:
        output += f"\nMore chunks available. To continue, use the same memory_ref and chunk_ids={next_ids}."
    else:
        output += "\nEnd of the requested selection."
    return output


@_one_at_a_time
async def read_memory(
    ctx: RunContext[FarmerContext], memory_ref: str, chunk_ids: list[int] | None = None,
    include_history: bool = False,
) -> str:
    """Open complete chunks from an episode. The program handles page sizes.

    Args:
        memory_ref: Episode reference from Level 2 or a result. Never invent one.
        chunk_ids: Optional chunk numbers from contents/results; omit to open the first page.
        include_history: Show earlier-version summaries/references instead of current detail.
    """
    memory_ref = _reference(memory_ref)
    if memory_ref is None:
        return "Use a memory_ref from a result; that reference is invalid."
    if _budget(ctx) < 512:
        return BUDGET_REACHED
    if chunk_ids is not None and (not chunk_ids or len(chunk_ids) > 50 or any(type(n) is not int or n < 1 for n in chunk_ids)):
        return "Use chunk numbers shown for this episode (at most 50)."
    if include_history:
        body, error = await _request(ctx, "history", memory_ref=memory_ref, limit=LIST_LIMIT)
        if error:
            return error
        rows = _rows(body, "history")
        if rows is None:
            return UNAVAILABLE
        return _finish(ctx, _format(rows,
            prefix="Earlier accounts, not additional current facts. Read their references for detail.",
            footer="\nOnly part of the history is shown." if body.get("truncated") else ""))
    body, error = await _request(ctx, "read", memory_ref=memory_ref, chunk_ids=chunk_ids)
    if error:
        return error
    if not isinstance(body.get("entry"), dict) or _rows(body, "chunks") is None:
        return UNAVAILABLE
    try:
        output = _render_read(body)
    except (KeyError, TypeError, ValueError):
        return UNAVAILABLE
    return _finish(ctx, output)


async def prepare_memory_tool(ctx: RunContext[FarmerContext], tool_def: ToolDefinition) -> ToolDefinition | None:
    if (not MEMORY_ENABLED or not (ctx.deps.mobile or "").strip()
            or getattr(ctx.deps, "persona", "farmer") != "farmer"):
        return None
    return tool_def
