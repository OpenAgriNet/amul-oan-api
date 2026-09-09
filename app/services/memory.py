"""
Memory integration (memory_v0).

Reads a farmer's accumulated memory directly from Qdrant and turns it into a
short, bounded context block for the prompt. Deliberately narrow:

- The global MEMORY_ENABLED switch defaults off. Once on, per-farmer settings
  default on for now; an explicit Qdrant off flag suppresses reply use. The
  background writer never changes those settings.
- **Read-only, and never on the critical path for long.** One quick lookup with a
  hard timeout; on any error or timeout it returns nothing rather than delaying or
  failing a turn. Writing memory is the separate background job's problem, not this
  module's.
- **Bounded output.** The injected block is capped, so memory can never quietly eat
  the prompt budget.
- **Filtered evidence.** The writer checks identifying/sensitive content and the
  reader hides identifier metadata in reply reads. These checks reduce exposure;
  they do not establish perfect detection of identifiers in generated prose.

Design: oan_horizon/future_work/memory_amul/memory-overview.md (Sections 3a, 6).
"""
import asyncio
import os

import httpx

from helpers.utils import get_logger
from app.services.memory_store import MemoryStore, validate_config

logger = get_logger(__name__)

# The reader checks the farmer's separate settings point in Qdrant before reads.
# The background writer and model cannot change that setting.
MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
if MEMORY_ENABLED:
    validate_config()
MEMORY_TIMEOUT_SECONDS = float(os.getenv("MEMORY_TIMEOUT_SECONDS", "2.0"))
MEMORY_TOP_K = int(os.getenv("MEMORY_TOP_K", "3"))
# Bounds the standing record, episode summaries, contents, dates and available keys.
MEMORY_MAX_CHARS = int(os.getenv("MEMORY_MAX_CHARS", "4500"))


async def fetch_memory_context(farmer_id: str | None, query: str) -> str:
    """Return a short context block for this turn, or "" when there is nothing to add.

    Never raises: any failure (unavailable store, timeout, unexpected shape) yields "",
    so a memory problem cannot fail the turn; waiting is bounded by the deadline.
    """
    # Only the global switch is checked here. Whether memory is on for *this*
    # farmer is decided (and enforced) by the Qdrant reader, which returns
    # nothing when it's off for them — so there is no second place to keep in sync.
    if not MEMORY_ENABLED or not farmer_id:
        return ""

    try:
        async with httpx.AsyncClient(timeout=MEMORY_TIMEOUT_SECONDS) as client:
            store = MemoryStore(client, farmer_id)
            async def lookup():
                if not await store.enabled():
                    return "", [], []
                return await asyncio.gather(
                    _standing_summary(store, farmer_id),
                    _relevant_entries(store, farmer_id, query),
                    _available_filters(store, farmer_id), return_exceptions=True)
            standing, recalled, filters = await asyncio.wait_for(lookup(), timeout=MEMORY_TIMEOUT_SECONDS)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("memory: lookup failed for farmer, continuing without it: %s", exc)
        return ""

    standing_text = standing if isinstance(standing, str) else ""
    recalled_lines = recalled if isinstance(recalled, list) else []
    filter_lines = filters if isinstance(filters, list) else []
    if not standing_text and not recalled_lines:
        return ""

    omitted = False
    while standing_text or recalled_lines:
        parts: list[str] = []
        if standing_text:
            parts.append(f"About this farmer (from earlier conversations):\n{standing_text}")
        if recalled_lines:
            joined = "\n".join(f"- {line}" for line in recalled_lines)
            parts.append(
                "SEARCH SAMPLE ONLY: use list_memories to list or count remembered matters.\n"
                "Possibly relevant from earlier conversations "
                "(may be out of date — confirm with the farmer rather than stating it as fact). "
                "Dates are given so you can judge relevance yourself: an end date that has "
                "passed means the deadline or withdrawal period is over, NOT that the matter "
                "was settled — a request marked open only tells you its last recorded state:\n"
                f"{joined}"
            )
        if filter_lines:
            parts.append(
                "Extra details on record for this farmer, if you need to look something up:\n"
                + "\n".join(f"- {line}" for line in filter_lines)
            )
        if omitted:
            parts.append("(Some memory was omitted to stay within budget; this is not everything on record.)")
        block = "\n\n".join(parts)
        if len(block) <= MEMORY_MAX_CHARS:
            return block
        omitted = True
        # Drop whole optional rows, then whole episodes; preserve evidence warnings.
        if filter_lines:
            filter_lines.pop()
        elif recalled_lines:
            recalled_lines.pop()
        else:
            break
    return ""


async def _standing_summary(store: MemoryStore, farmer_id: str) -> str:
    """Layer 1: the standing record. A direct fetch of one document, no search.

    There is exactly one of these per farmer, at a fixed address, holding a fixed set
    of fields — so this is a point read, not a query. Only the fields that actually
    have content come back, which is what keeps the always-injected block short for
    a farmer we barely know.
    """
    body = await store.profile()
    if body.get("memory_enabled") is False:
        return ""
    fields = body.get("profile") or {}
    if not fields:
        return ""
    return "\n".join(f"- {k.replace('_', ' ')}: {v}" for k, v in fields.items())


async def _relevant_entries(store: MemoryStore, farmer_id: str, query: str) -> list[str]:
    """Level 2: per-turn search over the summary and inline contents."""
    body = await store.search(query, limit=MEMORY_TOP_K)
    if body.get("memory_enabled") is False:
        return []
    lines: list[str] = []
    for hit in body.get("hits", []):
        # The service excludes internal records; legacy memories remain readable.
        headline = (hit.get("headline") or "").strip()
        if not headline:
            continue
        bits = [f"[memory_ref: {hit['id']}] {headline}"]
        status = hit.get("status")
        if status and status != "n/a":
            bits.append(f"[status: {status}]")
        # Dates are SHOWN, never used to hide a memory — the reader filters
        # nothing on dates by design. That puts the judgement here, in the model's
        # hands, where it can be reasoned about and said out loud, instead of in a
        # filter that silently removes things nobody can then reason about.
        bits.append(f"[last mentioned: {hit.get('source_ts') or 'date not recorded'}]")
        if hit.get("first_source_ts") and hit["first_source_ts"] != hit.get("source_ts"):
            bits.append(f"[first mentioned: {hit['first_source_ts']}]")
        if hit.get("expires_at"):
            bits.append(f"[had an end date of {hit['expires_at']}]")
        times = hit.get("times_raised")
        if isinstance(times, int) and times > 1:
            bits.append(f"[raised in {times} separate conversations]")
        if hit.get("chunk_count"):
            bits.append(f"[{hit['chunk_count']} chunks]")
        for row in hit.get("contents", [])[:5]:
            bits.append(f"\n  Chunks {row['start_chunk']}-{row['end_chunk']}: {row['text']}")
        lines.append(" ".join(bits))
    return lines


async def _available_filters(store: MemoryStore, farmer_id: str) -> list[str]:
    """What extra details exist on record for THIS farmer, and their values.

    Included only when standing context or headline matches are also present.
    Uses the farmer-scoped keys endpoint, never the global one — the global view
    crosses farmer boundaries and must never reach a conversation.
    """
    body = await store.keys()
    if body.get("memory_enabled") is False:
        return []
    lines: list[str] = []
    for k in body.get("keys", [])[:8]:
        name = (k.get("filter_path") or "").removeprefix("metadata.")
        if not name:
            continue
        desc = k.get("description")
        vals = k.get("values_for_this_farmer") or []
        bits = [name]
        if desc:
            bits.append(f"({desc})")
        if vals:
            bits.append(f"— currently: {', '.join(str(v) for v in vals[:4])}")
        lines.append(" ".join(bits))
    return lines
