"""Doctor-persona document search against a pinned Qdrant search service.

The farmer and voice agents search through Beckn (vet BPP). The doctor persona
can instead call the Qdrant search service directly, so its evidence comes from
a collection frozen for the doctor rollout. Post-processing mirrors the vet BPP
in Qdrant mode (services/vet-kb-bpp/src/qdrant.ts, rerank.ts, catalog.ts in
amul-bpp): 120 hybrid candidates, bm25lite rerank, at most 2 chunks per
document, top 12, then the same numbered-snippet text the Beckn tool returns.

With DOCTOR_SEARCH_URL unset the tool is the farmer search_documents, unchanged.
"""

import re
from typing import Any

import httpx

from agents.tools.search import (
    _TOKEN_RE,
    _expand_veterinary_synonyms,
    _validate_search_query,
    search_documents,
)
from agents.tools.telemetry import report_tool
from app.config import settings
from helpers.utils import get_logger

logger = get_logger(__name__)

# Vet BPP settings on dev (amul-net-bpp-vet, SEARCH_BACKEND=qdrant).
_FINAL_TOP_K = 12
_CANDIDATES = 120
_MAX_PER_DOC = 2
_HYBRID_ALPHA = 0.6
_HYBRID_RRF_K = 60
_EXCLUDE_REFERENCE = False
_TIMEOUT_S = 20.0

_META_FIELDS = (
    "name", "name_en", "name_gu", "filename", "title_en", "title_gu",
    "category_tags", "description", "doc_short_description", "doc_llm_description",
)


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _token_overlap_score(query: str, text: str) -> float:
    q_tokens = set(_TOKEN_RE.findall(_normalize_text(query)))
    t_tokens = set(_TOKEN_RE.findall(_normalize_text(text)))
    if not q_tokens or not t_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / len(q_tokens)


def _rerank_hits(query: str, hits: list[dict]) -> list[dict]:
    """bm25lite: 0.62*semantic_norm + 0.30*lexical + 0.08*lexical_meta - 0.12*is_reference."""
    if not hits:
        return hits
    raw = [float(h.get("_score") or 0.0) for h in hits]
    low, high = min(raw), max(raw)
    denom = high - low if high > low else 1.0
    scored = []
    for hit, score in zip(hits, raw):
        meta = " ".join(str(hit.get(k) or "") for k in _META_FIELDS)
        lexical_meta = _token_overlap_score(query, meta)
        lexical = max(_token_overlap_score(query, str(hit.get("text") or "")), lexical_meta)
        penalty = -0.12 if hit.get("is_reference") else 0.0
        scored.append((0.62 * (score - low) / denom + 0.30 * lexical + 0.08 * lexical_meta + penalty, hit))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [hit for _, hit in scored]


def _doc_key(hit: dict) -> str:
    for key in ("doc_id", "filename", "name_en", "name", "_id"):
        value = str(hit.get(key) or "").strip()
        if value:
            return value
    return ""


def _apply_doc_diversity(hits: list[dict], top_k: int, max_per_doc: int) -> list[dict]:
    selected: list[dict] = []
    counts: dict[str, int] = {}
    for hit in hits:
        key = _doc_key(hit)
        if counts.get(key, 0) >= max_per_doc:
            continue
        counts[key] = counts.get(key, 0) + 1
        selected.append(hit)
        if len(selected) >= top_k:
            break
    for hit in hits:
        if len(selected) >= top_k:
            break
        if hit not in selected:
            selected.append(hit)
    return selected


def _title(hit: dict) -> str:
    for key in ("title_en", "name_en", "title_gu", "name_gu"):
        value = str(hit.get(key) or "").strip()
        if value:
            return value
    return "Untitled passage"


def _format(hits: list[dict]) -> str:
    lines = []
    for i, hit in enumerate(hits, 1):
        text = str(hit.get("text") or "").strip() or _title(hit)
        src = str(hit.get("filename") or hit.get("source") or "")
        lines.append(f"{i}. {_title(hit)}\n{text}" + (f"\n(source: {src})" if src else ""))
    return "\n\n".join(lines)


async def _qdrant_search(query: str) -> list[dict]:
    cleaned = _normalize_text(query)
    payload = {
        "q": cleaned if cleaned.startswith("query:") else f"query: {cleaned}",
        "limit": _CANDIDATES,
        "alpha": _HYBRID_ALPHA,
        "rrf_k": _HYBRID_RRF_K,
        "exclude_reference": _EXCLUDE_REFERENCE,
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
        r = await client.post(
            f"{settings.doctor_search_url.rstrip('/')}/search",
            json=payload,
            headers={"Authorization": f"Bearer {settings.doctor_search_api_key}"},
        )
        r.raise_for_status()
        hits = r.json().get("hits")
    if not isinstance(hits, list):
        raise ValueError("search service response has no hits list")
    return _apply_doc_diversity(_rerank_hits(query, hits), _FINAL_TOP_K, _MAX_PER_DOC)


async def doctor_search_documents(query: str, top_k: int = 8) -> str:
    # The docstring is the tool description the model sees: kept identical to
    # search_documents so the doctor prompt and tool-calling stay the same.
    """Search veterinary/agricultural documents through Beckn discovery.

    Args:
        query: Concise English keywords preserving the farmer's intent.
        top_k: Requested maximum number of results.

    Returns:
        Formatted document results from the Beckn provider.
    """
    if not settings.doctor_search_url:
        return await search_documents(query, top_k)
    normalized = _expand_veterinary_synonyms(_validate_search_query(query))
    logger.info("Doctor document search via Qdrant query=%s", normalized)
    try:
        hits = await _qdrant_search(normalized)
    except Exception as exc:
        logger.warning("Doctor document search failed query=%s: %r", normalized, exc)
        report_tool("unavailable", f"doctor search service failed: {exc!r}"[:300])
        return "The document search is temporarily unavailable. Please try again shortly."
    hits = hits[: max(1, min(top_k, 20))]
    if not hits:
        report_tool("no_result", "no documents returned")
        return "No relevant documents were found on the network for this query."
    return _format(hits)
