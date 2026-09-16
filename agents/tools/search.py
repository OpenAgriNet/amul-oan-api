"""Beckn-backed veterinary and agricultural document search."""

import re

from pydantic_ai import ModelRetry

from agents.tools.beckn.network import network_search_documents
from helpers.utils import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[\w\-]+", re.UNICODE)
_REFUSAL_OR_META_PATTERNS = (
    "i can only answer",
    "your query appears to be",
    "would you like to ask about",
    "not within the agricultural scope",
    "out of scope",
    "not related to",
    "i cannot help",
    "i'm unable to",
    "as an ai",
    "search results for",
    "based on the provided documents",
)
_WRONG_INTENT_HINTS = ("hf receipts", "tracking numbers", "track number")


def _validate_search_query(query: str) -> str:
    """Validate the compact keyword query sent to Beckn discovery."""
    normalized = re.sub(r"\s+", " ", (query or "").strip())
    if not normalized:
        raise ModelRetry("INVALID_QUERY: EMPTY_QUERY. Provide a focused agricultural search query.")

    lowered = normalized.lower()
    if any(pattern in lowered for pattern in _REFUSAL_OR_META_PATTERNS):
        raise ModelRetry(
            "INVALID_QUERY: REFUSAL_TEXT_LEAK. Provide only concise domain keywords, "
            "never policy/refusal/meta text."
        )
    if any(pattern in lowered for pattern in _WRONG_INTENT_HINTS):
        raise ModelRetry(
            "INVALID_QUERY: OFF_TOPIC_QUERY. Regenerate query aligned to user intent "
            "and agricultural topic."
        )

    token_count = len(_TOKEN_RE.findall(lowered))
    if token_count > 20:
        raise ModelRetry(
            "INVALID_QUERY: QUERY_TOO_LONG. Use 2-12 concise keywords capturing "
            "entity/problem/task."
        )
    sentence_markers = ("?", ".", "!", " because ", " please ", " should ", " would ")
    if token_count >= 12 and any(marker in lowered for marker in sentence_markers):
        raise ModelRetry(
            "INVALID_QUERY: NARRATIVE_QUERY. Use compact keyword query, not a sentence "
            "or explanation."
        )
    return normalized


def _expand_veterinary_synonyms(query: str) -> str:
    """Add stable clinical aliases for known corpus vocabulary mismatches."""
    normalized = re.sub(r"\s+", " ", (query or "").strip())
    lowered = normalized.lower()
    additions: list[str] = []
    if "milk fever" in lowered:
        if "hypocalc" not in lowered:
            additions.append("hypocalcemia")
        if "parturient paresis" not in lowered:
            additions.append("parturient paresis")
    if "calf" in lowered and ("scour" in lowered or "diarrh" in lowered):
        if "scour" not in lowered:
            additions.append("scours")
        if "diarrh" not in lowered:
            additions.append("diarrhea")
        if "ethnoveterinary" not in lowered:
            additions.extend(
                term
                for term in ("oral rehydration", "electrolytes", "dehydration")
                if term not in lowered
            )
    return " ".join([normalized, *additions]).strip()


async def search_documents(query: str, top_k: int = 8) -> str:
    """Search veterinary/agricultural documents through Beckn discovery.

    Args:
        query: Concise English keywords preserving the farmer's intent.
        top_k: Requested maximum number of results.

    Returns:
        Formatted document results from the Beckn provider.
    """
    normalized = _expand_veterinary_synonyms(_validate_search_query(query))
    logger.info("Veterinary document search via Beckn query=%s", normalized)
    return await network_search_documents(normalized, top_k)
