"""
Marqo client implementation for vector search.
The Marqo Python client is synchronous; we run it in asyncio.to_thread() to avoid
blocking the event loop when serving many concurrent requests.
"""
import asyncio
import os
import re
import marqo
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from pydantic_ai import ModelRetry
from helpers.utils import get_logger
# NOTE: This is a hack to add Gujarati terms to the search results.
from agents.tools.terms import normalize_text_with_glossary

logger = get_logger(__name__)
_index_capabilities_cache: Dict[str, Dict[str, Any]] = {}
_TOKEN_RE = re.compile(r"[\w\-]+", re.UNICODE)


def _validate_search_query(query: str) -> str:
    """Normalize query before retrieval.

    Raises:
        ModelRetry: only when query is empty.
    """
    normalized = re.sub(r"\s+", " ", (query or "").strip())

    if not normalized:
        logger.warning("Search query validation failed: empty query")
        raise ModelRetry("INVALID_QUERY: EMPTY_QUERY. Provide a focused agricultural search query.")

    logger.info("Search query validation passed: query=%s", normalized)
    return normalized


def _marqo_search_sync(endpoint_url: str, index_name: str, search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Synchronous Marqo search; call via asyncio.to_thread() from async code."""
    client = marqo.Client(url=endpoint_url)
    result = client.index(index_name).search(**search_params)
    return result.get("hits", [])


def _get_index_capabilities_sync(endpoint_url: str, index_name: str) -> Dict[str, Any]:
    """
    Fetch and cache index capabilities once per endpoint/index pair.
    Useful for startup-like compatibility checks across index schema changes.
    """
    cache_key = f"{endpoint_url}::{index_name}"
    cached = _index_capabilities_cache.get(cache_key)
    if cached is not None:
        return cached

    client = marqo.Client(url=endpoint_url)
    try:
        index_info = client.get_index(index_name)
        tensor_fields = set(index_info.get("tensorFields", []) if isinstance(index_info, dict) else [])
        all_fields = index_info.get("allFields", []) if isinstance(index_info, dict) else []
        field_names = {f.get("name") for f in all_fields if isinstance(f, dict) and f.get("name")}
        capabilities = {
            "exists": True,
            "tensor_fields": sorted(tensor_fields),
            "has_text_tensor": "text" in tensor_fields,
            "has_text_for_embedding_tensor": "text_for_embedding" in tensor_fields,
            "has_is_reference_filter": "is_reference" in field_names,
            "field_names": sorted(field_names),
        }
    except Exception as e:
        capabilities = {
            "exists": False,
            "error": str(e),
            "tensor_fields": [],
            "has_text_tensor": False,
            "has_text_for_embedding_tensor": False,
            "has_is_reference_filter": False,
            "field_names": [],
        }

    _index_capabilities_cache[cache_key] = capabilities
    return capabilities


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _prepare_query_for_e5(query: str) -> str:
    cleaned = query.strip()
    if cleaned.lower().startswith("query:"):
        return cleaned
    return f"query: {cleaned}"


def _doc_key(hit: Dict[str, Any]) -> str:
    return (
        str(hit.get("doc_id") or "").strip()
        or str(hit.get("filename") or "").strip()
        or str(hit.get("name_en") or "").strip()
        or str(hit.get("name") or "").strip()
        or str(hit.get("_id") or "").strip()
    )


def _apply_doc_diversity(hits: List[Dict[str, Any]], top_k: int, max_per_doc: int) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    per_doc_counts: Dict[str, int] = {}

    for hit in hits:
        key = _doc_key(hit)
        count = per_doc_counts.get(key, 0)
        if count >= max_per_doc:
            continue
        per_doc_counts[key] = count + 1
        selected.append(hit)
        if len(selected) >= top_k:
            break

    if len(selected) < top_k:
        for hit in hits:
            if hit in selected:
                continue
            selected.append(hit)
            if len(selected) >= top_k:
                break
    return selected


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _tokenize(value: str) -> List[str]:
    return _TOKEN_RE.findall(_normalize_text(value))


def _token_overlap_score(query: str, text: str) -> float:
    q_tokens = set(_tokenize(query))
    t_tokens = set(_tokenize(text))
    if not q_tokens or not t_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / len(q_tokens)


def _metadata_blob(hit: Dict[str, Any]) -> str:
    return " ".join(
        str(hit.get(k) or "")
        for k in (
            "name",
            "name_en",
            "name_gu",
            "filename",
            "title_en",
            "title_gu",
            "category_tags",
            "description",
            "doc_short_description",
            "doc_llm_description",
        )
    )


def _rerank_hits(query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not hits:
        return hits

    raw_scores = [float(h.get("_score", h.get("score", 0.0)) or 0.0) for h in hits]
    min_score = min(raw_scores)
    max_score = max(raw_scores)
    denom = (max_score - min_score) if max_score > min_score else 1.0

    rescored: List[Dict[str, Any]] = []
    for hit, raw in zip(hits, raw_scores):
        semantic = (raw - min_score) / denom
        text = str(hit.get("text") or "")
        metadata_text = _metadata_blob(hit)
        lexical_text = _token_overlap_score(query, text)
        lexical_meta = _token_overlap_score(query, metadata_text)
        lexical = max(lexical_text, lexical_meta)

        metadata_boost = 0.08 * lexical_meta
        reference_penalty = -0.12 if bool(hit.get("is_reference", False)) else 0.0
        rerank_score = (0.62 * semantic) + (0.30 * lexical) + metadata_boost + reference_penalty

        enriched = dict(hit)
        enriched["_rerank_score"] = rerank_score
        rescored.append(enriched)

    rescored.sort(key=lambda x: float(x.get("_rerank_score", 0.0)), reverse=True)
    return rescored

DocumentType = Literal['video', 'document']

class SearchHit(BaseModel):
    """Individual search hit from elasticsearch"""
    name: str = ""
    text: str = ""
    doc_id: str = ""
    type: str = "document"  # Default to document since index only contains documents
    source: str = ""  # Make optional since it might not be in all results
    score: float = Field(default=0.0)
    id: str = Field(default="")
    
    class Config:
        # Allow extra fields from Marqo that we don't need
        extra = "ignore"
        # Handle both _score and score fields
        populate_by_name = True

    @property
    def processed_text(self) -> str:
        """Returns the text with cleaned up whitespace and newlines"""
        # Replace multiple newlines with a single line
        cleaned = re.sub(r'\n{2,}', '\n\n', self.text)
        cleaned = re.sub(r'\t+', '\t', cleaned)
        # NOTE: This is a hack to add Gujarati terms to the search results.
        cleaned = normalize_text_with_glossary(cleaned)
        return cleaned

    def __str__(self) -> str:
        # All results are documents in this index
        return f"**{self.name}**\n" + "```\n" + self.processed_text +  "\n```\n"


async def search_documents(
    query: str, 
    top_k: int = 10, 
) -> str:
    """
    Semantic search for documents. Use this tool to search for relevant documents.
    
    Args:
        query: The search query in *English* (required)
        top_k: Maximum number of results to return (default: 10)
        
    Returns:
        search_results: Formatted list of documents
    """
    try:
        query = _validate_search_query(query)
        endpoint_url = os.getenv('MARQO_ENDPOINT_URL')
        if not endpoint_url:
            raise ValueError("Marqo endpoint URL is required")
        index_name = os.getenv('MARQO_INDEX_NAME', 'sunbird-va-index')
        if not index_name:
            raise ValueError("Marqo index name is required")

        capabilities = await asyncio.to_thread(_get_index_capabilities_sync, endpoint_url, index_name)
        if capabilities.get("exists"):
            logger.info(
                "Index capabilities: tensor_fields=%s, text_tensor=%s, text_for_embedding_tensor=%s, has_is_reference=%s",
                capabilities.get("tensor_fields", []),
                capabilities.get("has_text_tensor"),
                capabilities.get("has_text_for_embedding_tensor"),
                capabilities.get("has_is_reference_filter"),
            )
        else:
            logger.warning("Could not inspect index '%s': %s", index_name, capabilities.get("error"))

        logger.info(f"Searching for '{query}' in index '{index_name}'")

        use_e5_query_prefix = _env_bool("MARQO_USE_E5_QUERY_PREFIX", True)
        exclude_reference_chunks = _env_bool("MARQO_EXCLUDE_REFERENCE", True)
        max_per_doc = int(os.getenv("MARQO_MAX_CHUNKS_PER_DOC", "2"))
        candidate_multiplier = int(os.getenv("MARQO_CANDIDATE_MULTIPLIER", "10"))
        candidate_cap = int(os.getenv("MARQO_CANDIDATE_CAP", "120"))
        search_limit = min(max(top_k * max(candidate_multiplier, 1), top_k), max(candidate_cap, top_k))
        effective_query = _prepare_query_for_e5(query) if use_e5_query_prefix else query

        search_mode = (os.getenv("MARQO_SEARCH_MODE", "hybrid") or "hybrid").strip().lower()
        search_params: Dict[str, Any] = {
            "q": effective_query,
            "limit": search_limit,
        }
        if search_mode == "hybrid":
            search_params["search_method"] = "hybrid"
            search_params["hybrid_parameters"] = {
                "retrievalMethod": "disjunction",
                "rankingMethod": "rrf",
                "alpha": 0.5,
                "rrfK": 60,
            }
        elif search_mode == "tensor":
            search_params["search_method"] = "tensor"
        elif search_mode == "lexical":
            search_params["search_method"] = "lexical"
        else:
            raise ValueError(f"Unsupported MARQO_SEARCH_MODE={search_mode}")

        if exclude_reference_chunks and capabilities.get("has_is_reference_filter", False):
            search_params["filter_string"] = "is_reference:false"

        # Marqo client is sync; run in thread pool to avoid blocking the event loop
        try:
            results = await asyncio.to_thread(
                _marqo_search_sync, endpoint_url, index_name, search_params
            )
        except Exception:
            if search_mode == "hybrid":
                logger.warning("Hybrid search failed, retrying with tensor search for query '%s'", query)
                fallback_params = {
                    "q": effective_query,
                    "limit": search_limit,
                    "search_method": "tensor",
                }
                if exclude_reference_chunks and capabilities.get("has_is_reference_filter", False):
                    fallback_params["filter_string"] = "is_reference:false"
                results = await asyncio.to_thread(
                    _marqo_search_sync, endpoint_url, index_name, fallback_params
                )
            else:
                raise

        results = _rerank_hits(query, results)
        results = _apply_doc_diversity(results, top_k=top_k, max_per_doc=max_per_doc)

        logger.info("Search completed: query=%s hits=%s", query, len(results))

        if len(results) == 0:
            return f"No results found for `{query}`"
        else:
            # Process hits and handle missing fields
            search_hits = []
            for hit in results:
                # Map Marqo fields to our model
                processed_hit = {
                    "name": hit.get("name") or hit.get("name_en") or hit.get("name_gu") or hit.get("filename", ""),
                    "text": hit.get("text", ""),
                    "doc_id": hit.get("doc_id", hit.get("_id", "")),
                    "type": hit.get("type", "document"),
                    "source": hit.get("source", ""),
                    "score": hit.get("_rerank_score", hit.get("_score", hit.get("score", 0.0))),
                    "id": hit.get("_id", hit.get("id", ""))
                }
                search_hits.append(SearchHit(**processed_hit))            
            # Convert back to dict format for compatibility
            document_string = '\n\n----\n\n'.join([str(document) for document in search_hits])
            return "> Search Results for `" + query + "`\n\n" + document_string
    except Exception as e:
        logger.error(f"Error searching documents: {e} for query: {query}")
        raise ModelRetry(f"Error searching documents, please try again")


async def search_videos(
    query: str, 
    top_k: int = 3, 
) -> str:
    """
    Semantic search for videos. Use this tool when recommending videos to the farmer.
    
    Args:
        query: The search query in *English* (required)
        top_k: Maximum number of results to return (default: 3)
        
    Returns:
        search_results: Formatted list of videos
    """
    try:
        endpoint_url = os.getenv('MARQO_ENDPOINT_URL')
        if not endpoint_url:
            raise ValueError("Marqo endpoint URL is required")
        index_name = os.getenv('MARQO_INDEX_NAME', 'sunbird-va-index')
        if not index_name:
            raise ValueError("Marqo index name is required")

        logger.info(f"Searching for '{query}' in index '{index_name}'")
        search_params = {
            "q": query,
            "limit": top_k,
            "search_method": "tensor",
        }
        # Marqo client is sync; run in thread pool to avoid blocking the event loop
        results = await asyncio.to_thread(
            _marqo_search_sync, endpoint_url, index_name, search_params
        )

        if len(results) == 0:
            return f"No videos found for `{query}`"
        else:            
            search_hits = [SearchHit(**hit) for hit in results]            
            video_string = '\n\n----\n\n'.join([str(document) for document in search_hits])
            return "> Videos for `" + query + "`\n\n" + video_string
        
    except Exception as e:
        logger.error(f"Error searching documents: {e} for query: {query}")
        raise ModelRetry(f"Error searching documents, please try again")
