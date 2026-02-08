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


def _marqo_search_sync(endpoint_url: str, index_name: str, search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Synchronous Marqo search; call via asyncio.to_thread() from async code."""
    client = marqo.Client(url=endpoint_url)
    result = client.index(index_name).search(**search_params)
    return result.get("hits", [])

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
            "search_method": "hybrid",
            "hybrid_parameters": {
                "retrievalMethod": "disjunction",
                "rankingMethod": "rrf",
                "alpha": 0.5,
                "rrfK": 60,
            },
        }
        # Marqo client is sync; run in thread pool to avoid blocking the event loop
        results = await asyncio.to_thread(
            _marqo_search_sync, endpoint_url, index_name, search_params
        )

        if len(results) == 0:
            return f"No results found for `{query}`"
        else:
            # Process hits and handle missing fields
            search_hits = []
            for hit in results:
                # Map Marqo fields to our model
                processed_hit = {
                    "name": hit.get("name", ""),
                    "text": hit.get("text", ""),
                    "doc_id": hit.get("doc_id", hit.get("_id", "")),
                    "type": hit.get("type", "document"),
                    "source": hit.get("source", ""),
                    "score": hit.get("_score", hit.get("score", 0.0)),
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