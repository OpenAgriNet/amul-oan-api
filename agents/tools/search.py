"""
Marqo client implementation for vector search.
"""
import os
import re
import marqo
from typing import Optional, Literal
from pydantic import BaseModel, Field
from pydantic_ai import ModelRetry, RunContext
from helpers.utils import get_logger
from agents.deps import FarmerContext
# NOTE: This is a hack to add Gujarati terms to the search results.
from agents.tools.terms import normalize_text_with_glossary

logger = get_logger(__name__)

DocumentType = Literal['video', 'document']

class SearchHit(BaseModel):
    """Individual search hit from elasticsearch"""
    name_en: str = ""
    name_gu: str = ""
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

    def get_string(self, lang_code: str) -> str:
        # All results are documents in this index
        if lang_code == 'gu':
            return f"**{self.name_gu}**\n" + "```\n" + self.processed_text +  "\n```\n"
        else:
            return f"**{self.name_en}**\n" + "```\n" + self.processed_text +  "\n```\n"


async def search_documents(
    ctx: RunContext,
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
        # Initialize Marqo client
        endpoint_url = os.getenv('MARQO_ENDPOINT_URL')
        if not endpoint_url:
            raise ValueError("Marqo endpoint URL is required")
        
        index_name = os.getenv('MARQO_INDEX_NAME', 'sunbird-va-index')
        if not index_name:
            raise ValueError("Marqo index name is required")
        
        client = marqo.Client(url=endpoint_url)
        logger.info(f"Searching for '{query}' in index '{index_name}'")
            
        # Perform search
        # Note: No filter_string needed - the index contains only documents
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
        
        results = client.index(index_name).search(**search_params)['hits']
        
        if len(results) == 0:
            return f"No results found for `{query}`"
        else:
            # Process hits and handle missing fields
            search_hits = []
            for hit in results:
                # Map Marqo fields to our model
                processed_hit = {
                    "text": hit.get("text", ""),
                    "doc_id": hit.get("doc_id", hit.get("_id", "")),
                    "type": hit.get("type", "document"),
                    "score": hit.get("_score", hit.get("score", 0.0)),
                    "source": hit.get("source", ""),
                    "id": hit.get("_id", hit.get("id", "")),
                    "name_en": hit.get("name_en", ""),
                    "name_gu": hit.get("name_gu", ""),
                }
                search_hits.append(SearchHit(**processed_hit))            
            # Convert back to dict format for compatibility
            document_string = '\n\n----\n\n'.join([document.get_string(ctx.deps.lang_code) for document in search_hits])
            return "> Search Results for `" + query + "`\n\n" + document_string
    except Exception as e:
        logger.error(f"Error searching documents: {e} for query: {query}")
        raise ModelRetry(f"Error searching documents, please try again")