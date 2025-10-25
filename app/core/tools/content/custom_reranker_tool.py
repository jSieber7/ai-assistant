"""
Custom Reranker Tool
A tool for reranking documents using a dummy keyword-based implementation
"""

import logging
from typing import Dict, Any, List, Optional
import asyncio
import numpy as np
from pydantic import BaseModel, Field

from app.core.tools.base.base import BaseTool, ToolResult
from app.core.config import settings

logger = logging.getLogger(__name__)


class RerankRequest(BaseModel):
    """Request model for reranking"""

    query: str = Field(..., description="Query to rank documents against")
    documents: List[str] = Field(..., description="Documents to rerank")
    top_n: Optional[int] = Field(None, description="Number of top results to return")
    model: str = Field("custom-reranker", description="Model to use")


class RerankResponse(BaseModel):
    """Response model for reranking results"""

    results: List[Dict[str, Any]]
    model: str
    query: str
    total_documents: int
    cached: bool = False


class CustomRerankerTool(BaseTool):
    """Tool for reranking documents using dummy keyword-based implementation"""

    name = "custom_reranker"
    description = (
        "Rerank documents based on keyword relevance to a query using dummy implementation"
    )

    def __init__(self):
        super().__init__()
        self.settings = settings
        self._model = None
        self._initialized = False

    async def _initialize_model(self):
        """Initialize the dummy reranker model"""
        # Always use dummy implementation
        self._initialized = False
        logger.info("Using dummy reranker implementation")

    async def execute(self, **kwargs) -> ToolResult:
        """Execute reranking tool"""
        try:
            # Extract parameters
            query = kwargs.get("query")
            documents = kwargs.get("documents", [])
            top_n = kwargs.get("top_n")
            model = kwargs.get("model", "custom-reranker")

            # Validate inputs
            if not query:
                return ToolResult(
                    success=False,
                    error="Query is required for reranking",
                    data=None,
                    tool_name=self.name,
                    execution_time=0.0,
                )

            if not documents or not isinstance(documents, list):
                return ToolResult(
                    success=False,
                    error="Documents must be a non-empty list",
                    data=None,
                    tool_name=self.name,
                    execution_time=0.0,
                )

            if len(documents) == 0:
                return ToolResult(
                    success=False,
                    error="Documents list cannot be empty",
                    data=None,
                    tool_name=self.name,
                    execution_time=0.0,
                )

            # Always use dummy implementation
            logger.info("Using dummy reranking implementation")
            return await self._fallback_rerank(query, documents, top_n, model)

            if response:
                return ToolResult(
                    success=True,
                    data=response,
                    tool_name=self.name,
                    execution_time=0.0,
                )
            else:
                return ToolResult(
                    success=False,
                    error="Failed to rerank documents",
                    data=None,
                    tool_name=self.name,
                    execution_time=0.0,
                )

        except Exception as e:
            logger.error(f"Error in custom reranker tool: {str(e)}")
            return ToolResult(
                success=False,
                error=f"Reranking failed: {str(e)}",
                data=None,
                tool_name=self.name,
                execution_time=0.0,
            )


    async def _fallback_rerank(
        self, query: str, documents: List[str], top_n: Optional[int], model: str
    ) -> ToolResult:
        """Fallback keyword-based reranking when model is not available"""
        try:
            # Simple keyword matching as fallback
            query_terms = set(query.lower().split())
            results = []
            
            for idx, doc in enumerate(documents):
                doc_terms = set(doc.lower().split())
                # Calculate Jaccard similarity
                intersection = len(query_terms.intersection(doc_terms))
                union = len(query_terms.union(doc_terms))
                score = intersection / union if union > 0 else 0.0
                
                results.append({
                    "index": idx,
                    "document": doc,
                    "relevance_score": score
                })
            
            # Sort by relevance score (descending)
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            # Apply top_n limit if specified
            if top_n is not None and top_n > 0:
                results = results[:top_n]
            
            response = {
                "results": results,
                "model": f"{model}-fallback",
                "query": query,
                "total_documents": len(documents),
                "cached": False
            }
            
            return ToolResult(
                success=True,
                data=response,
                tool_name=self.name,
                execution_time=0.0,
            )
            
        except Exception as e:
            logger.error(f"Error in fallback reranking: {str(e)}")
            return ToolResult(
                success=False,
                error=f"Fallback reranking failed: {str(e)}",
                data=None,
                tool_name=self.name,
                execution_time=0.0,
            )

    async def rerank_search_results(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convenience method to rerank search results

        Args:
            query: The search query
            search_results: List of search results with 'content' and 'url' fields
            top_n: Number of top results to return

        Returns:
            Reranked list of search results
        """
        try:
            # Extract content from search results
            documents = []
            for result in search_results:
                content = result.get("content", "")
                if content:
                    documents.append(content)

            if not documents:
                logger.warning("No documents found in search results")
                return search_results

            # Call reranker
            result = await self.execute(query=query, documents=documents, top_n=top_n)

            if not result.success:
                logger.error(f"Reranking failed: {result.error}")
                return search_results

            # Map reranked results back to original search results
            reranked_data = result.data
            reranked_results = []

            for item in reranked_data.get("results", []):
                original_index = item.get("index")
                if 0 <= original_index < len(search_results):
                    # Add relevance score to original result
                    original_result = search_results[original_index].copy()
                    original_result["relevance_score"] = item.get(
                        "relevance_score", 0.0
                    )
                    reranked_results.append(original_result)

            logger.info(f"Reranked {len(reranked_results)} search results")
            return reranked_results

        except Exception as e:
            logger.error(f"Error reranking search results: {str(e)}")
            return search_results

    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query to rank documents against",
                    },
                    "documents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Documents to rerank",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top results to return (optional)",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use for reranking (optional)",
                        "default": "custom-reranker",
                    },
                },
                "required": ["query", "documents"],
            },
        }