"""
Jina AI Reranker Tool
A tool for reranking documents using Jina AI's reranker service
"""

import logging
from typing import Dict, Any, List, Optional
import httpx
from pydantic import BaseModel, Field

from app.core.tools.base import BaseTool, ToolResult
from app.core.config import settings

logger = logging.getLogger(__name__)


class RerankRequest(BaseModel):
    """Request model for reranking"""

    query: str = Field(..., description="Query to rank documents against")
    documents: List[str] = Field(..., description="Documents to rerank")
    top_n: Optional[int] = Field(None, description="Number of top results to return")
    model: str = Field("jina-reranker-v2-base-multilingual", description="Model to use")


class RerankResponse(BaseModel):
    """Response model for reranking results"""

    results: List[Dict[str, Any]]
    model: str
    query: str
    total_documents: int
    cached: bool = False


class JinaRerankerTool(BaseTool):
    """Tool for reranking documents using Jina AI"""

    name = "jina_reranker"
    description = (
        "Rerank documents based on semantic relevance to a query using Jina AI"
    )

    def __init__(self):
        super().__init__()
        self.settings = settings
        self.http_client: Optional[httpx.AsyncClient] = None

    async def _initialize_client(self):
        """Initialize HTTP client if not already done"""
        if not self.http_client:
            self.http_client = httpx.AsyncClient(
                timeout=self.settings.jina_reranker_timeout,
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=50),
            )

    async def _cleanup_client(self):
        """Cleanup HTTP client"""
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None

    async def execute(self, **kwargs) -> ToolResult:
        """Execute the reranking tool"""
        try:
            # Extract parameters
            query = kwargs.get("query")
            documents = kwargs.get("documents", [])
            top_n = kwargs.get("top_n")
            model = kwargs.get("model", self.settings.jina_reranker_model)

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

            # Check if reranker is enabled
            if not self.settings.jina_reranker_enabled:
                return ToolResult(
                    success=False,
                    error="Jina reranker is not enabled",
                    data=None,
                    tool_name=self.name,
                    execution_time=0.0,
                )

            # Initialize client
            await self._initialize_client()

            # Prepare request
            request_data = {"query": query, "documents": documents, "model": model}

            if top_n is not None:
                request_data["top_n"] = top_n

            # Call reranker service
            response = await self._call_reranker_service(request_data)

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
                    error="Failed to get response from reranker service",
                    data=None,
                    tool_name=self.name,
                    execution_time=0.0,
                )

        except Exception as e:
            logger.error(f"Error in Jina reranker tool: {str(e)}")
            return ToolResult(
                success=False,
                error=f"Reranking failed: {str(e)}",
                data=None,
                tool_name=self.name,
                execution_time=0.0,
            )
        finally:
            await self._cleanup_client()

    async def _call_reranker_service(
        self, request_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Call the Jina reranker service"""
        try:
            url = f"{self.settings.jina_reranker_url}/rerank"

            logger.info(f"Calling Jina reranker service at {url}")
            logger.debug(f"Request data: {request_data}")

            response = await self.http_client.post(
                url, json=request_data, headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                # Handle both sync and async response.json()
                try:
                    result = await response.json()
                except TypeError:
                    result = response.json()
                logger.info(
                    f"Successfully reranked {len(request_data['documents'])} documents"
                )
                return result
            else:
                logger.error(
                    f"Jina reranker service returned status {response.status_code}: {response.text}"
                )
                return None

        except httpx.RequestError as e:
            logger.error(f"Error calling Jina reranker service: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error calling Jina reranker service: {str(e)}")
            return None

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
        """Get the tool schema"""
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
                        "default": "jina-reranker-v2-base-multilingual",
                    },
                },
                "required": ["query", "documents"],
            },
        }
