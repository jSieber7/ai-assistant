"""
Retrieval Service for RAG Pipeline

This service handles the retrieval phase:
- Retrieve Top 50 Chunks from DB → Re-rank with Jina Reranker → Top 5 Relevant Chunks
"""

import logging
import time
from typing import Dict, Any, List, Optional
from langchain_core.documents import Document

from ..storage.milvus_client import MilvusClient
from ..tools.execution.registry import ToolRegistry
from ..tools.execution.dynamic_executor import DynamicToolExecutor, TaskRequest, TaskType
from ..config import settings

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Service for handling document retrieval and reranking.

    This service retrieves relevant documents from vector storage
    and reranks them using the Jina reranker for improved relevance.
    """

    def __init__(
        self,
        vector_client: MilvusClient,
        tool_registry: ToolRegistry,
        retrieval_k: int = 50,
        rerank_top_n: int = 5,
        enable_reranking: Optional[bool] = None,
    ):
        """
        Initialize the retrieval service.

        Args:
            vector_client: Vector database client
            tool_registry: Registry of available tools
            retrieval_k: Number of documents to retrieve initially
            rerank_top_n: Number of top documents after reranking
            enable_reranking: Whether to enable reranking (defaults to settings)
        """
        self.vector_client = vector_client
        self.tool_registry = tool_registry
        self.dynamic_executor = DynamicToolExecutor(tool_registry)
        self.retrieval_k = retrieval_k
        self.rerank_top_n = rerank_top_n

        # Determine if reranking should be enabled
        if enable_reranking is None:
            # Prefer custom reranker, fall back to Ollama reranker
            self.enable_reranking = settings.custom_reranker_enabled or settings.ollama_reranker_enabled
        else:
            self.enable_reranking = enable_reranking

        # Service statistics
        self._stats = {
            "retrievals_performed": 0,
            "documents_retrieved": 0,
            "rerankings_performed": 0,
            "reranking_failures": 0,
            "total_retrieval_time": 0.0,
            "total_reranking_time": 0.0,
        }

    async def retrieve_and_rerank(
        self,
        query: str,
        collection_name: str,
        k: Optional[int] = None,
        top_n: Optional[int] = None,
        search_params: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Retrieve and rerank documents for a given query.

        Args:
            query: Search query
            collection_name: Name of the collection to search
            k: Number of documents to retrieve (defaults to self.retrieval_k)
            top_n: Number of top documents after reranking (defaults to self.rerank_top_n)
            search_params: Additional search parameters

        Returns:
            List of reranked documents
        """
        start_time = time.time()

        try:
            # Use defaults if not provided
            k = k or self.retrieval_k
            top_n = top_n or self.rerank_top_n

            # Step 1: Retrieve documents from vector database
            retrieved_docs = await self._retrieve_documents(
                query, collection_name, k, search_params
            )

            if not retrieved_docs:
                logger.warning(f"No documents retrieved for query: {query}")
                return []

            logger.info(f"Retrieved {len(retrieved_docs)} documents for query: {query}")

            # Step 2: Rerank documents if enabled
            if self.enable_reranking and len(retrieved_docs) > 1:
                reranked_docs = await self._rerank_documents(
                    query, retrieved_docs, top_n
                )
            else:
                # Just take the top_n documents without reranking
                reranked_docs = retrieved_docs[:top_n]
                logger.info("Reranking disabled, using top documents by similarity")

            # Update statistics
            retrieval_time = time.time() - start_time
            self._update_stats(len(retrieved_docs), retrieval_time)

            logger.info(
                f"Retrieval and reranking completed in {retrieval_time:.2f}s. "
                f"Final result: {len(reranked_docs)} documents"
            )

            return reranked_docs

        except Exception as e:
            retrieval_time = time.time() - start_time
            error_msg = f"Document retrieval and reranking failed: {str(e)}"
            logger.error(error_msg)
            return []

    async def _retrieve_documents(
        self,
        query: str,
        collection_name: str,
        k: int,
        search_params: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Retrieve documents from vector database.

        Args:
            query: Search query
            collection_name: Name of the collection to search
            k: Number of documents to retrieve
            search_params: Additional search parameters

        Returns:
            List of retrieved documents with scores
        """
        try:
            # Default search parameters
            if search_params is None:
                search_params = {
                    "metric_type": settings.milvus_settings.metric_type,
                    "params": {"ef": 64}
                    if settings.milvus_settings.index_type == "HNSW"
                    else {},
                }

            # Perform similarity search
            retrieved_docs_with_scores = await self.vector_client.similarity_search(
                collection_name=collection_name,
                query=query,
                k=k,
                search_params=search_params,
            )

            # Extract documents and add scores to metadata
            retrieved_docs = []
            for doc, score in retrieved_docs_with_scores:
                doc.metadata["similarity_score"] = score
                doc.metadata["retrieval_method"] = "similarity_search"
                doc.metadata["retrieval_query"] = query
                doc.metadata["retrieved_at"] = time.time()
                retrieved_docs.append(doc)

            logger.debug(
                f"Retrieved {len(retrieved_docs)} documents from collection {collection_name}"
            )
            return retrieved_docs

        except Exception as e:
            logger.error(f"Error retrieving documents from {collection_name}: {str(e)}")
            return []

    async def _rerank_documents(
        self, query: str, documents: List[Document], top_n: int
    ) -> List[Document]:
        """
        Rerank documents using Jina reranker.

        Args:
            query: Search query
            documents: List of documents to rerank
            top_n: Number of top documents to return

        Returns:
            List of reranked documents
        """
        start_time = time.time()

        try:
            # Extract content for reranking
            doc_contents = [doc.page_content for doc in documents]

            # Determine which reranker to use
            if settings.custom_reranker_enabled:
                required_tool = "custom_reranker"
                model = settings.custom_reranker_model
            elif settings.ollama_reranker_enabled:
                required_tool = "ollama_reranker"
                model = getattr(settings, 'ollama_reranker_model', 'nomic-embed-text')
            else:
                # No reranker available, disable reranking
                logger.warning("No reranker is enabled, skipping reranking")
                return documents[:top_n]
            
            # Create rerank task request
            rerank_request = TaskRequest(
                task_type=TaskType.RERANK,
                query=query,
                context={"documents": doc_contents},
                required_tools=[required_tool],
                parameters={
                    "documents": doc_contents,
                    "top_n": top_n,
                    "model": model,
                },
                max_tools=1,
            )

            # Execute reranking
            rerank_result = await self.dynamic_executor.execute_task(rerank_request)

            if not rerank_result.success:
                logger.error(f"Reranking task failed: {rerank_result.error}")
                self._stats["reranking_failures"] += 1
                # Fallback to original order
                return documents[:top_n]

            # Process reranking results
            reranked_docs = await self._process_reranking_results(
                rerank_result.data, documents, top_n
            )

            # Update statistics
            reranking_time = time.time() - start_time
            self._stats["rerankings_performed"] += 1
            self._stats["total_reranking_time"] += reranking_time

            logger.info(
                f"Reranking completed in {reranking_time:.2f}s. "
                f"Returned {len(reranked_docs)} documents"
            )

            return reranked_docs

        except Exception as e:
            reranking_time = time.time() - start_time
            logger.error(f"Error during reranking: {str(e)}")
            self._stats["reranking_failures"] += 1
            self._stats["total_reranking_time"] += reranking_time

            # Fallback to original order
            return documents[:top_n]

    async def _process_reranking_results(
        self,
        rerank_data: Dict[str, Any],
        original_documents: List[Document],
        top_n: int,
    ) -> List[Document]:
        """
        Process reranking results and reorder documents.

        Args:
            rerank_data: Reranking results from Jina
            original_documents: Original list of documents
            top_n: Number of top documents to return

        Returns:
            List of reranked documents
        """
        try:
            if not isinstance(rerank_data, dict) or "results" not in rerank_data:
                logger.warning("Invalid reranking data format")
                return original_documents[:top_n]

            reranked_docs = []
            rerank_results = rerank_data.get("results", [])

            for item in rerank_results:
                original_index = item.get("index")
                relevance_score = item.get("relevance_score", 0.0)

                if 0 <= original_index < len(original_documents):
                    # Get the original document
                    doc = original_documents[original_index].copy()

                    # Add reranking metadata
                    doc.metadata["rerank_score"] = relevance_score
                    doc.metadata["rerank_index"] = len(reranked_docs)
                    doc.metadata["reranked_at"] = time.time()
                    if settings.custom_reranker_enabled:
                        reranking_method = "custom_reranker"
                    elif settings.ollama_reranker_enabled:
                        reranking_method = "ollama_reranker"
                    else:
                        reranking_method = "unknown"
                    
                    doc.metadata["reranking_method"] = reranking_method

                    reranked_docs.append(doc)

                    # Stop if we have enough documents
                    if len(reranked_docs) >= top_n:
                        break

            logger.info(f"Successfully reranked {len(reranked_docs)} documents")
            return reranked_docs

        except Exception as e:
            logger.error(f"Error processing reranking results: {str(e)}")
            return original_documents[:top_n]

    async def retrieve_only(
        self,
        query: str,
        collection_name: str,
        k: Optional[int] = None,
        search_params: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Retrieve documents without reranking.

        Args:
            query: Search query
            collection_name: Name of the collection to search
            k: Number of documents to retrieve
            search_params: Additional search parameters

        Returns:
            List of retrieved documents
        """
        k = k or self.retrieval_k
        return await self._retrieve_documents(query, collection_name, k, search_params)

    async def rerank_only(
        self, query: str, documents: List[Document], top_n: Optional[int] = None
    ) -> List[Document]:
        """
        Rerank existing documents without retrieval.

        Args:
            query: Search query
            documents: List of documents to rerank
            top_n: Number of top documents to return

        Returns:
            List of reranked documents
        """
        top_n = top_n or self.rerank_top_n

        if not self.enable_reranking:
            logger.warning("Reranking is disabled")
            return documents[:top_n]

        return await self._rerank_documents(query, documents, top_n)

    def _update_stats(self, retrieved_count: int, retrieval_time: float):
        """Update service statistics."""
        self._stats["retrievals_performed"] += 1
        self._stats["documents_retrieved"] += retrieved_count
        self._stats["total_retrieval_time"] += retrieval_time

    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "service_name": "RetrievalService",
            "retrievals_performed": self._stats["retrievals_performed"],
            "documents_retrieved": self._stats["documents_retrieved"],
            "rerankings_performed": self._stats["rerankings_performed"],
            "reranking_failures": self._stats["reranking_failures"],
            "total_retrieval_time": self._stats["total_retrieval_time"],
            "total_reranking_time": self._stats["total_reranking_time"],
            "avg_retrieval_time": (
                self._stats["total_retrieval_time"]
                / max(self._stats["retrievals_performed"], 1)
            ),
            "avg_reranking_time": (
                self._stats["total_reranking_time"]
                / max(self._stats["rerankings_performed"], 1)
            ),
            "reranking_success_rate": (
                (
                    self._stats["rerankings_performed"]
                    - self._stats["reranking_failures"]
                )
                / max(self._stats["rerankings_performed"], 1)
            ),
            "config": {
                "retrieval_k": self.retrieval_k,
                "rerank_top_n": self.rerank_top_n,
                "enable_reranking": self.enable_reranking,
                "custom_reranker_enabled": settings.custom_reranker_enabled,
                "ollama_reranker_enabled": settings.ollama_reranker_enabled,
            },
        }

    def reset_stats(self):
        """Reset service statistics."""
        self._stats = {
            "retrievals_performed": 0,
            "documents_retrieved": 0,
            "rerankings_performed": 0,
            "reranking_failures": 0,
            "total_retrieval_time": 0.0,
            "total_reranking_time": 0.0,
        }

    async def validate_collection(self, collection_name: str) -> Dict[str, Any]:
        """
        Validate that a collection exists and is accessible.

        Args:
            collection_name: Name of the collection to validate

        Returns:
            Validation results
        """
        try:
            if not self.vector_client:
                return {"valid": False, "error": "No vector client configured"}

            # Try to get collection stats
            stats = await self.vector_client.get_collection_stats(collection_name)

            return {"valid": True, "stats": stats, "collection_name": collection_name}

        except Exception as e:
            return {"valid": False, "error": str(e), "collection_name": collection_name}
