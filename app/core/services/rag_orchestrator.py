"""
RAG Orchestrator for Complete Pipeline

This service coordinates all services for the complete RAG pipeline:
- User Query → Agent LLM → Search Query Generation → SearxNG Tool →
- List of URLs → Firecrawl/Playwright Tool → Scraped Markdown Pages →
- Chunk Documents → Embed Chunks → Vector DB → Retrieve Top 50 Chunks →
- Re-rank with Jina Reranker → Top 5 Relevant Chunks → Final LLM Prompt →
- Final Synthesized Answer
"""

import logging
import time
import uuid
from typing import Dict, Any, List, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from .query_processing import QueryProcessingService
from .search import SearchService
from .ingestion import IngestionService
from .retrieval import RetrievalService
from .synthesis import SynthesisService
from ..tools.execution.registry import ToolRegistry
from ..storage.milvus_client import MilvusClient

logger = logging.getLogger(__name__)


class RAGOrchestrator:
    """
    Orchestrator for the complete RAG pipeline.

    This service coordinates all the individual services to provide
    a complete RAG (Retrieval-Augmented Generation) solution.
    """

    def __init__(
        self,
        query_service: QueryProcessingService,
        search_service: SearchService,
        ingestion_service: IngestionService,
        retrieval_service: RetrievalService,
        synthesis_service: SynthesisService,
        auto_cleanup: bool = True,
    ):
        """
        Initialize the RAG orchestrator.

        Args:
            query_service: Query processing service
            search_service: Search and scraping service
            ingestion_service: Document ingestion service
            retrieval_service: Document retrieval service
            synthesis_service: Answer synthesis service
            auto_cleanup: Whether to automatically cleanup temporary collections
        """
        self.query_service = query_service
        self.search_service = search_service
        self.ingestion_service = ingestion_service
        self.retrieval_service = retrieval_service
        self.synthesis_service = synthesis_service
        self.auto_cleanup = auto_cleanup

        # Service statistics
        self._stats = {
            "total_queries_processed": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_execution_time": 0.0,
            "avg_execution_time": 0.0,
            "temporary_collections_created": 0,
            "temporary_collections_cleaned": 0,
        }

        # Track active collections for cleanup
        self._active_collections: set[str] = set()

    async def process_query(
        self,
        user_query: str,
        context: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the complete RAG pipeline.

        Args:
            user_query: Original user query
            context: Additional context for processing
            options: Processing options

        Returns:
            Dictionary with complete pipeline results
        """
        start_time = time.time()
        session_id = str(uuid.uuid4())

        try:
            logger.info(
                f"Starting RAG pipeline for query: {user_query} (session: {session_id})"
            )

            # Initialize result dictionary
            result = {
                "query": user_query,
                "session_id": session_id,
                "success": False,
                "answer": "",
                "sources": [],
                "metadata": {},
                "pipeline_steps": {},
                "execution_time": 0.0,
            }

            # Step 1: Query Processing
            optimized_query = await self._step_query_processing(
                user_query, context, result["pipeline_steps"]
            )

            # Step 2: Search and Scraping
            scraped_documents = await self._step_search_and_scraping(
                optimized_query, context, result["pipeline_steps"], options
            )

            if not scraped_documents:
                result["error"] = "No relevant information found during search"
                result["answer"] = (
                    "I couldn't find relevant information for your query. Please try rephrasing or providing more specific terms."
                )
                return await self._finalize_result(result, start_time)

            # Step 3: Ingestion
            collection_name = await self._step_ingestion(
                scraped_documents, session_id, result["pipeline_steps"]
            )

            # Step 4: Retrieval and Reranking
            retrieved_documents = await self._step_retrieval(
                optimized_query, collection_name, result["pipeline_steps"], options
            )

            # Step 5: Synthesis
            synthesis_result = await self._step_synthesis(
                user_query, retrieved_documents, context, result["pipeline_steps"]
            )

            # Compile final result
            result.update(synthesis_result)
            result["success"] = True
            result["metadata"]["collection_name"] = collection_name

            # Cleanup if enabled
            if self.auto_cleanup:
                await self._cleanup_collection(collection_name)

            return await self._finalize_result(result, start_time)

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"RAG pipeline failed: {str(e)}"
            logger.error(error_msg)

            return {
                "query": user_query,
                "session_id": session_id,
                "success": False,
                "answer": "I encountered an error while processing your request. Please try again.",
                "error": error_msg,
                "execution_time": execution_time,
                "metadata": {"error_type": type(e).__name__},
            }

    async def _step_query_processing(
        self,
        user_query: str,
        context: Optional[Dict[str, Any]],
        pipeline_steps: Dict[str, Any],
    ) -> str:
        """Step 1: Query Processing"""
        step_start = time.time()

        try:
            optimized_query = await self.query_service.generate_search_query(
                user_query, context
            )

            pipeline_steps["query_processing"] = {
                "success": True,
                "original_query": user_query,
                "optimized_query": optimized_query,
                "execution_time": time.time() - step_start,
            }

            logger.info(
                f"Query optimization completed: '{user_query}' → '{optimized_query}'"
            )
            return optimized_query

        except Exception as e:
            pipeline_steps["query_processing"] = {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - step_start,
            }
            logger.error(f"Query processing failed: {str(e)}")
            return user_query  # Fallback to original query

    async def _step_search_and_scraping(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
        pipeline_steps: Dict[str, Any],
        options: Optional[Dict[str, Any]],
    ) -> List[Document]:
        """Step 2: Search and Scraping"""
        step_start = time.time()

        try:
            # Extract search options
            search_params = options.get("search_params", {}) if options else {}

            scraped_documents = await self.search_service.search_and_scrape(
                query, context, search_params
            )

            pipeline_steps["search_and_scraping"] = {
                "success": True,
                "query": query,
                "documents_found": len(scraped_documents),
                "execution_time": time.time() - step_start,
                "search_stats": self.search_service.get_service_stats(),
            }

            logger.info(
                f"Search and scraping completed: {len(scraped_documents)} documents"
            )
            return scraped_documents

        except Exception as e:
            pipeline_steps["search_and_scraping"] = {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - step_start,
            }
            logger.error(f"Search and scraping failed: {str(e)}")
            return []

    async def _step_ingestion(
        self, documents: List[Document], session_id: str, pipeline_steps: Dict[str, Any]
    ) -> str:
        """Step 3: Ingestion"""
        step_start = time.time()

        try:
            # Create temporary collection
            collection_name = await self.ingestion_service.create_temporary_collection(
                session_id
            )
            self._active_collections.add(collection_name)
            self._stats["temporary_collections_created"] += 1

            # Ingest documents
            ingestion_result = await self.ingestion_service.ingest_documents(
                documents, collection_name
            )

            pipeline_steps["ingestion"] = {
                "success": ingestion_result["success"],
                "collection_name": collection_name,
                "documents_processed": ingestion_result["documents_processed"],
                "chunks_created": ingestion_result["chunks_created"],
                "chunks_ingested": ingestion_result["chunks_ingested"],
                "execution_time": time.time() - step_start,
                "ingestion_stats": self.ingestion_service.get_service_stats(),
            }

            logger.info(
                f"Ingestion completed: {ingestion_result['chunks_ingested']} chunks in {collection_name}"
            )
            return collection_name

        except Exception as e:
            pipeline_steps["ingestion"] = {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - step_start,
            }
            logger.error(f"Ingestion failed: {str(e)}")
            raise

    async def _step_retrieval(
        self,
        query: str,
        collection_name: str,
        pipeline_steps: Dict[str, Any],
        options: Optional[Dict[str, Any]],
    ) -> List[Document]:
        """Step 4: Retrieval and Reranking"""
        step_start = time.time()

        try:
            # Extract retrieval options
            retrieval_k = options.get("retrieval_k", 50) if options else 50
            rerank_top_n = options.get("rerank_top_n", 5) if options else 5

            retrieved_documents = await self.retrieval_service.retrieve_and_rerank(
                query, collection_name, retrieval_k, rerank_top_n
            )

            pipeline_steps["retrieval"] = {
                "success": True,
                "query": query,
                "documents_retrieved": len(retrieved_documents),
                "retrieval_k": retrieval_k,
                "rerank_top_n": rerank_top_n,
                "execution_time": time.time() - step_start,
                "retrieval_stats": self.retrieval_service.get_service_stats(),
            }

            logger.info(f"Retrieval completed: {len(retrieved_documents)} documents")
            return retrieved_documents

        except Exception as e:
            pipeline_steps["retrieval"] = {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - step_start,
            }
            logger.error(f"Retrieval failed: {str(e)}")
            return []

    async def _step_synthesis(
        self,
        original_query: str,
        documents: List[Document],
        context: Optional[Dict[str, Any]],
        pipeline_steps: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Step 5: Synthesis"""
        step_start = time.time()

        try:
            synthesis_result = await self.synthesis_service.synthesize_answer(
                original_query, documents, context
            )

            pipeline_steps["synthesis"] = {
                "success": synthesis_result["success"],
                "documents_used": len(documents),
                "execution_time": time.time() - step_start,
                "synthesis_stats": self.synthesis_service.get_service_stats(),
            }

            logger.info(
                f"Synthesis completed: {len(synthesis_result.get('answer', ''))} characters"
            )
            return synthesis_result

        except Exception as e:
            pipeline_steps["synthesis"] = {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - step_start,
            }
            logger.error(f"Synthesis failed: {str(e)}")

            return {
                "answer": "I encountered an error while generating the final answer.",
                "sources": [],
                "success": False,
                "error": str(e),
            }

    async def _finalize_result(
        self, result: Dict[str, Any], start_time: float
    ) -> Dict[str, Any]:
        """Finalize the pipeline result."""
        result["execution_time"] = time.time() - start_time

        # Update statistics
        self._stats["total_queries_processed"] += 1
        if result["success"]:
            self._stats["successful_queries"] += 1
        else:
            self._stats["failed_queries"] += 1

        self._stats["total_execution_time"] += result["execution_time"]
        self._stats["avg_execution_time"] = self._stats["total_execution_time"] / max(
            self._stats["total_queries_processed"], 1
        )

        logger.info(
            f"RAG pipeline completed in {result['execution_time']:.2f}s. "
            f"Success: {result['success']}"
        )

        return result

    async def _cleanup_collection(self, collection_name: str):
        """Clean up a temporary collection."""
        try:
            success = await self.ingestion_service.drop_collection(collection_name)
            if success:
                self._active_collections.discard(collection_name)
                self._stats["temporary_collections_cleaned"] += 1
                logger.info(f"Cleaned up collection: {collection_name}")
            else:
                logger.warning(f"Failed to cleanup collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error cleaning up collection {collection_name}: {str(e)}")

    async def cleanup_all_collections(self):
        """Clean up all active temporary collections."""
        collections_to_cleanup = list(self._active_collections)

        for collection_name in collections_to_cleanup:
            await self._cleanup_collection(collection_name)

        logger.info(f"Cleaned up {len(collections_to_cleanup)} collections")

    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        service_stats = {
            "query_processing": self.query_service.get_service_stats(),
            "search": self.search_service.get_service_stats(),
            "ingestion": self.ingestion_service.get_service_stats(),
            "retrieval": self.retrieval_service.get_service_stats(),
            "synthesis": self.synthesis_service.get_service_stats(),
        }

        return {
            "orchestrator_name": "RAGOrchestrator",
            "total_queries_processed": self._stats["total_queries_processed"],
            "successful_queries": self._stats["successful_queries"],
            "failed_queries": self._stats["failed_queries"],
            "success_rate": (
                self._stats["successful_queries"]
                / max(self._stats["total_queries_processed"], 1)
            ),
            "total_execution_time": self._stats["total_execution_time"],
            "avg_execution_time": self._stats["avg_execution_time"],
            "temporary_collections_created": self._stats[
                "temporary_collections_created"
            ],
            "temporary_collections_cleaned": self._stats[
                "temporary_collections_cleaned"
            ],
            "active_collections": len(self._active_collections),
            "auto_cleanup_enabled": self.auto_cleanup,
            "service_stats": service_stats,
        }

    def reset_stats(self):
        """Reset orchestrator statistics."""
        self._stats = {
            "total_queries_processed": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_execution_time": 0.0,
            "avg_execution_time": 0.0,
            "temporary_collections_created": 0,
            "temporary_collections_cleaned": 0,
        }

        # Reset service stats
        self.query_service.reset_stats()
        self.search_service.reset_stats()
        self.ingestion_service.reset_stats()
        self.retrieval_service.reset_stats()
        self.synthesis_service.reset_stats()

    @classmethod
    async def create_default(
        cls,
        tool_registry: ToolRegistry,
        llm: BaseChatModel,
        embeddings: Embeddings,
        vector_client: Optional[MilvusClient] = None,
        **kwargs,
    ) -> "RAGOrchestrator":
        """
        Create a default RAG orchestrator with standard configuration.

        Args:
            tool_registry: Tool registry instance
            llm: Language model instance
            embeddings: Embeddings instance
            vector_client: Vector database client (optional)
            **kwargs: Additional configuration options

        Returns:
            Configured RAG orchestrator instance
        """
        # Create vector client if not provided
        if not vector_client:
            vector_client = MilvusClient(embeddings)
            await vector_client.connect()

        # Create services
        query_service = QueryProcessingService(llm)
        search_service = SearchService(tool_registry)
        ingestion_service = IngestionService(embeddings, vector_client)
        retrieval_service = RetrievalService(vector_client, tool_registry)
        synthesis_service = SynthesisService(llm)

        # Create orchestrator
        return cls(
            query_service=query_service,
            search_service=search_service,
            ingestion_service=ingestion_service,
            retrieval_service=retrieval_service,
            synthesis_service=synthesis_service,
            **kwargs,
        )
