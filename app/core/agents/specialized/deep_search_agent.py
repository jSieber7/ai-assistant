"""
Deep Search Agent for advanced web search and RAG pipeline.

This agent implements a sophisticated search workflow that combines web search,
content scraping, vector storage, and re-ranking to provide comprehensive
answers to user queries.

This agent now uses the modular RAG services architecture for better
maintainability and reusability.
"""

import logging
import time
from typing import Dict, List, Any, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from app.core.agents.base.base import BaseAgent, AgentResult, AgentState
from app.core.tools.execution.registry import ToolRegistry
from app.core.services.rag_orchestrator import RAGOrchestrator
from app.core.services.query_processing import QueryProcessingService
from app.core.services.search import SearchService
from app.core.services.ingestion import IngestionService
from app.core.services.retrieval import RetrievalService
from app.core.services.synthesis import SynthesisService
from app.core.storage.milvus_client import MilvusClient

logger = logging.getLogger(__name__)


class DeepSearchAgent(BaseAgent):
    """
    Advanced agent for deep web search and RAG-based answer generation.

    This agent implements the following workflow using modular services:
    1. Generate optimized search query from user input (QueryProcessingService)
    2. Search using SearXNG tool and scrape with Firecrawl (SearchService)
    3. Split content and create embeddings (IngestionService)
    4. Store in temporary Milvus collection (IngestionService)
    5. Retrieve with Milvus retriever (RetrievalService)
    6. Re-rank with Jina Reranker (RetrievalService)
    7. Generate final response using LCEL chain (SynthesisService)
    8. Clean up temporary collection (RAGOrchestrator)
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        llm: BaseChatModel,
        embeddings: Embeddings,
        max_iterations: int = 1,
    ):
        super().__init__(tool_registry, max_iterations)
        self.llm = llm
        self.embeddings = embeddings

        # Initialize modular services
        self._initialize_modular_services()

        # Configuration
        self.max_search_results = 5
        self.retrieval_k = 50
        self.rerank_top_n = 5

    def _initialize_modular_services(self):
        """Initialize the modular RAG services."""
        try:
            # Initialize Milvus client
            self.milvus_client = MilvusClient(self.embeddings)

            # Create services
            self.query_service = QueryProcessingService(self.llm)
            self.search_service = SearchService(self.tool_registry)
            self.ingestion_service = IngestionService(
                self.embeddings, self.milvus_client
            )
            self.retrieval_service = RetrievalService(
                self.milvus_client, self.tool_registry
            )
            self.synthesis_service = SynthesisService(self.llm)

            # Create orchestrator
            self.rag_orchestrator = RAGOrchestrator(
                query_service=self.query_service,
                search_service=self.search_service,
                ingestion_service=self.ingestion_service,
                retrieval_service=self.retrieval_service,
                synthesis_service=self.synthesis_service,
                auto_cleanup=True,
            )

            logger.info("Modular RAG services initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize modular services: {str(e)}")
            raise

    @property
    def name(self) -> str:
        return "deep_search"

    @property
    def description(self) -> str:
        return "Advanced agent for deep web search and comprehensive answer generation using RAG pipeline"

    async def _process_message_impl(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        context: Dict[str, Any] = None,
    ) -> AgentResult:
        """
        Process a message using the Deep Search workflow.
        """
        start_time = time.time()
        self.state = AgentState.THINKING

        if conversation_id:
            self._current_conversation_id = conversation_id
        else:
            self.start_conversation()

        # Add user message to conversation history
        self.add_to_conversation("user", message, {"timestamp": start_time})

        try:
            # Initialize Milvus connection if needed
            if not await self._initialize_milvus():
                raise RuntimeError("Failed to initialize Milvus connection")

            # Prepare options for the orchestrator
            options = {
                "search_params": {
                    "results_count": self.max_search_results,
                    "category": "general",
                },
                "retrieval_k": self.retrieval_k,
                "rerank_top_n": self.rerank_top_n,
            }

            # Process using the RAG orchestrator
            self.state = AgentState.EXECUTING_TOOL
            result = await self.rag_orchestrator.process_query(
                message, context, options
            )

            # Convert orchestrator result to AgentResult
            if result["success"]:
                execution_time = time.time() - start_time

                # Add assistant response to conversation history
                self.add_to_conversation(
                    "assistant",
                    result["answer"],
                    {
                        "execution_time": execution_time,
                        "search_query": result.get("metadata", {}).get(
                            "optimized_query", message
                        ),
                        "documents_processed": result.get("metadata", {}).get(
                            "documents_used", 0
                        ),
                        "collection_name": result.get("metadata", {}).get(
                            "collection_name", ""
                        ),
                        "pipeline_steps": result.get("pipeline_steps", {}),
                    },
                )

                self._usage_count += 1
                self._last_used = time.time()
                self.state = AgentState.IDLE

                return AgentResult(
                    success=True,
                    response=result["answer"],
                    tool_results=[],
                    agent_name=self.name,
                    execution_time=execution_time,
                    conversation_id=self._current_conversation_id,
                    metadata={
                        "optimized_query": result.get("metadata", {}).get(
                            "optimized_query", message
                        ),
                        "documents_count": result.get("metadata", {}).get(
                            "documents_used", 0
                        ),
                        "chunks_count": result.get("metadata", {}).get(
                            "chunks_created", 0
                        ),
                        "collection_name": result.get("metadata", {}).get(
                            "collection_name", ""
                        ),
                        "pipeline_steps": result.get("pipeline_steps", {}),
                        "sources": result.get("sources", []),
                    },
                )
            else:
                return AgentResult(
                    success=False,
                    response=result.get(
                        "answer", "I couldn't process your request. Please try again."
                    ),
                    tool_results=[],
                    error=result.get("error", "Unknown error"),
                    agent_name=self.name,
                    execution_time=time.time() - start_time,
                    conversation_id=self._current_conversation_id,
                    metadata={"error": result.get("error", "Unknown error")},
                )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Deep search processing failed: {str(e)}"
            logger.error(error_msg)
            self.state = AgentState.ERROR

            return AgentResult(
                success=False,
                response="I encountered an error while processing your deep search request. Please try again.",
                tool_results=[],
                error=error_msg,
                agent_name=self.name,
                execution_time=execution_time,
                conversation_id=self._current_conversation_id,
                metadata={"error_type": type(e).__name__},
            )

    async def _initialize_milvus(self) -> bool:
        """Initialize Milvus connection."""
        try:
            return await self.milvus_client.connect()

        except Exception as e:
            logger.error(f"Failed to initialize Milvus: {str(e)}")
            return False

    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent statistics including service stats."""
        base_stats = {
            "agent_name": self.name,
            "max_search_results": self.max_search_results,
            "retrieval_k": self.retrieval_k,
            "rerank_top_n": self.rerank_top_n,
        }

        # Add service stats
        if hasattr(self, "rag_orchestrator"):
            base_stats["service_stats"] = self.rag_orchestrator.get_orchestrator_stats()

        return base_stats

    async def cleanup_resources(self):
        """Clean up agent resources."""
        try:
            # Cleanup orchestrator resources
            if hasattr(self, "rag_orchestrator"):
                await self.rag_orchestrator.cleanup_all_collections()

            # Cleanup Milvus connection
            if hasattr(self, "milvus_client"):
                await self.milvus_client.disconnect()

            logger.info("DeepSearchAgent resources cleaned up")

        except Exception as e:
            logger.error(f"Error during resource cleanup: {str(e)}")

    # Delegate methods to services for direct access if needed

    async def search_and_scrape(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> List:
        """Delegate to search service."""
        return await self.search_service.search_and_scrape(query, context)

    async def retrieve_and_rerank(
        self, query: str, collection_name: str, k: int = 50, top_n: int = 5
    ) -> List:
        """Delegate to retrieval service."""
        return await self.retrieval_service.retrieve_and_rerank(
            query, collection_name, k, top_n
        )

    async def synthesize_answer(
        self, query: str, documents: List, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Delegate to synthesis service."""
        result = await self.synthesis_service.synthesize_answer(
            query, documents, context
        )
        return result.get("answer", "")
