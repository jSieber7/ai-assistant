"""
Integration tests for RAG Orchestrator.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from langchain.chat_models.base import BaseChatModel
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document

from app.core.services.rag_orchestrator import RAGOrchestrator
from app.core.services.query_processing import QueryProcessingService
from app.core.services.search import SearchService
from app.core.services.ingestion import IngestionService
from app.core.services.retrieval import RetrievalService
from app.core.services.synthesis import SynthesisService


class TestRAGOrchestrator:
    """Integration tests for RAG Orchestrator."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = Mock(spec=BaseChatModel)
        return llm

    @pytest.fixture
    def mock_embeddings(self):
        """Create mock embeddings."""
        embeddings = Mock(spec=Embeddings)
        embeddings.aembed_documents = AsyncMock(
            return_value=[[0.1, 0.2, 0.3] for _ in range(5)]
        )
        embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
        return embeddings

    @pytest.fixture
    def mock_services(self, mock_llm, mock_embeddings):
        """Create mock services."""
        query_service = Mock(spec=QueryProcessingService)
        query_service.generate_search_query = AsyncMock(return_value="optimized query")

        search_service = Mock(spec=SearchService)
        search_service.search_and_scrape = AsyncMock(
            return_value=[
                Document(
                    page_content="Document 1 content", metadata={"source": "url1"}
                ),
                Document(
                    page_content="Document 2 content", metadata={"source": "url2"}
                ),
            ]
        )

        ingestion_service = Mock(spec=IngestionService)
        ingestion_service.create_temporary_collection = AsyncMock(
            return_value="test_collection"
        )
        ingestion_service.ingest_documents = AsyncMock(
            return_value={
                "success": True,
                "documents_processed": 2,
                "chunks_created": 5,
                "chunks_ingested": 5,
            }
        )
        ingestion_service.drop_collection = AsyncMock(return_value=True)

        retrieval_service = Mock(spec=RetrievalService)
        retrieval_service.retrieve_and_rerank = AsyncMock(
            return_value=[
                Document(page_content="Retrieved doc 1", metadata={"source": "url1"}),
                Document(page_content="Retrieved doc 2", metadata={"source": "url2"}),
            ]
        )

        synthesis_service = Mock(spec=SynthesisService)
        synthesis_service.synthesize_answer = AsyncMock(
            return_value={
                "answer": "This is the synthesized answer based on the documents.",
                "sources": [{"source": "url1"}, {"source": "url2"}],
                "success": True,
                "metadata": {"documents_used": 2},
            }
        )

        return {
            "query_service": query_service,
            "search_service": search_service,
            "ingestion_service": ingestion_service,
            "retrieval_service": retrieval_service,
            "synthesis_service": synthesis_service,
        }

    @pytest.fixture
    def orchestrator(self, mock_services):
        """Create RAG orchestrator with mock services."""
        return RAGOrchestrator(
            query_service=mock_services["query_service"],
            search_service=mock_services["search_service"],
            ingestion_service=mock_services["ingestion_service"],
            retrieval_service=mock_services["retrieval_service"],
            synthesis_service=mock_services["synthesis_service"],
            auto_cleanup=False,  # Disable auto cleanup for testing
        )

    @pytest.mark.asyncio
    async def test_process_query_success(self, orchestrator, mock_services):
        """Test successful query processing through the full pipeline."""
        user_query = "What is async programming in Python?"

        result = await orchestrator.process_query(user_query)

        # Verify the result
        assert result["success"] is True
        assert result["query"] == user_query
        assert (
            result["answer"] == "This is the synthesized answer based on the documents."
        )
        assert len(result["sources"]) == 2
        assert result["metadata"]["collection_name"] == "test_collection"
        assert "pipeline_steps" in result
        assert result["execution_time"] > 0

        # Verify all services were called
        mock_services["query_service"].generate_search_query.assert_called_once_with(
            user_query, None
        )
        mock_services["search_service"].search_and_scrape.assert_called_once()
        mock_services[
            "ingestion_service"
        ].create_temporary_collection.assert_called_once()
        mock_services["ingestion_service"].ingest_documents.assert_called_once()
        mock_services["retrieval_service"].retrieve_and_rerank.assert_called_once()
        mock_services["synthesis_service"].synthesize_answer.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_query_with_context(self, orchestrator, mock_services):
        """Test query processing with context."""
        user_query = "machine learning"
        context = {"domain": "healthcare", "time_range": "2023"}
        options = {"retrieval_k": 10, "rerank_top_n": 3}

        result = await orchestrator.process_query(user_query, context, options)

        assert result["success"] is True

        # Verify context was passed to query service
        mock_services["query_service"].generate_search_query.assert_called_once_with(
            user_query, context
        )

    @pytest.mark.asyncio
    async def test_process_query_no_search_results(self, mock_services):
        """Test query processing when no search results are found."""
        # Mock search service to return no results
        mock_services["search_service"].search_and_scrape = AsyncMock(return_value=[])

        orchestrator = RAGOrchestrator(
            query_service=mock_services["query_service"],
            search_service=mock_services["search_service"],
            ingestion_service=mock_services["ingestion_service"],
            retrieval_service=mock_services["retrieval_service"],
            synthesis_service=mock_services["synthesis_service"],
        )

        user_query = "nonexistent topic"

        result = await orchestrator.process_query(user_query)

        assert result["success"] is False
        assert "No relevant information found" in result["error"]
        assert "couldn't find relevant information" in result["answer"]

    @pytest.mark.asyncio
    async def test_process_query_synthesis_failure(self, mock_services):
        """Test query processing when synthesis fails."""
        # Mock synthesis service to fail
        mock_services["synthesis_service"].synthesize_answer = AsyncMock(
            return_value={
                "answer": "Failed to synthesize answer",
                "sources": [],
                "success": False,
                "error": "Synthesis error",
            }
        )

        orchestrator = RAGOrchestrator(
            query_service=mock_services["query_service"],
            search_service=mock_services["search_service"],
            ingestion_service=mock_services["ingestion_service"],
            retrieval_service=mock_services["retrieval_service"],
            synthesis_service=mock_services["synthesis_service"],
        )

        user_query = "test query"

        result = await orchestrator.process_query(user_query)

        assert result["success"] is False
        assert result["answer"] == "Failed to synthesize answer"
        assert result["error"] == "Synthesis error"

    @pytest.mark.asyncio
    async def test_process_query_with_auto_cleanup(self, mock_services):
        """Test query processing with automatic cleanup enabled."""
        orchestrator = RAGOrchestrator(
            query_service=mock_services["query_service"],
            search_service=mock_services["search_service"],
            ingestion_service=mock_services["ingestion_service"],
            retrieval_service=mock_services["retrieval_service"],
            synthesis_service=mock_services["synthesis_service"],
            auto_cleanup=True,
        )

        user_query = "test query"

        result = await orchestrator.process_query(user_query)

        assert result["success"] is True

        # Verify cleanup was called
        mock_services["ingestion_service"].drop_collection.assert_called_once_with(
            "test_collection"
        )

    @pytest.mark.asyncio
    async def test_cleanup_all_collections(self, orchestrator, mock_services):
        """Test cleanup of all active collections."""
        # Add some collections to the active set
        orchestrator._active_collections.add("collection1")
        orchestrator._active_collections.add("collection2")

        await orchestrator.cleanup_all_collections()

        # Verify cleanup was called for each collection
        assert mock_services["ingestion_service"].drop_collection.call_count == 2
        assert len(orchestrator._active_collections) == 0

    def test_get_orchestrator_stats(self, orchestrator, mock_services):
        """Test getting orchestrator statistics."""
        # Mock service stats
        mock_services["query_service"].get_service_stats.return_value = {
            "service": "query_stats"
        }
        mock_services["search_service"].get_service_stats.return_value = {
            "service": "search_stats"
        }
        mock_services["ingestion_service"].get_service_stats.return_value = {
            "service": "ingestion_stats"
        }
        mock_services["retrieval_service"].get_service_stats.return_value = {
            "service": "retrieval_stats"
        }
        mock_services["synthesis_service"].get_service_stats.return_value = {
            "service": "synthesis_stats"
        }

        stats = orchestrator.get_orchestrator_stats()

        assert stats["orchestrator_name"] == "RAGOrchestrator"
        assert stats["total_queries_processed"] == 0
        assert stats["successful_queries"] == 0
        assert stats["failed_queries"] == 0
        assert stats["auto_cleanup_enabled"] is False
        assert "service_stats" in stats
        assert len(stats["service_stats"]) == 5

    def test_reset_stats(self, orchestrator, mock_services):
        """Test resetting orchestrator statistics."""
        # Update some stats
        orchestrator._stats["total_queries_processed"] = 10
        orchestrator._stats["successful_queries"] = 8
        orchestrator._stats["failed_queries"] = 2

        # Reset stats
        orchestrator.reset_stats()

        # Verify stats are reset
        assert orchestrator._stats["total_queries_processed"] == 0
        assert orchestrator._stats["successful_queries"] == 0
        assert orchestrator._stats["failed_queries"] == 0

        # Verify service reset methods were called
        mock_services["query_service"].reset_stats.assert_called_once()
        mock_services["search_service"].reset_stats.assert_called_once()
        mock_services["ingestion_service"].reset_stats.assert_called_once()
        mock_services["retrieval_service"].reset_stats.assert_called_once()
        mock_services["synthesis_service"].reset_stats.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_query_pipeline_steps(self, orchestrator, mock_services):
        """Test that pipeline steps are properly recorded."""
        user_query = "test query"

        result = await orchestrator.process_query(user_query)

        pipeline_steps = result["pipeline_steps"]

        # Verify all pipeline steps are present
        assert "query_processing" in pipeline_steps
        assert "search_and_scraping" in pipeline_steps
        assert "ingestion" in pipeline_steps
        assert "retrieval" in pipeline_steps
        assert "synthesis" in pipeline_steps

        # Verify each step has required fields
        for step_name, step_data in pipeline_steps.items():
            assert "success" in step_data
            assert "execution_time" in step_data

    @pytest.mark.asyncio
    async def test_step_query_processing_failure(self, mock_services):
        """Test query processing step failure."""
        # Mock query service to fail
        mock_services["query_service"].generate_search_query = AsyncMock(
            side_effect=Exception("Query processing failed")
        )

        orchestrator = RAGOrchestrator(
            query_service=mock_services["query_service"],
            search_service=mock_services["search_service"],
            ingestion_service=mock_services["ingestion_service"],
            retrieval_service=mock_services["retrieval_service"],
            synthesis_service=mock_services["synthesis_service"],
        )

        user_query = "test query"

        result = await orchestrator.process_query(user_query)

        # Should still succeed with fallback to original query
        assert result["success"] is True

        # Verify pipeline step shows failure but fallback worked
        pipeline_steps = result["pipeline_steps"]
        assert (
            pipeline_steps["query_processing"]["success"] is True
        )  # Fallback makes it successful
        assert "optimized query" in pipeline_steps["query_processing"]

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_create_default_orchestrator(self, mock_llm, mock_embeddings):
        """Test creating default orchestrator."""
        with (
            patch(
                "app.core.services.rag_orchestrator.QueryProcessingService"
            ) as mock_query_service,
            patch(
                "app.core.services.rag_orchestrator.SearchService"
            ) as mock_search_service,
            patch(
                "app.core.services.rag_orchestrator.IngestionService"
            ) as mock_ingestion_service,
            patch(
                "app.core.services.rag_orchestrator.RetrievalService"
            ) as mock_retrieval_service,
            patch(
                "app.core.services.rag_orchestrator.SynthesisService"
            ) as mock_synthesis_service,
            patch(
                "app.core.services.rag_orchestrator.MilvusClient"
            ) as mock_milvus_client,
        ):
            # Mock MilvusClient
            mock_client = Mock()
            mock_client.connect = AsyncMock(return_value=True)
            mock_milvus_client.return_value = mock_client

            # Mock tool registry
            mock_tool_registry = Mock()

            orchestrator = await RAGOrchestrator.create_default(
                tool_registry=mock_tool_registry,
                llm=mock_llm,
                embeddings=mock_embeddings,
            )

            assert orchestrator is not None
            assert orchestrator.auto_cleanup is True

            # Verify services were created
            mock_query_service.assert_called_once_with(mock_llm)
            mock_search_service.assert_called_once_with(mock_tool_registry)
            mock_ingestion_service.assert_called_once_with(mock_embeddings, mock_client)
            mock_retrieval_service.assert_called_once_with(
                mock_client, mock_tool_registry
            )
            mock_synthesis_service.assert_called_once_with(mock_llm)
