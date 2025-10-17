"""
Integration tests for Jina Reranker service
"""

import pytest
from unittest.mock import AsyncMock, patch
import httpx

from app.core.tools.jina_reranker_tool import JinaRerankerTool
from app.core.config import settings


class TestJinaRerankerTool:
    """Test cases for Jina Reranker tool"""

    @pytest.fixture
    def settings(self):
        """Get test settings"""
        test_settings = settings
        test_settings.jina_reranker_enabled = True
        test_settings.jina_reranker_url = "http://test-jina-reranker:8080"
        test_settings.jina_reranker_timeout = 10
        return test_settings

    @pytest.fixture
    def reranker_tool(self):
        """Create Jina Reranker tool instance"""
        tool = JinaRerankerTool()
        # Mock settings
        tool.settings = AsyncMock()
        tool.settings.jina_reranker_enabled = True
        tool.settings.jina_reranker_url = "http://test-jina-reranker:8080"
        tool.settings.jina_reranker_timeout = 10
        tool.settings.jina_reranker_model = "jina-reranker-v2-base-multilingual"
        return tool

    @pytest.mark.asyncio
    async def test_rerank_success(self, reranker_tool):
        """Test successful reranking"""
        # Mock HTTP client
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json = AsyncMock(return_value={
            "results": [
                {"index": 1, "document": "Document 2", "relevance_score": 0.9},
                {"index": 0, "document": "Document 1", "relevance_score": 0.7},
                {"index": 2, "document": "Document 3", "relevance_score": 0.3},
            ],
            "model": "jina-reranker-v2-base-multilingual",
            "query": "test query",
            "total_documents": 3,
            "cached": False,
        })

        with patch("httpx.AsyncClient.post", return_value=mock_response):
            result = await reranker_tool.execute(
                query="test query",
                documents=["Document 1", "Document 2", "Document 3"],
                top_n=3,
            )

        assert result.success is True
        assert "results" in result.data
        assert len(result.data["results"]) == 3
        assert result.data["results"][0]["relevance_score"] == 0.9

    @pytest.mark.asyncio
    async def test_rerank_with_top_n(self, reranker_tool):
        """Test reranking with top_n parameter"""
        # Mock HTTP client
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json = AsyncMock(return_value={
            "results": [
                {"index": 1, "document": "Document 2", "relevance_score": 0.9},
                {"index": 0, "document": "Document 1", "relevance_score": 0.7},
            ],
            "model": "jina-reranker-v2-base-multilingual",
            "query": "test query",
            "total_documents": 3,
            "cached": False,
        })

        with patch("httpx.AsyncClient.post", return_value=mock_response):
            result = await reranker_tool.execute(
                query="test query",
                documents=["Document 1", "Document 2", "Document 3"],
                top_n=2,
            )

        assert result.success is True
        assert len(result.data["results"]) == 2

    @pytest.mark.asyncio
    async def test_rerank_disabled(self, reranker_tool):
        """Test reranking when service is disabled"""
        # Mock settings to disable service
        reranker_tool.settings.jina_reranker_enabled = False

        result = await reranker_tool.execute(
            query="test query", documents=["Document 1", "Document 2"]
        )

        assert result.success is False
        assert "disabled" in result.error.lower()

    @pytest.mark.asyncio
    async def test_rerank_missing_query(self, reranker_tool):
        """Test reranking with missing query"""
        result = await reranker_tool.execute(documents=["Document 1", "Document 2"])

        assert result.success is False
        assert "query" in result.error.lower()

    @pytest.mark.asyncio
    async def test_rerank_empty_documents(self, reranker_tool):
        """Test reranking with empty documents list"""
        result = await reranker_tool.execute(query="test query", documents=[])

        assert result.success is False
        assert "documents" in result.error.lower()

    @pytest.mark.asyncio
    async def test_rerank_http_error(self, reranker_tool):
        """Test reranking with HTTP error"""
        # Mock HTTP client to return error
        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("httpx.AsyncClient.post", return_value=mock_response):
            result = await reranker_tool.execute(
                query="test query", documents=["Document 1", "Document 2"]
            )

        assert result.success is False
        assert "failed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_rerank_connection_error(self, reranker_tool):
        """Test reranking with connection error"""
        # Mock HTTP client to raise connection error
        with patch(
            "httpx.AsyncClient.post",
            side_effect=httpx.RequestError("Connection failed"),
        ):
            result = await reranker_tool.execute(
                query="test query", documents=["Document 1", "Document 2"]
            )

        assert result.success is False
        assert "failed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_rerank_search_results(self, reranker_tool):
        """Test reranking search results"""
        # Mock HTTP client
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json = AsyncMock(return_value={
            "results": [
                {"index": 1, "document": "Content 2", "relevance_score": 0.9},
                {"index": 0, "document": "Content 1", "relevance_score": 0.7},
            ],
            "model": "jina-reranker-v2-base-multilingual",
            "query": "test query",
            "total_documents": 2,
            "cached": False,
        })

        search_results = [
            {"content": "Content 1", "url": "http://example.com/1"},
            {"content": "Content 2", "url": "http://example.com/2"},
        ]

        with patch("httpx.AsyncClient.post", return_value=mock_response):
            reranked_results = await reranker_tool.rerank_search_results(
                query="test query", search_results=search_results, top_n=2
            )

        assert len(reranked_results) == 2
        assert reranked_results[0]["relevance_score"] == 0.9
        assert reranked_results[0]["url"] == "http://example.com/2"

    @pytest.mark.asyncio
    async def test_rerank_search_results_fallback(self, reranker_tool):
        """Test reranking search results with fallback on error"""
        # Mock HTTP client to return error
        mock_response = AsyncMock()
        mock_response.status_code = 500

        search_results = [
            {"content": "Content 1", "url": "http://example.com/1"},
            {"content": "Content 2", "url": "http://example.com/2"},
        ]

        with patch("httpx.AsyncClient.post", return_value=mock_response):
            reranked_results = await reranker_tool.rerank_search_results(
                query="test query", search_results=search_results, top_n=2
            )

        # Should return original results on error
        assert len(reranked_results) == 2
        assert reranked_results[0]["url"] == "http://example.com/1"

    def test_get_schema(self, reranker_tool):
        """Test tool schema"""
        schema = reranker_tool.get_schema()

        assert schema["name"] == "jina_reranker"
        assert "query" in schema["parameters"]["properties"]
        assert "documents" in schema["parameters"]["properties"]
        assert "top_n" in schema["parameters"]["properties"]
        assert "model" in schema["parameters"]["properties"]

        # Check required parameters
        assert "query" in schema["parameters"]["required"]
        assert "documents" in schema["parameters"]["required"]


@pytest.mark.asyncio
async def test_jina_reranker_integration():
    """Integration test for Jina Reranker service"""
    # This test requires the actual service to be running
    # Skip if service is not available
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8080/health", timeout=5)
            if response.status_code != 200:
                pytest.skip("Jina Reranker service not available")
    except Exception:
        pytest.skip("Jina Reranker service not available")

    # Test actual service
    test_settings = settings
    test_settings.jina_reranker_enabled = True
    test_settings.jina_reranker_url = "http://localhost:8080"

    tool = JinaRerankerTool()
    tool.settings = test_settings

    result = await tool.execute(
        query="machine learning",
        documents=[
            "Machine learning is a subset of artificial intelligence",
            "Deep learning uses neural networks",
            "Python is a programming language",
        ],
        top_n=2,
    )

    assert result.success is True
    assert "results" in result.data
    assert len(result.data["results"]) <= 2