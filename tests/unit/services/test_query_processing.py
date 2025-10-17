"""
Unit tests for Query Processing Service.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from langchain.chat_models.base import BaseChatModel

from app.core.services.query_processing import QueryProcessingService


class TestQueryProcessingService:
    """Test cases for QueryProcessingService."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = Mock(spec=BaseChatModel)
        response = Mock()
        response.content = "optimized search query about Python async programming"
        llm.ainvoke = AsyncMock(return_value=response)
        return llm

    @pytest.fixture
    def query_service(self, mock_llm):
        """Create QueryProcessingService instance."""
        return QueryProcessingService(mock_llm)

    @pytest.mark.asyncio
    async def test_generate_search_query(self, query_service, mock_llm):
        """Test search query generation."""
        user_query = "What is async programming in Python?"
        
        result = await query_service.generate_search_query(user_query)
        
        assert result == "optimized search query about Python async programming"
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_search_query_with_context(self, query_service, mock_llm):
        """Test search query generation with context."""
        user_query = "machine learning"
        context = {"domain": "healthcare", "time_range": "2023"}
        
        result = await query_service.generate_search_query(user_query, context)
        
        assert result == "optimized search query about Python async programming"
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_search_query_fallback(self, query_service, mock_llm):
        """Test fallback to original query when LLM fails."""
        user_query = "original query"
        mock_llm.ainvoke.side_effect = Exception("LLM error")
        
        result = await query_service.generate_search_query(user_query)
        
        assert result == user_query

    @pytest.mark.asyncio
    async def test_generate_search_query_empty_response(self, query_service, mock_llm):
        """Test fallback when LLM returns empty response."""
        user_query = "original query"
        response = Mock()
        response.content = ""
        mock_llm.ainvoke.return_value = response
        
        result = await query_service.generate_search_query(user_query)
        
        assert result == user_query

    @pytest.mark.asyncio
    async def test_batch_generate_queries(self, query_service, mock_llm):
        """Test batch query generation."""
        queries = ["query 1", "query 2", "query 3"]
        
        results = await query_service.batch_generate_queries(queries)
        
        assert len(results) == 3
        assert all(result == "optimized search query about Python async programming" for result in results)
        assert mock_llm.ainvoke.call_count == 3

    def test_validate_query_valid(self, query_service):
        """Test query validation with valid query."""
        query = "What is async programming in Python?"
        
        result = query_service.validate_query(query)
        
        assert result["valid"] is True
        assert len(result["issues"]) == 0

    def test_validate_query_empty(self, query_service):
        """Test query validation with empty query."""
        query = ""
        
        result = query_service.validate_query(query)
        
        assert result["valid"] is False
        assert "Query is empty" in result["issues"]

    def test_validate_query_too_short(self, query_service):
        """Test query validation with too short query."""
        query = "hi"
        
        result = query_service.validate_query(query)
        
        assert result["valid"] is False
        assert "Query is too short" in result["issues"]

    def test_validate_query_too_long(self, query_service):
        """Test query validation with too long query."""
        query = "a" * 201
        
        result = query_service.validate_query(query)
        
        assert result["valid"] is True  # Still valid but with warning
        assert "Query is too long" in result["issues"]

    def test_validate_query_advanced(self, query_service):
        """Test query validation with advanced operators."""
        query = 'site:example.com "exact phrase" filetype:pdf'
        
        result = query_service.validate_query(query)
        
        assert result["valid"] is True
        assert result["complexity"] == "advanced"
        assert result["query_type"] == "advanced"

    def test_validate_query_unmatched_quotes(self, query_service):
        """Test query validation with unmatched quotes."""
        query = '"unmatched quote'
        
        result = query_service.validate_query(query)
        
        assert result["valid"] is True  # Still valid but with issue
        assert "Unmatched quotes" in result["issues"]

    def test_get_service_stats(self, query_service):
        """Test service statistics."""
        stats = query_service.get_service_stats()
        
        assert stats["service_name"] == "QueryProcessingService"
        assert stats["llm_configured"] is True
        assert stats["prompt_template_loaded"] is True
        assert "query_optimization" in stats["supported_features"]
        assert "batch_processing" in stats["supported_features"]