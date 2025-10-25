"""
Unit tests for SearXNG Tool.

This module tests the SearXNG web search tool.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import uuid
from typing import List, Dict, Any, Optional, Tuple
import json
import aiohttp

from app.core.tools.web.searxng_tool import SearXNGTool


class TestSearXNGTool:
    """Test SearXNGTool class"""

    @pytest.fixture
    def searxng_tool(self):
        """Create a SearXNG tool instance"""
        return SearXNGTool()

    def test_tool_init(self, searxng_tool):
        """Test SearXNGTool initialization"""
        assert searxng_tool.name == "searxng_search"
        assert searxng_tool.description == "Perform web searches using SearXNG"
        assert isinstance(searxng_tool.keywords, list)
        assert "search" in searxng_tool.keywords
        assert searxng_tool.category == "web"

    def test_tool_keywords(self, searxng_tool):
        """Test SearXNGTool keywords"""
        keywords = searxng_tool.keywords
        
        assert isinstance(keywords, list)
        assert "search" in keywords
        assert "web" in keywords
        assert "searxng" in keywords

    def test_tool_parameters(self, searxng_tool):
        """Test SearXNGTool parameters"""
        parameters = searxng_tool.parameters
        
        assert isinstance(parameters, dict)
        assert "query" in parameters
        assert "engines" in parameters
        assert "language" in parameters
        assert "time_range" in parameters
        assert "num_results" in parameters
        
        # Check query parameter
        query_param = parameters["query"]
        assert query_param["type"] == "string"
        assert query_param["required"] is True
        assert "description" in query_param
        
        # Check optional parameters
        engines_param = parameters["engines"]
        assert engines_param["type"] == "array"
        assert engines_param["required"] is False
        
        num_results_param = parameters["num_results"]
        assert num_results_param["type"] == "integer"
        assert num_results_param["default"] == 10

    def test_tool_properties(self, searxng_tool):
        """Test SearXNGTool properties"""
        assert searxng_tool.name == "searxng_search"
        assert searxng_tool.description == "Perform web searches using SearXNG"
        assert searxng_tool.category == "web"
        assert searxng_tool.version == "1.0.0"
        assert searxng_tool.requires_api_key is False
        assert searxng_tool.requires_auth is False

    @pytest.mark.asyncio
    async def test_execute_basic_search(self, searxng_tool):
        """Test basic search execution"""
        mock_response_data = {
            "results": [
                {
                    "title": "Test Result 1",
                    "url": "https://example.com/1",
                    "content": "This is the first test result",
                    "engine": "google",
                    "score": 0.95
                },
                {
                    "title": "Test Result 2",
                    "url": "https://example.com/2",
                    "content": "This is the second test result",
                    "engine": "bing",
                    "score": 0.85
                }
            ],
            "number_of_results": 2
        }
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_response_data
            mock_response.status = 200
            
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            result = await searxng_tool.execute(query="test query")
            
            assert isinstance(result, dict)
            assert "results" in result
            assert "query" in result
            assert "num_results" in result
            
            assert result["query"] == "test query"
            assert len(result["results"]) == 2
            assert result["num_results"] == 2
            
            # Check first result
            first_result = result["results"][0]
            assert first_result["title"] == "Test Result 1"
            assert first_result["url"] == "https://example.com/1"
            assert first_result["content"] == "This is the first test result"

    @pytest.mark.asyncio
    async def test_execute_with_parameters(self, searxng_tool):
        """Test search execution with parameters"""
        mock_response_data = {
            "results": [
                {
                    "title": "Filtered Result",
                    "url": "https://example.com/filtered",
                    "content": "Filtered search result",
                    "engine": "duckduckgo",
                    "score": 0.90
                }
            ],
            "number_of_results": 1
        }
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_response_data
            mock_response.status = 200
            
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            result = await searxng_tool.execute(
                query="filtered query",
                engines=["duckduckgo"],
                language="en",
                time_range="day",
                num_results=5
            )
            
            assert result["query"] == "filtered query"
            assert len(result["results"]) == 1
            
            # Check that parameters were passed correctly
            call_args = mock_session.post.call_args
            assert "q=filtered query" in call_args[1]["data"]
            assert "engines=duckduckgo" in call_args[1]["data"]
            assert "language=en" in call_args[1]["data"]
            assert "time_range=day" in call_args[1]["data"]
            assert "num_results=5" in call_args[1]["data"]

    @pytest.mark.asyncio
    async def test_execute_no_results(self, searxng_tool):
        """Test search execution with no results"""
        mock_response_data = {
            "results": [],
            "number_of_results": 0
        }
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_response_data
            mock_response.status = 200
            
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            result = await searxng_tool.execute(query="no results query")
            
            assert result["query"] == "no results query"
            assert result["results"] == []
            assert result["num_results"] == 0

    @pytest.mark.asyncio
    async def test_execute_http_error(self, searxng_tool):
        """Test search execution with HTTP error"""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text.return_value = "Internal Server Error"
            
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            with pytest.raises(Exception, match="Search failed with status 500"):
                await searxng_tool.execute(query="test query")

    @pytest.mark.asyncio
    async def test_execute_network_error(self, searxng_tool):
        """Test search execution with network error"""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.post.side_effect = aiohttp.ClientError("Network error")
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            with pytest.raises(Exception, match="Network error"):
                await searxng_tool.execute(query="test query")

    @pytest.mark.asyncio
    async def test_execute_json_error(self, searxng_tool):
        """Test search execution with JSON parsing error"""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
            mock_response.status = 200
            mock_response.text.return_value = "Invalid JSON response"
            
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            with pytest.raises(Exception, match="Invalid JSON"):
                await searxng_tool.execute(query="test query")

    @pytest.mark.asyncio
    async def test_execute_with_custom_searxng_url(self, searxng_tool):
        """Test search execution with custom SearXNG URL"""
        # Set custom URL
        searxng_tool.searxng_url = "https://custom-searxng.example.com"
        
        mock_response_data = {
            "results": [
                {
                    "title": "Custom Result",
                    "url": "https://example.com/custom",
                    "content": "Custom search result"
                }
            ],
            "number_of_results": 1
        }
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_response_data
            mock_response.status = 200
            
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            await searxng_tool.execute(query="test query")
            
            # Check that custom URL was used
            call_args = mock_session.post.call_args
            assert "https://custom-searxng.example.com" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_execute_with_empty_query(self, searxng_tool):
        """Test search execution with empty query"""
        with pytest.raises(ValueError, match="Query parameter is required"):
            await searxng_tool.execute(query="")

    @pytest.mark.asyncio
    async def test_execute_with_missing_query(self, searxng_tool):
        """Test search execution with missing query"""
        with pytest.raises(ValueError, match="Query parameter is required"):
            await searxng_tool.execute()

    def test_should_use_with_search_keywords(self, searxng_tool):
        """Test should_use method with search keywords"""
        assert searxng_tool.should_use("search for information") is True
        assert searxng_tool.should_use("find web results") is True
        assert searxng_tool.should_use("look up online") is True

    def test_should_use_without_search_keywords(self, searxng_tool):
        """Test should_use method without search keywords"""
        assert searxng_tool.should_use("create a file") is False
        assert searxng_tool.should_use("calculate something") is False
        assert searxng_tool.should_use("send an email") is False

    def test_should_use_with_context(self, searxng_tool):
        """Test should_use method with context"""
        context = {"available_tools": ["searxng_search", "calculator"]}
        
        assert searxng_tool.should_use("search the web", context) is True
        assert searxng_tool.should_use("find information online", context) is True

    def test_validate_parameters_valid(self, searxng_tool):
        """Test validate_parameters with valid parameters"""
        assert searxng_tool.validate_parameters(query="test query") is True
        assert searxng_tool.validate_parameters(
            query="test query",
            engines=["google"],
            language="en",
            num_results=10
        ) is True

    def test_validate_parameters_missing_required(self, searxng_tool):
        """Test validate_parameters with missing required parameter"""
        assert searxng_tool.validate_parameters() is False
        assert searxng_tool.validate_parameters(query="") is False

    def test_validate_parameters_invalid_types(self, searxng_tool):
        """Test validate_parameters with invalid parameter types"""
        # Invalid num_results type
        assert searxng_tool.validate_parameters(query="test", num_results="ten") is False
        
        # Invalid engines type
        assert searxng_tool.validate_parameters(query="test", engines="google") is False

    @pytest.mark.asyncio
    async def test_execute_with_timeout(self, searxng_tool):
        """Test execute_with_timeout method"""
        mock_response_data = {
            "results": [
                {
                    "title": "Timeout Test Result",
                    "url": "https://example.com/timeout",
                    "content": "Test result with timeout"
                }
            ],
            "number_of_results": 1
        }
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_response_data
            mock_response.status = 200
            
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            result = await searxng_tool.execute_with_timeout(
                query="timeout test",
                timeout=5.0
            )
            
            assert result.success is True
            assert result.data["query"] == "timeout test"
            assert len(result.data["results"]) == 1

    @pytest.mark.asyncio
    async def test_get_usage_stats(self, searxng_tool):
        """Test get_usage_stats method"""
        # Initially should have no stats
        stats = searxng_tool.get_usage_stats()
        assert stats["execution_count"] == 0
        assert stats["total_execution_time"] == 0.0
        
        # Execute a search to generate stats
        mock_response_data = {
            "results": [],
            "number_of_results": 0
        }
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_response_data
            mock_response.status = 200
            
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            await searxng_tool.execute(query="test")
            
            # Check stats after execution
            stats = searxng_tool.get_usage_stats()
            assert stats["execution_count"] == 1
            assert stats["total_execution_time"] > 0

    def test_tool_category_and_version(self, searxng_tool):
        """Test tool category and version"""
        assert searxng_tool.category == "web"
        assert searxng_tool.version == "1.0.0"

    def test_tool_auth_requirements(self, searxng_tool):
        """Test tool authentication requirements"""
        assert searxng_tool.requires_api_key is False
        assert searxng_tool.requires_auth is False

    @pytest.mark.asyncio
    async def test_execute_with_special_characters(self, searxng_tool):
        """Test search execution with special characters in query"""
        mock_response_data = {
            "results": [
                {
                    "title": "Special Characters Result",
                    "url": "https://example.com/special",
                    "content": "Result with special characters"
                }
            ],
            "number_of_results": 1
        }
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_response_data
            mock_response.status = 200
            
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            # Query with special characters
            result = await searxng_tool.execute(query="test with & special < characters >")
            
            assert result["query"] == "test with & special < characters >"
            assert len(result["results"]) == 1

    @pytest.mark.asyncio
    async def test_execute_with_unicode_characters(self, searxng_tool):
        """Test search execution with unicode characters in query"""
        mock_response_data = {
            "results": [
                {
                    "title": "Unicode Result",
                    "url": "https://example.com/unicode",
                    "content": "Result with unicode: ñáéíóú"
                }
            ],
            "number_of_results": 1
        }
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_response_data
            mock_response.status = 200
            
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            # Query with unicode characters
            result = await searxng_tool.execute(query="búsqueda con caracteres especiales: ñáéíóú")
            
            assert result["query"] == "búsqueda con caracteres especiales: ñáéíóú"
            assert len(result["results"]) == 1