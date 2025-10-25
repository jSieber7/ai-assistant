"""
Unit tests for Firecrawl Tool.

This module tests the Firecrawl web scraping tool.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import uuid
from typing import List, Dict, Any, Optional, Tuple
import json
import aiohttp

from app.core.tools.web.firecrawl_tool import FirecrawlTool


class TestFirecrawlTool:
    """Test FirecrawlTool class"""

    @pytest.fixture
    def firecrawl_tool(self):
        """Create a Firecrawl tool instance"""
        return FirecrawlTool(api_key="test-api-key")

    def test_tool_init(self, firecrawl_tool):
        """Test FirecrawlTool initialization"""
        assert firecrawl_tool.name == "firecrawl_scrape"
        assert firecrawl_tool.description == "Scrape web pages using Firecrawl API"
        assert isinstance(firecrawl_tool.keywords, list)
        assert "scrape" in firecrawl_tool.keywords
        assert firecrawl_tool.category == "web"
        assert firecrawl_tool.api_key == "test-api-key"

    def test_tool_init_without_api_key(self):
        """Test FirecrawlTool initialization without API key"""
        with patch.dict('os.environ', {'FIRECRAWL_API_KEY': 'env-api-key'}):
            tool = FirecrawlTool()
            assert tool.api_key == "env-api-key"

    def test_tool_init_without_any_api_key(self):
        """Test FirecrawlTool initialization without any API key"""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="API key is required"):
                FirecrawlTool()

    def test_tool_keywords(self, firecrawl_tool):
        """Test FirecrawlTool keywords"""
        keywords = firecrawl_tool.keywords
        
        assert isinstance(keywords, list)
        assert "scrape" in keywords
        assert "web" in keywords
        assert "firecrawl" in keywords
        assert "extract" in keywords

    def test_tool_parameters(self, firecrawl_tool):
        """Test FirecrawlTool parameters"""
        parameters = firecrawl_tool.parameters
        
        assert isinstance(parameters, dict)
        assert "url" in parameters
        assert "formats" in parameters
        assert "only_main_content" in parameters
        assert "include_tags" in parameters
        assert "exclude_tags" in parameters
        assert "wait_for" in parameters
        assert "timeout" in parameters
        
        # Check url parameter
        url_param = parameters["url"]
        assert url_param["type"] == "string"
        assert url_param["required"] is True
        assert "description" in url_param
        
        # Check optional parameters
        formats_param = parameters["formats"]
        assert formats_param["type"] == "array"
        assert formats_param["required"] is False
        assert "markdown" in formats_param["default"]
        assert "html" in formats_param["default"]

    def test_tool_properties(self, firecrawl_tool):
        """Test FirecrawlTool properties"""
        assert firecrawl_tool.name == "firecrawl_scrape"
        assert firecrawl_tool.description == "Scrape web pages using Firecrawl API"
        assert firecrawl_tool.category == "web"
        assert firecrawl_tool.version == "1.0.0"
        assert firecrawl_tool.requires_api_key is True
        assert firecrawl_tool.requires_auth is False

    @pytest.mark.asyncio
    async def test_execute_basic_scrape(self, firecrawl_tool):
        """Test basic scraping execution"""
        mock_response_data = {
            "success": True,
            "data": {
                "markdown": "# Test Page\n\nThis is a test page with some content.",
                "html": "<html><body><h1>Test Page</h1><p>This is a test page with some content.</p></body></html>",
                "links": ["https://example.com/page1", "https://example.com/page2"],
                "images": ["https://example.com/image1.jpg", "https://example.com/image2.jpg"],
                "metadata": {
                    "title": "Test Page",
                    "description": "A test page for scraping",
                    "language": "en"
                }
            }
        }
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_response_data
            mock_response.status = 200
            
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            result = await firecrawl_tool.execute(url="https://example.com")
            
            assert isinstance(result, dict)
            assert "success" in result
            assert "data" in result
            assert "url" in result
            
            assert result["success"] is True
            assert result["url"] == "https://example.com"
            
            # Check scraped content
            data = result["data"]
            assert "markdown" in data
            assert "html" in data
            assert "links" in data
            assert "images" in data
            assert "metadata" in data
            
            assert "# Test Page" in data["markdown"]
            assert "Test Page" in data["metadata"]["title"]

    @pytest.mark.asyncio
    async def test_execute_with_formats(self, firecrawl_tool):
        """Test scraping execution with specific formats"""
        mock_response_data = {
            "success": True,
            "data": {
                "raw": "<html><body>Raw content</body></html>",
                "links": ["https://example.com/link"]
            }
        }
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_response_data
            mock_response.status = 200
            
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            result = await firecrawl_tool.execute(
                url="https://example.com",
                formats=["raw", "links"]
            )
            
            assert result["success"] is True
            data = result["data"]
            assert "raw" in data
            assert "links" in data
            assert "markdown" not in data  # Should not be included
            assert "html" not in data     # Should not be included

    @pytest.mark.asyncio
    async def test_execute_with_content_filtering(self, firecrawl_tool):
        """Test scraping execution with content filtering"""
        mock_response_data = {
            "success": True,
            "data": {
                "markdown": "# Main Content\n\nThis is the main content only.",
                "html": "<main><h1>Main Content</h1><p>This is the main content only.</p></main>"
            }
        }
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_response_data
            mock_response.status = 200
            
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            result = await firecrawl_tool.execute(
                url="https://example.com",
                only_main_content=True,
                include_tags=["main", "article"],
                exclude_tags=["nav", "footer", "aside"]
            )
            
            assert result["success"] is True
            assert "Main Content" in result["data"]["markdown"]

    @pytest.mark.asyncio
    async def test_execute_with_wait_options(self, firecrawl_tool):
        """Test scraping execution with wait options"""
        mock_response_data = {
            "success": True,
            "data": {
                "markdown": "# Dynamic Content\n\nThis content loaded after JavaScript.",
                "html": "<html><body><div id='dynamic'>Dynamic content</div></body></html>"
            }
        }
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_response_data
            mock_response.status = 200
            
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            result = await firecrawl_tool.execute(
                url="https://example.com",
                wait_for=3000,  # Wait 3 seconds
                timeout=60000   # 60 second timeout
            )
            
            assert result["success"] is True
            assert "Dynamic Content" in result["data"]["markdown"]

    @pytest.mark.asyncio
    async def test_execute_api_error(self, firecrawl_tool):
        """Test scraping execution with API error"""
        mock_response_data = {
            "success": False,
            "error": "Failed to scrape URL: 404 Not Found"
        }
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_response_data
            mock_response.status = 400
            
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            with pytest.raises(Exception, match="Failed to scrape URL"):
                await firecrawl_tool.execute(url="https://example.com/not-found")

    @pytest.mark.asyncio
    async def test_execute_http_error(self, firecrawl_tool):
        """Test scraping execution with HTTP error"""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text.return_value = "Internal Server Error"
            
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            with pytest.raises(Exception, match="Scraping failed with status 500"):
                await firecrawl_tool.execute(url="https://example.com")

    @pytest.mark.asyncio
    async def test_execute_network_error(self, firecrawl_tool):
        """Test scraping execution with network error"""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.post.side_effect = aiohttp.ClientError("Network error")
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            with pytest.raises(Exception, match="Network error"):
                await firecrawl_tool.execute(url="https://example.com")

    @pytest.mark.asyncio
    async def test_execute_with_invalid_url(self, firecrawl_tool):
        """Test scraping execution with invalid URL"""
        with pytest.raises(ValueError, match="Invalid URL format"):
            await firecrawl_tool.execute(url="not-a-valid-url")

    @pytest.mark.asyncio
    async def test_execute_with_empty_url(self, firecrawl_tool):
        """Test scraping execution with empty URL"""
        with pytest.raises(ValueError, match="URL parameter is required"):
            await firecrawl_tool.execute(url="")

    @pytest.mark.asyncio
    async def test_execute_with_missing_url(self, firecrawl_tool):
        """Test scraping execution with missing URL"""
        with pytest.raises(ValueError, match="URL parameter is required"):
            await firecrawl_tool.execute()

    def test_should_use_with_scrape_keywords(self, firecrawl_tool):
        """Test should_use method with scrape keywords"""
        assert firecrawl_tool.should_use("scrape the website") is True
        assert firecrawl_tool.should_use("extract content from page") is True
        assert firecrawl_tool.should_use("get web page content") is True

    def test_should_use_without_scrape_keywords(self, firecrawl_tool):
        """Test should_use method without scrape keywords"""
        assert firecrawl_tool.should_use("create a file") is False
        assert firecrawl_tool.should_use("calculate something") is False
        assert firecrawl_tool.should_use("send an email") is False

    def test_should_use_with_url(self, firecrawl_tool):
        """Test should_use method with URL in text"""
        assert firecrawl_tool.should_use("scrape https://example.com") is True
        assert firecrawl_tool.should_use("extract content from http://website.org/page") is True

    def test_validate_parameters_valid(self, firecrawl_tool):
        """Test validate_parameters with valid parameters"""
        assert firecrawl_tool.validate_parameters(url="https://example.com") is True
        assert firecrawl_tool.validate_parameters(
            url="https://example.com",
            formats=["markdown"],
            only_main_content=True
        ) is True

    def test_validate_parameters_missing_required(self, firecrawl_tool):
        """Test validate_parameters with missing required parameter"""
        assert firecrawl_tool.validate_parameters() is False
        assert firecrawl_tool.validate_parameters(url="") is False

    def test_validate_parameters_invalid_types(self, firecrawl_tool):
        """Test validate_parameters with invalid parameter types"""
        # Invalid formats type
        assert firecrawl_tool.validate_parameters(url="https://example.com", formats="markdown") is False
        
        # Invalid only_main_content type
        assert firecrawl_tool.validate_parameters(url="https://example.com", only_main_content="true") is False

    @pytest.mark.asyncio
    async def test_execute_with_timeout(self, firecrawl_tool):
        """Test execute_with_timeout method"""
        mock_response_data = {
            "success": True,
            "data": {
                "markdown": "# Timeout Test\n\nContent for timeout test."
            }
        }
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_response_data
            mock_response.status = 200
            
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            result = await firecrawl_tool.execute_with_timeout(
                url="https://example.com",
                timeout=5.0
            )
            
            assert result.success is True
            assert result.data["success"] is True
            assert "Timeout Test" in result.data["data"]["markdown"]

    @pytest.mark.asyncio
    async def test_get_usage_stats(self, firecrawl_tool):
        """Test get_usage_stats method"""
        # Initially should have no stats
        stats = firecrawl_tool.get_usage_stats()
        assert stats["execution_count"] == 0
        assert stats["total_execution_time"] == 0.0
        
        # Execute a scrape to generate stats
        mock_response_data = {
            "success": True,
            "data": {
                "markdown": "# Test\n\nContent."
            }
        }
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_response_data
            mock_response.status = 200
            
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            await firecrawl_tool.execute(url="https://example.com")
            
            # Check stats after execution
            stats = firecrawl_tool.get_usage_stats()
            assert stats["execution_count"] == 1
            assert stats["total_execution_time"] > 0

    def test_tool_category_and_version(self, firecrawl_tool):
        """Test tool category and version"""
        assert firecrawl_tool.category == "web"
        assert firecrawl_tool.version == "1.0.0"

    def test_tool_auth_requirements(self, firecrawl_tool):
        """Test tool authentication requirements"""
        assert firecrawl_tool.requires_api_key is True
        assert firecrawl_tool.requires_auth is False

    @pytest.mark.asyncio
    async def test_execute_with_complex_html(self, firecrawl_tool):
        """Test scraping execution with complex HTML content"""
        mock_response_data = {
            "success": True,
            "data": {
                "markdown": "# Complex Page\n\nThis is a complex page with tables and lists.\n\n| Column 1 | Column 2 |\n|----------|----------|\n| Cell 1   | Cell 2   |\n\n- Item 1\n- Item 2\n- Item 3",
                "html": "<html><body><h1>Complex Page</h1><p>This is a complex page with tables and lists.</p><table><tr><th>Column 1</th><th>Column 2</th></tr><tr><td>Cell 1</td><td>Cell 2</td></tr></table><ul><li>Item 1</li><li>Item 2</li><li>Item 3</li></ul></body></html>",
                "links": ["https://example.com/page1", "https://example.com/page2"],
                "images": ["https://example.com/image1.jpg"],
                "metadata": {
                    "title": "Complex Page",
                    "description": "A complex page with structured content",
                    "language": "en",
                    "author": "Test Author"
                }
            }
        }
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_response_data
            mock_response.status = 200
            
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            result = await firecrawl_tool.execute(
                url="https://example.com/complex",
                formats=["markdown", "html", "links", "images", "metadata"]
            )
            
            assert result["success"] is True
            data = result["data"]
            
            # Check all formats are present
            assert "markdown" in data
            assert "html" in data
            assert "links" in data
            assert "images" in data
            assert "metadata" in data
            
            # Check complex content was parsed correctly
            assert "Column 1" in data["markdown"]
            assert "Item 1" in data["markdown"]
            assert "Test Author" in data["metadata"]["author"]

    @pytest.mark.asyncio
    async def test_execute_with_javascript_rendered_content(self, firecrawl_tool):
        """Test scraping execution with JavaScript rendered content"""
        mock_response_data = {
            "success": True,
            "data": {
                "markdown": "# Dynamic Page\n\nThis content was rendered by JavaScript.",
                "html": "<html><body><div id='app'><h1>Dynamic Page</h1><p>This content was rendered by JavaScript.</p></div></body></html>"
            }
        }
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_response_data
            mock_response.status = 200
            
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            result = await firecrawl_tool.execute(
                url="https://example.com/spa",
                wait_for=2000  # Wait for JavaScript to render
            )
            
            assert result["success"] is True
            assert "Dynamic Page" in result["data"]["markdown"]
            assert "JavaScript" in result["data"]["markdown"]

    @pytest.mark.asyncio
    async def test_execute_with_large_content(self, firecrawl_tool):
        """Test scraping execution with large content"""
        # Create large markdown content
        large_content = "# Large Page\n\n" + "This is a large page. " * 1000
        
        mock_response_data = {
            "success": True,
            "data": {
                "markdown": large_content,
                "html": f"<html><body><p>{large_content}</p></body></html>"
            }
        }
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_response_data
            mock_response.status = 200
            
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            result = await firecrawl_tool.execute(url="https://example.com/large")
            
            assert result["success"] is True
            assert len(result["data"]["markdown"]) > 10000  # Verify large content was handled

    @pytest.mark.asyncio
    async def test_execute_with_unicode_content(self, firecrawl_tool):
        """Test scraping execution with unicode content"""
        mock_response_data = {
            "success": True,
            "data": {
                "markdown": "# Unicode Page\n\nContenido con caracteres especiales: ñáéíóú ë ï ö ü 中文 русский العربية",
                "html": "<html><body><h1>Unicode Page</h1><p>Contenido con caracteres especiales: ñáéíóú ë ï ö ü 中文 русский العربية</p></body></html>"
            }
        }
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_response_data
            mock_response.status = 200
            
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            result = await firecrawl_tool.execute(url="https://example.com/unicode")
            
            assert result["success"] is True
            assert "ñáéíóú" in result["data"]["markdown"]
            assert "中文" in result["data"]["markdown"]
            assert "русский" in result["data"]["markdown"]
            assert "العربية" in result["data"]["markdown"]