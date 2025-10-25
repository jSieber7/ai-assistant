"""
Unit tests for Standalone Scraper Tool.

This module tests the standalone web scraping tool that doesn't require external services.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import uuid
from typing import List, Dict, Any, Optional, Tuple
import json
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re

from app.core.tools.web.standalone_scraper import StandaloneScraper


class TestStandaloneScraper:
    """Test StandaloneScraper class"""

    @pytest.fixture
    def standalone_scraper(self):
        """Create a standalone scraper instance"""
        return StandaloneScraper()

    def test_tool_init(self, standalone_scraper):
        """Test StandaloneScraper initialization"""
        assert standalone_scraper.name == "standalone_scraper"
        assert standalone_scraper.description == "Scrape web pages without external dependencies"
        assert isinstance(standalone_scraper.keywords, list)
        assert "scrape" in standalone_scraper.keywords
        assert standalone_scraper.category == "web"

    def test_tool_keywords(self, standalone_scraper):
        """Test StandaloneScraper keywords"""
        keywords = standalone_scraper.keywords
        
        assert isinstance(keywords, list)
        assert "scrape" in keywords
        assert "web" in keywords
        assert "extract" in keywords
        assert "html" in keywords
        assert "content" in keywords

    def test_tool_parameters(self, standalone_scraper):
        """Test StandaloneScraper parameters"""
        parameters = standalone_scraper.parameters
        
        assert isinstance(parameters, dict)
        assert "url" in parameters
        assert "format" in parameters
        assert "selector" in parameters
        assert "remove_tags" in parameters
        assert "timeout" in parameters
        assert "headers" in parameters
        
        # Check url parameter
        url_param = parameters["url"]
        assert url_param["type"] == "string"
        assert url_param["required"] is True
        assert "description" in url_param
        
        # Check format parameter
        format_param = parameters["format"]
        assert format_param["type"] == "string"
        assert format_param["required"] is False
        assert "text" in format_param["enum"]
        assert "markdown" in format_param["enum"]
        assert "html" in format_param["enum"]
        assert "json" in format_param["enum"]

    def test_tool_properties(self, standalone_scraper):
        """Test StandaloneScraper properties"""
        assert standalone_scraper.name == "standalone_scraper"
        assert standalone_scraper.description == "Scrape web pages without external dependencies"
        assert standalone_scraper.category == "web"
        assert standalone_scraper.version == "1.0.0"
        assert standalone_scraper.requires_api_key is False
        assert standalone_scraper.requires_auth is False

    @pytest.mark.asyncio
    async def test_execute_basic_scrape_text(self, standalone_scraper):
        """Test basic scraping with text format"""
        html_content = """
        <html>
            <head>
                <title>Test Page</title>
                <meta name="description" content="Test description">
            </head>
            <body>
                <h1>Main Title</h1>
                <p>This is a paragraph of text.</p>
                <div class="content">
                    <p>More content here.</p>
                </div>
            </body>
        </html>
        """
        
        mock_response = AsyncMock()
        mock_response.text.return_value = html_content
        mock_response.status = 200
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            result = await standalone_scraper.execute(
                url="https://example.com",
                format="text"
            )
            
            assert isinstance(result, dict)
            assert "success" in result
            assert "url" in result
            assert "content" in result
            assert "format" in result
            assert "metadata" in result
            
            assert result["success"] is True
            assert result["url"] == "https://example.com"
            assert result["format"] == "text"
            assert "Main Title" in result["content"]
            assert "This is a paragraph" in result["content"]
            
            # Check metadata
            metadata = result["metadata"]
            assert metadata["title"] == "Test Page"
            assert metadata["description"] == "Test description"

    @pytest.mark.asyncio
    async def test_execute_scrape_markdown(self, standalone_scraper):
        """Test scraping with markdown format"""
        html_content = """
        <html>
            <body>
                <h1>Markdown Test</h1>
                <h2>Subtitle</h2>
                <p>This is a <strong>paragraph</strong> with <em>emphasis</em>.</p>
                <ul>
                    <li>Item 1</li>
                    <li>Item 2</li>
                </ul>
                <a href="https://example.com">Link</a>
            </body>
        </html>
        """
        
        mock_response = AsyncMock()
        mock_response.text.return_value = html_content
        mock_response.status = 200
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            result = await standalone_scraper.execute(
                url="https://example.com",
                format="markdown"
            )
            
            assert result["success"] is True
            assert result["format"] == "markdown"
            
            content = result["content"]
            assert "# Markdown Test" in content
            assert "## Subtitle" in content
            assert "**paragraph**" in content
            assert "*emphasis*" in content
            assert "- Item 1" in content
            assert "[Link](https://example.com)" in content

    @pytest.mark.asyncio
    async def test_execute_scrape_html(self, standalone_scraper):
        """Test scraping with HTML format"""
        html_content = """
        <html>
            <body>
                <h1>HTML Test</h1>
                <p>Original HTML content.</p>
            </body>
        </html>
        """
        
        mock_response = AsyncMock()
        mock_response.text.return_value = html_content
        mock_response.status = 200
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            result = await standalone_scraper.execute(
                url="https://example.com",
                format="html"
            )
            
            assert result["success"] is True
            assert result["format"] == "html"
            assert "HTML Test" in result["content"]
            assert "Original HTML content" in result["content"]

    @pytest.mark.asyncio
    async def test_execute_scrape_json(self, standalone_scraper):
        """Test scraping with JSON format"""
        html_content = """
        <html>
            <body>
                <h1>JSON Test</h1>
                <div class="article">
                    <h2>Article Title</h2>
                    <p>Article content goes here.</p>
                    <span class="author">John Doe</span>
                </div>
            </body>
        </html>
        """
        
        mock_response = AsyncMock()
        mock_response.text.return_value = html_content
        mock_response.status = 200
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            result = await standalone_scraper.execute(
                url="https://example.com",
                format="json"
            )
            
            assert result["success"] is True
            assert result["format"] == "json"
            
            # Parse the JSON content
            json_content = json.loads(result["content"])
            assert "title" in json_content
            assert "content" in json_content
            assert "metadata" in json_content
            assert json_content["title"] == "JSON Test"
            assert "Article Title" in json_content["content"]
            assert "John Doe" in json_content["content"]

    @pytest.mark.asyncio
    async def test_execute_with_selector(self, standalone_scraper):
        """Test scraping with CSS selector"""
        html_content = """
        <html>
            <body>
                <div class="main-content">
                    <h1>Main Title</h1>
                    <p>Main content paragraph.</p>
                </div>
                <div class="sidebar">
                    <p>Sidebar content.</p>
                </div>
            </body>
        </html>
        """
        
        mock_response = AsyncMock()
        mock_response.text.return_value = html_content
        mock_response.status = 200
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            result = await standalone_scraper.execute(
                url="https://example.com",
                format="text",
                selector=".main-content"
            )
            
            assert result["success"] is True
            assert "Main Title" in result["content"]
            assert "Main content paragraph" in result["content"]
            assert "Sidebar content" not in result["content"]  # Should be excluded

    @pytest.mark.asyncio
    async def test_execute_with_remove_tags(self, standalone_scraper):
        """Test scraping with tag removal"""
        html_content = """
        <html>
            <body>
                <h1>Test Page</h1>
                <p>Content paragraph.</p>
                <script>alert('test');</script>
                <style>body { color: red; }</style>
                <nav>Navigation menu</nav>
                <footer>Footer content</footer>
                <main>Main content</main>
            </body>
        </html>
        """
        
        mock_response = AsyncMock()
        mock_response.text.return_value = html_content
        mock_response.status = 200
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            result = await standalone_scraper.execute(
                url="https://example.com",
                format="text",
                remove_tags=["script", "style", "nav", "footer"]
            )
            
            assert result["success"] is True
            assert "Test Page" in result["content"]
            assert "Content paragraph" in result["content"]
            assert "Main content" in result["content"]
            assert "alert('test');" not in result["content"]  # Script should be removed
            assert "color: red" not in result["content"]      # Style should be removed
            assert "Navigation menu" not in result["content"]  # Nav should be removed
            assert "Footer content" not in result["content"]   # Footer should be removed

    @pytest.mark.asyncio
    async def test_execute_with_custom_headers(self, standalone_scraper):
        """Test scraping with custom headers"""
        html_content = "<html><body><h1>Test with headers</h1></body></html>"
        
        mock_response = AsyncMock()
        mock_response.text.return_value = html_content
        mock_response.status = 200
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            custom_headers = {
                "User-Agent": "Mozilla/5.0 (Custom Bot)",
                "Accept-Language": "en-US,en;q=0.9"
            }
            
            result = await standalone_scraper.execute(
                url="https://example.com",
                headers=custom_headers
            )
            
            assert result["success"] is True
            
            # Check that headers were passed
            call_args = mock_session.get.call_args
            assert call_args[1]["headers"] == custom_headers

    @pytest.mark.asyncio
    async def test_execute_with_timeout(self, standalone_scraper):
        """Test scraping with custom timeout"""
        html_content = "<html><body><h1>Timeout test</h1></body></html>"
        
        mock_response = AsyncMock()
        mock_response.text.return_value = html_content
        mock_response.status = 200
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            result = await standalone_scraper.execute(
                url="https://example.com",
                timeout=30
            )
            
            assert result["success"] is True
            
            # Check that timeout was passed
            call_args = mock_session.get.call_args
            assert call_args[1]["timeout"] == 30

    @pytest.mark.asyncio
    async def test_execute_http_error(self, standalone_scraper):
        """Test scraping with HTTP error"""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 404
            mock_response.text.return_value = "Not Found"
            
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            with pytest.raises(Exception, match="Failed to fetch URL: 404"):
                await standalone_scraper.execute(url="https://example.com/not-found")

    @pytest.mark.asyncio
    async def test_execute_network_error(self, standalone_scraper):
        """Test scraping with network error"""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get.side_effect = aiohttp.ClientError("Network error")
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            with pytest.raises(Exception, match="Network error"):
                await standalone_scraper.execute(url="https://example.com")

    @pytest.mark.asyncio
    async def test_execute_with_invalid_url(self, standalone_scraper):
        """Test scraping with invalid URL"""
        with pytest.raises(ValueError, match="Invalid URL format"):
            await standalone_scraper.execute(url="not-a-valid-url")

    @pytest.mark.asyncio
    async def test_execute_with_empty_url(self, standalone_scraper):
        """Test scraping with empty URL"""
        with pytest.raises(ValueError, match="URL parameter is required"):
            await standalone_scraper.execute(url="")

    @pytest.mark.asyncio
    async def test_execute_with_missing_url(self, standalone_scraper):
        """Test scraping with missing URL"""
        with pytest.raises(ValueError, match="URL parameter is required"):
            await standalone_scraper.execute()

    def test_should_use_with_scrape_keywords(self, standalone_scraper):
        """Test should_use method with scrape keywords"""
        assert standalone_scraper.should_use("scrape the website") is True
        assert standalone_scraper.should_use("extract content from page") is True
        assert standalone_scraper.should_use("get web page content") is True

    def test_should_use_without_scrape_keywords(self, standalone_scraper):
        """Test should_use method without scrape keywords"""
        assert standalone_scraper.should_use("create a file") is False
        assert standalone_scraper.should_use("calculate something") is False
        assert standalone_scraper.should_use("send an email") is False

    def test_should_use_with_url(self, standalone_scraper):
        """Test should_use method with URL in text"""
        assert standalone_scraper.should_use("scrape https://example.com") is True
        assert standalone_scraper.should_use("extract content from http://website.org/page") is True

    def test_validate_parameters_valid(self, standalone_scraper):
        """Test validate_parameters with valid parameters"""
        assert standalone_scraper.validate_parameters(url="https://example.com") is True
        assert standalone_scraper.validate_parameters(
            url="https://example.com",
            format="markdown",
            selector=".content"
        ) is True

    def test_validate_parameters_missing_required(self, standalone_scraper):
        """Test validate_parameters with missing required parameter"""
        assert standalone_scraper.validate_parameters() is False
        assert standalone_scraper.validate_parameters(url="") is False

    def test_validate_parameters_invalid_types(self, standalone_scraper):
        """Test validate_parameters with invalid parameter types"""
        # Invalid timeout type
        assert standalone_scraper.validate_parameters(url="https://example.com", timeout="30") is False
        
        # Invalid format
        assert standalone_scraper.validate_parameters(url="https://example.com", format="invalid") is False

    @pytest.mark.asyncio
    async def test_execute_with_timeout_wrapper(self, standalone_scraper):
        """Test execute_with_timeout method"""
        html_content = "<html><body><h1>Timeout test</h1></body></html>"
        
        mock_response = AsyncMock()
        mock_response.text.return_value = html_content
        mock_response.status = 200
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            result = await standalone_scraper.execute_with_timeout(
                url="https://example.com",
                timeout=5.0
            )
            
            assert result.success is True
            assert result.data["success"] is True
            assert "Timeout test" in result.data["content"]

    @pytest.mark.asyncio
    async def test_get_usage_stats(self, standalone_scraper):
        """Test get_usage_stats method"""
        # Initially should have no stats
        stats = standalone_scraper.get_usage_stats()
        assert stats["execution_count"] == 0
        assert stats["total_execution_time"] == 0.0
        
        # Execute a scrape to generate stats
        html_content = "<html><body><h1>Test</h1></body></html>"
        
        mock_response = AsyncMock()
        mock_response.text.return_value = html_content
        mock_response.status = 200
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            await standalone_scraper.execute(url="https://example.com")
            
            # Check stats after execution
            stats = standalone_scraper.get_usage_stats()
            assert stats["execution_count"] == 1
            assert stats["total_execution_time"] > 0

    def test_tool_category_and_version(self, standalone_scraper):
        """Test tool category and version"""
        assert standalone_scraper.category == "web"
        assert standalone_scraper.version == "1.0.0"

    def test_tool_auth_requirements(self, standalone_scraper):
        """Test tool authentication requirements"""
        assert standalone_scraper.requires_api_key is False
        assert standalone_scraper.requires_auth is False

    @pytest.mark.asyncio
    async def test_extract_links_from_content(self, standalone_scraper):
        """Test link extraction from content"""
        html_content = """
        <html>
            <body>
                <a href="https://example.com/page1">Page 1</a>
                <a href="/relative-page">Relative Page</a>
                <a href="#section">Anchor Link</a>
                <a href="mailto:test@example.com">Email</a>
            </body>
        </html>
        """
        
        mock_response = AsyncMock()
        mock_response.text.return_value = html_content
        mock_response.status = 200
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            result = await standalone_scraper.execute(
                url="https://example.com",
                format="json"
            )
            
            assert result["success"] is True
            
            # Parse JSON and check links
            json_content = json.loads(result["content"])
            assert "links" in json_content
            
            links = json_content["links"]
            assert any(link["url"] == "https://example.com/page1" for link in links)
            assert any(link["url"] == "https://example.com/relative-page" for link in links)
            # Anchor and email links should be filtered out

    @pytest.mark.asyncio
    async def test_extract_images_from_content(self, standalone_scraper):
        """Test image extraction from content"""
        html_content = """
        <html>
            <body>
                <img src="https://example.com/image1.jpg" alt="Image 1">
                <img src="/relative-image.png" alt="Relative Image">
                <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==" alt="Base64 Image">
            </body>
        </html>
        """
        
        mock_response = AsyncMock()
        mock_response.text.return_value = html_content
        mock_response.status = 200
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            result = await standalone_scraper.execute(
                url="https://example.com",
                format="json"
            )
            
            assert result["success"] is True
            
            # Parse JSON and check images
            json_content = json.loads(result["content"])
            assert "images" in json_content
            
            images = json_content["images"]
            assert any(img["src"] == "https://example.com/image1.jpg" for img in images)
            assert any(img["src"] == "https://example.com/relative-image.png" for img in images)
            # Base64 images should be included

    @pytest.mark.asyncio
    async def test_handle_unicode_content(self, standalone_scraper):
        """Test handling of unicode content"""
        html_content = """
        <html>
            <body>
                <h1>Unicode Test</h1>
                <p>Content with unicode: ñáéíóú ë ï ö ü 中文 русский العربية</p>
            </body>
        </html>
        """
        
        mock_response = AsyncMock()
        mock_response.text.return_value = html_content
        mock_response.status = 200
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            result = await standalone_scraper.execute(
                url="https://example.com",
                format="text"
            )
            
            assert result["success"] is True
            assert "ñáéíóú" in result["content"]
            assert "中文" in result["content"]
            assert "русский" in result["content"]
            assert "العربية" in result["content"]

    @pytest.mark.asyncio
    async def test_handle_large_content(self, standalone_scraper):
        """Test handling of large content"""
        # Create large HTML content
        large_content = "<html><body>"
        for i in range(1000):
            large_content += f"<p>This is paragraph {i} with some content.</p>"
        large_content += "</body></html>"
        
        mock_response = AsyncMock()
        mock_response.text.return_value = large_content
        mock_response.status = 200
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            result = await standalone_scraper.execute(
                url="https://example.com/large",
                format="text"
            )
            
            assert result["success"] is True
            assert len(result["content"]) > 50000  # Verify large content was handled
            assert "paragraph 500" in result["content"]  # Verify middle content is present

    @pytest.mark.asyncio
    async def test_handle_malformed_html(self, standalone_scraper):
        """Test handling of malformed HTML"""
        html_content = """
        <html>
            <body>
                <h1>Malformed HTML
                <p>Unclosed paragraph
                <div>Nested div without closing
                <span>More content
            </body>
        """
        
        mock_response = AsyncMock()
        mock_response.text.return_value = html_content
        mock_response.status = 200
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            result = await standalone_scraper.execute(
                url="https://example.com/malformed",
                format="text"
            )
            
            assert result["success"] is True
            assert "Malformed HTML" in result["content"]
            assert "Unclosed paragraph" in result["content"]
            # BeautifulSoup should handle malformed HTML gracefully