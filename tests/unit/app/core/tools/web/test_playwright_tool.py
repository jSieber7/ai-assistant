"""
Unit tests for Playwright Tool.

This module tests the Playwright browser automation tool.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import uuid
from typing import List, Dict, Any, Optional, Tuple
import json
import aiohttp
from pathlib import Path

from app.core.tools.web.playwright_tool import PlaywrightTool


class TestPlaywrightTool:
    """Test PlaywrightTool class"""

    @pytest.fixture
    def playwright_tool(self):
        """Create a Playwright tool instance"""
        return PlaywrightTool()

    def test_tool_init(self, playwright_tool):
        """Test PlaywrightTool initialization"""
        assert playwright_tool.name == "playwright_automation"
        assert playwright_tool.description == "Automate browser interactions using Playwright"
        assert isinstance(playwright_tool.keywords, list)
        assert "browser" in playwright_tool.keywords
        assert playwright_tool.category == "web"
        assert playwright_tool.browser_type == "chromium"  # Default browser

    def test_tool_init_with_browser_type(self):
        """Test PlaywrightTool initialization with custom browser type"""
        tool = PlaywrightTool(browser_type="firefox")
        assert tool.browser_type == "firefox"
        
        tool = PlaywrightTool(browser_type="webkit")
        assert tool.browser_type == "webkit"

    def test_tool_init_with_invalid_browser_type(self):
        """Test PlaywrightTool initialization with invalid browser type"""
        with pytest.raises(ValueError, match="Invalid browser type"):
            PlaywrightTool(browser_type="invalid")

    def test_tool_keywords(self, playwright_tool):
        """Test PlaywrightTool keywords"""
        keywords = playwright_tool.keywords
        
        assert isinstance(keywords, list)
        assert "browser" in keywords
        assert "automation" in keywords
        assert "playwright" in keywords
        assert "scraping" in keywords
        assert "screenshot" in keywords

    def test_tool_parameters(self, playwright_tool):
        """Test PlaywrightTool parameters"""
        parameters = playwright_tool.parameters
        
        assert isinstance(parameters, dict)
        assert "action" in parameters
        assert "url" in parameters
        assert "selector" in parameters
        assert "text" in parameters
        assert "wait_for" in parameters
        assert "timeout" in parameters
        assert "screenshot" in parameters
        
        # Check action parameter
        action_param = parameters["action"]
        assert action_param["type"] == "string"
        assert action_param["required"] is True
        assert "navigate" in action_param["enum"]
        assert "click" in action_param["enum"]
        assert "type" in action_param["enum"]
        assert "screenshot" in action_param["enum"]
        
        # Check optional parameters
        url_param = parameters["url"]
        assert url_param["type"] == "string"
        assert url_param["required"] is False

    def test_tool_properties(self, playwright_tool):
        """Test PlaywrightTool properties"""
        assert playwright_tool.name == "playwright_automation"
        assert playwright_tool.description == "Automate browser interactions using Playwright"
        assert playwright_tool.category == "web"
        assert playwright_tool.version == "1.0.0"
        assert playwright_tool.requires_api_key is False
        assert playwright_tool.requires_auth is False

    @pytest.mark.asyncio
    async def test_execute_navigate_action(self, playwright_tool):
        """Test navigate action execution"""
        mock_page = AsyncMock()
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.goto.return_value = None
        mock_page.title.return_value = "Test Page"
        mock_page.url.return_value = "https://example.com"
        
        with patch('playwright.async_api.async_playwright') as mock_playwright:
            mock_playwright_instance = AsyncMock()
            mock_playwright.return_value.__aenter__.return_value = mock_playwright_instance
            mock_playwright_instance.chromium.launch.return_value = mock_browser
            
            result = await playwright_tool.execute(
                action="navigate",
                url="https://example.com"
            )
            
            assert isinstance(result, dict)
            assert "success" in result
            assert "action" in result
            assert "url" in result
            assert "title" in result
            
            assert result["success"] is True
            assert result["action"] == "navigate"
            assert result["url"] == "https://example.com"
            assert result["title"] == "Test Page"
            
            mock_page.goto.assert_called_once_with("https://example.com")

    @pytest.mark.asyncio
    async def test_execute_click_action(self, playwright_tool):
        """Test click action execution"""
        mock_page = AsyncMock()
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.click.return_value = None
        mock_page.title.return_value = "Test Page"
        
        with patch('playwright.async_api.async_playwright') as mock_playwright:
            mock_playwright_instance = AsyncMock()
            mock_playwright.return_value.__aenter__.return_value = mock_playwright_instance
            mock_playwright_instance.chromium.launch.return_value = mock_browser
            
            result = await playwright_tool.execute(
                action="click",
                selector="#submit-button",
                url="https://example.com"
            )
            
            assert result["success"] is True
            assert result["action"] == "click"
            assert result["selector"] == "#submit-button"
            
            mock_page.goto.assert_called_once_with("https://example.com")
            mock_page.click.assert_called_once_with("#submit-button")

    @pytest.mark.asyncio
    async def test_execute_type_action(self, playwright_tool):
        """Test type action execution"""
        mock_page = AsyncMock()
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.type.return_value = None
        mock_page.title.return_value = "Test Page"
        
        with patch('playwright.async_api.async_playwright') as mock_playwright:
            mock_playwright_instance = AsyncMock()
            mock_playwright.return_value.__aenter__.return_value = mock_playwright_instance
            mock_playwright_instance.chromium.launch.return_value = mock_browser
            
            result = await playwright_tool.execute(
                action="type",
                selector="#search-input",
                text="search query",
                url="https://example.com"
            )
            
            assert result["success"] is True
            assert result["action"] == "type"
            assert result["selector"] == "#search-input"
            assert result["text"] == "search query"
            
            mock_page.goto.assert_called_once_with("https://example.com")
            mock_page.type.assert_called_once_with("#search-input", "search query")

    @pytest.mark.asyncio
    async def test_execute_screenshot_action(self, playwright_tool):
        """Test screenshot action execution"""
        mock_page = AsyncMock()
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        
        # Mock screenshot data
        screenshot_data = b"fake-screenshot-data"
        
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.screenshot.return_value = screenshot_data
        mock_page.title.return_value = "Test Page"
        
        with patch('playwright.async_api.async_playwright') as mock_playwright:
            mock_playwright_instance = AsyncMock()
            mock_playwright.return_value.__aenter__.return_value = mock_playwright_instance
            mock_playwright_instance.chromium.launch.return_value = mock_browser
            
            with patch('base64.b64encode') as mock_b64:
                mock_b64.return_value = b"ZmFrZS1zY3JlZW5zaG90LWRhdGE="
                
                result = await playwright_tool.execute(
                    action="screenshot",
                    url="https://example.com"
                )
            
            assert result["success"] is True
            assert result["action"] == "screenshot"
            assert "screenshot" in result
            assert result["screenshot"] == "ZmFrZS1zY3JlZW5zaG90LWRhdGE="
            
            mock_page.goto.assert_called_once_with("https://example.com")
            mock_page.screenshot.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_get_text_action(self, playwright_tool):
        """Test get_text action execution"""
        mock_page = AsyncMock()
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_element = AsyncMock()
        
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.query_selector.return_value = mock_element
        mock_element.text_content.return_value = "Extracted text content"
        mock_page.title.return_value = "Test Page"
        
        with patch('playwright.async_api.async_playwright') as mock_playwright:
            mock_playwright_instance = AsyncMock()
            mock_playwright.return_value.__aenter__.return_value = mock_playwright_instance
            mock_playwright_instance.chromium.launch.return_value = mock_browser
            
            result = await playwright_tool.execute(
                action="get_text",
                selector="#content",
                url="https://example.com"
            )
            
            assert result["success"] is True
            assert result["action"] == "get_text"
            assert result["selector"] == "#content"
            assert result["text"] == "Extracted text content"
            
            mock_page.goto.assert_called_once_with("https://example.com")
            mock_page.query_selector.assert_called_once_with("#content")
            mock_element.text_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_wait_for_action(self, playwright_tool):
        """Test wait_for action execution"""
        mock_page = AsyncMock()
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.wait_for_selector.return_value = None
        mock_page.title.return_value = "Test Page"
        
        with patch('playwright.async_api.async_playwright') as mock_playwright:
            mock_playwright_instance = AsyncMock()
            mock_playwright.return_value.__aenter__.return_value = mock_playwright_instance
            mock_playwright_instance.chromium.launch.return_value = mock_browser
            
            result = await playwright_tool.execute(
                action="wait_for",
                selector="#dynamic-content",
                url="https://example.com"
            )
            
            assert result["success"] is True
            assert result["action"] == "wait_for"
            assert result["selector"] == "#dynamic-content"
            
            mock_page.goto.assert_called_once_with("https://example.com")
            mock_page.wait_for_selector.assert_called_once_with("#dynamic-content")

    @pytest.mark.asyncio
    async def test_execute_with_timeout(self, playwright_tool):
        """Test execute action with custom timeout"""
        mock_page = AsyncMock()
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.goto.return_value = None
        mock_page.title.return_value = "Test Page"
        
        with patch('playwright.async_api.async_playwright') as mock_playwright:
            mock_playwright_instance = AsyncMock()
            mock_playwright.return_value.__aenter__.return_value = mock_playwright_instance
            mock_playwright_instance.chromium.launch.return_value = mock_browser
            
            result = await playwright_tool.execute(
                action="navigate",
                url="https://example.com",
                timeout=30000  # 30 seconds
            )
            
            assert result["success"] is True
            
            # Check that timeout was passed to page methods
            mock_page.goto.assert_called_once_with("https://example.com", timeout=30000)

    @pytest.mark.asyncio
    async def test_execute_with_invalid_action(self, playwright_tool):
        """Test execute with invalid action"""
        with pytest.raises(ValueError, match="Invalid action"):
            await playwright_tool.execute(
                action="invalid_action",
                url="https://example.com"
            )

    @pytest.mark.asyncio
    async def test_execute_with_missing_action(self, playwright_tool):
        """Test execute with missing action"""
        with pytest.raises(ValueError, match="Action parameter is required"):
            await playwright_tool.execute(url="https://example.com")

    @pytest.mark.asyncio
    async def test_execute_browser_error(self, playwright_tool):
        """Test execute with browser error"""
        with patch('playwright.async_api.async_playwright') as mock_playwright:
            mock_playwright_instance = AsyncMock()
            mock_playwright.return_value.__aenter__.return_value = mock_playwright_instance
            mock_playwright_instance.chromium.launch.side_effect = Exception("Browser launch failed")
            
            with pytest.raises(Exception, match="Browser launch failed"):
                await playwright_tool.execute(
                    action="navigate",
                    url="https://example.com"
                )

    @pytest.mark.asyncio
    async def test_execute_page_error(self, playwright_tool):
        """Test execute with page error"""
        mock_page = AsyncMock()
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.goto.side_effect = Exception("Navigation failed")
        
        with patch('playwright.async_api.async_playwright') as mock_playwright:
            mock_playwright_instance = AsyncMock()
            mock_playwright.return_value.__aenter__.return_value = mock_playwright_instance
            mock_playwright_instance.chromium.launch.return_value = mock_browser
            
            with pytest.raises(Exception, match="Navigation failed"):
                await playwright_tool.execute(
                    action="navigate",
                    url="https://example.com"
                )

    def test_should_use_with_browser_keywords(self, playwright_tool):
        """Test should_use method with browser keywords"""
        assert playwright_tool.should_use("automate browser") is True
        assert playwright_tool.should_use("take screenshot") is True
        assert playwright_tool.should_use("click on element") is True
        assert playwright_tool.should_use("fill form") is True

    def test_should_use_without_browser_keywords(self, playwright_tool):
        """Test should_use method without browser keywords"""
        assert playwright_tool.should_use("create a file") is False
        assert playwright_tool.should_use("calculate something") is False
        assert playwright_tool.should_use("send an email") is False

    def test_validate_parameters_valid(self, playwright_tool):
        """Test validate_parameters with valid parameters"""
        assert playwright_tool.validate_parameters(action="navigate", url="https://example.com") is True
        assert playwright_tool.validate_parameters(
            action="click",
            selector="#button",
            url="https://example.com"
        ) is True

    def test_validate_parameters_missing_required(self, playwright_tool):
        """Test validate_parameters with missing required parameter"""
        assert playwright_tool.validate_parameters() is False
        assert playwright_tool.validate_parameters(action="") is False

    def test_validate_parameters_invalid_types(self, playwright_tool):
        """Test validate_parameters with invalid parameter types"""
        # Invalid timeout type
        assert playwright_tool.validate_parameters(
            action="navigate",
            url="https://example.com",
            timeout="30000"
        ) is False

    @pytest.mark.asyncio
    async def test_execute_with_timeout_wrapper(self, playwright_tool):
        """Test execute_with_timeout method"""
        mock_page = AsyncMock()
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.goto.return_value = None
        mock_page.title.return_value = "Timeout Test Page"
        
        with patch('playwright.async_api.async_playwright') as mock_playwright:
            mock_playwright_instance = AsyncMock()
            mock_playwright.return_value.__aenter__.return_value = mock_playwright_instance
            mock_playwright_instance.chromium.launch.return_value = mock_browser
            
            result = await playwright_tool.execute_with_timeout(
                action="navigate",
                url="https://example.com",
                timeout=5.0
            )
            
            assert result.success is True
            assert result.data["success"] is True
            assert result.data["action"] == "navigate"
            assert result.data["title"] == "Timeout Test Page"

    @pytest.mark.asyncio
    async def test_get_usage_stats(self, playwright_tool):
        """Test get_usage_stats method"""
        # Initially should have no stats
        stats = playwright_tool.get_usage_stats()
        assert stats["execution_count"] == 0
        assert stats["total_execution_time"] == 0.0
        
        # Execute an action to generate stats
        mock_page = AsyncMock()
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.goto.return_value = None
        mock_page.title.return_value = "Test Page"
        
        with patch('playwright.async_api.async_playwright') as mock_playwright:
            mock_playwright_instance = AsyncMock()
            mock_playwright.return_value.__aenter__.return_value = mock_playwright_instance
            mock_playwright_instance.chromium.launch.return_value = mock_browser
            
            await playwright_tool.execute(
                action="navigate",
                url="https://example.com"
            )
            
            # Check stats after execution
            stats = playwright_tool.get_usage_stats()
            assert stats["execution_count"] == 1
            assert stats["total_execution_time"] > 0

    def test_tool_category_and_version(self, playwright_tool):
        """Test tool category and version"""
        assert playwright_tool.category == "web"
        assert playwright_tool.version == "1.0.0"

    def test_tool_auth_requirements(self, playwright_tool):
        """Test tool authentication requirements"""
        assert playwright_tool.requires_api_key is False
        assert playwright_tool.requires_auth is False

    @pytest.mark.asyncio
    async def test_execute_with_headless_option(self, playwright_tool):
        """Test execute with headless option"""
        mock_page = AsyncMock()
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.goto.return_value = None
        mock_page.title.return_value = "Test Page"
        
        with patch('playwright.async_api.async_playwright') as mock_playwright:
            mock_playwright_instance = AsyncMock()
            mock_playwright.return_value.__aenter__.return_value = mock_playwright_instance
            mock_playwright_instance.chromium.launch.return_value = mock_browser
            
            # Test with headless=True (default)
            result = await playwright_tool.execute(
                action="navigate",
                url="https://example.com",
                headless=True
            )
            
            assert result["success"] is True
            mock_playwright_instance.chromium.launch.assert_called_with(headless=True)
            
            # Test with headless=False
            result = await playwright_tool.execute(
                action="navigate",
                url="https://example.com",
                headless=False
            )
            
            assert result["success"] is True
            mock_playwright_instance.chromium.launch.assert_called_with(headless=False)

    @pytest.mark.asyncio
    async def test_execute_with_viewport_option(self, playwright_tool):
        """Test execute with viewport option"""
        mock_page = AsyncMock()
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.goto.return_value = None
        mock_page.title.return_value = "Test Page"
        
        with patch('playwright.async_api.async_playwright') as mock_playwright:
            mock_playwright_instance = AsyncMock()
            mock_playwright.return_value.__aenter__.return_value = mock_playwright_instance
            mock_playwright_instance.chromium.launch.return_value = mock_browser
            
            result = await playwright_tool.execute(
                action="navigate",
                url="https://example.com",
                viewport={"width": 1920, "height": 1080}
            )
            
            assert result["success"] is True
            mock_browser.new_context.assert_called_with(viewport={"width": 1920, "height": 1080})

    @pytest.mark.asyncio
    async def test_execute_with_user_agent_option(self, playwright_tool):
        """Test execute with user agent option"""
        mock_page = AsyncMock()
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.goto.return_value = None
        mock_page.title.return_value = "Test Page"
        
        with patch('playwright.async_api.async_playwright') as mock_playwright:
            mock_playwright_instance = AsyncMock()
            mock_playwright.return_value.__aenter__.return_value = mock_playwright_instance
            mock_playwright_instance.chromium.launch.return_value = mock_browser
            
            result = await playwright_tool.execute(
                action="navigate",
                url="https://example.com",
                user_agent="Mozilla/5.0 (Custom User Agent)"
            )
            
            assert result["success"] is True
            mock_browser.new_context.assert_called_with(user_agent="Mozilla/5.0 (Custom User Agent)")

    @pytest.mark.asyncio
    async def test_execute_with_multiple_actions(self, playwright_tool):
        """Test execute with multiple actions in sequence"""
        mock_page = AsyncMock()
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_element = AsyncMock()
        
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.goto.return_value = None
        mock_page.type.return_value = None
        mock_page.click.return_value = None
        mock_page.query_selector.return_value = mock_element
        mock_element.text_content.return_value = "Search results"
        mock_page.title.return_value = "Search Results"
        
        with patch('playwright.async_api.async_playwright') as mock_playwright:
            mock_playwright_instance = AsyncMock()
            mock_playwright.return_value.__aenter__.return_value = mock_playwright_instance
            mock_playwright_instance.chromium.launch.return_value = mock_browser
            
            # Navigate
            result1 = await playwright_tool.execute(
                action="navigate",
                url="https://example.com"
            )
            
            # Type in search box
            result2 = await playwright_tool.execute(
                action="type",
                selector="#search",
                text="test query"
            )
            
            # Click search button
            result3 = await playwright_tool.execute(
                action="click",
                selector="#search-button"
            )
            
            # Get results
            result4 = await playwright_tool.execute(
                action="get_text",
                selector="#results"
            )
            
            assert result1["success"] is True
            assert result2["success"] is True
            assert result3["success"] is True
            assert result4["success"] is True
            assert result4["text"] == "Search results"
            
            # Verify all actions were called
            mock_page.goto.assert_called_with("https://example.com")
            mock_page.type.assert_called_with("#search", "test query")
            mock_page.click.assert_called_with("#search-button")
            mock_page.query_selector.assert_called_with("#results")