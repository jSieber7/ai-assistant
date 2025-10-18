"""
Unit tests for the Playwright tool.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from app.core.tools.web.playwright_tool import PlaywrightTool
from app.core.tools.base.base import ToolExecutionError


class TestPlaywrightTool:
    """Test cases for PlaywrightTool"""

    @pytest.fixture
    def playwright_tool(self):
        """Create a PlaywrightTool instance for testing"""
        return PlaywrightTool(headless=True, browser_type="chromium")

    def test_tool_properties(self, playwright_tool):
        """Test basic tool properties"""
        assert playwright_tool.name == "playwright_automation"
        assert "browser automation" in playwright_tool.description.lower()
        assert "playwright" in playwright_tool.keywords
        assert "scrape" in playwright_tool.keywords

    def test_parameters(self, playwright_tool):
        """Test tool parameters"""
        params = playwright_tool.parameters

        assert "url" in params
        assert params["url"]["required"] is True
        assert params["url"]["type"] is str

        assert "actions" in params
        assert params["actions"]["type"] == List[Dict[str, Any]]
        assert params["actions"]["required"] is False

        assert "screenshot" in params
        assert params["screenshot"]["type"] is bool
        assert params["screenshot"]["default"] is False

    @pytest.mark.asyncio
    async def test_unsupported_browser_type(self):
        """Test error for unsupported browser type"""
        tool = PlaywrightTool(browser_type="unsupported")

        with patch(
            "app.core.tools.playwright_tool.async_playwright"
        ) as mock_playwright:
            mock_playwright_instance = MagicMock()
            mock_playwright.return_value = mock_playwright_instance
            mock_playwright_instance.start = AsyncMock()

            # Initialize browser and expect error
            with pytest.raises(ToolExecutionError, match="Unsupported browser type"):
                await tool._ensure_browser()

    def test_cleanup_no_browser(self, playwright_tool):
        """Test cleanup when no browser is initialized"""
        # Should not raise an error
        import asyncio

        asyncio.run(playwright_tool.cleanup())

        # Check that resources are still None
        assert playwright_tool._context is None
        assert playwright_tool._browser is None
        assert playwright_tool._playwright is None
