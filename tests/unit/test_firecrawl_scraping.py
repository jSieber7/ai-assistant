"""
Unit tests for Firecrawl web scraping functionality.

These tests verify the Firecrawl scraper tool and agent functionality
with mock Firecrawl API components.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from app.core.tools.firecrawl_tool import FirecrawlTool
from app.core.agents.firecrawl_agent import FirecrawlAgent
from app.core.config import settings


class TestFirecrawlTool:
    """Test Firecrawl scraper tool functionality"""

    @pytest.fixture
    def scraper_tool(self):
        """Create a Firecrawl scraper tool instance"""
        return FirecrawlTool()

    @pytest.fixture
    def mock_httpx(self):
        """Mock HTTPX client"""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": {
                    "markdown": "Test content",
                    "raw": "<html>Test content</html>",
                    "metadata": {
                        "title": "Test Page",
                        "description": "Test description",
                    },
                    "links": [{"url": "https://example.com", "text": "Example"}],
                    "images": [
                        {"src": "https://example.com/image.jpg", "alt": "Test image"}
                    ],
                }
            }

            mock_instance = AsyncMock()
            mock_instance.post.return_value = mock_response
            mock_client.return_value = mock_instance

            yield mock_instance

    @pytest.fixture
    def mock_httpx_error(self):
        """Mock HTTPX client with error response"""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.text = "Unauthorized"

            mock_instance = AsyncMock()
            mock_instance.post.return_value = mock_response
            mock_client.return_value = mock_instance

            yield mock_instance

    def test_tool_initialization(self, scraper_tool):
        """Test tool initialization"""
        assert scraper_tool.name == "firecrawl_scrape"
        assert "scrape" in scraper_tool.description.lower()
        assert "firecrawl" in scraper_tool.keywords

    @pytest.mark.asyncio
    async def test_execute_with_basic_scraping(self, scraper_tool, mock_httpx):
        """Test basic scraping functionality"""
        # Mock Firecrawl settings
        with patch.object(settings.firecrawl_settings, "enabled", True):
            with patch.object(settings.firecrawl_settings, "scraping_enabled", True):
                with patch.object(settings.firecrawl_settings, "extract_links", True):
                    with patch.object(
                        settings.firecrawl_settings, "extract_images", True
                    ):
                        # Mock Docker health check
                        with patch.object(
                            scraper_tool, "_check_docker_health", return_value=True
                        ):
                            result = await scraper_tool.execute(
                                url="https://example.com"
                            )

                            assert result["url"] == "https://example.com"
                            assert result["title"] == "Test Page"
                            assert "Test content" in result["content"]
                            assert result["link_count"] > 0
                            assert result["image_count"] > 0

    @pytest.mark.asyncio
    async def test_execute_with_custom_formats(self, scraper_tool, mock_httpx):
        """Test scraping with custom formats"""
        with patch.object(settings.firecrawl_settings, "enabled", True):
            with patch.object(settings.firecrawl_settings, "scraping_enabled", True):
                # Mock Docker health check
                with patch.object(
                    scraper_tool, "_check_docker_health", return_value=True
                ):
                    result = await scraper_tool.execute(
                        url="https://example.com",
                        formats=["markdown", "html"],
                        wait_for=5000,
                        screenshot=True,
                    )

                    assert result["url"] == "https://example.com"
                    assert "markdown" in result["formats"]

    @pytest.mark.asyncio
    async def test_batch_scraping(self, scraper_tool, mock_httpx):
        """Test batch scraping functionality"""
        urls = ["https://example1.com", "https://example2.com"]

        with patch.object(settings.firecrawl_settings, "enabled", True):
            with patch.object(settings.firecrawl_settings, "scraping_enabled", True):
                with patch.object(
                    settings.firecrawl_settings, "max_concurrent_scrapes", 2
                ):
                    # Mock Docker health check
                    with patch.object(
                        scraper_tool, "_check_docker_health", return_value=True
                    ):
                        results = await scraper_tool.batch_scrape(urls=urls)

                        assert len(results) == 2
                        for i, result in enumerate(results):
                            assert result["url"] == urls[i]
                            assert "content" in result

    @pytest.mark.asyncio
    async def test_api_error_handling(self, scraper_tool, mock_httpx_error):
        """Test API error handling"""
        with patch.object(settings.firecrawl_settings, "enabled", True):
            with patch.object(settings.firecrawl_settings, "scraping_enabled", True):
                with pytest.raises(Exception) as exc_info:
                    await scraper_tool.execute(url="https://example.com")

                assert "Firecrawl" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_disabled_settings(self, scraper_tool):
        """Test behavior when Firecrawl is disabled"""
        with patch.object(settings.firecrawl_settings, "scraping_enabled", False):
            with pytest.raises(Exception) as exc_info:
                await scraper_tool.execute(url="https://example.com")

            assert "not enabled" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_docker_mode_configuration(self):
        """Test Docker mode configuration"""
        with (
            patch.object(settings.firecrawl_settings, "deployment_mode", "docker"),
            patch.object(
                settings.firecrawl_settings, "docker_url", "http://firecrawl-api:3002"
            ),
            patch.object(settings.firecrawl_settings, "enabled", True),
        ):
            tool = FirecrawlTool()
            # Tool doesn't have base_url and api_key attributes anymore
            # It uses effective configuration from settings when _get_client is called
            assert tool._client is None

    @pytest.mark.asyncio
    async def test_effective_configuration_properties(self):
        """Test effective configuration properties"""
        # Test Docker mode
        with (
            patch.object(settings.firecrawl_settings, "deployment_mode", "docker"),
            patch.object(
                settings.firecrawl_settings, "docker_url", "http://firecrawl-api:3002"
            ),
        ):
            assert (
                settings.firecrawl_settings.effective_url == "http://firecrawl-api:3002"
            )
            assert settings.firecrawl_settings.effective_api_key is None


class TestFirecrawlAgent:
    """Test Firecrawl scraper agent functionality"""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for agent testing"""
        llm = AsyncMock()

        # Mock analyze response
        analyze_response = Mock()
        analyze_response.generations = [
            [
                Mock(
                    text='{"content_type": "general", "recommended_formats": ["markdown"], "extraction_targets": ["content"], "challenges": [], "recommendations": []}'
                )
            ]
        ]

        # Mock summary response
        summary_response = Mock()
        summary_response.generations = [
            [
                Mock(
                    text='{"overall_assessment": "Success", "content_quality_score": 90, "data_completeness": "Good", "issues_found": [], "recommendations": [], "key_insights": ["Test completed"]}'
                )
            ]
        ]

        llm.agenerate.side_effect = [analyze_response, summary_response]
        return llm

    @pytest.fixture
    def scraper_agent(self, mock_llm):
        """Create a Firecrawl scraper agent instance"""
        return FirecrawlAgent(llm=mock_llm, max_iterations=3)

    def test_agent_initialization(self, scraper_agent):
        """Test agent initialization"""
        assert scraper_agent.name == "firecrawl_agent"
        assert "firecrawl" in scraper_agent.description.lower()
        assert "scraping" in scraper_agent.description.lower()

    @pytest.mark.asyncio
    async def test_url_extraction(self, scraper_agent):
        """Test URL extraction from query"""
        query = "Please scrape https://example.com and also check www.test.org"
        urls = scraper_agent._extract_urls_from_query(query)

        assert len(urls) == 2
        assert "https://example.com" in urls
        assert "https://www.test.org" in urls

    @pytest.mark.asyncio
    async def test_task_analysis(self, scraper_agent, mock_llm):
        """Test scraping task analysis"""
        query = "Scrape product information from https://shop.example.com"

        analysis = await scraper_agent._analyze_scraping_task(query)

        assert "content_type" in analysis
        assert "recommended_formats" in analysis
        assert "extraction_targets" in analysis

    @pytest.mark.asyncio
    async def test_execute_with_valid_urls(self, scraper_agent, mock_llm):
        """Test agent execution with valid URLs"""
        query = "Scrape content from https://example.com"

        with patch.object(scraper_agent, "_get_scraper_tool") as mock_get_tool:
            mock_tool = AsyncMock()
            mock_tool.execute.return_value = {
                "url": "https://example.com",
                "content": "Test content",
                "title": "Test Page",
                "link_count": 0,
                "image_count": 0,
            }
            mock_get_tool.return_value = mock_tool

            result = await scraper_agent.execute(query)

            assert result["success"] is True
            assert result["total_urls"] == 1
            assert len(result["results"]) == 1

    @pytest.mark.asyncio
    async def test_execute_with_no_urls(self, scraper_agent, mock_llm):
        """Test agent execution with no URLs found"""
        query = "Scrape some content for me"

        result = await scraper_agent.execute(query)

        assert result["success"] is False
        assert "No valid URLs found" in result["error"]

    @pytest.mark.asyncio
    async def test_batch_scrape(self, scraper_agent, mock_llm):
        """Test batch scraping functionality"""
        urls = ["https://example1.com", "https://example2.com"]

        with patch.object(scraper_agent, "_get_scraper_tool") as mock_get_tool:
            mock_tool = AsyncMock()
            mock_tool.batch_scrape.return_value = [
                {
                    "url": "https://example1.com",
                    "content": "Content 1",
                    "success": True,
                },
                {
                    "url": "https://example2.com",
                    "content": "Content 2",
                    "success": True,
                },
            ]
            mock_get_tool.return_value = mock_tool

            result = await scraper_agent.batch_scrape(urls)

            assert result["success"] is True
            assert result["total_urls"] == 2
            assert result["successful"] == 2
            assert result["failed"] == 0

    @pytest.mark.asyncio
    async def test_process_message_impl(self, scraper_agent, mock_llm):
        """Test internal message processing"""
        message = "Scrape https://example.com"

        with patch.object(scraper_agent, "execute") as mock_execute:
            mock_execute.return_value = {
                "success": True,
                "summary": {"overall_assessment": "Success"},
                "total_urls": 1,
            }

            result = await scraper_agent._process_message_impl(message)

            assert result.success is True
            assert "Success" in result.response
            assert result.agent_name == "firecrawl_agent"

    def test_get_agent_stats(self, scraper_agent):
        """Test agent statistics"""
        stats = scraper_agent.get_agent_stats()

        assert "specialization" in stats
        assert stats["specialization"] == "firecrawl_web_scraping"
        assert "scraper_tool_available" in stats


class TestFirecrawlIntegration:
    """Test Firecrawl integration scenarios"""

    @pytest.mark.asyncio
    async def test_firecrawl_initialization(self):
        """Test Firecrawl initialization with Docker mode"""
        from app.core.config import initialize_firecrawl_system
        from app.core.tools.registry import tool_registry
        from app.core.agents.registry import agent_registry

        # Mock Firecrawl settings
        with (
            patch.object(settings.firecrawl_settings, "enabled", True),
            patch.object(settings.firecrawl_settings, "deployment_mode", "docker"),
        ):
            # Mock tool and agent registries
            with (
                patch.object(tool_registry, "register"),
                patch.object(agent_registry, "register"),
            ):
                # Mock LLM
                with patch("app.core.config.get_llm") as mock_get_llm:
                    mock_get_llm.return_value = AsyncMock()

                    # This should not raise an exception
                    initialize_firecrawl_system()

    @pytest.mark.asyncio
    async def test_firecrawl_disabled(self):
        """Test behavior when Firecrawl is disabled"""
        from app.core.config import initialize_firecrawl_system

        with patch.object(settings.firecrawl_settings, "enabled", False):
            # Should return without initializing Firecrawl
            result = initialize_firecrawl_system()
            assert result is None


@pytest.mark.integration
class TestFirecrawlScrapingIntegration:
    """Integration tests for Firecrawl scraping (requires external services)"""

    @pytest.mark.skip(reason="Requires Firecrawl API key")
    @pytest.mark.asyncio
    async def test_live_firecrawl_scraping(self):
        """Test actual Firecrawl scraping (requires valid API key)"""
        # This test requires actual Firecrawl credentials
        # and should be run in a controlled environment
        pytest.skip("Requires Firecrawl API key")
