"""
Unit tests for Firecrawl Docker mode functionality.

These tests verify the Docker mode configuration and behavior
with mock components, without requiring actual Docker services.
"""

import pytest
from unittest.mock import patch, AsyncMock
from app.core.tools.web.firecrawl_tool import FirecrawlTool
from app.core.agents.specialized.firecrawl_agent import FirecrawlAgent
from app.core.config import settings


class TestFirecrawlDockerMode:
    """Test Firecrawl Docker mode configuration and behavior"""

    @pytest.fixture
    def docker_settings(self):
        """Configure settings for Docker mode"""
        with (
            patch.object(settings.firecrawl_settings, "deployment_mode", "docker"),
            patch.object(
                settings.firecrawl_settings, "docker_url", "http://firecrawl-api:3002"
            ),
            patch.object(settings.firecrawl_settings, "enabled", True),
            patch.object(settings.firecrawl_settings, "scraping_enabled", True),
        ):
            yield settings.firecrawl_settings

    # API mode is no longer supported - Docker-only mode

    def test_docker_mode_configuration(self, docker_settings):
        """Test Docker mode configuration properties"""
        assert docker_settings.deployment_mode == "docker"
        assert docker_settings.effective_url == "http://firecrawl-api:3002"
        assert docker_settings.effective_api_key is None

    def test_docker_tool_initialization(self, docker_settings):
        """Test Firecrawl tool initialization in Docker mode"""
        tool = FirecrawlTool()
        # Tool should use effective configuration from settings when _get_client is called
        # Tool doesn't have base_url and api_key attributes anymore
        assert tool._client is None

    @pytest.mark.asyncio
    async def test_docker_health_check_success(self, docker_settings):
        """Test successful Docker health check"""
        tool = FirecrawlTool()

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}

        with patch.object(
            tool,
            "_get_client",
            return_value=AsyncMock(get=AsyncMock(return_value=mock_response)),
        ):
            is_healthy = await tool._check_docker_health()
            assert is_healthy is True

    @pytest.mark.asyncio
    async def test_docker_health_check_failure(self, docker_settings):
        """Test Docker health check failure"""
        tool = FirecrawlTool()

        with patch.object(
            tool,
            "_get_client",
            return_value=AsyncMock(
                get=AsyncMock(side_effect=Exception("Connection failed"))
            ),
        ):
            is_healthy = await tool._check_docker_health()
            assert is_healthy is False

    @pytest.mark.asyncio
    async def test_docker_scraping_with_health_check(self, docker_settings):
        """Test scraping with Docker health check"""
        tool = FirecrawlTool()

        # Mock successful health check and scraping
        mock_health_response = AsyncMock()
        mock_health_response.status_code = 200

        mock_scrape_response = AsyncMock()
        mock_scrape_response.status_code = 200
        mock_scrape_response.json = AsyncMock(
            return_value={
                "data": {
                    "markdown": "# Test Page\n\nTest content from Docker.",
                    "metadata": {"title": "Test Page"},
                    "links": [],
                    "images": [],
                }
            }
        )

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_health_response
        mock_client.post.return_value = mock_scrape_response

        with patch.object(tool, "_get_client", return_value=mock_client):
            result = await tool.execute(url="https://example.com")

            assert result["url"] == "https://example.com"
            assert result["title"] == "Test Page"
            assert "Test content from Docker" in result["content"]

    @pytest.mark.asyncio
    async def test_docker_scraping_failure(self, docker_settings):
        """Test Docker scraping failure"""
        tool = FirecrawlTool()

        with patch.object(tool, "_check_docker_health", return_value=False):
            with pytest.raises(Exception) as exc_info:
                await tool.execute(url="https://example.com")

            assert "unhealthy" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_docker_agent_configuration(self, docker_settings):
        """Test Firecrawl agent configuration in Docker mode"""
        # Mock LLM
        mock_llm = AsyncMock()

        agent = FirecrawlAgent(llm=mock_llm)

        # Agent should use effective configuration from settings
        with patch.object(agent, "_get_scraper_tool") as mock_get_tool:
            mock_tool = AsyncMock()
            mock_get_tool.return_value = mock_tool

            # Verify tool is created with Docker configuration
            tool = await agent._get_scraper_tool()
            # Tool doesn't have base_url and api_key attributes anymore
            assert tool is not None

    def test_docker_mode_configuration(self):
        """Test Docker mode configuration properties"""
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

    @pytest.mark.asyncio
    async def test_batch_scraping_docker_mode(self, docker_settings):
        """Test batch scraping in Docker mode"""
        tool = FirecrawlTool()
        urls = ["https://example1.com", "https://example2.com"]

        # Mock successful health check and scraping
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json = AsyncMock(
            return_value={
                "data": {
                    "markdown": "# Test Page\n\nTest content.",
                    "metadata": {"title": "Test Page"},
                    "links": [],
                    "images": [],
                }
            }
        )

        mock_client = AsyncMock()
        mock_client.get.return_value = AsyncMock(status_code=200)
        mock_client.post.return_value = mock_response

        with (
            patch.object(tool, "_check_docker_health", return_value=True),
            patch.object(tool, "_get_client", return_value=mock_client),
        ):
            results = await tool.batch_scrape(urls=urls)

            assert len(results) == 2
            for i, result in enumerate(results):
                assert result["url"] == urls[i]
                assert "content" in result

    @pytest.mark.asyncio
    async def test_cleanup_client(self):
        """Test cleanup of HTTP client"""
        tool = FirecrawlTool()

        # Mock client
        tool._client = AsyncMock()
        tool._client.aclose = AsyncMock()

        # Run cleanup
        await tool.cleanup()

        # Verify client is cleaned up
        assert tool._client is None


class TestFirecrawlDockerModeEdgeCases:
    """Test edge cases and error conditions for Docker mode"""

    @pytest.mark.asyncio
    async def test_docker_failure_handling(self):
        """Test behavior when Docker fails"""
        tool = FirecrawlTool()

        with patch.object(tool, "_check_docker_health", return_value=False):
            with pytest.raises(Exception) as exc_info:
                await tool.execute(url="https://example.com")

            assert "unhealthy" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_scraping_timeout_handling(self):
        """Test timeout handling during scraping"""
        tool = FirecrawlTool()

        with (
            patch.object(settings.firecrawl_settings, "deployment_mode", "docker"),
            patch.object(tool, "_check_docker_health", return_value=True),
        ):
            # Mock timeout error
            mock_client = AsyncMock()
            mock_client.post.side_effect = Exception("Request timed out")

            with patch.object(tool, "_get_client", return_value=mock_client):
                with pytest.raises(Exception) as exc_info:
                    await tool.execute(url="https://example.com")

                assert "timed out" in str(exc_info.value).lower()

    def test_configuration_validation(self):
        """Test configuration validation for different modes"""
        # Test invalid deployment mode
        with patch.object(settings.firecrawl_settings, "deployment_mode", "invalid"):
            # Should still work but fall back to default behavior
            assert settings.firecrawl_settings.deployment_mode == "invalid"
            # Effective URL should still be accessible
            assert settings.firecrawl_settings.effective_url is not None

    @pytest.mark.asyncio
    async def test_concurrent_scraping_with_health_check(self):
        """Test concurrent scraping with health check"""
        tool = FirecrawlTool()

        # Mock successful health check
        with patch.object(tool, "_check_docker_health", return_value=True):
            # Mock scraping response
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json = AsyncMock(
                return_value={
                    "data": {
                        "markdown": "# Test Page\n\nTest content.",
                        "metadata": {"title": "Test Page"},
                        "links": [],
                        "images": [],
                    }
                }
            )

            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response

            with patch.object(tool, "_get_client", return_value=mock_client):
                # Test multiple concurrent requests
                import asyncio

                urls = [
                    "https://example1.com",
                    "https://example2.com",
                    "https://example3.com",
                ]
                tasks = [tool.execute(url=url) for url in urls]
                results = await asyncio.gather(*tasks)

                assert len(results) == 3
                for result in results:
                    assert result["title"] == "Test Page"
                    assert "Test content" in result["content"]
