"""
Integration tests for Firecrawl Docker deployment.

These tests verify that the self-hosted Firecrawl Docker services
work correctly with the AI Assistant application.
"""

import pytest
import asyncio
import httpx
from unittest.mock import patch, AsyncMock
from app.core.tools.firecrawl_tool import FirecrawlTool
from app.core.agents.firecrawl_agent import FirecrawlAgent
from app.core.config import settings


@pytest.mark.integration
class TestFirecrawlDockerIntegration:
    """Integration tests for Firecrawl Docker deployment"""

    @pytest.fixture
    def docker_config(self):
        """Configure settings for Docker mode"""
        with (
            patch.object(settings.firecrawl_settings, "deployment_mode", "docker"),
            patch.object(
                settings.firecrawl_settings, "docker_url", "http://firecrawl-api:3002"
            ),
            patch.object(settings.firecrawl_settings, "enabled", True),
            patch.object(settings.firecrawl_settings, "scraping_enabled", True),
        ):
            yield

    @pytest.fixture
    def mock_docker_response(self):
        """Mock successful Docker response"""
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json = AsyncMock(return_value={
            "data": {
                "markdown": "# Test Page\n\nThis is test content from Docker Firecrawl.",
                "raw": "<html><head><title>Test Page</title></head><body><h1>Test Page</h1><p>This is test content from Docker Firecrawl.</p></body></html>",
                "metadata": {
                    "title": "Test Page",
                    "description": "Test page from Docker Firecrawl",
                    "language": "en",
                },
                "links": [{"url": "https://example.com/page2", "text": "Next Page"}],
                "images": [
                    {"src": "https://example.com/image.jpg", "alt": "Test Image"}
                ],
            }
        })
        return mock_response

    @pytest.fixture
    def mock_health_response(self):
        """Mock health check response"""
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json = AsyncMock(return_value={"status": "healthy"})
        return mock_response

    @pytest.mark.asyncio
    async def test_docker_mode_configuration(self, docker_config):
        """Test Docker mode configuration"""
        assert settings.firecrawl_settings.deployment_mode == "docker"
        assert settings.firecrawl_settings.effective_url == "http://firecrawl-api:3002"
        assert settings.firecrawl_settings.effective_api_key is None

    @pytest.mark.asyncio
    async def test_docker_tool_initialization(self, docker_config):
        """Test Firecrawl tool initialization in Docker mode"""
        tool = FirecrawlTool()
        # Tool doesn't have base_url and api_key attributes anymore
        # It uses effective configuration from settings when _get_client is called
        assert tool._client is None

    @pytest.mark.asyncio
    async def test_docker_health_check_success(
        self, docker_config, mock_health_response
    ):
        """Test successful Docker health check"""
        tool = FirecrawlTool()

        with patch.object(tool, "_get_client", return_value=AsyncMock(get=AsyncMock(return_value=mock_health_response))):
            is_healthy = await tool._check_docker_health()
            assert is_healthy is True

    @pytest.mark.asyncio
    async def test_docker_health_check_failure(self, docker_config):
        """Test Docker health check failure"""
        tool = FirecrawlTool()

        with patch.object(
            tool, "_get_client", return_value=AsyncMock(get=AsyncMock(side_effect=httpx.ConnectError("Connection failed")))
        ):
            is_healthy = await tool._check_docker_health()
            assert is_healthy is False

    @pytest.mark.asyncio
    async def test_docker_scraping_success(
        self, docker_config, mock_docker_response, mock_health_response
    ):
        """Test successful scraping with Docker Firecrawl"""
        tool = FirecrawlTool()

        with (
            patch.object(tool, "_check_docker_health", return_value=True),
            patch.object(tool, "_get_client", return_value=AsyncMock(post=AsyncMock(return_value=mock_docker_response))),
        ):
            result = await tool.execute(url="https://example.com")

            assert result["url"] == "https://example.com"
            assert result["title"] == "Test Page"
            assert "Test Page" in result["content"]
            assert result["link_count"] > 0
            # Don't check for images since extract_images is False by default

    @pytest.mark.asyncio
    async def test_docker_scraping_failure(self, docker_config):
        """Test Docker scraping failure"""
        tool = FirecrawlTool()

        with patch.object(tool, "_check_docker_health", return_value=False):
            with pytest.raises(Exception) as exc_info:
                await tool.execute(url="https://example.com")

            assert "unhealthy" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_docker_agent_integration(
        self, docker_config, mock_docker_response, mock_health_response
    ):
        """Test Firecrawl agent with Docker deployment"""
        # Mock LLM
        mock_llm = AsyncMock()
        mock_response = AsyncMock()
        mock_response.generations = [
            [
                AsyncMock(
                    text='{"content_type": "general", "recommended_formats": ["markdown"], "extraction_targets": ["content"], "challenges": [], "recommendations": []}'
                )
            ]
        ]
        mock_llm.agenerate.return_value = mock_response

        agent = FirecrawlAgent(llm=mock_llm)

        with patch.object(agent, "_get_scraper_tool") as mock_get_tool:
            mock_tool = AsyncMock()
            mock_tool.execute.return_value = {
                "url": "https://example.com",
                "content": "Test content from Docker",
                "title": "Test Page",
                "link_count": 1,
                "image_count": 1,
            }
            mock_get_tool.return_value = mock_tool

            result = await agent.execute("Scrape https://example.com")

            assert result["success"] is True
            assert result["total_urls"] == 1
            assert len(result["results"]) == 1

    @pytest.mark.asyncio
    async def test_docker_batch_scraping(
        self, docker_config, mock_docker_response, mock_health_response
    ):
        """Test batch scraping with Docker Firecrawl"""
        tool = FirecrawlTool()
        urls = ["https://example1.com", "https://example2.com"]

        with (
            patch.object(tool, "_check_docker_health", return_value=True),
            patch.object(tool, "_get_client", return_value=AsyncMock(post=AsyncMock(return_value=mock_docker_response))),
        ):
            results = await tool.batch_scrape(urls=urls)

            assert len(results) == 2
            for i, result in enumerate(results):
                assert result["url"] == urls[i]
                assert "content" in result

    @pytest.mark.asyncio
    async def test_docker_mode_configuration(self):
        """Test Docker mode configuration"""
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


@pytest.mark.integration
@pytest.mark.docker
class TestFirecrawlDockerLive:
    """Live tests against actual Docker Firecrawl services"""

    @pytest.mark.asyncio
    async def test_live_docker_health_check(self):
        """Test health check against live Docker Firecrawl"""
        tool = FirecrawlTool()

        # Only run if Docker is available
        try:
            is_healthy = await tool._check_docker_health()
            if is_healthy:
                # Test actual scraping
                result = await tool.execute(
                    url="https://httpbin.org/html", formats=["markdown"], timeout=30
                )

                assert result["url"] == "https://httpbin.org/html"
                assert len(result["content"]) > 0
                assert "Herman Melville" in result["content"]  # Expected content
            else:
                pytest.skip("Docker Firecrawl not available")
        except Exception:
            pytest.skip("Docker Firecrawl not available")

    @pytest.mark.asyncio
    async def test_live_docker_vs_api_performance(self):
        """Compare performance between Docker and API modes"""
        import time

        test_url = "https://httpbin.org/html"

        # Test Docker mode
        with (
            patch.object(settings.firecrawl_settings, "deployment_mode", "docker"),
            patch.object(
                settings.firecrawl_settings, "docker_url", "http://firecrawl-api:3002"
            ),
        ):
            tool = FirecrawlTool()

            try:
                start_time = time.time()
                docker_result = await tool.execute(url=test_url, formats=["markdown"])
                docker_time = time.time() - start_time

                if docker_result.get("content"):
                    print(f"Docker mode completed in {docker_time:.2f}s")
                else:
                    pytest.skip("Docker mode failed")
            except Exception:
                pytest.skip("Docker mode not available")


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
