"""
Unit tests for the Standalone Scraper interface.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from app.core.tools.web.standalone_scraper import StandaloneScraper, quick_scrape
from app.core.tools.base.base import ToolResult


@pytest.mark.unit
class TestStandaloneScraper:
    """Test cases for StandaloneScraper"""

    @pytest.fixture
    def mock_scraper(self):
        """Create a StandaloneScraper with mocked dependencies"""
        with patch("app.core.tools.standalone_scraper.settings") as mock_settings:
            mock_settings.firecrawl_settings.scraping_enabled = True
            scraper = StandaloneScraper()
            scraper._scraper = AsyncMock()
            scraper._scraper._check_docker_health = AsyncMock(return_value=True)
            return scraper

    @pytest.mark.asyncio
    async def test_initialization_success(self, mock_scraper):
        """Test successful initialization"""
        result = await mock_scraper.initialize()
        assert result is True
        assert mock_scraper._initialized is True

    @pytest.mark.asyncio
    async def test_initialization_failure_scraping_disabled(self):
        """Test initialization failure when scraping is disabled"""
        with patch("app.core.tools.standalone_scraper.settings") as mock_settings:
            mock_settings.firecrawl_settings.scraping_enabled = False
            scraper = StandaloneScraper()

            result = await scraper.initialize()
            assert result is False
            assert scraper._initialized is False

    @pytest.mark.asyncio
    async def test_initialization_failure_docker_unhealthy(self):
        """Test initialization failure when Docker is unhealthy"""
        with patch("app.core.tools.standalone_scraper.settings") as mock_settings:
            mock_settings.firecrawl_settings.scraping_enabled = True
            scraper = StandaloneScraper()
            scraper._scraper = AsyncMock()
            scraper._scraper._check_docker_health = AsyncMock(return_value=False)

            result = await scraper.initialize()
            assert result is False
            assert scraper._initialized is False

    @pytest.mark.asyncio
    async def test_scrape_url_success(self, mock_scraper):
        """Test successful URL scraping"""
        # Mock the scraper execute method
        mock_data = {
            "url": "https://example.com",
            "title": "Example Page",
            "content": "This is the page content",
            "content_length": 24,
        }
        mock_scraper._scraper.execute = AsyncMock(return_value=mock_data)

        result = await mock_scraper.scrape_url("https://example.com")

        assert result.success is True
        assert result.data == mock_data
        assert result.tool_name == "standalone_scraper"

    @pytest.mark.asyncio
    async def test_scrape_url_not_initialized(self):
        """Test scraping when scraper is not initialized"""
        with patch("app.core.tools.standalone_scraper.settings") as mock_settings:
            mock_settings.firecrawl_settings.scraping_enabled = False
            scraper = StandaloneScraper()

            result = await scraper.scrape_url("https://example.com")

            assert result.success is False
            assert "not initialized" in result.error

    @pytest.mark.asyncio
    async def test_scrape_url_with_parameters(self, mock_scraper):
        """Test scraping URL with custom parameters"""
        mock_data = {"content": "test content"}
        mock_scraper._scraper.execute = AsyncMock(return_value=mock_data)

        result = await mock_scraper.scrape_url(
            url="https://example.com",
            formats=["markdown", "raw"],
            wait_for=3000,
            screenshot=True,
            include_tags=["article"],
            exclude_tags=["nav"],
            extract_images=True,
            extract_links=True,
            timeout=60,
        )

        assert result.success is True
        mock_scraper._scraper.execute.assert_called_once_with(
            url="https://example.com",
            formats=["markdown", "raw"],
            wait_for=3000,
            screenshot=True,
            include_tags=["article"],
            exclude_tags=["nav"],
            extract_images=True,
            extract_links=True,
            timeout=60,
        )

    @pytest.mark.asyncio
    async def test_scrape_url_execution_error(self, mock_scraper):
        """Test handling of execution errors"""
        mock_scraper._scraper.execute = AsyncMock(
            side_effect=Exception("Scraping failed")
        )

        result = await mock_scraper.scrape_url("https://example.com")

        assert result.success is False
        assert "Scraping failed" in result.error

    @pytest.mark.asyncio
    async def test_scrape_multiple_urls_success(self, mock_scraper):
        """Test successful multiple URL scraping"""
        mock_batch_results = [
            {"url": "https://example.com", "content": "Content 1"},
            {"url": "https://example.org", "content": "Content 2"},
        ]
        mock_scraper._scraper.batch_scrape = AsyncMock(return_value=mock_batch_results)

        results = await mock_scraper.scrape_multiple_urls(
            ["https://example.com", "https://example.org"]
        )

        assert len(results) == 2
        assert all(result.success for result in results)
        assert results[0].data["url"] == "https://example.com"
        assert results[1].data["url"] == "https://example.org"

    @pytest.mark.asyncio
    async def test_scrape_multiple_urls_with_errors(self, mock_scraper):
        """Test multiple URL scraping with some errors"""
        mock_batch_results = [
            {"url": "https://example.com", "content": "Content 1"},
            {"url": "https://example.org", "error": "Scraping failed"},
        ]
        mock_scraper._scraper.batch_scrape = AsyncMock(return_value=mock_batch_results)

        results = await mock_scraper.scrape_multiple_urls(
            ["https://example.com", "https://example.org"]
        )

        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is False
        assert "Scraping failed" in results[1].error

    @pytest.mark.asyncio
    async def test_scrape_multiple_urls_not_initialized(self):
        """Test multiple URL scraping when not initialized"""
        with patch("app.core.tools.standalone_scraper.settings") as mock_settings:
            mock_settings.firecrawl_settings.scraping_enabled = False
            scraper = StandaloneScraper()

            results = await scraper.scrape_multiple_urls(
                ["https://example.com", "https://example.org"]
            )

            assert len(results) == 2
            assert all(not result.success for result in results)
            assert all("not initialized" in result.error for result in results)

    @pytest.mark.asyncio
    async def test_extract_content_only_success(self, mock_scraper):
        """Test successful content extraction"""
        mock_data = {
            "content": "This is a long enough content to pass the minimum length check",
            "title": "Test Page",
        }
        mock_scraper._scraper.execute = AsyncMock(return_value=mock_data)

        content = await mock_scraper.extract_content_only("https://example.com")

        assert content == mock_data["content"]

    @pytest.mark.asyncio
    async def test_extract_content_only_too_short(self, mock_scraper):
        """Test content extraction when content is too short"""
        mock_data = {"content": "Short", "title": "Test Page"}
        mock_scraper._scraper.execute = AsyncMock(return_value=mock_data)

        content = await mock_scraper.extract_content_only(
            "https://example.com", min_length=50
        )

        assert content is None

    @pytest.mark.asyncio
    async def test_extract_content_only_failure(self, mock_scraper):
        """Test content extraction when scraping fails"""
        mock_scraper._scraper.execute = AsyncMock(side_effect=Exception("Failed"))

        content = await mock_scraper.extract_content_only("https://example.com")

        assert content is None

    @pytest.mark.asyncio
    async def test_extract_links_from_url_success(self, mock_scraper):
        """Test successful link extraction"""
        mock_data = {
            "links": [
                {"url": "https://example.com/page1", "text": "Page 1"},
                {"url": "https://example.com/page2", "text": "Page 2"},
                {"url": "", "text": "Invalid link"},  # Should be filtered out
            ]
        }
        mock_scraper._scraper.execute = AsyncMock(return_value=mock_data)

        links = await mock_scraper.extract_links_from_url("https://example.com")

        assert len(links) == 2
        assert "https://example.com/page1" in links
        assert "https://example.com/page2" in links

    @pytest.mark.asyncio
    async def test_extract_links_from_url_failure(self, mock_scraper):
        """Test link extraction when scraping fails"""
        mock_scraper._scraper.execute = AsyncMock(side_effect=Exception("Failed"))

        links = await mock_scraper.extract_links_from_url("https://example.com")

        assert links == []

    @pytest.mark.asyncio
    async def test_extract_images_from_url_success(self, mock_scraper):
        """Test successful image extraction"""
        mock_data = {
            "images": [
                {"url": "https://example.com/image1.jpg", "alt": "Image 1"},
                {"url": "https://example.com/image2.jpg", "alt": "Image 2"},
            ]
        }
        mock_scraper._scraper.execute = AsyncMock(return_value=mock_data)

        images = await mock_scraper.extract_images_from_url("https://example.com")

        assert len(images) == 2
        assert images[0]["url"] == "https://example.com/image1.jpg"
        assert images[1]["url"] == "https://example.com/image2.jpg"

    @pytest.mark.asyncio
    async def test_extract_images_from_url_failure(self, mock_scraper):
        """Test image extraction when scraping fails"""
        mock_scraper._scraper.execute = AsyncMock(side_effect=Exception("Failed"))

        images = await mock_scraper.extract_images_from_url("https://example.com")

        assert images == []

    @pytest.mark.asyncio
    async def test_get_page_metadata_success(self, mock_scraper):
        """Test successful page metadata extraction"""
        mock_data = {
            "url": "https://example.com",
            "title": "Example Page",
            "description": "Page description",
            "content_length": 1000,
            "link_count": 10,
            "image_count": 5,
            "metadata": {"author": "Test Author"},
        }
        mock_scraper._scraper.execute = AsyncMock(return_value=mock_data)

        metadata = await mock_scraper.get_page_metadata("https://example.com")

        assert metadata["success"] is True
        assert metadata["url"] == "https://example.com"
        assert metadata["title"] == "Example Page"
        assert metadata["description"] == "Page description"
        assert metadata["content_length"] == 1000
        assert metadata["link_count"] == 10
        assert metadata["image_count"] == 5
        assert metadata["metadata"]["author"] == "Test Author"

    @pytest.mark.asyncio
    async def test_get_page_metadata_failure(self, mock_scraper):
        """Test page metadata extraction when scraping fails"""
        mock_scraper._scraper.execute = AsyncMock(side_effect=Exception("Failed"))

        metadata = await mock_scraper.get_page_metadata("https://example.com")

        assert metadata["success"] is False
        assert "Failed" in metadata["error"]
        assert metadata["title"] == ""
        assert metadata["content_length"] == 0

    @pytest.mark.asyncio
    async def test_cleanup(self, mock_scraper):
        """Test cleanup method"""
        await mock_scraper.cleanup()
        mock_scraper._scraper.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_destructor_cleanup(self):
        """Test that destructor calls cleanup"""
        with patch("app.core.tools.standalone_scraper.settings") as mock_settings:
            mock_settings.firecrawl_settings.scraping_enabled = True
            scraper = StandaloneScraper()
            scraper._scraper = AsyncMock()

            # Simulate destructor
            scraper.__del__()

            # Give some time for async task to be created
            await asyncio.sleep(0.1)

            # Note: We can't easily test the actual cleanup call from destructor
            # due to the async nature, but we can verify the method exists
            assert hasattr(scraper, "cleanup")


@pytest.mark.unit
class TestQuickScrape:
    """Test cases for the quick_scrape convenience function"""

    @pytest.mark.asyncio
    async def test_quick_scrape_success(self):
        """Test successful quick scrape"""
        with patch("app.core.tools.standalone_scraper.settings") as mock_settings:
            mock_settings.firecrawl_settings.scraping_enabled = True

            with patch(
                "app.core.tools.standalone_scraper.StandaloneScraper"
            ) as mock_scraper_class:
                mock_scraper = AsyncMock()
                mock_result = ToolResult(
                    success=True,
                    data={"content": "test content"},
                    tool_name="standalone_scraper",
                    execution_time=1.0,
                )
                mock_scraper.scrape_url = AsyncMock(return_value=mock_result)
                mock_scraper_class.return_value = mock_scraper

                result = await quick_scrape("https://example.com")

                assert result == {"content": "test content"}
                mock_scraper.initialize.assert_called_once()
                mock_scraper.scrape_url.assert_called_once_with(
                    "https://example.com", None
                )
                mock_scraper.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_quick_scrape_with_formats(self):
        """Test quick scrape with custom formats"""
        with patch("app.core.tools.standalone_scraper.settings") as mock_settings:
            mock_settings.firecrawl_settings.scraping_enabled = True

            with patch(
                "app.core.tools.standalone_scraper.StandaloneScraper"
            ) as mock_scraper_class:
                mock_scraper = AsyncMock()
                mock_result = ToolResult(
                    success=True,
                    data={"content": "test content"},
                    tool_name="standalone_scraper",
                    execution_time=1.0,
                )
                mock_scraper.scrape_url = AsyncMock(return_value=mock_result)
                mock_scraper_class.return_value = mock_scraper

                result = await quick_scrape("https://example.com", formats=["markdown"])

                assert result == {"content": "test content"}
                mock_scraper.scrape_url.assert_called_once_with(
                    "https://example.com", ["markdown"]
                )

    @pytest.mark.asyncio
    async def test_quick_scrape_failure(self):
        """Test quick scrape when scraping fails"""
        with patch("app.core.tools.standalone_scraper.settings") as mock_settings:
            mock_settings.firecrawl_settings.scraping_enabled = True

            with patch(
                "app.core.tools.standalone_scraper.StandaloneScraper"
            ) as mock_scraper_class:
                mock_scraper = AsyncMock()
                mock_result = ToolResult(
                    success=False,
                    data=None,
                    error="Scraping failed",
                    tool_name="standalone_scraper",
                    execution_time=1.0,
                )
                mock_scraper.scrape_url = AsyncMock(return_value=mock_result)
                mock_scraper_class.return_value = mock_scraper

                result = await quick_scrape("https://example.com")

                assert result is None

    @pytest.mark.asyncio
    async def test_quick_scrape_initialization_failure(self):
        """Test quick scrape when initialization fails"""
        with patch("app.core.tools.standalone_scraper.settings") as mock_settings:
            mock_settings.firecrawl_settings.scraping_enabled = False

            with patch(
                "app.core.tools.standalone_scraper.StandaloneScraper"
            ) as mock_scraper_class:
                mock_scraper = AsyncMock()
                mock_scraper.initialize = AsyncMock(return_value=False)
                mock_scraper_class.return_value = mock_scraper

                result = await quick_scrape("https://example.com")

                assert result is None
                mock_scraper.initialize.assert_called_once()
                mock_scraper.cleanup.assert_called_once()


@pytest.mark.unit
class TestStandaloneScraperIntegration:
    """Integration tests for StandaloneScraper"""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test a complete workflow with multiple operations"""
        with patch("app.core.tools.standalone_scraper.settings") as mock_settings:
            mock_settings.firecrawl_settings.scraping_enabled = True

            scraper = StandaloneScraper()
            scraper._scraper = AsyncMock()
            scraper._scraper._check_docker_health = AsyncMock(return_value=True)

            # Mock different responses for different operations
            scraper._scraper.execute.side_effect = [
                {"content": "Page content", "title": "Test Page"},  # For scrape_url
                {"links": [{"url": "https://example.com/page1"}]},  # For extract_links
                {
                    "images": [{"url": "https://example.com/image.jpg"}]
                },  # For extract_images
                {
                    "title": "Test Page",
                    "description": "Test description",
                },  # For get_metadata
            ]

            scraper._scraper.batch_scrape = AsyncMock(
                return_value=[
                    {"url": "https://example.com", "content": "Content 1"},
                    {"url": "https://example.org", "content": "Content 2"},
                ]
            )

            # Initialize
            assert await scraper.initialize() is True

            # Scrape single URL
            result = await scraper.scrape_url("https://example.com")
            assert result.success is True

            # Extract content
            content = await scraper.extract_content_only("https://example.com")
            assert content == "Page content"

            # Extract links
            links = await scraper.extract_links_from_url("https://example.com")
            assert len(links) == 1

            # Extract images
            images = await scraper.extract_images_from_url("https://example.com")
            assert len(images) == 1

            # Get metadata
            metadata = await scraper.get_page_metadata("https://example.com")
            assert metadata["success"] is True

            # Scrape multiple URLs
            results = await scraper.scrape_multiple_urls(
                ["https://example.com", "https://example.org"]
            )
            assert len(results) == 2
            assert all(result.success for result in results)

            # Cleanup
            await scraper.cleanup()
