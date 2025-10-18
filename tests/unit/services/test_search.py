"""
Unit tests for Search Service.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from langchain.docstore.document import Document

from app.core.services.search import SearchService
from app.core.tools.registry import ToolRegistry
from app.core.tools.dynamic_executor import TaskResult, ToolResult


class TestSearchService:
    """Test cases for SearchService."""

    @pytest.fixture
    def mock_tool_registry(self):
        """Create a mock tool registry."""
        return Mock(spec=ToolRegistry)

    @pytest.fixture
    def search_service(self, mock_tool_registry):
        """Create SearchService instance."""
        return SearchService(
            tool_registry=mock_tool_registry,
            max_search_results=3,
            max_concurrent_scrapes=2,
            scrape_timeout=30,
        )

    @pytest.fixture
    def sample_search_results(self):
        """Create sample search results."""
        return {
            "results": [
                {
                    "title": "Python Async Programming",
                    "url": "https://example.com/python-async",
                    "content": "Content about async programming",
                    "engine": "google",
                },
                {
                    "title": "Asyncio Tutorial",
                    "url": "https://example.com/asyncio-tutorial",
                    "content": "Tutorial on asyncio",
                    "engine": "bing",
                },
            ],
            "total_results": 2,
            "search_time": 0.5,
        }

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents."""
        return [
            Document(
                page_content="This is a detailed article about Python async programming...",
                metadata={
                    "source": "https://example.com/python-async",
                    "title": "Python Async Programming",
                    "scraped_at": 1234567890,
                },
            ),
            Document(
                page_content="This tutorial covers asyncio in depth...",
                metadata={
                    "source": "https://example.com/asyncio-tutorial",
                    "title": "Asyncio Tutorial",
                    "scraped_at": 1234567891,
                },
            ),
        ]

    @pytest.mark.asyncio
    async def test_search_and_scrape_success(
        self, search_service, sample_search_results, sample_documents
    ):
        """Test successful search and scrape."""
        query = "Python async programming"

        # Mock dynamic executor
        with patch.object(
            search_service.dynamic_executor, "execute_task"
        ) as mock_execute:
            # Mock search result
            search_task_result = TaskResult(
                success=True,
                data=sample_search_results,
                tool_results=[],
                execution_time=0.5,
                task_type=search_service.dynamic_executor._task_tool_mapping["search"][
                    0
                ],
            )

            # Mock scrape results
            scrape_task_result = TaskResult(
                success=True,
                data=sample_documents[0].page_content,
                tool_results=[
                    ToolResult(
                        success=True,
                        data={
                            "content": sample_documents[0].page_content,
                            "title": sample_documents[0].metadata["title"],
                            "url": sample_documents[0].metadata["source"],
                        },
                        tool_name="firecrawl_scrape",
                        execution_time=1.0,
                    )
                ],
                execution_time=1.0,
                task_type="scrape",
            )

            mock_execute.side_effect = [
                search_task_result,
                scrape_task_result,
                scrape_task_result,
            ]

            result = await search_service.search_and_scrape(query)

            assert len(result) == 2
            assert all(isinstance(doc, Document) for doc in result)
            assert result[0].metadata["source"] == "https://example.com/python-async"
            assert (
                result[1].metadata["source"] == "https://example.com/asyncio-tutorial"
            )

    @pytest.mark.asyncio
    async def test_search_and_scrape_no_results(self, search_service):
        """Test search and scrape with no results."""
        query = "nonexistent query"

        with patch.object(
            search_service.dynamic_executor, "execute_task"
        ) as mock_execute:
            # Mock empty search result
            search_task_result = TaskResult(
                success=True,
                data={"results": []},
                tool_results=[],
                execution_time=0.5,
                task_type="search",
            )

            mock_execute.return_value = search_task_result

            result = await search_service.search_and_scrape(query)

            assert len(result) == 0

    @pytest.mark.asyncio
    async def test_search_and_scrape_search_failure(self, search_service):
        """Test search and scrape with search failure."""
        query = "test query"

        with patch.object(
            search_service.dynamic_executor, "execute_task"
        ) as mock_execute:
            # Mock failed search result
            search_task_result = TaskResult(
                success=False,
                data=None,
                tool_results=[],
                execution_time=0.5,
                task_type="search",
                error="Search failed",
            )

            mock_execute.return_value = search_task_result

            result = await search_service.search_and_scrape(query)

            assert len(result) == 0

    @pytest.mark.asyncio
    async def test_search_only(self, search_service, sample_search_results):
        """Test search only functionality."""
        query = "Python async programming"

        with patch.object(
            search_service.dynamic_executor, "execute_task"
        ) as mock_execute:
            search_task_result = TaskResult(
                success=True,
                data=sample_search_results,
                tool_results=[],
                execution_time=0.5,
                task_type="search",
            )

            mock_execute.return_value = search_task_result

            result = await search_service.search_only(query)

            assert result == sample_search_results

    @pytest.mark.asyncio
    async def test_scrape_urls_only(self, search_service, sample_documents):
        """Test scrape URLs only functionality."""
        urls = [
            "https://example.com/python-async",
            "https://example.com/asyncio-tutorial",
        ]

        with patch.object(
            search_service.dynamic_executor, "execute_task"
        ) as mock_execute:
            scrape_task_result = TaskResult(
                success=True,
                data=sample_documents[0].page_content,
                tool_results=[
                    ToolResult(
                        success=True,
                        data={
                            "content": sample_documents[0].page_content,
                            "title": sample_documents[0].metadata["title"],
                            "url": sample_documents[0].metadata["source"],
                        },
                        tool_name="firecrawl_scrape",
                        execution_time=1.0,
                    )
                ],
                execution_time=1.0,
                task_type="scrape",
            )

            mock_execute.side_effect = [scrape_task_result, scrape_task_result]

            result = await search_service.scrape_urls_only(urls)

            assert len(result) == 2
            assert all(isinstance(doc, Document) for doc in result)

    def test_extract_urls_valid(self, search_service):
        """Test URL extraction from valid search results."""
        search_results = {
            "results": [
                {"url": "https://example.com/page1"},
                {"url": "http://test.org/page2"},
                {"url": "https://another.net/page3"},
            ]
        }

        urls = search_service._extract_urls(search_results)

        assert len(urls) == 3
        assert "https://example.com/page1" in urls
        assert "http://test.org/page2" in urls
        assert "https://another.net/page3" in urls

    def test_extract_urls_invalid(self, search_service):
        """Test URL extraction with invalid URLs."""
        search_results = {
            "results": [
                {"url": "invalid-url"},
                {"url": ""},
                {"url": "https://valid.com/page"},
            ]
        }

        urls = search_service._extract_urls(search_results)

        assert len(urls) == 1
        assert urls[0] == "https://valid.com/page"

    def test_is_valid_url_valid(self, search_service):
        """Test URL validation with valid URLs."""
        valid_urls = [
            "https://example.com/page",
            "http://test.org/article",
            "https://localhost:8080/api",
            "https://192.168.1.1:3000/resource",
        ]

        for url in valid_urls:
            assert search_service._is_valid_url(url) is True

    def test_is_valid_url_invalid(self, search_service):
        """Test URL validation with invalid URLs."""
        invalid_urls = [
            "not-a-url",
            "ftp://invalid-protocol.com",
            "https://example.com/file.pdf",  # File extension to skip
            "https://example.com/image.jpg",  # Image to skip
            "https://" + "a" * 2050,  # Too long
        ]

        for url in invalid_urls:
            assert search_service._is_valid_url(url) is False

    def test_get_service_stats(self, search_service):
        """Test service statistics."""
        stats = search_service.get_service_stats()

        assert stats["service_name"] == "SearchService"
        assert stats["searches_performed"] == 0
        assert stats["urls_scraped"] == 0
        assert stats["successful_scrapes"] == 0
        assert stats["failed_scrapes"] == 0
        assert stats["config"]["max_search_results"] == 3
        assert stats["config"]["max_concurrent_scrapes"] == 2
        assert stats["config"]["scrape_timeout"] == 30

    def test_reset_stats(self, search_service):
        """Test statistics reset."""
        # Manually update some stats
        search_service._stats["searches_performed"] = 5
        search_service._stats["successful_scrapes"] = 10

        # Reset stats
        search_service.reset_stats()

        # Check stats are reset
        assert search_service._stats["searches_performed"] == 0
        assert search_service._stats["successful_scrapes"] == 0

    @pytest.mark.asyncio
    async def test_search_and_scrape_with_context(
        self, search_service, sample_search_results
    ):
        """Test search and scrape with context."""
        query = "Python async programming"
        context = {"domain": "technology", "time_range": "2023"}
        search_params = {"results_count": 5, "category": "technology"}

        with patch.object(
            search_service.dynamic_executor, "execute_task"
        ) as mock_execute:
            search_task_result = TaskResult(
                success=True,
                data=sample_search_results,
                tool_results=[],
                execution_time=0.5,
                task_type="search",
            )

            mock_execute.return_value = search_task_result

            await search_service.search_and_scrape(query, context, search_params)

            # Verify the context was passed to the search
            mock_execute.assert_called_once()
            call_args = mock_execute.call_args[0][0]
            assert call_args.context["domain"] == "technology"
            assert call_args.context["time_range"] == "2023"
            assert call_args.parameters["results_count"] == 5
