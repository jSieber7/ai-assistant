"""
Search Service for RAG Pipeline

This service handles the search and scraping phase:
- SearxNG Tool → List of URLs → Firecrawl/Playwright Tool → Scraped Markdown Pages
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Union
from langchain.docstore.document import Document

from ..tools.registry import ToolRegistry
from ..tools.dynamic_executor import DynamicToolExecutor, TaskRequest, TaskType
from ..config import settings

logger = logging.getLogger(__name__)


class SearchService:
    """
    Service for handling web search and content scraping.

    This service coordinates search and scraping tools to find and extract
    relevant content from the web based on optimized queries.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        max_search_results: int = 5,
        max_concurrent_scrapes: int = 3,
        scrape_timeout: int = 30,
    ):
        """
        Initialize the search service.

        Args:
            tool_registry: Registry of available tools
            max_search_results: Maximum number of search results to process
            max_concurrent_scrapes: Maximum concurrent scraping operations
            scrape_timeout: Timeout for scraping operations
        """
        self.tool_registry = tool_registry
        self.dynamic_executor = DynamicToolExecutor(tool_registry)
        self.max_search_results = max_search_results
        self.max_concurrent_scrapes = max_concurrent_scrapes
        self.scrape_timeout = scrape_timeout

        # Service statistics
        self._stats = {
            "searches_performed": 0,
            "urls_scraped": 0,
            "successful_scrapes": 0,
            "failed_scrapes": 0,
            "total_content_length": 0,
        }

    async def search_and_scrape(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        search_params: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Execute search and scrape pipeline.

        Args:
            query: Search query
            context: Additional context for search
            search_params: Additional search parameters

        Returns:
            List of scraped documents
        """
        start_time = time.time()

        try:
            # Step 1: Execute search
            search_results = await self._execute_search(query, context, search_params)

            if not search_results:
                logger.warning(f"No search results found for query: {query}")
                return []

            # Step 2: Extract URLs from search results
            urls = self._extract_urls(search_results)

            if not urls:
                logger.warning(f"No URLs found in search results for query: {query}")
                return []

            logger.info(f"Found {len(urls)} URLs to scrape for query: {query}")

            # Step 3: Scrape content from URLs
            scraped_documents = await self._scrape_urls(urls, query)

            # Update statistics
            self._stats["searches_performed"] += 1
            self._stats["urls_scraped"] += len(urls)
            self._stats["successful_scrapes"] += len(scraped_documents)
            self._stats["failed_scrapes"] += len(urls) - len(scraped_documents)
            self._stats["total_content_length"] += sum(
                len(doc.page_content) for doc in scraped_documents
            )

            execution_time = time.time() - start_time
            logger.info(
                f"Search and scrape completed in {execution_time:.2f}s. "
                f"Scraped {len(scraped_documents)} documents from {len(urls)} URLs"
            )

            return scraped_documents

        except Exception as e:
            logger.error(f"Search and scrape failed for query '{query}': {str(e)}")
            return []

    async def _execute_search(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        search_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Execute search using SearXNG tool.

        Args:
            query: Search query
            context: Additional context
            search_params: Search parameters

        Returns:
            Search results data
        """
        try:
            # Prepare search parameters
            params = {
                "results_count": self.max_search_results,
                "category": "general",
                "language": "auto",
                "time_range": "",
                "safesearch": 0,
            }

            # Override with provided parameters
            if search_params:
                params.update(search_params)

            # Add context to search request
            search_context = params.copy()
            if context:
                search_context.update(context)

            # Create search task request
            search_request = TaskRequest(
                task_type=TaskType.SEARCH,
                query=query,
                context=search_context,
                required_tools=["searxng_search"],
                parameters=params,
                max_tools=1,
            )

            # Execute search
            search_result = await self.dynamic_executor.execute_task(search_request)

            if not search_result.success:
                logger.error(f"Search task failed: {search_result.error}")
                return None

            logger.info(f"Search completed successfully for query: {query}")
            return search_result.data

        except Exception as e:
            logger.error(f"Search execution failed: {str(e)}")
            return None

    def _extract_urls(self, search_results: Dict[str, Any]) -> List[str]:
        """
        Extract URLs from search results.

        Args:
            search_results: Search results data

        Returns:
            List of URLs
        """
        urls = []

        try:
            if isinstance(search_results, dict) and "results" in search_results:
                for result in search_results.get("results", []):
                    url = result.get("url", "")
                    if url and self._is_valid_url(url):
                        urls.append(url)

            logger.debug(f"Extracted {len(urls)} valid URLs from search results")
            return urls

        except Exception as e:
            logger.error(f"Error extracting URLs from search results: {str(e)}")
            return []

    def _is_valid_url(self, url: str) -> bool:
        """
        Validate URL format and accessibility.

        Args:
            url: URL to validate

        Returns:
            True if URL is valid
        """
        try:
            import re

            url_pattern = re.compile(
                r"^https?://"  # http:// or https://
                r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain...
                r"localhost|"  # localhost...
                r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
                r"(?::\d+)?"  # optional port
                r"(?:/?|[/?]\S+)$",
                re.IGNORECASE,
            )

            if not url_pattern.match(url):
                return False

            # Additional checks
            if len(url) > 2048:  # URL too long
                return False

            # Skip certain file types
            skip_extensions = [".pdf", ".jpg", ".jpeg", ".png", ".gif", ".zip", ".exe"]
            if any(url.lower().endswith(ext) for ext in skip_extensions):
                return False

            return True

        except Exception:
            return False

    async def _scrape_urls(
        self, urls: List[str], original_query: str
    ) -> List[Document]:
        """
        Scrape content from multiple URLs concurrently.

        Args:
            urls: List of URLs to scrape
            original_query: Original search query for context

        Returns:
            List of scraped documents
        """
        if not urls:
            return []

        # Create semaphore to limit concurrent scraping
        semaphore = asyncio.Semaphore(self.max_concurrent_scrapes)

        async def scrape_single_url(url: str) -> Optional[Document]:
            async with semaphore:
                return await self._scrape_single_url(url, original_query)

        # Execute scraping concurrently
        tasks = [scrape_single_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        documents = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error scraping URL {urls[i]}: {str(result)}")
            elif result is not None:
                documents.append(result)

        logger.info(f"Successfully scraped {len(documents)} out of {len(urls)} URLs")
        return documents

    async def _scrape_single_url(
        self, url: str, original_query: str
    ) -> Optional[Document]:
        """
        Scrape content from a single URL.

        Args:
            url: URL to scrape
            original_query: Original search query for context

        Returns:
            Document with scraped content or None if failed
        """
        try:
            # Prepare scrape parameters
            scrape_params = {
                "formats": ["markdown"],
                "wait_for": 2000,
                "timeout": self.scrape_timeout,
                "extract_links": True,
                "extract_images": False,
                "include_tags": ["article", "main", "content"],
                "exclude_tags": ["nav", "footer", "aside", "script", "style"],
            }

            # Create scrape task request
            scrape_request = TaskRequest(
                task_type=TaskType.SCRAPE,
                query=f"Scrape content from {url}",
                context={"url": url, "original_query": original_query},
                required_tools=["firecrawl_scrape"],
                parameters=scrape_params,
                max_tools=1,
            )

            # Execute scraping
            scrape_result = await self.dynamic_executor.execute_task(scrape_request)

            if not scrape_result.success:
                logger.warning(f"Failed to scrape {url}: {scrape_result.error}")
                return None

            # Extract content from result
            scrape_data = scrape_result.data
            if not isinstance(scrape_data, dict):
                logger.warning(f"Invalid scrape data format for {url}")
                return None

            content = scrape_data.get("content", "")
            title = scrape_data.get("title", "")

            # Filter out very short content
            if not content or len(content.strip()) < 100:
                logger.warning(
                    f"Content too short for {url}: {len(content)} characters"
                )
                return None

            # Create document
            doc = Document(
                page_content=content,
                metadata={
                    "source": url,
                    "title": title,
                    "scraped_at": time.time(),
                    "original_query": original_query,
                    "content_length": len(content),
                    "scrape_method": "firecrawl",
                },
            )

            logger.debug(f"Successfully scraped {url}: {len(content)} characters")
            return doc

        except Exception as e:
            logger.error(f"Error scraping URL {url}: {str(e)}")
            return None

    async def search_only(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        search_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Execute search only without scraping.

        Args:
            query: Search query
            context: Additional context
            search_params: Search parameters

        Returns:
            Search results data
        """
        return await self._execute_search(query, context, search_params)

    async def scrape_urls_only(
        self, urls: List[str], context: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Scrape URLs only without search.

        Args:
            urls: List of URLs to scrape
            context: Additional context

        Returns:
            List of scraped documents
        """
        original_query = context.get("original_query", "") if context else ""
        return await self._scrape_urls(urls, original_query)

    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "service_name": "SearchService",
            "searches_performed": self._stats["searches_performed"],
            "urls_scraped": self._stats["urls_scraped"],
            "successful_scrapes": self._stats["successful_scrapes"],
            "failed_scrapes": self._stats["failed_scrapes"],
            "success_rate": (
                self._stats["successful_scrapes"] / max(self._stats["urls_scraped"], 1)
            ),
            "total_content_length": self._stats["total_content_length"],
            "avg_content_length": (
                self._stats["total_content_length"]
                / max(self._stats["successful_scrapes"], 1)
            ),
            "config": {
                "max_search_results": self.max_search_results,
                "max_concurrent_scrapes": self.max_concurrent_scrapes,
                "scrape_timeout": self.scrape_timeout,
            },
        }

    def reset_stats(self):
        """Reset service statistics."""
        self._stats = {
            "searches_performed": 0,
            "urls_scraped": 0,
            "successful_scrapes": 0,
            "failed_scrapes": 0,
            "total_content_length": 0,
        }
