"""
Firecrawl API integration tool for web scraping

This module provides a tool for scraping web content using Firecrawl API
with advanced options and comprehensive data extraction.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
import httpx

from .base import BaseTool, ToolExecutionError

logger = logging.getLogger(__name__)


class FirecrawlTool(BaseTool):
    """Scrape web content using Firecrawl API with advanced options"""

    def __init__(
        self, api_key: Optional[str] = None, base_url: str = "https://api.firecrawl.dev"
    ):
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url
        self._client = None

    @property
    def name(self) -> str:
        return "firecrawl_scrape"

    @property
    def description(self) -> str:
        return "Scrape web content using Firecrawl API with advanced rendering and data extraction"

    @property
    def keywords(self) -> List[str]:
        return [
            "scrape",
            "web",
            "firecrawl",
            "crawl",
            "extract",
            "content",
            "markdown",
            "api",
        ]

    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "url": {
                "type": str,
                "description": "URL to scrape",
                "required": True,
            },
            "formats": {
                "type": List[str],
                "description": "Output formats (markdown, raw, html, etc.)",
                "required": False,
                "default": ["markdown", "raw"],
            },
            "wait_for": {
                "type": int,
                "description": "Time to wait for page load in milliseconds",
                "required": False,
                "default": 2000,
            },
            "screenshot": {
                "type": bool,
                "description": "Take a screenshot of the page",
                "required": False,
                "default": False,
            },
            "include_tags": {
                "type": List[str],
                "description": "HTML tags to include in extraction",
                "required": False,
                "default": ["article", "main", "content"],
            },
            "exclude_tags": {
                "type": List[str],
                "description": "HTML tags to exclude from extraction",
                "required": False,
                "default": ["nav", "footer", "aside", "script", "style"],
            },
            "extract_images": {
                "type": bool,
                "description": "Extract images from page",
                "required": False,
                "default": False,
            },
            "extract_links": {
                "type": bool,
                "description": "Extract links from page",
                "required": False,
                "default": True,
            },
            "timeout": {
                "type": int,
                "description": "Timeout in seconds",
                "required": False,
                "default": 30,
            },
        }

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None:
            from ..config import settings

            # Use settings if no api_key provided
            api_key = self.api_key or settings.firecrawl_settings.api_key
            base_url = self.base_url or settings.firecrawl_settings.base_url

            if not api_key:
                raise ToolExecutionError("Firecrawl API key is not configured")

            self._client = httpx.AsyncClient(
                base_url=base_url,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30.0,
            )
        return self._client

    def _extract_content(self, data: Dict[str, Any], url: str) -> Dict[str, Any]:
        """Extract and structure content from Firecrawl response"""
        from ..config import settings

        # Get the main data from response
        response_data = data.get("data", {})

        # Extract title
        title = response_data.get("metadata", {}).get("title", "")

        # Extract content based on available formats
        content = ""
        if "markdown" in response_data:
            content = response_data["markdown"]
        elif "raw" in response_data:
            content = response_data["raw"]
        elif "html" in response_data:
            content = response_data["html"]

        # Extract metadata
        metadata = response_data.get("metadata", {})
        description = metadata.get("description", "")

        # Extract links if enabled
        links = []
        if settings.firecrawl_settings.extract_links and "links" in response_data:
            links = response_data["links"]

        # Extract images if enabled
        images = []
        if settings.firecrawl_settings.extract_images and "images" in response_data:
            images = response_data["images"]

        return {
            "url": url,
            "title": title,
            "content": content,
            "description": description,
            "links": links,
            "images": images,
            "metadata": metadata,
            "content_length": len(content),
            "link_count": len(links),
            "image_count": len(images),
            "formats": list(response_data.keys()),
        }

    async def execute(
        self,
        url: str,
        formats: Optional[List[str]] = None,
        wait_for: Optional[int] = None,
        screenshot: Optional[bool] = None,
        include_tags: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None,
        extract_images: Optional[bool] = None,
        extract_links: Optional[bool] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute web scraping using Firecrawl API"""
        from ..config import settings

        if not settings.firecrawl_settings.scraping_enabled:
            raise ToolExecutionError("Firecrawl scraping is not enabled")

        # Validate URL
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        try:
            # Build options from parameters and settings
            options = {
                "formats": formats or settings.firecrawl_settings.formats,
                "waitFor": wait_for or settings.firecrawl_settings.wait_for,
                "screenshot": (
                    screenshot
                    if screenshot is not None
                    else settings.firecrawl_settings.screenshot
                ),
                "includeTags": include_tags or settings.firecrawl_settings.include_tags,
                "excludeTags": exclude_tags or settings.firecrawl_settings.exclude_tags,
            }

            # Get client and make request
            client = self._get_client()
            response = await client.post(
                "/v1/scrape",
                json={"url": url, "options": options},
                timeout=timeout or settings.firecrawl_settings.scrape_timeout,
            )

            if response.status_code == 200:
                data = response.json()

                # Extract and process content
                scraped_data = self._extract_content(data, url)

                # Override extraction settings if explicitly provided
                if extract_images is not None:
                    scraped_data["extract_images"] = extract_images
                if extract_links is not None:
                    scraped_data["extract_links"] = extract_links

                logger.info(f"Successfully scraped {url} with Firecrawl")
                return scraped_data
            else:
                error_msg = (
                    f"Firecrawl API error: {response.status_code} - {response.text}"
                )
                logger.error(error_msg)
                raise ToolExecutionError(error_msg)

        except httpx.TimeoutException:
            error_msg = f"Firecrawl request timed out for {url}"
            logger.error(error_msg)
            raise ToolExecutionError(error_msg)
        except Exception as e:
            error_msg = f"Firecrawl scraping failed for {url}: {str(e)}"
            logger.error(error_msg)
            raise ToolExecutionError(error_msg)

    async def batch_scrape(
        self,
        urls: List[str],
        formats: Optional[List[str]] = None,
        wait_for: Optional[int] = None,
        screenshot: Optional[bool] = None,
        include_tags: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None,
        timeout: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Scrape multiple URLs concurrently"""
        from ..config import settings

        # Limit concurrent requests
        semaphore = asyncio.Semaphore(
            settings.firecrawl_settings.max_concurrent_scrapes
        )

        async def scrape_with_semaphore(url: str) -> Dict[str, Any]:
            async with semaphore:
                try:
                    return await self.execute(
                        url=url,
                        formats=formats,
                        wait_for=wait_for,
                        screenshot=screenshot,
                        include_tags=include_tags,
                        exclude_tags=exclude_tags,
                        timeout=timeout,
                    )
                except Exception as e:
                    return {
                        "url": url,
                        "error": str(e),
                        "success": False,
                    }

        tasks = [scrape_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    {
                        "url": urls[i],
                        "error": str(result),
                        "success": False,
                    }
                )
            else:
                processed_results.append(result)

        return processed_results

    def cleanup(self):
        """Clean up resources"""
        if self._client:
            try:
                # Try to close the client if there's an event loop running
                loop = asyncio.get_running_loop()
                loop.create_task(self._client.aclose())
            except RuntimeError:
                # No event loop running, create a new one to close the client
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._client.aclose())
                loop.close()
            self._client = None
            logger.info("Firecrawl HTTP client cleaned up")

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except Exception:
            # Ignore errors during cleanup in destructor
            pass
