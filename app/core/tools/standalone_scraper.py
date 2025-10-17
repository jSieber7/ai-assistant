"""
Standalone Scraper interface for independent web scraping operations.

This module provides a simple interface for using the FirecrawlTool
independently from any agent or workflow.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from .firecrawl_tool import FirecrawlTool
from .base import ToolResult
from ..config import settings

logger = logging.getLogger(__name__)


class StandaloneScraper:
    """
    Standalone interface for web scraping using Firecrawl.
    
    This class provides a simple interface for scraping web content
    without requiring any agent or complex workflow setup.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the standalone scraper.
        
        Args:
            api_key: Firecrawl API key (not used in Docker mode)
        """
        self._scraper = FirecrawlTool(api_key=api_key)
        self._initialized = False
    
    async def initialize(self) -> bool:
        """
        Initialize the scraper and check connectivity.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Check if scraping is enabled
            if not settings.firecrawl_settings.scraping_enabled:
                logger.error("Firecrawl scraping is not enabled in settings")
                return False
            
            # Check Docker health
            if not await self._scraper._check_docker_health():
                logger.error("Firecrawl Docker instance is not healthy")
                return False
            
            self._initialized = True
            logger.info("Standalone scraper initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize standalone scraper: {str(e)}")
            return False
    
    async def scrape_url(
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
    ) -> ToolResult:
        """
        Scrape a single URL.
        
        Args:
            url: URL to scrape
            formats: Output formats (markdown, raw, html, etc.)
            wait_for: Time to wait for page load in milliseconds
            screenshot: Take a screenshot of the page
            include_tags: HTML tags to include in extraction
            exclude_tags: HTML tags to exclude from extraction
            extract_images: Extract images from page
            extract_links: Extract links from page
            timeout: Timeout in seconds
            
        Returns:
            ToolResult with scraped data
        """
        if not self._initialized:
            if not await self.initialize():
                return ToolResult(
                    success=False,
                    data=None,
                    error="Scraper not initialized",
                    tool_name="standalone_scraper",
                    execution_time=0.0
                )
        
        try:
            result = await self._scraper.execute(
                url=url,
                formats=formats,
                wait_for=wait_for,
                screenshot=screenshot,
                include_tags=include_tags,
                exclude_tags=exclude_tags,
                extract_images=extract_images,
                extract_links=extract_links,
                timeout=timeout,
            )
            
            # Convert to ToolResult if needed
            if not isinstance(result, ToolResult):
                result = ToolResult(
                    success=True,
                    data=result,
                    tool_name="standalone_scraper",
                    execution_time=0.0,
                    metadata={"method": "direct_scrape"}
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {str(e)}")
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                tool_name="standalone_scraper",
                execution_time=0.0
            )
    
    async def scrape_multiple_urls(
        self,
        urls: List[str],
        formats: Optional[List[str]] = None,
        wait_for: Optional[int] = None,
        screenshot: Optional[bool] = None,
        include_tags: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None,
        timeout: Optional[int] = None,
        max_concurrent: Optional[int] = None,
    ) -> List[ToolResult]:
        """
        Scrape multiple URLs concurrently.
        
        Args:
            urls: List of URLs to scrape
            formats: Output formats (markdown, raw, html, etc.)
            wait_for: Time to wait for page load in milliseconds
            screenshot: Take screenshots of pages
            include_tags: HTML tags to include in extraction
            exclude_tags: HTML tags to exclude from extraction
            timeout: Timeout in seconds
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of ToolResults, one for each URL
        """
        if not self._initialized:
            if not await self.initialize():
                return [ToolResult(
                    success=False,
                    data=None,
                    error="Scraper not initialized",
                    tool_name="standalone_scraper",
                    execution_time=0.0
                ) for _ in urls]
        
        try:
            # Use the batch_scrape functionality from FirecrawlTool
            batch_results = await self._scraper.batch_scrape(
                urls=urls,
                formats=formats,
                wait_for=wait_for,
                screenshot=screenshot,
                include_tags=include_tags,
                exclude_tags=exclude_tags,
                timeout=timeout,
            )
            
            # Convert batch results to ToolResults
            tool_results = []
            for i, batch_result in enumerate(batch_results):
                url = urls[i] if i < len(urls) else "unknown"
                
                if batch_result.get("success", True) and "error" not in batch_result:
                    tool_results.append(ToolResult(
                        success=True,
                        data=batch_result,
                        tool_name="standalone_scraper",
                        execution_time=0.0,
                        metadata={"url": url, "method": "batch_scrape"}
                    ))
                else:
                    tool_results.append(ToolResult(
                        success=False,
                        data=None,
                        error=batch_result.get("error", "Unknown error"),
                        tool_name="standalone_scraper",
                        execution_time=0.0,
                        metadata={"url": url}
                    ))
            
            return tool_results
            
        except Exception as e:
            logger.error(f"Error in batch scraping: {str(e)}")
            return [ToolResult(
                success=False,
                data=None,
                error=str(e),
                tool_name="standalone_scraper",
                execution_time=0.0
            ) for _ in urls]
    
    async def extract_content_only(self, url: str, min_length: int = 100) -> Optional[str]:
        """
        Extract only the text content from a URL.
        
        Args:
            url: URL to scrape
            min_length: Minimum content length to return
            
        Returns:
            Text content or None if extraction failed or content too short
        """
        result = await self.scrape_url(url, formats=["markdown"])
        
        if result.success:
            content = result.data.get("content", "")
            if len(content.strip()) >= min_length:
                return content
        
        return None
    
    async def extract_links_from_url(self, url: str) -> List[str]:
        """
        Extract all links from a URL.
        
        Args:
            url: URL to scrape
            
        Returns:
            List of links found on the page
        """
        result = await self.scrape_url(
            url,
            formats=["markdown"],
            extract_links=True,
            extract_images=False
        )
        
        if result.success:
            links = result.data.get("links", [])
            return [link.get("url", "") for link in links if link.get("url")]
        
        return []
    
    async def extract_images_from_url(self, url: str) -> List[Dict[str, Any]]:
        """
        Extract all images from a URL.
        
        Args:
            url: URL to scrape
            
        Returns:
            List of image information dictionaries
        """
        result = await self.scrape_url(
            url,
            formats=["markdown"],
            extract_images=True,
            extract_links=False
        )
        
        if result.success:
            return result.data.get("images", [])
        
        return []
    
    async def get_page_metadata(self, url: str) -> Dict[str, Any]:
        """
        Get metadata about a page without full content extraction.
        
        Args:
            url: URL to scrape
            
        Returns:
            Dictionary with page metadata
        """
        result = await self.scrape_url(
            url,
            formats=["markdown"],
            extract_links=False,
            extract_images=False,
            wait_for=1000  # Shorter wait for metadata
        )
        
        if result.success:
            data = result.data
            return {
                "url": url,
                "title": data.get("title", ""),
                "description": data.get("description", ""),
                "content_length": data.get("content_length", 0),
                "link_count": data.get("link_count", 0),
                "image_count": data.get("image_count", 0),
                "metadata": data.get("metadata", {}),
                "success": True
            }
        
        return {
            "url": url,
            "title": "",
            "description": "",
            "content_length": 0,
            "link_count": 0,
            "image_count": 0,
            "metadata": {},
            "success": False,
            "error": result.error
        }
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            await self._scraper.cleanup()
            logger.info("Standalone scraper cleaned up")
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            loop.create_task(self.cleanup())
        except RuntimeError:
            # No event loop running
            pass


# Convenience function for quick scraping
async def quick_scrape(url: str, formats: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """
    Quick convenience function to scrape a URL.
    
    Args:
        url: URL to scrape
        formats: Output formats
        
    Returns:
        Scraped data or None if failed
    """
    scraper = StandaloneScraper()
    try:
        result = await scraper.scrape_url(url, formats=formats)
        return result.data if result.success else None
    finally:
        await scraper.cleanup()