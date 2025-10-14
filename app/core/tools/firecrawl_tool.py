"""
Firecrawl API integration tool for web scraping
"""

import asyncio
import httpx
from typing import Dict, Any, List
from app.core.tools.base import BaseTool, ToolResult


class FirecrawlTool(BaseTool):
    """Tool for scraping web content using Firecrawl API"""

    def __init__(self, api_key: str, base_url: str = "https://api.firecrawl.dev"):
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )

    @property
    def name(self) -> str:
        return "firecrawl_scrape"

    @property
    def description(self) -> str:
        return "Scrape web content using Firecrawl API with advanced options"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "url": {"type": "string", "required": True, "description": "URL to scrape"},
            "options": {
                "type": "object",
                "required": False,
                "description": "Firecrawl options (formats, waitFor, etc.)",
                "properties": {
                    "formats": {"type": "array", "items": {"type": "string"}},
                    "waitFor": {"type": "number"},
                    "screenshot": {"type": "boolean"},
                    "includeTags": {"type": "array", "items": {"type": "string"}},
                    "excludeTags": {"type": "array", "items": {"type": "string"}},
                },
            },
        }

    async def execute(self, url: str, options: Dict[str, Any] = None) -> ToolResult:
        try:
            # Default options for content extraction
            scrape_options = {
                "formats": ["markdown", "raw"],
                "waitFor": 2000,
                "screenshot": False,
                "includeTags": ["article", "main", "content"],
                "excludeTags": ["nav", "footer", "aside", "script", "style"],
            }

            if options:
                scrape_options.update(options)

            # Make request to Firecrawl API
            response = await self.client.post(
                "/v1/scrape", json={"url": url, "options": scrape_options}
            )

            if response.status_code == 200:
                data = response.json()

                # Extract and process content
                content = {
                    "url": url,
                    "title": data.get("data", {}).get("metadata", {}).get("title", ""),
                    "markdown": data.get("data", {}).get("markdown", ""),
                    "raw": data.get("data", {}).get("raw", ""),
                    "links": data.get("data", {}).get("links", []),
                    "images": data.get("data", {}).get("images", []),
                    "metadata": data.get("data", {}).get("metadata", {}),
                }

                return ToolResult(success=True, data=content, tool_name=self.name)
            else:
                return ToolResult(
                    success=False,
                    error=f"Firecrawl API error: {response.status_code} - {response.text}",
                    tool_name=self.name,
                )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Firecrawl tool error: {str(e)}",
                tool_name=self.name,
            )

    async def batch_scrape(
        self, urls: List[str], options: Dict[str, Any] = None
    ) -> List[ToolResult]:
        """Scrape multiple URLs concurrently"""
        tasks = [self.execute(url, options) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)
