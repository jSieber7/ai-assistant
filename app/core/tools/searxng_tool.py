"""
SearXNG search tool for the AI Assistant tool system.

This module provides a tool for performing web searches using SearXNG.
"""

import aiohttp
import asyncio
from typing import Dict, Any, List, Optional
from urllib.parse import urlencode, quote_plus
from bs4 import BeautifulSoup
from .base import BaseTool, ToolExecutionError


class SearXNGTool(BaseTool):
    """Perform web searches using SearXNG"""

    @property
    def name(self) -> str:
        return "searxng_search"

    @property
    def description(self) -> str:
        return "Search the web using SearXNG privacy-respecting search engine"

    @property
    def keywords(self) -> List[str]:
        return [
            "search",
            "web",
            "internet",
            "google",
            "find",
            "lookup",
            "query",
            "searxng",
        ]

    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "query": {
                "type": str,
                "description": "Search query",
                "required": True,
            },
            "category": {
                "type": str,
                "description": "Search category (general, images, videos, news, map, music, it, science)",
                "required": False,
                "default": "general",
            },
            "language": {
                "type": str,
                "description": "Search language (e.g., 'en', 'en-US', 'auto')",
                "required": False,
                "default": "auto",
            },
            "time_range": {
                "type": str,
                "description": "Time range (day, week, month, year)",
                "required": False,
                "default": "",
            },
            "safesearch": {
                "type": int,
                "description": "Safe search level (0=none, 1=moderate, 2=strict)",
                "required": False,
                "default": 0,
            },
            "results_count": {
                "type": int,
                "description": "Number of results to return (max 20)",
                "required": False,
                "default": 10,
            },
        }

    async def execute(
        self,
        query: str,
        category: str = "general",
        language: str = "auto",
        time_range: str = "",
        safesearch: int = 0,
        results_count: int = 10,
    ) -> Dict[str, Any]:
        """Execute web search using SearXNG"""
        from ..config import settings

        searxng_url = settings.searxng_url

        if not searxng_url:
            raise ToolExecutionError("SearXNG URL not configured")

        # Validate parameters
        if results_count < 1 or results_count > 20:
            results_count = min(max(results_count, 1), 20)

        if safesearch not in [0, 1, 2]:
            safesearch = 0

        # Prepare search parameters for POST request (HTML format)
        params = {
            "q": query,
            "language": language,
            "safesearch": str(safesearch),
            "theme": "simple",
        }

        # Add category if not general
        if category != "general":
            params[f"category_{category}"] = "1"
        else:
            params["category_general"] = "1"

        # Add time range if specified
        if time_range:
            params["time_range"] = time_range

        # Construct search URL
        search_url = f"{searxng_url.rstrip('/')}/search"

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "X-Forwarded-For": "127.0.0.1",
                "X-Real-IP": "127.0.0.1",
            }
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.post(search_url, data=params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ToolExecutionError(
                            f"SearXNG search failed with status {response.status}: {error_text}"
                        )

                    # Parse HTML response
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract search results from HTML
                    processed_results = []
                    result_elements = soup.find_all('article', class_='result')
                    
                    for element in result_elements[:results_count]:
                        title_elem = element.find('h3')
                        if not title_elem:
                            continue
                            
                        title_link = title_elem.find('a')
                        if not title_link:
                            continue
                            
                        title = title_link.get_text(strip=True)
                        url = title_link.get('href', '')
                        
                        # Extract content/description
                        content_elem = element.find('p', class_='content')
                        content = content_elem.get_text(strip=True) if content_elem else ""
                        
                        # Extract engine information
                        engine_elem = element.find('div', class_='engines')
                        engine = engine_elem.get_text(strip=True) if engine_elem else ""
                        
                        processed_results.append({
                            "title": title,
                            "url": url,
                            "content": content,
                            "engine": engine,
                            "category": category,
                            "score": 0,  # Not available in HTML
                        })

                    return {
                        "query": query,
                        "results": processed_results,
                        "total_results": len(processed_results),
                        "search_time": 0,  # Not available in HTML
                        "engines": [],  # Not easily extracted from HTML
                        "answers": [],  # Not easily extracted from HTML
                        "infoboxes": [],  # Not easily extracted from HTML
                        "suggestions": [],  # Not easily extracted from HTML
                    }

        except aiohttp.ClientError as e:
            raise ToolExecutionError(f"Failed to connect to SearXNG: {e}")
        except Exception as e:
            raise ToolExecutionError(f"Search failed: {e}")