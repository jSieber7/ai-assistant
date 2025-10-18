"""
Content processor for multi-writer system
"""

from typing import List, Dict, Any
from app.core.tools.web.firecrawl_tool import FirecrawlTool


class ContentProcessor:
    """Processes web content for multi-writer system"""

    def __init__(self, firecrawl_api_key: str):
        self.firecrawl = FirecrawlTool(firecrawl_api_key)

    async def process_sources(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process multiple sources and prepare content for writers
        """
        results = {
            "processed_content": [],
            "failed_sources": [],
            "metadata": {"total_sources": len(sources), "successful": 0, "failed": 0},
        }

        # Extract URLs and scrape content
        urls = [source.get("url") for source in sources if source.get("url")]

        if urls:
            scrape_results = await self.firecrawl.batch_scrape(urls)

            for i, result in enumerate(scrape_results):
                if result.success:
                    # Process and clean content
                    content = self._clean_content(result.data)
                    content["source_info"] = sources[i]
                    results["processed_content"].append(content)
                    results["metadata"]["successful"] += 1
                else:
                    results["failed_sources"].append(
                        {"source": sources[i], "error": result.error}
                    )
                    results["metadata"]["failed"] += 1

        return results

    def _clean_content(self, raw_content: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and structure raw content for AI processing"""
        return {
            "title": raw_content.get("title", ""),
            "content": raw_content.get("markdown", ""),
            "raw_content": raw_content.get("raw", ""),
            "url": raw_content.get("url", ""),
            "word_count": len(raw_content.get("markdown", "").split()),
            "key_points": self._extract_key_points(raw_content.get("markdown", "")),
            "links": raw_content.get("links", []),
            "images": raw_content.get("images", []),
        }

    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from content using simple heuristics"""
        lines = content.split("\n")
        key_points = []

        for line in lines:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            # Look for bullet points, numbered lists, or emphasized text
            if (
                line.startswith(("#", "*", "-", "•"))
                or line[0].isdigit()
                and "." in line[:5]
                or "**" in line
                or "*" in line
            ):
                key_points.append(line)

        return key_points[:10]  # Limit to top 10 points
