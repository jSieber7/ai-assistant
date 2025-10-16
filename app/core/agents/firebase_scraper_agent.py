"""
Firebase Scraper Agent for the AI Assistant system.

This agent specializes in web scraping tasks using Firebase infrastructure
with advanced features like Selenium rendering and Firestore storage.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models.base import BaseChatModel

from .base import BaseAgent, AgentResult
from ..tools.registry import tool_registry
from ..tools.firebase_scraper_tool import FirebaseScraperTool

logger = logging.getLogger(__name__)


class FirebaseScraperAgent(BaseAgent):
    """Agent specialized in web scraping using Firebase infrastructure"""

    def __init__(
        self,
        llm: BaseChatModel,
        tool_registry=tool_registry,
        max_iterations: int = 3,
        timeout: int = 60,
    ):
        # BaseAgent only takes tool_registry and max_iterations
        super().__init__(tool_registry, max_iterations)
        self.llm = llm
        self._scraper_tool = None
        self._specialized_prompt = self._create_specialized_prompt()

    @property
    def name(self) -> str:
        return "firebase_scraper_agent"

    @property
    def description(self) -> str:
        return "Specialized agent for web scraping using Firebase with advanced rendering and storage"

    def _create_specialized_prompt(self) -> str:
        """Create specialized system prompt for Firebase scraping tasks"""
        return """You are a Firebase Web Scraping Specialist Agent. Your expertise includes:

WEB SCRAPING CAPABILITIES:
- Extract content from websites using both HTTP requests and Selenium rendering
- Handle JavaScript-heavy websites that require browser automation
- Extract structured data including text content, links, images, and metadata
- Handle pagination and multi-page content extraction

FIREBASE INTEGRATION:
- Store scraped data in Firebase Firestore for persistent storage
- Use Firebase Storage for image and file storage when needed
- Maintain data consistency and proper indexing
- Handle authentication and security with Firebase services

ADVANCED FEATURES:
- JavaScript rendering for dynamic content
- Image extraction and processing
- Link discovery and crawling
- Content cleaning and normalization
- Error handling and retry mechanisms

TASK EXECUTION STRATEGY:
1. Analyze the scraping request and determine the best approach
2. Choose between simple HTTP scraping or Selenium rendering
3. Extract content with proper error handling
4. Store results in Firebase Firestore
5. Provide comprehensive results with metadata

RESPONSE FORMAT:
Always provide structured responses including:
- Scraped content summary
- Data quality assessment
- Storage information (Firestore document ID)
- Any issues encountered
- Recommendations for further processing

Remember to handle errors gracefully and provide helpful feedback to users."""

    async def _get_scraper_tool(self) -> FirebaseScraperTool:
        """Get or create the Firebase scraper tool"""
        if self._scraper_tool is None:
            self._scraper_tool = FirebaseScraperTool()
        return self._scraper_tool

    async def _analyze_scraping_task(self, query: str) -> Dict[str, Any]:
        """Analyze the scraping task to determine the best approach"""
        analysis_prompt = f"""
        Analyze this web scraping request and determine the best approach:
        
        USER REQUEST: {query}
        
        Please analyze:
        1. What type of content is being requested?
        2. Does it likely require JavaScript rendering?
        3. What specific data should be extracted?
        4. Any potential challenges or considerations?
        
        Respond with a JSON analysis including:
        - content_type: "article", "product", "news", "general", etc.
        - requires_js: true/false
        - extraction_targets: list of data points to extract
        - challenges: list of potential issues
        - recommendations: list of recommendations
        """

        messages = [
            SystemMessage(content=self._specialized_prompt),
            HumanMessage(content=analysis_prompt),
        ]

        response = await self.llm.agenerate([messages])
        analysis_text = response.generations[0][0].text

        import json

        try:
            # Extract JSON from response
            if "```json" in analysis_text:
                json_str = analysis_text.split("```json")[1].split("```")[0].strip()
            elif "```" in analysis_text:
                json_str = analysis_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = analysis_text.strip()

            return json.loads(json_str)
        except Exception:
            # Fallback analysis
            return {
                "content_type": "general",
                "requires_js": "javascript" in query.lower()
                or "dynamic" in query.lower(),
                "extraction_targets": ["content", "links", "metadata"],
                "challenges": ["Unknown website structure"],
                "recommendations": ["Use Selenium for robust scraping"],
            }

    async def execute(
        self, query: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute web scraping task using Firebase infrastructure"""
        logger.info(f"FirebaseScraperAgent executing query: {query}")

        # Analyze the scraping task
        task_analysis = await self._analyze_scraping_task(query)

        # Extract URLs from query
        urls = self._extract_urls_from_query(query)
        if not urls:
            return {
                "success": False,
                "error": "No valid URLs found in the query",
                "analysis": task_analysis,
            }

        # Get the scraper tool
        scraper_tool = await self._get_scraper_tool()

        try:
            results = []
            for url in urls:
                # Determine scraping method based on analysis
                use_selenium = task_analysis.get("requires_js", True)

                # Execute scraping
                scraped_data = await scraper_tool.execute(
                    url=url,
                    use_selenium=use_selenium,
                    store_in_firestore=True,
                    extract_images=task_analysis.get("extract_images", False),
                    extract_links=True,
                    timeout=30,
                )

                results.append(
                    {
                        "url": url,
                        "data": scraped_data,
                        "method": "selenium" if use_selenium else "http",
                        "analysis": task_analysis,
                    }
                )

            # Generate comprehensive summary
            summary = await self._generate_summary(results, task_analysis)

            return {
                "success": True,
                "results": results,
                "summary": summary,
                "total_urls": len(urls),
                "total_content": sum(
                    len(r["data"].get("content", "")) for r in results
                ),
                "analysis": task_analysis,
            }

        except Exception as e:
            logger.error(f"Firebase scraping failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "analysis": task_analysis,
                "urls_attempted": urls,
            }

    def _extract_urls_from_query(self, query: str) -> List[str]:
        """Extract URLs from the query text"""
        import re

        # Simple URL extraction
        url_pattern = r"https?://[^\s]+|www\.[^\s]+"
        urls = re.findall(url_pattern, query)

        # Clean and validate URLs
        valid_urls = []
        for url in urls:
            if not url.startswith("http"):
                url = f"https://{url}"
            # Basic validation
            if "." in url and len(url) > 10:
                valid_urls.append(url)

        return valid_urls

    async def _generate_summary(
        self, results: List[Dict], analysis: Dict
    ) -> Dict[str, Any]:
        """Generate a comprehensive summary of scraping results"""
        if not results:
            return {"message": "No results to summarize"}

        summary_prompt = f"""
        Generate a comprehensive summary of the web scraping results:
        
        TASK ANALYSIS: {analysis}
        
        RESULTS:
        {json.dumps(results, indent=2)}
        
        Please provide a structured summary including:
        - Overall success assessment
        - Content quality evaluation
        - Data completeness
        - Any issues encountered
        - Recommendations for next steps
        
        Format as JSON with these fields:
        - overall_assessment
        - content_quality_score (0-100)
        - data_completeness
        - issues_found
        - recommendations
        - key_insights
        """

        messages = [
            SystemMessage(content=self._specialized_prompt),
            HumanMessage(content=summary_prompt),
        ]

        response = await self.llm.agenerate([messages])
        summary_text = response.generations[0][0].text

        try:
            # Extract JSON from response
            if "```json" in summary_text:
                json_str = summary_text.split("```json")[1].split("```")[0].strip()
            else:
                json_str = summary_text.strip()

            return json.loads(json_str)
        except Exception:
            # Fallback summary
            return {
                "overall_assessment": "Scraping completed",
                "content_quality_score": 80,
                "data_completeness": "Good",
                "issues_found": [],
                "recommendations": ["Review extracted content for accuracy"],
                "key_insights": [f"Scraped {len(results)} URLs successfully"],
            }

    async def batch_scrape(
        self, urls: List[str], use_selenium: bool = True
    ) -> Dict[str, Any]:
        """Perform batch scraping of multiple URLs"""
        scraper_tool = await self._get_scraper_tool()

        try:
            results = await scraper_tool.batch_scrape(
                urls=urls,
                use_selenium=use_selenium,
                store_in_firestore=True,
                timeout=30,
            )

            # Filter successful results
            successful_results = []
            for i, result in enumerate(results):
                if not isinstance(result, Exception):
                    successful_results.append(
                        {"url": urls[i], "data": result, "success": True}
                    )
                else:
                    successful_results.append(
                        {"url": urls[i], "error": str(result), "success": False}
                    )

            return {
                "success": True,
                "results": successful_results,
                "total_urls": len(urls),
                "successful": len([r for r in successful_results if r["success"]]),
                "failed": len([r for r in successful_results if not r["success"]]),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "total_urls": len(urls),
                "successful": 0,
                "failed": len(urls),
            }

    async def _process_message_impl(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        context: Dict[str, Any] = None,
    ) -> AgentResult:
        """
        Internal implementation of message processing for Firebase scraping

        Args:
            message: User message to process
            conversation_id: Optional conversation ID for context
            context: Additional context information

        Returns:
            AgentResult with response and tool execution results
        """
        import time

        start_time = time.time()

        try:
            # Execute the scraping task
            result = await self.execute(message, context)

            # Convert to AgentResult format
            if result.get("success", False):
                return AgentResult(
                    success=True,
                    response=result.get("summary", {}).get(
                        "overall_assessment", "Scraping completed successfully"
                    ),
                    agent_name=self.name,
                    execution_time=time.time() - start_time,
                    metadata=result,
                )
            else:
                return AgentResult(
                    success=False,
                    response=f"Scraping failed: {result.get('error', 'Unknown error')}",
                    agent_name=self.name,
                    execution_time=time.time() - start_time,
                    error=result.get("error"),
                    metadata=result,
                )
        except Exception as e:
            return AgentResult(
                success=False,
                response=f"Error during scraping: {str(e)}",
                agent_name=self.name,
                execution_time=time.time() - start_time,
                error=str(e),
                metadata={"error": str(e)},
            )

    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        base_stats = {
            "specialization": "firebase_web_scraping",
            "scraper_tool_available": self._scraper_tool is not None,
            "firebase_initialized": (
                getattr(self._scraper_tool, "_firebase_initialized", False)
                if self._scraper_tool
                else False
            ),
        }
        return base_stats
