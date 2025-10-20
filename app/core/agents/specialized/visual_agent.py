"""
Visual LMM Agent for AI Assistant System

This agent specializes in visual analysis and browser control with visual understanding,
integrating image processing, visual analysis, and web automation capabilities.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, Union
from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models.base import BaseChatModel

from app.core.agents.base.base import BaseAgent, AgentResult
from app.core.tools.execution.registry import ToolRegistry
from app.core.tools.visual.image_processor import ImageProcessorTool
from app.core.tools.visual.visual_analyzer import VisualAnalyzerTool
from app.core.tools.web.playwright_tool import PlaywrightTool
from app.core.tools.web.firecrawl_tool import FirecrawlTool

logger = logging.getLogger(__name__)


class VisualAgent(BaseAgent):
    """Agent specialized in visual analysis and browser control with visual understanding"""
    
    def __init__(
        self,
        llm: BaseChatModel,
        tool_registry: ToolRegistry,
        max_iterations: int = 3,
        default_visual_model: str = "openai_vision:gpt-4-vision-preview",
        enable_browser_control: bool = True,
    ):
        # BaseAgent only takes tool_registry and max_iterations
        super().__init__(tool_registry, max_iterations)
        self.llm = llm
        self.default_visual_model = default_visual_model
        self.enable_browser_control = enable_browser_control
        
        # Initialize visual tools
        self.image_processor = ImageProcessorTool()
        self.visual_analyzer = VisualAnalyzerTool(default_model=default_visual_model)
        self.playwright_tool = PlaywrightTool() if enable_browser_control else None
        self.firecrawl_tool = FirecrawlTool()
        
        # Agent configuration
        self._specialized_prompt = self._create_specialized_prompt()
        self._max_concurrent_analyses = 3
        self._screenshot_quality = 85
        self._default_image_format = "JPEG"
    
    @property
    def name(self) -> str:
        return "visual_agent"
    
    @property
    def description(self) -> str:
        return "Specialized agent for visual analysis, image processing, and browser control with visual understanding"
    
    def _create_specialized_prompt(self) -> str:
        """Create specialized system prompt for visual analysis tasks"""
        return """You are a Visual Analysis Specialist Agent with advanced capabilities in:

VISUAL ANALYSIS CAPABILITIES:
- Analyze images from URLs, files, or base64 data
- Extract text from images using OCR
- Detect and identify objects, people, and scenes
- Compare multiple images for similarities and differences
- Provide detailed image descriptions for accessibility
- Analyze technical aspects of images (composition, lighting, etc.)

WEB VISUAL CAPABILITIES:
- Take screenshots of web pages and analyze them
- Extract and analyze images from web content
- Control browsers with visual understanding
- Perform visual web scraping and content extraction
- Analyze web page layouts and designs
- Identify interactive elements visually

INTEGRATION APPROACH:
1. Analyze the user's request to determine visual needs
2. Extract or capture relevant images
3. Process and optimize images for analysis
4. Use appropriate visual models for analysis
5. Integrate visual insights with web automation when needed
6. Provide comprehensive visual reports

VISUAL TASK TYPES:
- Image description and analysis
- OCR and text extraction
- Object detection and identification
- Image comparison and change detection
- Web page screenshot analysis
- Visual web scraping
- Browser automation with visual feedback
- Accessibility analysis

RESPONSE FORMAT:
Always provide structured responses including:
- Visual analysis results
- Confidence levels when applicable
- Recommendations for further visual processing
- Integration with web automation if relevant
- Technical details about the analysis process

Remember to handle visual tasks efficiently and provide actionable insights from visual content."""
    
    async def _analyze_visual_request(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze the visual request to determine the best approach"""
        analysis_prompt = f"""
        Analyze this visual analysis request and determine the best approach:
        
        USER REQUEST: {query}
        
        Please analyze and provide a JSON response with:
        - task_type: "image_analysis", "web_visual", "ocr", "comparison", "browser_control", "mixed"
        - visual_sources: list of potential image sources (URLs, files, screenshots needed)
        - analysis_types: list of analyses needed (describe, ocr, objects, compare, etc.)
        - web_actions: list of web actions needed (screenshot, scrape, navigate, etc.)
        - priority: primary focus of the request
        - complexity: "simple", "moderate", "complex"
        - recommended_approach: step-by-step approach
        """
        
        try:
            messages = [
                SystemMessage(content=self._specialized_prompt),
                HumanMessage(content=analysis_prompt),
            ]
            
            response = await self.llm.agenerate([messages])
            analysis_text = response.generations[0][0].text
            
            # Extract JSON from response
            if "```json" in analysis_text:
                json_str = analysis_text.split("```json")[1].split("```")[0].strip()
            elif "```" in analysis_text:
                json_str = analysis_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = analysis_text.strip()
            
            return json.loads(json_str)
            
        except Exception as e:
            logger.warning(f"Request analysis failed: {str(e)}")
            # Fallback analysis
            return {
                "task_type": "image_analysis",
                "visual_sources": [],
                "analysis_types": ["describe"],
                "web_actions": [],
                "priority": "general_analysis",
                "complexity": "moderate",
                "recommended_approach": "Extract images and analyze with visual model",
            }
    
    def _extract_image_sources(self, query: str, context: Dict[str, Any] = None) -> List[str]:
        """Extract potential image sources from the query"""
        sources = []
        
        # Extract URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, query)
        sources.extend(urls)
        
        # Check for image file extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        for ext in image_extensions:
            pattern = rf'\S+{re.escape(ext)}'
            matches = re.findall(pattern, query, re.IGNORECASE)
            sources.extend(matches)
        
        # Check for base64 images
        base64_pattern = r'data:image/[^;]+;base64,[A-Za-z0-9+/]+={0,2}'
        base64_matches = re.findall(base64_pattern, query)
        sources.extend(base64_matches)
        
        # Check context for additional sources
        if context:
            if "images" in context:
                sources.extend(context["images"])
            if "urls" in context:
                sources.extend(context["urls"])
        
        return list(set(sources))  # Remove duplicates
    
    async def _capture_web_screenshots(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Capture screenshots of web pages"""
        if not self.playwright_tool or not urls:
            return []
        
        screenshots = []
        
        for url in urls:
            try:
                result = await self.playwright_tool.execute(
                    url=url,
                    screenshot=True,
                    wait_for=3000,
                    extract_text=False,
                )
                
                if result.get("screenshot"):
                    screenshots.append({
                        "url": url,
                        "screenshot": result["screenshot"],
                        "title": result.get("title", ""),
                        "timestamp": result.get("metadata", {}).get("timestamp"),
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to capture screenshot for {url}: {str(e)}")
        
        return screenshots
    
    async def _extract_web_images(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Extract images from web pages"""
        if not urls:
            return []
        
        extracted_images = []
        
        for url in urls:
            try:
                result = await self.firecrawl_tool.execute(
                    url=url,
                    formats=["markdown"],
                    extract_images=True,
                    extract_links=False,
                )
                
                if result.get("images"):
                    for image in result["images"]:
                        extracted_images.append({
                            "source_url": url,
                            "image_url": image.get("src", ""),
                            "alt_text": image.get("alt", ""),
                            "title": image.get("title", ""),
                        })
                        
            except Exception as e:
                logger.warning(f"Failed to extract images from {url}: {str(e)}")
        
        return extracted_images
    
    async def _perform_visual_analysis(
        self,
        images: List[str],
        analysis_types: List[str],
        model: str = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Perform visual analysis on images"""
        if not images:
            return {"success": False, "error": "No images to analyze"}
        
        model = model or self.default_visual_model
        results = {}
        
        # Perform different types of analysis
        for analysis_type in analysis_types:
            try:
                result = await self.visual_analyzer.execute(
                    images=images,
                    analysis_type=analysis_type,
                    model=model,
                    process_images=True,
                    context=context,
                )
                
                results[analysis_type] = result
                
            except Exception as e:
                logger.error(f"Failed {analysis_type} analysis: {str(e)}")
                results[analysis_type] = {
                    "success": False,
                    "error": str(e),
                    "analysis_type": analysis_type,
                }
        
        return {
            "success": True,
            "results": results,
            "image_count": len(images),
            "analysis_types": analysis_types,
            "model": model,
        }
    
    async def _generate_comprehensive_response(
        self,
        query: str,
        analysis_results: Dict[str, Any],
        request_analysis: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> str:
        """Generate a comprehensive response based on visual analysis results"""
        
        response_prompt = f"""
        Based on the visual analysis results, provide a comprehensive response to the user's request.
        
        USER REQUEST: {query}
        
        REQUEST ANALYSIS: {json.dumps(request_analysis, indent=2)}
        
        VISUAL ANALYSIS RESULTS: {json.dumps(analysis_results, indent=2)}
        
        Please provide:
        1. A clear, direct answer to the user's question
        2. Key insights from the visual analysis
        3. Any relevant details or observations
        4. Recommendations or next steps if applicable
        5. Technical details about the analysis process
        
        Format your response in a clear, structured way that addresses the user's needs.
        """
        
        try:
            messages = [
                SystemMessage(content=self._specialized_prompt),
                HumanMessage(content=response_prompt),
            ]
            
            response = await self.llm.agenerate([messages])
            return response.generations[0][0].text.strip()
            
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            return f"I've completed the visual analysis, but I'm having trouble formulating the response. The analysis was successful with {len(analysis_results.get('results', {}))} different types of analysis performed."
    
    async def _process_message_impl(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        context: Dict[str, Any] = None,
    ) -> AgentResult:
        """
        Internal implementation of message processing for visual analysis
        
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
            # Analyze the visual request
            request_analysis = await self._analyze_visual_request(message, context)
            
            # Extract image sources from the query
            image_sources = self._extract_image_sources(message, context)
            
            # Collect all images to analyze
            all_images = []
            
            # Add direct image sources
            all_images.extend(image_sources)
            
            # Handle web visual tasks
            web_urls = []
            if request_analysis.get("task_type") in ["web_visual", "browser_control", "mixed"]:
                # Extract URLs from query
                url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
                web_urls = re.findall(url_pattern, message)
                
                # Capture screenshots if needed
                if "screenshot" in request_analysis.get("web_actions", []):
                    screenshots = await self._capture_web_screenshots(web_urls)
                    for screenshot in screenshots:
                        all_images.append(f"data:image/jpeg;base64,{screenshot['screenshot']}")
                
                # Extract images from web pages
                if "extract_images" in request_analysis.get("web_actions", []):
                    web_images = await self._extract_web_images(web_urls)
                    for web_image in web_images:
                        if web_image.get("image_url"):
                            all_images.append(web_image["image_url"])
            
            # Perform visual analysis if we have images
            analysis_results = {}
            if all_images:
                analysis_types = request_analysis.get("analysis_types", ["describe"])
                analysis_results = await self._perform_visual_analysis(
                    images=all_images,
                    analysis_types=analysis_types,
                    context=context
                )
            
            # Generate comprehensive response
            response = await self._generate_comprehensive_response(
                query=message,
                analysis_results=analysis_results,
                request_analysis=request_analysis,
                context=context
            )
            
            execution_time = time.time() - start_time
            
            # Prepare metadata
            metadata = {
                "request_analysis": request_analysis,
                "image_sources": image_sources,
                "web_urls": web_urls,
                "total_images": len(all_images),
                "analysis_results": analysis_results,
                "visual_model_used": self.default_visual_model,
            }
            
            return AgentResult(
                success=True,
                response=response,
                agent_name=self.name,
                execution_time=execution_time,
                conversation_id=self._current_conversation_id,
                metadata=metadata,
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Visual agent processing failed: {str(e)}"
            logger.error(error_msg)
            
            return AgentResult(
                success=False,
                response=f"I encountered an error while processing your visual request: {str(e)}",
                agent_name=self.name,
                execution_time=execution_time,
                conversation_id=self._current_conversation_id,
                error=error_msg,
                metadata={"error": str(e)},
            )
    
    async def analyze_image_url(
        self,
        image_url: str,
        analysis_type: str = "describe",
        model: str = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Analyze a single image from URL"""
        return await self.visual_analyzer.execute(
            images=image_url,
            analysis_type=analysis_type,
            model=model,
            context=context,
        )
    
    async def analyze_webpage_visual(
        self,
        url: str,
        analysis_types: List[str] = None,
        include_screenshot: bool = True,
        include_extracted_images: bool = True,
        model: str = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Analyze a web page visually"""
        analysis_types = analysis_types or ["describe", "analyze"]
        model = model or self.default_visual_model
        
        images_to_analyze = []
        
        # Take screenshot if requested
        if include_screenshot and self.playwright_tool:
            try:
                screenshot_result = await self.playwright_tool.execute(
                    url=url,
                    screenshot=True,
                    wait_for=3000,
                )
                
                if screenshot_result.get("screenshot"):
                    images_to_analyze.append(f"data:image/jpeg;base64,{screenshot_result['screenshot']}")
                    
            except Exception as e:
                logger.warning(f"Failed to capture screenshot for {url}: {str(e)}")
        
        # Extract images from page if requested
        if include_extracted_images:
            try:
                extracted_images = await self._extract_web_images([url])
                for img in extracted_images:
                    if img.get("image_url"):
                        images_to_analyze.append(img["image_url"])
                        
            except Exception as e:
                logger.warning(f"Failed to extract images from {url}: {str(e)}")
        
        # Perform analysis
        if images_to_analyze:
            return await self._perform_visual_analysis(
                images=images_to_analyze,
                analysis_types=analysis_types,
                model=model,
                context=context,
            )
        else:
            return {
                "success": False,
                "error": "No images found to analyze",
                "url": url,
            }
    
    async def compare_webpages_visual(
        self,
        urls: List[str],
        comparison_focus: str = "general",
        model: str = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Compare multiple web pages visually"""
        if len(urls) < 2:
            return {"success": False, "error": "At least 2 URLs are required for comparison"}
        
        # Capture screenshots of all pages
        screenshots = []
        for url in urls:
            try:
                screenshot_result = await self.playwright_tool.execute(
                    url=url,
                    screenshot=True,
                    wait_for=3000,
                )
                
                if screenshot_result.get("screenshot"):
                    screenshots.append(f"data:image/jpeg;base64,{screenshot_result['screenshot']}")
                    
            except Exception as e:
                logger.warning(f"Failed to capture screenshot for {url}: {str(e)}")
        
        # Compare screenshots
        if screenshots:
            return await self.visual_analyzer.compare_images(
                images=screenshots,
                comparison_focus=comparison_focus,
                model=model,
                context=context,
            )
        else:
            return {
                "success": False,
                "error": "No screenshots captured for comparison",
                "urls": urls,
            }
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        base_stats = {
            "specialization": "visual_analysis",
            "default_visual_model": self.default_visual_model,
            "browser_control_enabled": self.enable_browser_control,
            "max_concurrent_analyses": self._max_concurrent_analyses,
            "screenshot_quality": self._screenshot_quality,
            "default_image_format": self._default_image_format,
        }
        return base_stats
    
    async def cleanup(self):
        """Clean up resources"""
        await self.image_processor.cleanup()
        await self.visual_analyzer.cleanup()
        
        if self.playwright_tool:
            await self.playwright_tool.cleanup()
        
        logger.info("Visual agent resources cleaned up")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.cleanup())
        except RuntimeError:
            # No event loop running, can't cleanup
            pass