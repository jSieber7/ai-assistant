"""
Visual Browser Control Tool for Visual LMM System

This module provides advanced browser control capabilities with visual understanding,
enabling agents to interact with web pages based on visual descriptions and context.
"""

import asyncio
import logging
import base64
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass

from app.core.tools.base.base import BaseTool, ToolExecutionError
from app.core.tools.web.playwright_tool import PlaywrightTool
from app.core.tools.visual.visual_analyzer import VisualAnalyzerTool

logger = logging.getLogger(__name__)


@dataclass
class VisualElement:
    """Represents a visually identified element on a web page"""
    
    selector: str
    description: str
    bbox: Dict[str, int]  # bounding box: {x, y, width, height}
    confidence: float
    element_type: str  # button, link, input, text, etc.
    text_content: Optional[str] = None
    screenshot_region: Optional[str] = None  # base64 screenshot of the element


class VisualBrowserTool(BaseTool):
    """Advanced browser control tool with visual understanding"""
    
    def __init__(
        self,
        visual_model: str = "openai_vision:gpt-4-vision-preview",
        headless: bool = True,
        browser_type: str = "chromium",
        screenshot_quality: int = 85,
    ):
        super().__init__()
        self.visual_model = visual_model
        self.screenshot_quality = screenshot_quality
        
        # Initialize underlying tools
        self.playwright_tool = PlaywrightTool(headless=headless, browser_type=browser_type)
        self.visual_analyzer = VisualAnalyzerTool(default_model=visual_model)
        
        # Visual interaction patterns
        self.interaction_patterns = {
            "click": ["button", "link", "clickable element", "interactive element"],
            "type": ["input field", "text box", "search box", "form field", "textarea"],
            "select": ["dropdown", "select menu", "option", "radio button", "checkbox"],
            "hover": ["menu item", "navigation element", "expandable element"],
        }
    
    @property
    def name(self) -> str:
        return "visual_browser"
    
    @property
    def description(self) -> str:
        return "Advanced browser control with visual understanding for interacting with web pages based on visual descriptions"
    
    @property
    def keywords(self) -> List[str]:
        return [
            "browser",
            "visual",
            "click",
            "interact",
            "navigate",
            "screenshot",
            "element",
            "find",
            "automation",
        ]
    
    @property
    def categories(self) -> List[str]:
        return ["visual", "browser", "automation", "web"]
    
    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "url": {
                "type": str,
                "description": "URL to navigate to",
                "required": False,
            },
            "action": {
                "type": str,
                "description": "Action to perform: navigate, click, type, select, hover, screenshot, analyze",
                "required": False,
                "default": "analyze",
            },
            "visual_description": {
                "type": str,
                "description": "Visual description of the element to interact with (e.g., 'red submit button', 'search box in the header')",
                "required": False,
            },
            "text": {
                "type": str,
                "description": "Text to type (for 'type' action)",
                "required": False,
            },
            "selector": {
                "type": str,
                "description": "CSS selector (alternative to visual_description)",
                "required": False,
            },
            "wait_for": {
                "type": int,
                "description": "Time to wait after action in milliseconds",
                "required": False,
                "default": 2000,
            },
            "analysis_type": {
                "type": str,
                "description": "Type of visual analysis: 'elements', 'layout', 'content', 'accessibility'",
                "required": False,
                "default": "elements",
            },
            "screenshot": {
                "type": bool,
                "description": "Take screenshot after action",
                "required": False,
                "default": True,
            },
        }
    
    async def _take_screenshot(self, page) -> str:
        """Take a screenshot and return as base64"""
        try:
            screenshot_bytes = await page.screenshot()
            return base64.b64encode(screenshot_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to take screenshot: {str(e)}")
            raise ToolExecutionError(f"Screenshot failed: {str(e)}")
    
    async def _analyze_page_screenshot(
        self,
        screenshot_base64: str,
        analysis_type: str = "elements",
        context: str = ""
    ) -> Dict[str, Any]:
        """Analyze a page screenshot using visual understanding"""
        
        analysis_prompts = {
            "elements": (
                "Analyze this web page screenshot and identify all interactive elements. "
                "For each element, provide: description, element type (button, link, input, etc.), "
                "approximate position, and any visible text. Format as JSON with 'elements' array."
            ),
            "layout": (
                "Analyze the layout and structure of this web page. "
                "Describe the main sections, navigation, content areas, and overall organization. "
                "Format as JSON with 'layout' object containing sections and their descriptions."
            ),
            "content": (
                "Analyze the content of this web page. "
                "Identify the main topics, key information, forms, and important elements. "
                "Format as JSON with 'content' object containing main topics and key elements."
            ),
            "accessibility": (
                "Analyze this web page for accessibility. "
                "Identify potential issues, missing alt text, color contrast problems, "
                "and navigation challenges. Format as JSON with 'accessibility' object."
            ),
        }
        
        prompt = analysis_prompts.get(analysis_type, analysis_prompts["elements"])
        if context:
            prompt = f"{prompt}\n\nContext: {context}"
        
        try:
            result = await self.visual_analyzer.execute(
                images=f"data:image/jpeg;base64,{screenshot_base64}",
                custom_prompt=prompt,
                model=self.visual_model,
            )
            
            if result.get("success"):
                # Try to parse JSON from the analysis
                try:
                    analysis_text = result.get("analysis", "")
                    if "```json" in analysis_text:
                        json_str = analysis_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in analysis_text:
                        json_str = analysis_text.split("```")[1].split("```")[0].strip()
                    else:
                        # Try to find JSON in the text
                        import re
                        json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(0)
                        else:
                            json_str = analysis_text
                    
                    return json.loads(json_str)
                    
                except json.JSONDecodeError:
                    # Fallback to raw text
                    return {
                        "analysis_type": analysis_type,
                        "raw_analysis": result.get("analysis", ""),
                        "parsing_failed": True,
                    }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error"),
                    "analysis_type": analysis_type,
                }
                
        except Exception as e:
            logger.error(f"Visual analysis failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "analysis_type": analysis_type,
            }
    
    async def _find_element_by_description(
        self,
        page,
        visual_description: str,
        screenshot_base64: str
    ) -> Optional[VisualElement]:
        """Find an element on the page based on visual description"""
        
        prompt = f"""
        Analyze this web page screenshot and find the element that matches this description: "{visual_description}"
        
        Please provide:
        1. CSS selector for the element (if possible to determine)
        2. Bounding box coordinates (x, y, width, height)
        3. Confidence level (0-1) that this is the right element
        4. Element type (button, link, input, etc.)
        5. Any visible text content
        
        Format as JSON with these exact keys: selector, bbox, confidence, element_type, text_content
        If you cannot find a matching element, set confidence to 0 and selector to null.
        """
        
        try:
            result = await self.visual_analyzer.execute(
                images=f"data:image/jpeg;base64,{screenshot_base64}",
                custom_prompt=prompt,
                model=self.visual_model,
            )
            
            if result.get("success"):
                analysis_text = result.get("analysis", "")
                
                # Extract JSON from response
                if "```json" in analysis_text:
                    json_str = analysis_text.split("```json")[1].split("```")[0].strip()
                elif "```" in analysis_text:
                    json_str = analysis_text.split("```")[1].split("```")[0].strip()
                else:
                    # Try to find JSON in the text
                    import re
                    json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                    else:
                        return None
                
                element_data = json.loads(json_str)
                
                # Check confidence threshold
                if element_data.get("confidence", 0) > 0.5:
                    return VisualElement(
                        selector=element_data.get("selector"),
                        description=visual_description,
                        bbox=element_data.get("bbox", {}),
                        confidence=element_data.get("confidence", 0),
                        element_type=element_data.get("element_type", "unknown"),
                        text_content=element_data.get("text_content"),
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to find element by description: {str(e)}")
            return None
    
    async def _execute_visual_action(
        self,
        page,
        action: str,
        visual_element: VisualElement,
        text: str = None,
        wait_for: int = 2000
    ) -> Dict[str, Any]:
        """Execute an action on a visually identified element"""
        
        try:
            if action == "click":
                await page.click(visual_element.selector)
                action_result = "Clicked element"
                
            elif action == "type":
                if not text:
                    raise ToolExecutionError("Text parameter is required for 'type' action")
                await page.fill(visual_element.selector, text)
                action_result = f"Typed text: '{text}'"
                
            elif action == "hover":
                await page.hover(visual_element.selector)
                action_result = "Hovered over element"
                
            elif action == "select":
                if not text:
                    raise ToolExecutionError("Text parameter is required for 'select' action")
                await page.select_option(visual_element.selector, text)
                action_result = f"Selected option: '{text}'"
                
            else:
                raise ToolExecutionError(f"Unsupported action: {action}")
            
            # Wait for any page changes
            await page.wait_for_timeout(wait_for)
            
            return {
                "success": True,
                "action": action,
                "element": {
                    "selector": visual_element.selector,
                    "description": visual_element.description,
                    "type": visual_element.element_type,
                },
                "result": action_result,
            }
            
        except Exception as e:
            error_msg = f"Failed to execute {action} on {visual_element.description}: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "action": action,
                "element": {
                    "selector": visual_element.selector,
                    "description": visual_element.description,
                    "type": visual_element.element_type,
                },
                "error": error_msg,
            }
    
    async def execute(
        self,
        url: str = None,
        action: str = "analyze",
        visual_description: str = None,
        text: str = None,
        selector: str = None,
        wait_for: int = 2000,
        analysis_type: str = "elements",
        screenshot: bool = True,
    ) -> Dict[str, Any]:
        """Execute visual browser control"""
        
        try:
            # Initialize browser if needed
            await self.playwright_tool._ensure_browser()
            
            # Create new page
            page = await self.playwright_tool._context.new_page()
            
            try:
                # Navigate to URL if provided
                if url:
                    await page.goto(url, timeout=30000)
                    await page.wait_for_timeout(2000)  # Wait for page to stabilize
                
                # Take initial screenshot
                screenshot_base64 = await self._take_screenshot(page)
                
                result = {
                    "url": url,
                    "action": action,
                    "title": await page.title(),
                    "screenshot_taken": True,
                    "screenshot_data": screenshot_base64 if screenshot else None,
                }
                
                # Handle different actions
                if action == "analyze":
                    # Perform visual analysis
                    analysis = await self._analyze_page_screenshot(
                        screenshot_base64, analysis_type, visual_description or ""
                    )
                    result["analysis"] = analysis
                    result["analysis_type"] = analysis_type
                    
                elif action in ["click", "type", "hover", "select"]:
                    # Find element and interact
                    if visual_description:
                        # Find element by visual description
                        visual_element = await self._find_element_by_description(
                            page, visual_description, screenshot_base64
                        )
                        
                        if not visual_element:
                            result["success"] = False
                            result["error"] = f"Could not find element matching: {visual_description}"
                            return result
                        
                    elif selector:
                        # Use provided selector
                        visual_element = VisualElement(
                            selector=selector,
                            description=selector,
                            bbox={},
                            confidence=1.0,
                            element_type="unknown",
                        )
                    else:
                        result["success"] = False
                        result["error"] = "Either visual_description or selector is required for interaction actions"
                        return result
                    
                    # Execute the action
                    action_result = await self._execute_visual_action(
                        page, action, visual_element, text, wait_for
                    )
                    
                    result.update(action_result)
                    
                    # Take screenshot after action if requested
                    if screenshot:
                        post_action_screenshot = await self._take_screenshot(page)
                        result["post_action_screenshot"] = post_action_screenshot
                
                elif action == "screenshot":
                    # Just take a screenshot
                    result["screenshot_data"] = screenshot_base64
                    result["success"] = True
                
                elif action == "navigate":
                    # Navigation already handled above
                    result["success"] = True
                    result["navigation_complete"] = True
                
                else:
                    result["success"] = False
                    result["error"] = f"Unsupported action: {action}"
                
                return result
                
            finally:
                # Clean up page
                await page.close()
                
        except Exception as e:
            error_msg = f"Visual browser control failed: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "url": url,
                "action": action,
            }
    
    async def multi_step_interaction(
        self,
        url: str,
        steps: List[Dict[str, Any]],
        screenshot_after_each_step: bool = True
    ) -> Dict[str, Any]:
        """Execute multiple interaction steps with visual understanding"""
        
        results = []
        
        try:
            # Initialize browser
            await self.playwright_tool._ensure_browser()
            
            # Create persistent page
            page = await self.playwright_tool._context.new_page()
            
            try:
                # Navigate to initial URL
                await page.goto(url, timeout=30000)
                await page.wait_for_timeout(2000)
                
                # Execute each step
                for i, step in enumerate(steps):
                    step_result = {
                        "step_number": i + 1,
                        "step_description": step.get("description", f"Step {i + 1}"),
                    }
                    
                    # Take screenshot before step
                    screenshot_before = await self._take_screenshot(page)
                    step_result["screenshot_before"] = screenshot_before
                    
                    # Execute the step
                    action = step.get("action", "analyze")
                    
                    if action in ["click", "type", "hover", "select"]:
                        visual_description = step.get("visual_description")
                        selector = step.get("selector")
                        text = step.get("text")
                        wait_for = step.get("wait_for", 2000)
                        
                        if visual_description:
                            visual_element = await self._find_element_by_description(
                                page, visual_description, screenshot_before
                            )
                            
                            if not visual_element:
                                step_result["success"] = False
                                step_result["error"] = f"Could not find element: {visual_description}"
                                results.append(step_result)
                                continue
                        elif selector:
                            visual_element = VisualElement(
                                selector=selector,
                                description=selector,
                                bbox={},
                                confidence=1.0,
                                element_type="unknown",
                            )
                        else:
                            step_result["success"] = False
                            step_result["error"] = "Either visual_description or selector required"
                            results.append(step_result)
                            continue
                        
                        # Execute action
                        action_result = await self._execute_visual_action(
                            page, action, visual_element, text, wait_for
                        )
                        step_result.update(action_result)
                        
                    elif action == "analyze":
                        analysis_type = step.get("analysis_type", "elements")
                        context = step.get("context", "")
                        
                        analysis = await self._analyze_page_screenshot(
                            screenshot_before, analysis_type, context
                        )
                        step_result["analysis"] = analysis
                        step_result["analysis_type"] = analysis_type
                        step_result["success"] = True
                    
                    elif action == "wait":
                        wait_time = step.get("duration", 2000)
                        await page.wait_for_timeout(wait_time)
                        step_result["success"] = True
                        step_result["waited_ms"] = wait_time
                    
                    # Take screenshot after step
                    if screenshot_after_each_step:
                        screenshot_after = await self._take_screenshot(page)
                        step_result["screenshot_after"] = screenshot_after
                    
                    results.append(step_result)
                
                return {
                    "success": True,
                    "url": url,
                    "total_steps": len(steps),
                    "results": results,
                }
                
            finally:
                await page.close()
                
        except Exception as e:
            error_msg = f"Multi-step interaction failed: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "url": url,
                "completed_steps": len(results),
            }
    
    async def cleanup(self):
        """Clean up browser resources"""
        await self.playwright_tool.cleanup()
        await self.visual_analyzer.cleanup()
        logger.info("Visual browser resources cleaned up")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.cleanup())
        except RuntimeError:
            # No event loop running, can't cleanup
            pass