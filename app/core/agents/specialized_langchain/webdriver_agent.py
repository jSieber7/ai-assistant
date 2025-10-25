"""
WebdriverAgent - A specialized agent for browser automation using Selenium and LangChain.

This agent can perform web browsing, form filling, clicking, screenshot capture,
and other browser automation tasks using WebDriver.
"""

import logging
import asyncio
import base64
import io
from typing import Dict, List, Optional, Any, AsyncGenerator, Union
from pydantic import BaseModel, Field
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.schema.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.common.keys import Keys
    from selenium.common.exceptions import WebDriverException, TimeoutException
    from webdriver_manager.chrome import ChromeDriverManager
    from PIL import Image
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    webdriver = None
    WebDriverWait = None
    ActionChains = None

from ...config import settings
from ...llm_providers import get_llm

logger = logging.getLogger(__name__)


class WebdriverState(BaseModel):
    """State for webdriver automation workflow"""
    url: Optional[str] = Field(default=None, description="Target URL")
    action: str = Field(description="Action to perform: navigate, click, fill, screenshot, extract")
    selector: Optional[str] = Field(default=None, description="CSS selector for element interaction")
    text: Optional[str] = Field(default=None, description="Text to type or search for")
    wait_time: int = Field(default=10, description="Time to wait for elements")
    messages: List[BaseMessage] = Field(default_factory=list, description="Conversation messages")
    screenshot_data: Optional[str] = Field(default=None, description="Base64 encoded screenshot")
    extracted_content: Optional[str] = Field(default=None, description="Extracted text content")
    current_url: Optional[str] = Field(default=None, description="Current page URL")
    page_title: Optional[str] = Field(default=None, description="Current page title")
    error: Optional[str] = Field(default=None, description="Error message if any")
    success: bool = Field(default=False, description="Whether action was successful")


class WebdriverAgent:
    """
    A specialized agent for browser automation using Selenium and LangChain.
    
    This agent can perform various browser automation tasks including navigation,
    element interaction, screenshot capture, and content extraction.
    """
    
    def __init__(self, llm=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize WebdriverAgent.
        
        Args:
            llm: LangChain LLM instance (if None, will get from settings)
            config: Additional configuration for agent
        """
        self.llm = llm
        self.config = config or {}
        self.name = "webdriver_agent"
        self.description = "Specialized agent for browser automation using Selenium"
        
        # WebDriver instance
        self.driver = None
        self.is_initialized = False
        
        # LangGraph workflow
        self.workflow = None
        self.checkpointer = MemorySaver()
        
        # Initialize components
        asyncio.create_task(self._initialize_async())
    
    async def _initialize_async(self):
        """Initialize components asynchronously"""
        try:
            if not self.llm:
                self.llm = await get_llm()
            
            # Check Selenium availability
            if not SELENIUM_AVAILABLE:
                logger.warning("Selenium not available - WebdriverAgent will be disabled")
                return
            
            # Create LangGraph workflow
            self.workflow = self._create_workflow()
            
            logger.info("WebdriverAgent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize WebdriverAgent: {str(e)}")
    
    def _initialize_driver(self) -> bool:
        """Initialize WebDriver instance"""
        if not SELENIUM_AVAILABLE:
            logger.error("Selenium not available")
            return False
        
        try:
            # Configure Chrome options
            chrome_options = Options()
            
            # Headless mode based on config
            if self.config.get("headless", True):
                chrome_options.add_argument("--headless")
            
            # Additional options for stability
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            
            # Initialize driver
            self.driver = webdriver.Chrome(
                service=webdriver.chrome.service.Service(ChromeDriverManager().install()),
                options=chrome_options
            )
            
            self.is_initialized = True
            logger.info("WebDriver initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {str(e)}")
            return False
    
    def _create_workflow(self) -> StateGraph:
        """Create LangGraph workflow for webdriver automation"""
        workflow = StateGraph(WebdriverState)
        
        # Define nodes
        def initialize_browser(state: WebdriverState) -> WebdriverState:
            """Initialize browser if not already done"""
            if not self.is_initialized:
                if self._initialize_driver():
                    state.messages.append(SystemMessage(content="Browser initialized successfully"))
                else:
                    state.error = "Failed to initialize browser"
                    state.success = False
                    return state
            else:
                state.messages.append(SystemMessage(content="Browser already initialized"))
            
            return state
        
        def navigate_to_url(state: WebdriverState) -> WebdriverState:
            """Navigate to specified URL"""
            if not self.driver or not state.url:
                state.error = "Browser not initialized or URL not provided"
                state.success = False
                return state
            
            try:
                self.driver.get(state.url)
                WebDriverWait(self.driver, state.wait_time).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                
                state.current_url = self.driver.current_url
                state.page_title = self.driver.title
                state.success = True
                state.messages.append(SystemMessage(content=f"Successfully navigated to {state.url}"))
                
            except TimeoutException:
                state.error = f"Timeout while navigating to {state.url}"
                state.success = False
            except Exception as e:
                state.error = f"Error navigating to {state.url}: {str(e)}"
                state.success = False
            
            return state
        
        def interact_with_element(state: WebdriverState) -> WebdriverState:
            """Interact with page element"""
            if not self.driver or not state.selector:
                state.error = "Browser not initialized or selector not provided"
                state.success = False
                return state
            
            try:
                # Wait for element
                element = WebDriverWait(self.driver, state.wait_time).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, state.selector))
                )
                
                # Perform action based on type
                if state.action == "click":
                    element.click()
                    state.messages.append(SystemMessage(content=f"Clicked element: {state.selector}"))
                
                elif state.action == "fill" and state.text:
                    element.clear()
                    element.send_keys(state.text)
                    state.messages.append(SystemMessage(content=f"Filled element {state.selector} with text"))
                
                elif state.action == "scroll":
                    self.driver.execute_script("arguments[0].scrollIntoView();", element)
                    state.messages.append(SystemMessage(content=f"Scrolled to element: {state.selector}"))
                
                state.success = True
                
            except TimeoutException:
                state.error = f"Element not found: {state.selector}"
                state.success = False
            except Exception as e:
                state.error = f"Error interacting with element {state.selector}: {str(e)}"
                state.success = False
            
            return state
        
        def capture_screenshot(state: WebdriverState) -> WebdriverState:
            """Capture screenshot of current page"""
            if not self.driver:
                state.error = "Browser not initialized"
                state.success = False
                return state
            
            try:
                # Capture screenshot
                screenshot = self.driver.get_screenshot_as_png()
                
                # Convert to base64
                buffered = io.BytesIO(screenshot)
                screenshot_base64 = base64.b64encode(buffered.getvalue()).decode()
                
                state.screenshot_data = screenshot_base64
                state.success = True
                state.messages.append(SystemMessage(content="Screenshot captured successfully"))
                
            except Exception as e:
                state.error = f"Error capturing screenshot: {str(e)}"
                state.success = False
            
            return state
        
        def extract_content(state: WebdriverState) -> WebdriverState:
            """Extract text content from page"""
            if not self.driver:
                state.error = "Browser not initialized"
                state.success = False
                return state
            
            try:
                # Extract page content
                if state.selector:
                    # Extract from specific element
                    element = WebDriverWait(self.driver, state.wait_time).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, state.selector))
                    )
                    content = element.text
                else:
                    # Extract entire page
                    content = self.driver.find_element(By.TAG_NAME, "body").text
                
                state.extracted_content = content
                state.success = True
                state.messages.append(SystemMessage(content="Content extracted successfully"))
                
            except TimeoutException:
                state.error = f"Element not found for extraction: {state.selector}"
                state.success = False
            except Exception as e:
                state.error = f"Error extracting content: {str(e)}"
                state.success = False
            
            return state
        
        def cleanup(state: WebdriverState) -> WebdriverState:
            """Clean up resources"""
            if self.driver:
                try:
                    self.driver.quit()
                    self.is_initialized = False
                    state.messages.append(SystemMessage(content="Browser closed"))
                except Exception as e:
                    logger.error(f"Error closing browser: {str(e)}")
            
            return state
        
        # Add nodes to workflow
        workflow.add_node("initialize_browser", initialize_browser)
        workflow.add_node("navigate_to_url", navigate_to_url)
        workflow.add_node("interact_with_element", interact_with_element)
        workflow.add_node("capture_screenshot", capture_screenshot)
        workflow.add_node("extract_content", extract_content)
        workflow.add_node("cleanup", cleanup)
        
        # Define conditional routing
        def route_action(state: WebdriverState) -> str:
            """Route to appropriate node based on action"""
            if state.action == "navigate":
                return "navigate_to_url"
            elif state.action in ["click", "fill", "scroll"]:
                return "interact_with_element"
            elif state.action == "screenshot":
                return "capture_screenshot"
            elif state.action == "extract":
                return "extract_content"
            else:
                return "cleanup"
        
        # Set up workflow
        workflow.set_entry_point("initialize_browser")
        workflow.add_conditional_edges(
            "initialize_browser",
            route_action,
            {
                "navigate_to_url": "navigate_to_url",
                "interact_with_element": "interact_with_element",
                "capture_screenshot": "capture_screenshot",
                "extract_content": "extract_content",
                "cleanup": "cleanup",
            }
        )
        workflow.add_edge("navigate_to_url", "cleanup")
        workflow.add_edge("interact_with_element", "cleanup")
        workflow.add_edge("capture_screenshot", "cleanup")
        workflow.add_edge("extract_content", "cleanup")
        workflow.add_edge("cleanup", END)
        
        # Compile workflow with checkpointer
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def execute_action(
        self,
        action: str,
        url: Optional[str] = None,
        selector: Optional[str] = None,
        text: Optional[str] = None,
        wait_time: int = 10,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute browser automation action.
        
        Args:
            action: Action to perform (navigate, click, fill, screenshot, extract)
            url: Target URL for navigation
            selector: CSS selector for element interaction
            text: Text to type or search for
            wait_time: Time to wait for elements
            thread_id: Thread ID for conversation tracking
            
        Returns:
            Dictionary containing action result and metadata
        """
        if not SELENIUM_AVAILABLE:
            return {
                "success": False,
                "error": "Selenium not available - install selenium and webdriver-manager",
            }
        
        if not self.workflow:
            await self._initialize_async()
            if not self.workflow:
                raise RuntimeError("Failed to initialize WebdriverAgent workflow")
        
        try:
            # Create initial state
            state = WebdriverState(
                action=action,
                url=url,
                selector=selector,
                text=text,
                wait_time=wait_time,
            )
            
            # Run workflow
            config = {"thread_id": thread_id or "default"} if thread_id else {}
            result = await self.workflow.ainvoke(state, config=config)
            
            return {
                "success": result.success,
                "action": action,
                "url": result.current_url,
                "title": result.page_title,
                "screenshot": result.screenshot_data,
                "content": result.extracted_content,
                "error": result.error,
                "messages": [msg.content for msg in result.messages],
            }
            
        except Exception as e:
            logger.error(f"Error in webdriver action: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "action": action,
            }
    
    async def close_browser(self) -> bool:
        """Close the browser and clean up resources"""
        try:
            if self.driver:
                self.driver.quit()
                self.driver = None
                self.is_initialized = False
                logger.info("Browser closed successfully")
                return True
            return True
        except Exception as e:
            logger.error(f"Error closing browser: {str(e)}")
            return False
    
    def __del__(self):
        """Cleanup when agent is destroyed"""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass