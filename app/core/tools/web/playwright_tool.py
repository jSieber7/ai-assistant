"""
Playwright tool for direct browser automation.

This tool provides direct access to Playwright functionality for web scraping,
automation, and testing tasks.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from playwright.async_api import async_playwright, Page

from app.core.tools.base.base import BaseTool, ToolExecutionError

logger = logging.getLogger(__name__)


class PlaywrightTool(BaseTool):
    """Tool for direct browser automation using Playwright"""

    def __init__(self, headless: bool = True, browser_type: str = "chromium"):
        super().__init__()
        self.headless = headless
        self.browser_type = browser_type
        self._playwright = None
        self._browser = None
        self._context = None

    @property
    def name(self) -> str:
        return "playwright_automation"

    @property
    def description(self) -> str:
        return "Direct browser automation using Playwright for web scraping and interaction"

    @property
    def keywords(self) -> List[str]:
        return [
            "browser",
            "automation",
            "playwright",
            "scrape",
            "interact",
            "click",
            "type",
            "screenshot",
        ]

    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "url": {
                "type": str,
                "description": "URL to navigate to",
                "required": True,
            },
            "actions": {
                "type": List[Dict[str, Any]],
                "description": "List of actions to perform (click, type, wait, etc.)",
                "required": False,
                "default": [],
            },
            "wait_for": {
                "type": int,
                "description": "Time to wait after page load in milliseconds",
                "required": False,
                "default": 3000,
            },
            "screenshot": {
                "type": bool,
                "description": "Take a screenshot of the final page",
                "required": False,
                "default": False,
            },
            "selector": {
                "type": str,
                "description": "CSS selector to extract content from",
                "required": False,
            },
            "extract_text": {
                "type": bool,
                "description": "Extract text content from the page",
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

    async def _ensure_browser(self):
        """Ensure browser is initialized"""
        if self._browser is None:
            playwright_instance = async_playwright()
            self._playwright = await playwright_instance.start()

            # Select browser type
            if self.browser_type == "chromium":
                self._browser = await self._playwright.chromium.launch(
                    headless=self.headless
                )
            elif self.browser_type == "firefox":
                self._browser = await self._playwright.firefox.launch(
                    headless=self.headless
                )
            elif self.browser_type == "webkit":
                self._browser = await self._playwright.webkit.launch(
                    headless=self.headless
                )
            else:
                raise ToolExecutionError(
                    f"Unsupported browser type: {self.browser_type}"
                )

            # Create context with default settings
            self._context = await self._browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            )

    async def _execute_actions(self, page: Page, actions: List[Dict[str, Any]]) -> None:
        """Execute a list of actions on the page"""
        for action in actions:
            action_type = action.get("type")

            if action_type == "click":
                selector = action.get("selector")
                if selector:
                    await page.click(selector)
                else:
                    raise ToolExecutionError("Click action requires 'selector'")

            elif action_type == "type":
                selector = action.get("selector")
                text = action.get("text")
                if selector and text:
                    await page.fill(selector, text)
                else:
                    raise ToolExecutionError(
                        "Type action requires 'selector' and 'text'"
                    )

            elif action_type == "wait":
                duration = action.get("duration", 1000)
                await page.wait_for_timeout(duration)

            elif action_type == "wait_for_selector":
                selector = action.get("selector")
                timeout = action.get("timeout", 30000)
                if selector:
                    await page.wait_for_selector(selector, timeout=timeout)
                else:
                    raise ToolExecutionError(
                        "wait_for_selector action requires 'selector'"
                    )

            elif action_type == "scroll":
                direction = action.get("direction", "down")
                distance = action.get("distance", 500)
                if direction == "down":
                    await page.mouse.wheel(0, distance)
                elif direction == "up":
                    await page.mouse.wheel(0, -distance)
                elif direction == "right":
                    await page.mouse.wheel(distance, 0)
                elif direction == "left":
                    await page.mouse.wheel(-distance, 0)

            elif action_type == "screenshot":
                path = action.get("path")
                await page.screenshot(path=path)

            else:
                logger.warning(f"Unknown action type: {action_type}")

    async def execute(
        self,
        url: str,
        actions: Optional[List[Dict[str, Any]]] = None,
        wait_for: Optional[int] = None,
        screenshot: Optional[bool] = None,
        selector: Optional[str] = None,
        extract_text: Optional[bool] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute browser automation using Playwright"""

        # Ensure browser is available
        await self._ensure_browser()

        # Set defaults
        actions = actions or []
        wait_for = wait_for if wait_for is not None else 3000
        screenshot = screenshot if screenshot is not None else False
        extract_text = extract_text if extract_text is not None else True
        timeout = timeout if timeout is not None else 30

        # Create new page
        page = await self._context.new_page()

        try:
            # Navigate to URL
            await page.goto(url, timeout=timeout * 1000)

            # Wait for page to load
            await page.wait_for_timeout(wait_for)

            # Execute actions
            if actions:
                await self._execute_actions(page, actions)

            # Extract content
            result = {
                "url": url,
                "title": await page.title(),
                "actions_executed": len(actions),
            }

            # Extract text content
            if extract_text:
                if selector:
                    # Extract from specific selector
                    element = await page.query_selector(selector)
                    if element:
                        result["content"] = await element.text_content()
                    else:
                        result["content"] = (
                            f"No element found with selector: {selector}"
                        )
                else:
                    # Extract entire page text
                    result["content"] = await page.text_content("body")

            # Extract metadata
            result["metadata"] = await page.evaluate("""
                () => {
                    return {
                        description: document.querySelector('meta[name="description"]')?.content || '',
                        keywords: document.querySelector('meta[name="keywords"]')?.content || '',
                        author: document.querySelector('meta[name="author"]')?.content || '',
                        viewport: {
                            width: window.innerWidth,
                            height: window.innerHeight
                        }
                    }
                }
            """)

            # Take screenshot if requested
            if screenshot:
                screenshot_bytes = await page.screenshot()
                import base64

                result["screenshot"] = base64.b64encode(screenshot_bytes).decode()

            logger.info(f"Successfully executed Playwright automation for {url}")
            return result

        except Exception as e:
            error_msg = f"Playwright automation failed for {url}: {str(e)}"
            logger.error(error_msg)
            raise ToolExecutionError(error_msg)

        finally:
            # Close page
            await page.close()

    async def cleanup(self):
        """Clean up browser resources"""
        if self._context:
            await self._context.close()
            self._context = None

        if self._browser:
            await self._browser.close()
            self._browser = None

        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

        logger.info("Playwright browser resources cleaned up")

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.cleanup())
        except RuntimeError:
            # No event loop running, can't cleanup
            pass
