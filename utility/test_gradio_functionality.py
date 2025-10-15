#!/usr/bin/env python3
"""
Test script to verify Gradio interface functionality with real providers.

This script can be run manually to test that all Gradio interface components
are fully functional and not just demo placeholders.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.ui.gradio_app import (
    get_models_list,
    get_providers_info,
    get_tools_info,
    get_agents_info,
    initialize_gradio_components,
    test_query,
    update_settings,
    create_gradio_app,
)
from app.core.config import settings

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GradioFunctionalityTester:
    """Test class for Gradio interface functionality."""

    def __init__(self):
        self.test_results = []
        self.passed = 0
        self.failed = 0

    def log_test(self, test_name: str, passed: bool, message: str = ""):
        """Log a test result."""
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
        if message:
            logger.info(f"  {message}")

        self.test_results.append((test_name, passed, message))
        if passed:
            self.passed += 1
        else:
            self.failed += 1

    async def test_model_listing(self):
        """Test model listing functionality."""
        logger.info("Testing model listing functionality...")

        try:
            models = get_models_list()

            if not models:
                self.log_test("Model listing", False, "No models returned")
                return

            if len(models) == 1 and models[0] == settings.default_model:
                self.log_test("Model listing", False, "Only default model returned")
                return

            self.log_test("Model listing", True, f"Found {len(models)} models")

            # Print first few models
            logger.info(f"  First 5 models: {models[:5]}")

        except Exception as e:
            self.log_test("Model listing", False, f"Exception: {str(e)}")

    def test_provider_info(self):
        """Test provider information functionality."""
        logger.info("Testing provider information functionality...")

        try:
            providers_info = get_providers_info()

            if not providers_info:
                self.log_test(
                    "Provider info", False, "No provider information returned"
                )
                return

            if "Unable to fetch" in providers_info:
                self.log_test(
                    "Provider info", False, "Failed to fetch provider information"
                )
                return

            # Check if providers are configured
            has_configured = "Configured" in providers_info
            has_unconfigured = "Not configured" in providers_info

            if not (has_configured or has_unconfigured):
                self.log_test(
                    "Provider info", False, "No provider status information found"
                )
                return

            self.log_test(
                "Provider info", True, "Provider information retrieved successfully"
            )
            logger.info(f"  Provider info: {providers_info[:100]}...")

        except Exception as e:
            self.log_test("Provider info", False, f"Exception: {str(e)}")

    def test_tools_info(self):
        """Test tools information functionality."""
        logger.info("Testing tools information functionality...")

        try:
            tools_info = get_tools_info()

            if not tools_info:
                self.log_test("Tools info", False, "No tools information returned")
                return

            if "Failed to initialize" in tools_info:
                self.log_test("Tools info", False, "Tool initialization failed")
                return

            has_tools = "No tools available" not in tools_info

            if has_tools:
                tool_count = tools_info.count("\n") if tools_info else 0
                self.log_test(
                    "Tools info", True, f"Found information for {tool_count} tools"
                )
            else:
                self.log_test(
                    "Tools info",
                    True,
                    "Tools system initialized but no tools available",
                )

            logger.info(f"  Tools info: {tools_info[:100]}...")

        except Exception as e:
            self.log_test("Tools info", False, f"Exception: {str(e)}")

    def test_agents_info(self):
        """Test agents information functionality."""
        logger.info("Testing agents information functionality...")

        try:
            agents_info = get_agents_info()

            if not agents_info:
                self.log_test("Agents info", False, "No agents information returned")
                return

            if "Failed to initialize" in agents_info:
                self.log_test("Agents info", False, "Agent initialization failed")
                return

            is_enabled = settings.agent_system_enabled

            if is_enabled:
                has_agents = "No agents available" not in agents_info

                if has_agents:
                    agent_count = agents_info.count("\n") if agents_info else 0
                    self.log_test(
                        "Agents info",
                        True,
                        f"Found information for {agent_count} agents",
                    )
                else:
                    self.log_test(
                        "Agents info",
                        True,
                        "Agent system enabled but no agents available",
                    )
            else:
                self.log_test("Agents info", True, "Agent system disabled as expected")

            logger.info(f"  Agents info: {agents_info[:100]}...")

        except Exception as e:
            self.log_test("Agents info", False, f"Exception: {str(e)}")

    def test_initialization(self):
        """Test Gradio components initialization."""
        logger.info("Testing Gradio components initialization...")

        try:
            success, status = initialize_gradio_components()

            if not success:
                self.log_test(
                    "Initialization", False, f"Initialization failed: {status}"
                )
                return

            self.log_test("Initialization", True, "Initialization successful")
            logger.info(f"  Status: {status}")

        except Exception as e:
            self.log_test("Initialization", False, f"Exception: {str(e)}")

    async def test_query_functionality(self):
        """Test query functionality."""
        logger.info("Testing query functionality...")

        try:
            result = await test_query(
                message="Hello, this is a test message",
                model=settings.default_model,
                temperature=0.7,
                max_tokens=100,
                use_agent_system=False,
            )

            if not result:
                self.log_test("Query functionality", False, "No response returned")
                return

            if result.startswith("Error:"):
                if "connect" in result.lower() or "timeout" in result.lower():
                    self.log_test(
                        "Query functionality",
                        False,
                        f"Connection error (expected in test): {result[:50]}...",
                    )
                else:
                    self.log_test(
                        "Query functionality", False, f"Query failed: {result[:50]}..."
                    )
                return

            if len(result) < 20:
                self.log_test("Query functionality", False, "Response too short")
                return

            self.log_test(
                "Query functionality",
                True,
                f"Query successful, response length: {len(result)}",
            )
            logger.info(f"  Response preview: {result[:100]}...")

        except Exception as e:
            self.log_test("Query functionality", False, f"Exception: {str(e)}")

    def test_settings_update(self):
        """Test settings update functionality."""
        logger.info("Testing settings update functionality...")

        try:
            # Get original settings
            original_tool_enabled = settings.tool_system_enabled
            original_debug = settings.debug

            # Update settings
            result = update_settings(
                tool_system_enabled=not original_tool_enabled,
                agent_system_enabled=not settings.agent_system_enabled,
                preferred_provider="ollama",
                enable_fallback=not settings.enable_fallback,
                debug_mode=not original_debug,
            )

            if not result:
                self.log_test(
                    "Settings update", False, "No response from settings update"
                )
                return

            if "Failed to update settings" in result:
                self.log_test(
                    "Settings update",
                    False,
                    f"Settings update failed: {result[:50]}...",
                )
                return

            # Check if settings were actually updated
            if settings.tool_system_enabled == original_tool_enabled:
                self.log_test(
                    "Settings update", False, "Settings not updated in memory"
                )
                return

            self.log_test("Settings update", True, "Settings updated successfully")
            logger.info(f"  Update result: {result[:100]}...")

            # Restore original settings
            update_settings(
                tool_system_enabled=original_tool_enabled,
                agent_system_enabled=settings.agent_system_enabled,
                preferred_provider=settings.preferred_provider,
                enable_fallback=settings.enable_fallback,
                debug_mode=original_debug,
            )

        except Exception as e:
            self.log_test("Settings update", False, f"Exception: {str(e)}")

    def test_app_creation(self):
        """Test Gradio app creation."""
        logger.info("Testing Gradio app creation...")

        try:
            app = create_gradio_app()

            if not app:
                self.log_test("App creation", False, "No app created")
                return

            if app.title != "AI Assistant Interface":
                self.log_test(
                    "App creation", False, f"Incorrect app title: {app.title}"
                )
                return

            self.log_test("App creation", True, "Gradio app created successfully")

        except Exception as e:
            self.log_test("App creation", False, f"Exception: {str(e)}")

    async def run_all_tests(self):
        """Run all tests."""
        logger.info("Starting Gradio functionality tests...")
        logger.info("=" * 60)

        # Run tests
        self.test_initialization()
        self.test_provider_info()
        self.test_tools_info()
        self.test_agents_info()
        await self.test_model_listing()
        await self.test_query_functionality()
        self.test_settings_update()
        self.test_app_creation()

        # Print results
        logger.info("=" * 60)
        logger.info(f"Test Results: {self.passed} passed, {self.failed} failed")

        if self.failed == 0:
            logger.info("🎉 All tests passed! Gradio interface is fully functional.")
        else:
            logger.warning(f"⚠️ {self.failed} tests failed. Check the logs for details.")

            # Print failed tests
            logger.info("Failed tests:")
            for test_name, passed, message in self.test_results:
                if not passed:
                    logger.info(f"  - {test_name}: {message}")

        return self.failed == 0


async def main():
    """Main function to run tests."""
    tester = GradioFunctionalityTester()
    success = await tester.run_all_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    # Run tests
    asyncio.run(main())
