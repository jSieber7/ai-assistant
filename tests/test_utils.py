"""
Shared utilities and fixtures for test files.

This module provides common functionality to avoid code duplication
across test files, including path setup, common imports, and shared fixtures.
"""

import os
import sys
import asyncio
from pathlib import Path


def setup_test_paths():
    """
    Set up Python path for test files to import from the app directory.
    This should be called at the top of any test file that needs to import app modules.
    """
    # Add the app directory to the path (works from any test location)
    project_root = Path(__file__).parent.parent
    app_path = project_root / "app"
    if str(app_path) not in sys.path:
        sys.path.insert(0, str(app_path))


def get_test_llm_config():
    """
    Return a standard test LLM configuration.
    """
    return {
        "openrouter_api_key": "test-key-123",
        "default_model": "deepseek/deepseek-v3.1-terminus",
        "environment": "testing"
    }


async def run_async_test(test_func, *args, **kwargs):
    """
    Helper to run async test functions with proper error handling.
    """
    try:
        return await test_func(*args, **kwargs)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


def create_test_chat_request(messages=None, model=None, stream=False):
    """
    Create a standardized test chat request.
    """
    if messages is None:
        messages = [{"role": "user", "content": "Hello, can you respond with just the word SUCCESS?"}]
    
    return {
        "messages": messages,
        "model": model or "deepseek/deepseek-v3.1-terminus",
        "stream": stream,
    }


class TestResult:
    """Standardized test result container."""
    
    def __init__(self, success=True, message="", data=None, error=None):
        self.success = success
        self.message = message
        self.data = data or {}
        self.error = error
    
    def __bool__(self):
        return self.success
    
    def __str__(self):
        if self.success:
            return f"✅ {self.message}"
        else:
            return f"❌ {self.message}: {self.error}"


# Common test data
TEST_MESSAGES = [
    "Hello, how are you?",
    "What tools are available?",
    "Can you help me calculate 15 + 27?",
    "What's the weather like today?",
]

TEST_QUERIES = [
    "Calculate 15 + 27",
    "What's the weather in London?",
    "Tell me a joke",
]