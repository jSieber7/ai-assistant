"""
Shared utilities and fixtures for test files.

This module provides common functionality to avoid code duplication
across test files, including path setup, common imports, and shared fixtures.
"""

import pytest
import sys
from pathlib import Path


@pytest.fixture(scope="session", autouse=True)
def setup_test_paths():
    """
    Set up Python path for test files to import from the app directory.
    This fixture is automatically used for all tests.
    """
    # Add the app directory to the path (works from any test location)
    project_root = Path(__file__).parent.parent
    app_path = project_root / "app"
    if str(app_path) not in sys.path:
        sys.path.insert(0, str(app_path))
    yield
    # Clean up if needed
    if str(app_path) in sys.path:
        sys.path.remove(str(app_path))


@pytest.fixture
def test_llm_config():
    """
    Return a standard test LLM configuration.
    """
    return {
        "openrouter_api_key": "test-key-123",
        "default_model": "deepseek/deepseek-v3.1-terminus",
        "environment": "testing",
    }


@pytest.fixture
def test_messages():
    """
    Return standard test messages for testing.
    """
    return [
        "Hello, how are you?",
        "What tools are available?",
        "Can you help me calculate 15 + 27?",
        "What's the weather like today?",
    ]


@pytest.fixture
def test_queries():
    """
    Return standard test queries for testing.
    """
    return [
        "Calculate 15 + 27",
        "What's the weather in London?",
        "Tell me a joke",
    ]


def create_test_chat_request(messages=None, model=None, stream=False):
    """
    Create a standardized test chat request.
    This is a helper function that can be used in tests.
    """
    if messages is None:
        messages = [
            {
                "role": "user",
                "content": "Hello, can you respond with just the word SUCCESS?",
            }
        ]

    return {
        "messages": messages,
        "model": model or "deepseek/deepseek-v3.1-terminus",
        "stream": stream,
    }


@pytest.fixture
def test_chat_request():
    """
    Return a standard test chat request.
    """
    return create_test_chat_request()


@pytest.fixture
def test_streaming_chat_request():
    """
    Return a standard test streaming chat request.
    """
    return create_test_chat_request(stream=True)


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


# Helper function for async tests
async def run_async_test(test_func, *args, **kwargs):
    """
    Helper to run async test functions with proper error handling.
    """
    try:
        return await test_func(*args, **kwargs)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise