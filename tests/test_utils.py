"""
Shared utilities and fixtures for test files.

This module provides common functionality to avoid code duplication
across test files, including path setup, common imports, and shared fixtures.
"""

import pytest
import sys
import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from unittest.mock import Mock, AsyncMock, patch

# Add the app directory to the path (works from any test location)
project_root = Path(__file__).parent.parent
app_path = project_root / "app"
if str(app_path) not in sys.path:
    sys.path.insert(0, str(app_path))


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


def create_test_chat_request(messages=None, model=None, stream=False, tools=None):
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

    request = {
        "messages": messages,
        "model": model or "deepseek/deepseek-v3.1-terminus",
        "stream": stream,
    }
    
    if tools is not None:
        request["tools"] = tools
    
    return request


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


@pytest.fixture
def test_chat_request_with_tools():
    """
    Return a standard test chat request with tools.
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get weather for"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    return create_test_chat_request(tools=tools)


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


class MockResponse:
    """Mock HTTP response for testing"""
    
    def __init__(self, json_data: Dict[str, Any], status_code: int = 200):
        self._json_data = json_data
        self.status_code = status_code
        self.headers = {"content-type": "application/json"}
    
    def json(self):
        return self._json_data
    
    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


class AsyncMockResponse:
    """Mock async HTTP response for testing"""
    
    def __init__(self, json_data: Dict[str, Any], status_code: int = 200):
        self._json_data = json_data
        self.status_code = status_code
        self.headers = {"content-type": "application/json"}
    
    async def json(self):
        return self._json_data
    
    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


class TestDatabase:
    """Mock database for testing"""
    
    def __init__(self):
        self.data = {}
        self.connections = []
    
    async def connect(self):
        """Mock database connection"""
        connection = Mock()
        self.connections.append(connection)
        return connection
    
    async def execute_query(self, query: str, params: Dict[str, Any] = None):
        """Mock query execution"""
        # Simple mock implementation
        if "SELECT" in query:
            return []
        return None
    
    async def execute_insert(self, table: str, data: Dict[str, Any]):
        """Mock insert execution"""
        record_id = len(self.data) + 1
        self.data[record_id] = data
        return {"id": record_id}
    
    async def execute_update(self, table: str, record_id: int, data: Dict[str, Any]):
        """Mock update execution"""
        if record_id in self.data:
            self.data[record_id].update(data)
            return {"rows_affected": 1}
        return {"rows_affected": 0}
    
    async def execute_delete(self, table: str, record_id: int):
        """Mock delete execution"""
        if record_id in self.data:
            del self.data[record_id]
            return {"rows_affected": 1}
        return {"rows_affected": 0}
    
    async def health_check(self):
        """Mock health check"""
        return {"status": "healthy"}


class TestCache:
    """Mock cache for testing"""
    
    def __init__(self):
        self.data = {}
    
    async def get(self, key: str):
        """Mock cache get"""
        return self.data.get(key)
    
    async def set(self, key: str, value: Any, ttl: int = 300):
        """Mock cache set"""
        self.data[key] = value
        return True
    
    async def delete(self, key: str):
        """Mock cache delete"""
        if key in self.data:
            del self.data[key]
            return True
        return False
    
    async def exists(self, key: str):
        """Mock cache exists"""
        return key in self.data
    
    async def clear(self):
        """Mock cache clear"""
        self.data.clear()
        return True


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


# Helper for creating temporary files
def create_temp_file(content: str = "test content", suffix: str = ".txt") -> Path:
    """Create a temporary file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
        f.write(content)
        return Path(f.name)


# Helper for creating temporary directories
def create_temp_dir(name: str = "test_dir") -> Path:
    """Create a temporary directory for testing"""
    temp_dir = Path(tempfile.mkdtemp(prefix=name))
    return temp_dir


# Helper for mocking HTTP responses
def create_mock_response(data: Dict[str, Any], status_code: int = 200) -> MockResponse:
    """Create a mock HTTP response"""
    return MockResponse(data, status_code)


def create_async_mock_response(data: Dict[str, Any], status_code: int = 200) -> AsyncMockResponse:
    """Create a mock async HTTP response"""
    return AsyncMockResponse(data, status_code)


# Helper for testing API responses
def assert_valid_api_response(response_data: Dict[str, Any], required_fields: List[str] = None):
    """Assert that API response is valid"""
    assert isinstance(response_data, dict), "Response should be a dictionary"
    
    if required_fields is None:
        required_fields = ["success", "data"]
    
    for field in required_fields:
        assert field in response_data, f"Missing required field: {field}"


# Helper for testing error responses
def assert_error_response(response_data: Dict[str, Any], expected_error: str = None):
    """Assert that error response is valid"""
    assert response_data.get("success") is False, "Response should indicate failure"
    assert "error" in response_data, "Response should contain error message"
    
    if expected_error:
        assert expected_error in response_data["error"], f"Expected error '{expected_error}' not found"


# Helper for performance testing
class PerformanceTimer:
    """Helper for timing operations"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start timing"""
        self.start_time = time.time()
    
    def stop(self):
        """Stop timing"""
        self.end_time = time.time()
    
    @property
    def duration(self) -> float:
        """Get duration in seconds"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    def assert_max_duration(self, max_seconds: float, operation: str = "operation"):
        """Assert operation completed within max duration"""
        assert self.duration > 0, f"Timer was not started and stopped for {operation}"
        assert self.duration <= max_seconds, f"{operation} took {self.duration:.2f}s, expected <= {max_seconds}s"


# Helper for creating mock LLM responses
def create_mock_llm_response(content: str, model: str = "test-model") -> Dict[str, Any]:
    """Create a mock LLM response"""
    return {
        "id": "test-response-id",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    }


# Helper for creating mock tool responses
def create_mock_tool_response(tool_name: str, result: Any) -> Dict[str, Any]:
    """Create a mock tool response"""
    return {
        "success": True,
        "tool_name": tool_name,
        "result": result,
        "execution_time": 0.5
    }


# Helper for testing streaming responses
async def collect_stream_response(response):
    """Collect all chunks from a streaming response"""
    chunks = []
    async for chunk in response:
        chunks.append(chunk)
    return chunks


# Helper for patching environment variables
class EnvPatcher:
    """Context manager for patching environment variables"""
    
    def __init__(self, **kwargs):
        self.env_vars = kwargs
        self.original_values = {}
    
    def __enter__(self):
        for key, value in self.env_vars.items():
            self.original_values[key] = os.environ.get(key)
            os.environ[key] = value
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for key, value in self.original_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# Helper for creating test data
class TestDataFactory:
    """Factory for creating test data"""
    
    @staticmethod
    def create_chat_message(role: str, content: str) -> Dict[str, str]:
        """Create a chat message"""
        return {"role": role, "content": content}
    
    @staticmethod
    def create_tool_call(name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create a tool call"""
        return {
            "id": f"call_{name}_{int(time.time())}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(parameters)
            }
        }
    
    @staticmethod
    def create_tool_response(name: str, result: Any) -> Dict[str, Any]:
        """Create a tool response"""
        return {
            "tool_call_id": f"call_{name}_{int(time.time())}",
            "role": "tool",
            "content": json.dumps(result)
        }
    
    @staticmethod
    def create_conversation(messages: List[Dict[str, str]], conversation_id: str = "test_conv") -> Dict[str, Any]:
        """Create a conversation object"""
        return {
            "conversation_id": conversation_id,
            "messages": messages,
            "created_at": time.time(),
            "updated_at": time.time()
        }


# Context manager for testing with temporary files
class TempFileContext:
    """Context manager for creating and cleaning up temporary files"""
    
    def __init__(self, content: str = "test content", suffix: str = ".txt"):
        self.content = content
        self.suffix = suffix
        self.file_path = None
    
    def __enter__(self):
        self.file_path = create_temp_file(self.content, self.suffix)
        return self.file_path
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_path and self.file_path.exists():
            self.file_path.unlink()


# Context manager for testing with temporary directories
class TempDirContext:
    """Context manager for creating and cleaning up temporary directories"""
    
    def __init__(self, name: str = "test_dir"):
        self.name = name
        self.dir_path = None
    
    def __enter__(self):
        self.dir_path = create_temp_dir(self.name)
        return self.dir_path
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import shutil
        if self.dir_path and self.dir_path.exists():
            shutil.rmtree(self.dir_path)