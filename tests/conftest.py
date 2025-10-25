"""
Pytest configuration and fixtures for the AI Assistant test suite.

This module provides common fixtures and configuration for all tests,
including database setup, mocking, and test utilities.
"""

import pytest
import asyncio
import os
import sys
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, Generator, Optional
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# FastAPI imports
from fastapi.testclient import TestClient

# App imports
from app.main import app
from app.core.config import settings
from app.core.secure_settings import SecureSettingsManager


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_environment():
    """Set up test environment variables."""
    # Set test environment
    original_env = {}
    test_env_vars = {
        "ENVIRONMENT": "testing",
        "DEBUG": "true",
        "LOG_LEVEL": "INFO",
        "AGENT_SYSTEM_ENABLED": "false",  # Disable for most tests
        "TOOL_SYSTEM_ENABLED": "true",
        "MONITORING_ENABLED": "true",
        "CACHE_ENABLED": "true",
        "DATABASE_URL": "postgresql://test:test@localhost/test_db",
        "REDIS_URL": "redis://localhost:6379/0",
    }
    
    # Store original values and set test values
    for key, value in test_env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield test_env_vars
    
    # Restore original environment
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture
def mock_settings():
    """Mock settings for testing"""
    settings_mock = Mock()
    
    # Core settings
    settings_mock.environment = "testing"
    settings_mock.debug = True
    settings_mock.host = "127.0.0.1"
    settings_mock.port = 8000
    
    # LLM provider settings
    settings_mock.openai_settings.enabled = False
    settings_mock.openrouter_api_key = "test-openrouter-key"
    settings_mock.ollama_settings.enabled = True
    settings_mock.ollama_settings.base_url = "http://localhost:11434"
    
    # System settings
    settings_mock.agent_system_enabled = False
    settings_mock.tool_system_enabled = True
    settings_mock.monitoring_enabled = True
    settings_mock.cache_enabled = True
    
    # Database settings
    settings_mock.database_url = "postgresql://test:test@localhost/test_db"
    settings_mock.redis_url = "redis://localhost:6379/0"
    
    # LangChain settings
    settings_mock.langchain_settings.integration_mode = "langchain"
    settings_mock.langchain_settings.llm_manager_enabled = True
    settings_mock.langchain_settings.tool_registry_enabled = True
    settings_mock.langchain_settings.agent_manager_enabled = True
    settings_mock.langchain_settings.memory_workflow_enabled = True
    
    return settings_mock


@pytest.fixture
def secure_settings_manager():
    """Create a secure settings manager for testing"""
    # Use a temporary directory for test settings
    test_settings_dir = Path(__file__).parent / "fixtures" / "test_settings"
    test_settings_dir.mkdir(exist_ok=True)
    
    manager = SecureSettingsManager(str(test_settings_dir))
    return manager


@pytest.fixture
def client():
    """Create a FastAPI test client"""
    return TestClient(app)


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider for testing"""
    provider = Mock()
    provider.provider_type = "openrouter"
    provider.create_llm = AsyncMock()
    provider.list_models = AsyncMock(return_value=[])
    provider.health_check = AsyncMock(return_value=True)
    return provider


@pytest.fixture
def mock_visual_llm_provider():
    """Mock visual LLM provider for testing"""
    provider = Mock()
    provider.provider_type = "openai_vision"
    provider.create_visual_llm = AsyncMock()
    provider.list_models = AsyncMock(return_value=[])
    provider.health_check = AsyncMock(return_value=True)
    provider.analyze_image = AsyncMock(return_value={"description": "Test image analysis"})
    return provider


@pytest.fixture
def mock_database_client():
    """Mock database client for testing"""
    client = Mock()
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.execute_query = AsyncMock(return_value=[])
    client.execute_insert = AsyncMock(return_value={"id": 1})
    client.execute_update = AsyncMock(return_value={"rows_affected": 1})
    client.execute_delete = AsyncMock(return_value={"rows_affected": 1})
    client.health_check = AsyncMock(return_value={"status": "healthy"})
    
    # Mock connection pool
    client.pool = Mock()
    client.pool.acquire = AsyncMock()
    client.pool.release = Mock()
    
    return client


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing"""
    client = Mock()
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.get = AsyncMock(return_value=None)
    client.set = AsyncMock(return_value=True)
    client.delete = AsyncMock(return_value=True)
    client.exists = AsyncMock(return_value=False)
    client.health_check = AsyncMock(return_value=True)
    return client


@pytest.fixture
def mock_langchain_integration():
    """Mock LangChain integration for testing"""
    integration = Mock()
    integration.initialize = AsyncMock()
    integration.shutdown = AsyncMock()
    integration.health_check = AsyncMock(return_value={
        "overall_status": "healthy",
        "components": {
            "llm_manager": {"status": "healthy"},
            "tool_registry": {"status": "healthy"},
            "agent_manager": {"status": "healthy"},
            "memory_manager": {"status": "healthy"},
            "monitoring": {"status": "healthy"}
        }
    })
    integration.get_integration_mode = Mock(return_value="langchain")
    integration.is_component_enabled = Mock(return_value=True)
    return integration


@pytest.fixture
def sample_chat_request():
    """Sample chat request for testing"""
    return {
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "model": "test-model",
        "stream": False,
        "temperature": 0.7,
        "max_tokens": 1000
    }


@pytest.fixture
def sample_chat_request_with_tools():
    """Sample chat request with tools for testing"""
    return {
        "messages": [
            {"role": "user", "content": "What's the weather like today?"}
        ],
        "model": "test-model",
        "stream": False,
        "tools": [
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
    }


@pytest.fixture
def sample_agent_request():
    """Sample agent request for testing"""
    return {
        "agent_name": "test_agent",
        "messages": [
            {"role": "user", "content": "Help me with a task"}
        ],
        "conversation_id": "test_conversation",
        "stream": False
    }


@pytest.fixture
def sample_tool_request():
    """Sample tool request for testing"""
    return {
        "tool_name": "test_tool",
        "parameters": {
            "query": "test query",
            "limit": 10
        }
    }


@pytest.fixture
def sample_conversation_data():
    """Sample conversation data for testing"""
    return {
        "conversation_id": "test_conversation",
        "title": "Test Conversation",
        "agent_name": "test_agent",
        "messages": [
            {"role": "user", "content": "Hello", "timestamp": "2023-01-01T00:00:00"},
            {"role": "assistant", "content": "Hi there!", "timestamp": "2023-01-01T00:00:01"}
        ]
    }


@pytest.fixture
def sample_metrics_data():
    """Sample metrics data for testing"""
    return {
        "component_type": "llm",
        "component_name": "openai",
        "metrics": {
            "request_count": 100,
            "success_rate": 0.95,
            "avg_duration": 1.5,
            "error_count": 5
        }
    }


@pytest.fixture
def sample_image_content():
    """Sample image content for testing"""
    return {
        "image_data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
        "media_type": "image/png",
        "filename": "test.png"
    }


@pytest.fixture
def temp_file(tmp_path):
    """Create a temporary file for testing"""
    def _create_temp_file(content="test content", suffix=".txt"):
        file_path = tmp_path / f"test_file{suffix}"
        file_path.write_text(content)
        return file_path
    
    return _create_temp_file


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing"""
    def _create_temp_dir(name="test_dir"):
        dir_path = tmp_path / name
        dir_path.mkdir(exist_ok=True)
        return dir_path
    
    return _create_temp_dir


class AsyncContextManager:
    """Helper class for creating async context managers for testing"""
    
    def __init__(self, return_value=None, side_effect=None):
        self.return_value = return_value
        self.side_effect = side_effect
    
    async def __aenter__(self):
        if self.side_effect:
            raise self.side_effect
        return self.return_value
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def async_context_manager():
    """Factory for creating async context managers"""
    def _create(return_value=None, side_effect=None):
        return AsyncContextManager(return_value, side_effect)
    return _create


@pytest.fixture
def mock_async_context():
    """Mock async context manager for testing"""
    return AsyncContextManager()


# Custom assertions for testing
class CustomAssertions:
    """Custom assertion helpers for testing"""
    
    @staticmethod
    def assert_valid_response(response_data: Dict[str, Any], required_fields: list):
        """Assert that response data contains all required fields"""
        for field in required_fields:
            assert field in response_data, f"Missing required field: {field}"
    
    @staticmethod
    def assert_valid_metrics(metrics_data: Dict[str, Any]):
        """Assert that metrics data is valid"""
        assert "component_type" in metrics_data
        assert "component_name" in metrics_data
        assert "metrics" in metrics_data
        assert isinstance(metrics_data["metrics"], dict)
    
    @staticmethod
    def assert_valid_conversation(conversation_data: Dict[str, Any]):
        """Assert that conversation data is valid"""
        assert "conversation_id" in conversation_data
        assert "messages" in conversation_data
        assert isinstance(conversation_data["messages"], list)
    
    @staticmethod
    def assert_valid_agent_result(result_data: Dict[str, Any]):
        """Assert that agent result is valid"""
        assert "result" in result_data or "messages" in result_data
        if "messages" in result_data:
            assert isinstance(result_data["messages"], list)


@pytest.fixture
def custom_assertions():
    """Fixture providing custom assertion helpers"""
    return CustomAssertions()


# Performance testing utilities
class PerformanceTracker:
    """Helper class for tracking performance in tests"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start tracking performance"""
        import time
        self.start_time = time.time()
    
    def stop(self):
        """Stop tracking performance"""
        import time
        self.end_time = time.time()
    
    @property
    def duration(self):
        """Get the duration of the tracked operation"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    def assert_max_duration(self, max_duration: float):
        """Assert that the operation completed within max_duration"""
        assert self.duration is not None, "Performance tracking not completed"
        assert self.duration <= max_duration, f"Operation took {self.duration}s, expected <= {max_duration}s"


@pytest.fixture
def performance_tracker():
    """Fixture providing performance tracker"""
    return PerformanceTracker()


# Test data generators
class TestDataGenerator:
    """Helper class for generating test data"""
    
    @staticmethod
    def generate_conversation_data(message_count: int = 2) -> Dict[str, Any]:
        """Generate conversation data with specified number of messages"""
        messages = []
        for i in range(message_count):
            role = "user" if i % 2 == 0 else "assistant"
            content = f"Message {i+1}"
            messages.append({
                "role": role,
                "content": content,
                "timestamp": f"2023-01-01T00:00:{i:02d}"
            })
        
        return {
            "conversation_id": "test_conversation",
            "agent_name": "test_agent",
            "messages": messages
        }
    
    @staticmethod
    def generate_metrics_data(component_type: str = "llm") -> Dict[str, Any]:
        """Generate metrics data for specified component type"""
        return {
            "component_type": component_type,
            "component_name": f"test_{component_type}",
            "metrics": {
                "request_count": 100,
                "success_rate": 0.95,
                "avg_duration": 1.5,
                "error_count": 5
            }
        }
    
    @staticmethod
    def generate_tool_list(count: int = 3) -> list:
        """Generate a list of tools"""
        tools = []
        for i in range(count):
            tools.append({
                "name": f"tool_{i+1}",
                "description": f"Test tool {i+1}",
                "parameters": {"param1": "value1", "param2": "value2"}
            })
        return tools
    
    @staticmethod
    def generate_agent_list(count: int = 3) -> list:
        """Generate a list of agents"""
        agents = []
        for i in range(count):
            agents.append({
                "name": f"agent_{i+1}",
                "description": f"Test agent {i+1}",
                "type": "workflow" if i % 2 == 0 else "react"
            })
        return agents


@pytest.fixture
def test_data_generator():
    """Fixture providing test data generator"""
    return TestDataGenerator()


# Configure pytest with custom markers
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "system: mark test as a system test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "langchain: mark test as LangChain specific")
    config.addinivalue_line("markers", "api: mark test as API specific")
    config.addinivalue_line("markers", "database: mark test as database specific")
    config.addinivalue_line("markers", "monitoring: mark test as monitoring specific")
    config.addinivalue_line("markers", "memory: mark test as memory management specific")
    config.addinivalue_line("markers", "agents: mark test as agent specific")
    config.addinivalue_line("markers", "tools: mark test as tool specific")
    config.addinivalue_line("markers", "llm: mark test as LLM specific")
    config.addinivalue_line("markers", "visual: mark test as visual LLM specific")
    config.addinivalue_line("markers", "multi_writer: mark test as multi-writer system specific")