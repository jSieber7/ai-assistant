"""
Pytest configuration and fixtures for the test suite.

This module provides common fixtures and configuration for all tests,
including database setup, mocking, and test utilities.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any, Generator
import os
import sys

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_settings():
    """Mock secure settings for testing"""
    settings = Mock()
    settings.get_setting.side_effect = lambda section, key, default=None: {
        # LangChain settings
        ('langchain', 'integration_mode'): 'langchain',
        ('langchain', 'llm_manager_enabled'): True,
        ('langchain', 'tool_registry_enabled'): True,
        ('langchain', 'agent_manager_enabled'): True,
        ('langchain', 'memory_workflow_enabled'): True,
        ('langchain', 'monitoring_enabled'): True,
        
        # LLM provider settings
        ('llm_providers', 'openai', 'api_key'): 'test-openai-key',
        ('llm_providers', 'openai', 'model'): 'gpt-3.5-turbo',
        ('llm_providers', 'openai', 'base_url'): 'https://api.openai.com/v1',
        ('llm_providers', 'openai', 'timeout'): 30,
        ('llm_providers', 'openai', 'max_retries'): 3,
        
        ('llm_providers', 'ollama', 'base_url'): 'http://localhost:11434',
        ('llm_providers', 'ollama', 'model'): 'llama2',
        ('llm_providers', 'ollama', 'timeout'): 60,
        ('llm_providers', 'ollama', 'max_retries'): 3,
        
        # Database settings
        ('database', 'host'): 'localhost',
        ('database', 'port'): '5432',
        ('database', 'name'): 'test_db',
        ('database', 'user'): 'test_user',
        ('database', 'password'): 'test_password',
        
        # Monitoring settings
        ('monitoring', 'metrics_retention_days'): 30,
        ('monitoring', 'performance_tracking_enabled'): True,
        ('monitoring', 'error_tracking_enabled'): True,
        
        # Memory settings
        ('memory', 'max_conversation_length'): 100,
        ('memory', 'conversation_timeout_hours'): 24,
        ('memory', 'cleanup_interval_hours'): 6,
        
        # Agent settings
        ('agents', 'default_timeout'): 60,
        ('agents', 'max_concurrent_invocations'): 10,
        ('agents', 'checkpoint_enabled'): True,
        
        # Tool settings
        ('tools', 'default_timeout'): 30,
        ('tools', 'max_concurrent_executions'): 5,
        ('tools', 'error_retry_attempts'): 3,
        
        # Default return for unknown settings
    }.get((section, key), default)
    return settings


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
    return client


@pytest.fixture
def mock_llm():
    """Mock LLM for testing"""
    llm = AsyncMock()
    llm.model_name = "test-model"
    llm.ainvoke.return_value.content = "Test response"
    llm.abatch.return_value = [Mock(content="Test response 1"), Mock(content="Test response 2")]
    return llm


@pytest.fixture
def mock_tool():
    """Mock tool for testing"""
    tool = AsyncMock()
    tool.name = "test_tool"
    tool.description = "Test tool description"
    tool.return_value = {"success": True, "result": "Tool executed successfully"}
    return tool


@pytest.fixture
def mock_agent():
    """Mock agent for testing"""
    agent = AsyncMock()
    agent.name = "test_agent"
    agent.description = "Test agent description"
    agent.ainvoke.return_value = {
        "result": "Agent executed successfully",
        "messages": [{"role": "assistant", "content": "Response"}]
    }
    return agent


@pytest.fixture
def sample_conversation_data():
    """Sample conversation data for testing"""
    return {
        "conversation_id": "test_conversation",
        "agent_name": "test_agent",
        "messages": [
            {"role": "human", "content": "Hello, world!", "timestamp": "2023-01-01T00:00:00"},
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
def sample_agent_workflow_data():
    """Sample agent workflow data for testing"""
    return {
        "agent_name": "research_agent",
        "input_data": {
            "messages": [{"role": "user", "content": "Research machine learning"}]
        },
        "config": {"thread_id": "test_thread"},
        "expected_result": {
            "result": "Research completed",
            "messages": [
                {"role": "user", "content": "Research machine learning"},
                {"role": "assistant", "content": "Research completed"}
            ]
        }
    }


@pytest.fixture
def sample_tool_execution_data():
    """Sample tool execution data for testing"""
    return {
        "tool_name": "search_tool",
        "parameters": {"query": "machine learning"},
        "expected_result": {
            "success": True,
            "result": "Search results for machine learning",
            "execution_time": 0.5
        }
    }


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
def temp_env_vars():
    """Temporary environment variables for testing"""
    original_env = {}
    
    def set_env(key: str, value: str):
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield set_env
    
    # Restore original environment variables
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


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


# Test markers for categorizing tests
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "langchain: mark test as LangChain specific"
    )
    config.addinivalue_line(
        "markers", "api: mark test as API specific"
    )
    config.addinivalue_line(
        "markers", "database: mark test as database specific"
    )


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
        assert "agent_name" in conversation_data
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


# Test data generators
class TestDataGenerator:
    """Helper class for generating test data"""
    
    @staticmethod
    def generate_conversation_data(message_count: int = 2) -> Dict[str, Any]:
        """Generate conversation data with specified number of messages"""
        messages = []
        for i in range(message_count):
            role = "human" if i % 2 == 0 else "assistant"
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