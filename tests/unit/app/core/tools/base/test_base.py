"""
Unit tests for Tool Base Components.

This module tests the base tool classes and configurations.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import uuid
from typing import List, Dict, Any, Optional, Tuple
import json

from app.core.tools.base.base import (
    ToolResult,
    ToolError,
    ToolTimeoutError,
    ToolConfigurationError,
    ToolExecutionError,
    BaseTool
)
from app.core.tools.base.config import ToolSystemSettings


class TestToolResult:
    """Test ToolResult class"""

    def test_tool_result_success(self):
        """Test ToolResult with success"""
        result = ToolResult(
            success=True,
            data={"result": "success"},
            message="Operation completed successfully"
        )
        
        assert result.success is True
        assert result.data == {"result": "success"}
        assert result.message == "Operation completed successfully"
        assert result.error is None
        assert result.metadata == {}

    def test_tool_result_failure(self):
        """Test ToolResult with failure"""
        result = ToolResult(
            success=False,
            error="Tool execution failed",
            message="Error occurred during execution"
        )
        
        assert result.success is False
        assert result.error == "Tool execution failed"
        assert result.message == "Error occurred during execution"
        assert result.data is None
        assert result.metadata == {}

    def test_tool_result_with_metadata(self):
        """Test ToolResult with metadata"""
        metadata = {"execution_time": 1.5, "tokens_used": 100}
        
        result = ToolResult(
            success=True,
            data={"result": "success"},
            metadata=metadata
        )
        
        assert result.success is True
        assert result.data == {"result": "success"}
        assert result.metadata == metadata

    def test_tool_result_to_dict(self):
        """Test ToolResult to_dict method"""
        metadata = {"test": True}
        
        result = ToolResult(
            success=True,
            data={"result": "test"},
            message="Test result",
            metadata=metadata
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["success"] is True
        assert result_dict["data"] == {"result": "test"}
        assert result_dict["message"] == "Test result"
        assert result_dict["error"] is None
        assert result_dict["metadata"] == metadata


class TestToolErrors:
    """Test tool error classes"""

    def test_tool_error(self):
        """Test ToolError base class"""
        error = ToolError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.args == ("Test error message",)

    def test_tool_error_with_code(self):
        """Test ToolError with error code"""
        error = ToolError("Test error", code=500)
        
        assert str(error) == "Test error"
        assert error.code == 500

    def test_tool_timeout_error(self):
        """Test ToolTimeoutError class"""
        error = ToolTimeoutError("Operation timed out after 30 seconds")
        
        assert isinstance(error, ToolError)
        assert str(error) == "Operation timed out after 30 seconds"

    def test_tool_configuration_error(self):
        """Test ToolConfigurationError class"""
        error = ToolConfigurationError("Invalid API key")
        
        assert isinstance(error, ToolError)
        assert str(error) == "Invalid API key"

    def test_tool_execution_error(self):
        """Test ToolExecutionError class"""
        error = ToolExecutionError("Failed to execute tool")
        
        assert isinstance(error, ToolError)
        assert str(error) == "Failed to execute tool"


class TestToolSystemSettings:
    """Test ToolSystemSettings class"""

    def test_tool_system_settings_defaults(self):
        """Test ToolSystemSettings default values"""
        settings = ToolSystemSettings()
        
        assert settings.max_concurrent_tools == 10
        assert settings.default_timeout == 30.0
        assert settings.enable_monitoring is True
        assert settings.enable_caching is True
        assert settings.cache_ttl == 3600

    def test_tool_system_settings_with_values(self):
        """Test ToolSystemSettings with provided values"""
        settings = ToolSystemSettings(
            max_concurrent_tools=20,
            default_timeout=60.0,
            enable_monitoring=False,
            enable_caching=False,
            cache_ttl=7200
        )
        
        assert settings.max_concurrent_tools == 20
        assert settings.default_timeout == 60.0
        assert settings.enable_monitoring is False
        assert settings.enable_caching is False
        assert settings.cache_ttl == 7200


class MockTool(BaseTool):
    """Mock tool for testing BaseTool"""
    
    def __init__(self, name="mock_tool", description="A mock tool for testing"):
        super().__init__()
        self._name = name
        self._description = description
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    @property
    def keywords(self) -> List[str]:
        return ["mock", "test"]
    
    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "input": {
                "type": "string",
                "description": "Input parameter",
                "required": True
            }
        }
    
    @property
    def category(self) -> str:
        return "test"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def requires_api_key(self) -> bool:
        return False
    
    @property
    def requires_auth(self) -> bool:
        return False
    
    async def execute(self, **kwargs) -> Any:
        if "input" in kwargs:
            return f"Processed: {kwargs['input']}"
        return "Processed: no input"


class TestBaseTool:
    """Test BaseTool class"""

    def test_base_tool_init(self):
        """Test BaseTool initialization"""
        tool = MockTool()
        
        assert tool.name == "mock_tool"
        assert tool.description == "A mock tool for testing"
        assert tool.keywords == ["mock", "test"]
        assert tool.category == "test"
        assert tool.version == "1.0.0"
        assert tool.requires_api_key is False
        assert tool.requires_auth is False
        assert tool.execution_count == 0
        assert tool.total_execution_time == 0.0
        assert tool.last_execution_time is None

    def test_base_tool_properties(self):
        """Test BaseTool properties"""
        tool = MockTool()
        
        # Test that properties return expected values
        assert isinstance(tool.name, str)
        assert isinstance(tool.description, str)
        assert isinstance(tool.keywords, list)
        assert isinstance(tool.parameters, dict)
        assert isinstance(tool.category, str)
        assert isinstance(tool.version, str)
        assert isinstance(tool.requires_api_key, bool)
        assert isinstance(tool.requires_auth, bool)

    def test_base_tool_parameters_structure(self):
        """Test BaseTool parameters structure"""
        tool = MockTool()
        
        parameters = tool.parameters
        
        assert isinstance(parameters, dict)
        assert "input" in parameters
        
        input_param = parameters["input"]
        assert "type" in input_param
        assert "description" in input_param
        assert "required" in input_param

    @pytest.mark.asyncio
    async def test_base_tool_execute(self):
        """Test BaseTool execute method"""
        tool = MockTool()
        
        result = await tool.execute(input="test input")
        
        assert result == "Processed: test input"
        assert tool.execution_count == 1
        assert tool.total_execution_time > 0
        assert tool.last_execution_time is not None

    @pytest.mark.asyncio
    async def test_base_tool_execute_without_input(self):
        """Test BaseTool execute without required input"""
        tool = MockTool()
        
        result = await tool.execute()
        
        assert result == "Processed: no input"
        assert tool.execution_count == 1

    @pytest.mark.asyncio
    async def test_execute_with_timeout_success(self):
        """Test execute_with_timeout with successful execution"""
        tool = MockTool()
        
        result = await tool.execute_with_timeout(input="test input", timeout=5.0)
        
        assert result.success is True
        assert result.data == "Processed: test input"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_execute_with_timeout_timeout(self):
        """Test execute_with_timeout with timeout"""
        class SlowTool(BaseTool):
            @property
            def name(self) -> str:
                return "slow_tool"
            
            @property
            def description(self) -> str:
                return "A slow tool"
            
            @property
            def keywords(self) -> List[str]:
                return ["slow"]
            
            @property
            def parameters(self) -> Dict[str, Dict[str, Any]]:
                return {}
            
            @property
            def category(self) -> str:
                return "test"
            
            @property
            def version(self) -> str:
                return "1.0.0"
            
            @property
            def requires_api_key(self) -> bool:
                return False
            
            @property
            def requires_auth(self) -> bool:
                return False
            
            async def execute(self, **kwargs) -> Any:
                await asyncio.sleep(2.0)  # Sleep longer than timeout
                return "slow result"
        
        tool = SlowTool()
        
        with pytest.raises(ToolTimeoutError):
            await tool.execute_with_timeout(timeout=1.0)

    @pytest.mark.asyncio
    async def test_execute_with_timeout_error(self):
        """Test execute_with_timeout with execution error"""
        class ErrorTool(BaseTool):
            @property
            def name(self) -> str:
                return "error_tool"
            
            @property
            def description(self) -> str:
                return "A tool that always errors"
            
            @property
            def keywords(self) -> List[str]:
                return ["error"]
            
            @property
            def parameters(self) -> Dict[str, Dict[str, Any]]:
                return {}
            
            @property
            def category(self) -> str:
                return "test"
            
            @property
            def version(self) -> str:
                return "1.0.0"
            
            @property
            def requires_api_key(self) -> bool:
                return False
            
            @property
            def requires_auth(self) -> bool:
                return False
            
            async def execute(self, **kwargs) -> Any:
                raise Exception("Tool execution failed")
        
        tool = ErrorTool()
        
        with pytest.raises(ToolExecutionError):
            await tool.execute_with_timeout()

    @pytest.mark.asyncio
    async def test_execute_with_timeout_monitoring(self):
        """Test execute_with_timeout with monitoring integration"""
        tool = MockTool()
        
        with patch('app.core.tools.base.base.track_tool_execution') as mock_track:
            mock_track.return_value = AsyncMock()
            
            await tool.execute_with_timeout(input="test input")
            
            # Check that monitoring was called
            mock_track.assert_called_once()

    def test_should_use(self):
        """Test should_use method"""
        tool = MockTool()
        
        # Test with matching keyword
        assert tool.should_use("Use mock tool to process data") is True
        
        # Test with non-matching keyword
        assert tool.should_use("Use search tool to find information") is False
        
        # Test with context
        context = {"available_tools": ["mock_tool", "search_tool"]}
        assert tool.should_use("mock", context) is True

    def test_should_use_with_context(self):
        """Test should_use method with context"""
        tool = MockTool()
        
        # Test with context containing the tool
        context = {"preferred_tool": "mock_tool"}
        assert tool.should_use("process data", context) is True
        
        # Test with context not containing the tool
        context = {"preferred_tool": "search_tool"}
        assert tool.should_use("process data", context) is False

    def test_get_usage_stats(self):
        """Test get_usage_stats method"""
        tool = MockTool()
        
        # Initially should have no stats
        stats = tool.get_usage_stats()
        assert stats["execution_count"] == 0
        assert stats["total_execution_time"] == 0.0
        assert stats["average_execution_time"] == 0.0
        assert stats["last_execution_time"] is None
        
        # Simulate some executions
        tool.execution_count = 5
        tool.total_execution_time = 10.5
        tool.last_execution_time = datetime.now()
        
        stats = tool.get_usage_stats()
        assert stats["execution_count"] == 5
        assert stats["total_execution_time"] == 10.5
        assert stats["average_execution_time"] == 2.1
        assert stats["last_execution_time"] is not None

    def test_validate_parameters_valid(self):
        """Test validate_parameters with valid parameters"""
        tool = MockTool()
        
        # Valid parameters
        result = tool.validate_parameters(input="test")
        assert result is True

    def test_validate_parameters_missing_required(self):
        """Test validate_parameters with missing required parameter"""
        tool = MockTool()
        
        # Missing required parameter
        result = tool.validate_parameters()
        assert result is False

    def test_validate_parameters_invalid_type(self):
        """Test validate_parameters with invalid parameter type"""
        tool = MockTool()
        
        # Invalid parameter type (should be string)
        result = tool.validate_parameters(input=123)
        # Should still be True as base implementation doesn't validate types
        assert result is True

    def test_validate_parameters_extra_parameters(self):
        """Test validate_parameters with extra parameters"""
        tool = MockTool()
        
        # Extra parameters should be allowed
        result = tool.validate_parameters(input="test", extra_param="value")
        assert result is True

    @pytest.mark.asyncio
    async def test_execution_stats_tracking(self):
        """Test that execution stats are properly tracked"""
        tool = MockTool()
        
        # Execute multiple times
        await tool.execute(input="test1")
        await tool.execute(input="test2")
        await tool.execute(input="test3")
        
        # Check stats
        assert tool.execution_count == 3
        assert tool.total_execution_time > 0
        assert tool.last_execution_time is not None
        
        # Check average calculation
        stats = tool.get_usage_stats()
        assert stats["average_execution_time"] == tool.total_execution_time / 3

    def test_tool_inheritance(self):
        """Test that custom tools can inherit from BaseTool"""
        class CustomTool(BaseTool):
            @property
            def name(self) -> str:
                return "custom_tool"
            
            @property
            def description(self) -> str:
                return "A custom tool"
            
            @property
            def keywords(self) -> List[str]:
                return ["custom"]
            
            @property
            def parameters(self) -> Dict[str, Dict[str, Any]]:
                return {}
            
            @property
            def category(self) -> str:
                return "custom"
            
            @property
            def version(self) -> str:
                return "1.0.0"
            
            @property
            def requires_api_key(self) -> bool:
                return False
            
            @property
            def requires_auth(self) -> bool:
                return False
            
            async def execute(self, **kwargs) -> Any:
                return "custom result"
        
        tool = CustomTool()
        
        assert isinstance(tool, BaseTool)
        assert tool.name == "custom_tool"
        assert tool.description == "A custom tool"
        assert tool.keywords == ["custom"]
        assert tool.category == "custom"
        assert tool.version == "1.0.0"
        assert tool.requires_api_key is False
        assert tool.requires_auth is False

    def test_abstract_methods(self):
        """Test that BaseTool requires abstract methods to be implemented"""
        # Should not be able to instantiate BaseTool directly
        with pytest.raises(TypeError):
            BaseTool()