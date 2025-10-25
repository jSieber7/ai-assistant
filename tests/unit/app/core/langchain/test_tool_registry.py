"""
Unit tests for LangChain Tool Registry.

This module tests tool registration, execution, and integration
with LangChain tool components.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, Callable

from langchain_core.tools import Tool
from langchain_core.tools.base import BaseTool

from app.core.langchain.tool_registry import LangChainToolRegistry
from app.core.secure_settings import secure_settings


class TestLangChainToolRegistry:
    """Test cases for LangChain Tool Registry"""
    
    @pytest.fixture
    async def tool_registry(self):
        """Create a LangChain Tool Registry instance for testing"""
        registry = LangChainToolRegistry()
        await registry.initialize()
        return registry
    
    @pytest.fixture
    def mock_settings(self):
        """Mock secure settings for testing"""
        mock_settings = Mock()
        mock_settings.get_setting.side_effect = lambda section, key, default=None: {
            ('langchain', 'tool_registry_enabled'): True,
            ('tools', 'execution_timeout'): 30,
            ('tools', 'max_concurrent_executions'): 10,
        }.get((section, key), default)
        return mock_settings
    
    @pytest.fixture
    def sample_tool(self):
        """Create a sample tool for testing"""
        def sample_function(input_text: str) -> str:
            """A sample tool function"""
            return f"Processed: {input_text}"
        
        return Tool(
            name="sample_tool",
            description="A sample tool for testing",
            func=sample_function
        )
    
    async def test_initialize_success(self, mock_settings):
        """Test successful initialization of tool registry"""
        with patch('app.core.langchain.tool_registry.secure_settings', mock_settings):
            registry = LangChainToolRegistry()
            
            # Test initialization
            await registry.initialize()
            
            # Verify initialization
            assert registry._initialized is True
            assert registry._monitoring is not None
            assert isinstance(registry._tools, dict)
    
    async def test_register_langchain_tool(self, tool_registry, sample_tool):
        """Test registering a LangChain tool"""
        # Register the tool
        await tool_registry.register_langchain_tool(sample_tool)
        
        # Verify registration
        assert "sample_tool" in tool_registry._tools
        assert tool_registry._tools["sample_tool"]["tool"] is sample_tool
        assert tool_registry._tools["sample_tool"]["type"] == "langchain"
    
    async def test_register_custom_tool(self, tool_registry):
        """Test registering a custom tool function"""
        def custom_tool(input_data: str) -> Dict[str, Any]:
            """A custom tool function"""
            return {"result": f"Custom: {input_data}"}
        
        # Register the custom tool
        await tool_registry.register_custom_tool(
            name="custom_tool",
            func=custom_tool,
            description="A custom tool for testing"
        )
        
        # Verify registration
        assert "custom_tool" in tool_registry._tools
        assert tool_registry._tools["custom_tool"]["type"] == "custom"
        assert callable(tool_registry._tools["custom_tool"]["tool"])
    
    async def test_get_tool(self, tool_registry, sample_tool):
        """Test getting a registered tool"""
        # Register the tool
        await tool_registry.register_langchain_tool(sample_tool)
        
        # Get the tool
        retrieved_tool = tool_registry.get_tool("sample_tool")
        
        assert retrieved_tool is sample_tool
    
    async def test_get_tool_not_found(self, tool_registry):
        """Test getting a tool that doesn't exist"""
        tool = tool_registry.get_tool("non_existent_tool")
        assert tool is None
    
    async def test_list_tools(self, tool_registry, sample_tool):
        """Test listing all registered tools"""
        # Register some tools
        await tool_registry.register_langchain_tool(sample_tool)
        
        def another_tool(input_data: str) -> str:
            return f"Another: {input_data}"
        
        await tool_registry.register_custom_tool(
            name="another_tool",
            func=another_tool,
            description="Another tool for testing"
        )
        
        # List tools
        tools = tool_registry.list_tools()
        
        assert isinstance(tools, list)
        assert len(tools) == 2
        tool_names = [tool.name if hasattr(tool, 'name') else tool.get('name') for tool in tools]
        assert "sample_tool" in tool_names
        assert "another_tool" in tool_names
    
    async def test_execute_tool_success(self, tool_registry, sample_tool):
        """Test successful tool execution"""
        # Register the tool
        await tool_registry.register_langchain_tool(sample_tool)
        
        # Execute the tool
        result = await tool_registry.execute_tool(
            tool_name="sample_tool",
            input_data="test input"
        )
        
        assert result == "Processed: test input"
    
    async def test_execute_tool_with_kwargs(self, tool_registry):
        """Test tool execution with keyword arguments"""
        def tool_with_kwargs(arg1: str, arg2: int, flag: bool = False) -> Dict[str, Any]:
            return {
                "arg1": arg1,
                "arg2": arg2,
                "flag": flag
            }
        
        # Register the tool
        await tool_registry.register_custom_tool(
            name="tool_with_kwargs",
            func=tool_with_kwargs,
            description="Tool with keyword arguments"
        )
        
        # Execute with kwargs
        result = await tool_registry.execute_tool(
            tool_name="tool_with_kwargs",
            input_data="test",
            arg2=42,
            flag=True
        )
        
        assert result["arg1"] == "test"
        assert result["arg2"] == 42
        assert result["flag"] is True
    
    async def test_execute_tool_not_found(self, tool_registry):
        """Test executing a tool that doesn't exist"""
        with pytest.raises(ValueError, match="Tool 'non_existent_tool' not found"):
            await tool_registry.execute_tool("non_existent_tool", "test input")
    
    async def test_execute_tool_error(self, tool_registry):
        """Test tool execution with error"""
        def error_tool(input_data: str) -> str:
            raise ValueError("Tool execution error")
        
        # Register the tool
        await tool_registry.register_custom_tool(
            name="error_tool",
            func=error_tool,
            description="Tool that always errors"
        )
        
        # Execute should raise the error
        with pytest.raises(ValueError, match="Tool execution error"):
            await tool_registry.execute_tool("error_tool", "test input")
    
    async def test_unregister_tool(self, tool_registry, sample_tool):
        """Test unregistering a tool"""
        # Register the tool
        await tool_registry.register_langchain_tool(sample_tool)
        
        # Verify it's registered
        assert "sample_tool" in tool_registry._tools
        
        # Unregister it
        result = await tool_registry.unregister_tool("sample_tool")
        
        assert result is True
        assert "sample_tool" not in tool_registry._tools
    
    async def test_unregister_tool_not_found(self, tool_registry):
        """Test unregistering a tool that doesn't exist"""
        result = await tool_registry.unregister_tool("non_existent_tool")
        assert result is False
    
    async def test_get_tool_info(self, tool_registry, sample_tool):
        """Test getting information about a tool"""
        # Register the tool
        await tool_registry.register_langchain_tool(sample_tool)
        
        # Get tool info
        info = tool_registry.get_tool_info("sample_tool")
        
        assert isinstance(info, dict)
        assert "name" in info
        assert "description" in info
        assert "type" in info
        assert "registered_at" in info
        assert info["name"] == "sample_tool"
    
    async def test_get_tool_info_not_found(self, tool_registry):
        """Test getting info for a tool that doesn't exist"""
        info = tool_registry.get_tool_info("non_existent_tool")
        assert info is None
    
    async def test_health_check(self, tool_registry, sample_tool):
        """Test health check functionality"""
        # Register some tools
        await tool_registry.register_langchain_tool(sample_tool)
        
        def another_tool(input_data: str) -> str:
            return f"Another: {input_data}"
        
        await tool_registry.register_custom_tool(
            name="another_tool",
            func=another_tool,
            description="Another tool for testing"
        )
        
        # Perform health check
        health = await tool_registry.health_check()
        
        assert isinstance(health, dict)
        assert "overall_status" in health
        assert "tools" in health
        assert "timestamp" in health
        assert "sample_tool" in health["tools"]
        assert "another_tool" in health["tools"]
    
    async def test_shutdown(self, tool_registry):
        """Test shutdown functionality"""
        # Mock monitoring shutdown
        tool_registry._monitoring.shutdown = AsyncMock()
        
        await tool_registry.shutdown()
        
        # Verify shutdown was called
        tool_registry._monitoring.shutdown.assert_called_once()
        
        # Verify registry is marked as not initialized
        assert tool_registry._initialized is False
        
        # Verify tools are cleared
        assert len(tool_registry._tools) == 0
    
    async def test_concurrent_executions(self, tool_registry, sample_tool):
        """Test concurrent tool executions"""
        # Register the tool
        await tool_registry.register_langchain_tool(sample_tool)
        
        # Execute tool concurrently
        tasks = [
            tool_registry.execute_tool("sample_tool", f"input_{i}")
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify all executions succeeded
        expected = [f"Processed: input_{i}" for i in range(5)]
        assert results == expected
    
    async def test_tool_validation(self, tool_registry):
        """Test tool validation during registration"""
        # Test invalid tool (no name)
        with pytest.raises(ValueError, match="Tool must have a name"):
            await tool_registry.register_custom_tool(
                name="",
                func=lambda x: x,
                description="Invalid tool"
            )
        
        # Test invalid tool (no function)
        with pytest.raises(ValueError, match="Tool must have a callable function"):
            await tool_registry.register_custom_tool(
                name="invalid_tool",
                func=None,
                description="Invalid tool"
            )
    
    async def test_monitoring_integration(self, tool_registry, sample_tool):
        """Test that monitoring is properly integrated"""
        # Mock monitoring to track calls
        tool_registry._monitoring.record_metric = AsyncMock()
        
        # Register and execute a tool
        await tool_registry.register_langchain_tool(sample_tool)
        await tool_registry.execute_tool("sample_tool", "test input")
        
        # Verify monitoring was called
        tool_registry._monitoring.record_metric.assert_called()
    
    async def test_tool_caching(self, tool_registry, sample_tool):
        """Test that tools are properly cached"""
        # Register a tool
        await tool_registry.register_langchain_tool(sample_tool)
        
        # Get the tool multiple times
        tool1 = tool_registry.get_tool("sample_tool")
        tool2 = tool_registry.get_tool("sample_tool")
        tool3 = tool_registry.get_tool("sample_tool")
        
        # All should be the same instance
        assert tool1 is tool2
        assert tool2 is tool3
    
    async def test_get_statistics(self, tool_registry, sample_tool):
        """Test getting tool registry statistics"""
        # Register some tools
        await tool_registry.register_langchain_tool(sample_tool)
        
        def another_tool(input_data: str) -> str:
            return f"Another: {input_data}"
        
        await tool_registry.register_custom_tool(
            name="another_tool",
            func=another_tool,
            description="Another tool for testing"
        )
        
        # Get statistics
        stats = tool_registry.get_statistics()
        
        assert isinstance(stats, dict)
        assert "total_tools_registered" in stats
        assert "tools_by_type" in stats
        assert "initialized" in stats
        assert stats["total_tools_registered"] == 2
        assert stats["tools_by_type"]["langchain"] == 1
        assert stats["tools_by_type"]["custom"] == 1
    
    async def test_tool_execution_timeout(self, tool_registry):
        """Test tool execution timeout"""
        async def slow_tool(input_data: str) -> str:
            await asyncio.sleep(2)  # Simulate slow operation
            return f"Slow: {input_data}"
        
        # Register the tool
        await tool_registry.register_custom_tool(
            name="slow_tool",
            func=slow_tool,
            description="A slow tool"
        )
        
        # Set a short timeout
        tool_registry._execution_timeout = 1.0
        
        # Execute should timeout
        with pytest.raises(asyncio.TimeoutError):
            await tool_registry.execute_tool("slow_tool", "test input")
    
    async def test_tool_error_handling(self, tool_registry):
        """Test error handling during tool execution"""
        def error_tool(input_data: str) -> str:
            raise RuntimeError("Tool runtime error")
        
        # Register the tool
        await tool_registry.register_custom_tool(
            name="error_tool",
            func=error_tool,
            description="Tool that errors"
        )
        
        # Execute should handle the error gracefully
        with pytest.raises(RuntimeError, match="Tool runtime error"):
            await tool_registry.execute_tool("error_tool", "test input")


if __name__ == "__main__":
    pytest.main([__file__])