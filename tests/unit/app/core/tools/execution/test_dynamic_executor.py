"""
Unit tests for Dynamic Tool Executor.

This module tests the dynamic tool execution system.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import uuid
from typing import List, Dict, Any, Optional, Tuple
import json
import time

from app.core.tools.execution.dynamic_executor import DynamicExecutor
from app.core.tools.base.base import BaseTool, ToolResult


class MockTool(BaseTool):
    """Mock tool for testing"""
    
    def __init__(self, name="mock_tool", should_fail=False, execution_time=0.1):
        super().__init__()
        self.name = name
        self.description = f"Mock tool {name}"
        self.keywords = ["mock", "test"]
        self.category = "test"
        self.version = "1.0.0"
        self.requires_api_key = False
        self.requires_auth = False
        self._should_fail = should_fail
        self._execution_time = execution_time
        self._execution_count = 0
    
    def get_parameters(self):
        return {
            "query": {
                "type": "string",
                "required": True,
                "description": "Test query parameter"
            }
        }
    
    async def execute(self, **kwargs):
        self._execution_count += 1
        await asyncio.sleep(self._execution_time)  # Simulate execution time
        
        if self._should_fail:
            raise Exception(f"Tool {self.name} failed as requested")
        
        return ToolResult(
            success=True,
            data={"result": f"Mock result for {self.name}", "query": kwargs.get("query")},
            message=f"Successfully executed {self.name}",
            execution_time=self._execution_time
        )
    
    def validate_parameters(self, **kwargs):
        return "query" in kwargs and kwargs["query"] is not None
    
    def should_use(self, text, context=None):
        return self.name in text.lower()


class TestDynamicExecutor:
    """Test DynamicExecutor class"""

    @pytest.fixture
    def executor(self):
        """Create a DynamicExecutor instance"""
        return DynamicExecutor()

    @pytest.fixture
    def mock_tools(self):
        """Create mock tools for testing"""
        return [
            MockTool("tool1"),
            MockTool("tool2"),
            MockTool("tool3", should_fail=True)
        ]

    def test_executor_init(self, executor):
        """Test DynamicExecutor initialization"""
        assert executor.name == "dynamic_executor"
        assert executor.description == "Dynamically executes tools based on context"
        assert isinstance(executor.tools, dict)
        assert len(executor.tools) == 0
        assert executor.default_timeout == 30.0
        assert executor.max_concurrent_executions == 10

    def test_executor_init_with_config(self):
        """Test DynamicExecutor initialization with custom config"""
        config = {
            "default_timeout": 60.0,
            "max_concurrent_executions": 5,
            "enable_caching": True
        }
        executor = DynamicExecutor(config)
        
        assert executor.default_timeout == 60.0
        assert executor.max_concurrent_executions == 5
        assert executor.enable_caching is True

    def test_register_tool(self, executor, mock_tools):
        """Test tool registration"""
        tool = mock_tools[0]
        
        executor.register_tool(tool)
        
        assert tool.name in executor.tools
        assert executor.tools[tool.name] == tool

    def test_register_tool_with_duplicate(self, executor, mock_tools):
        """Test tool registration with duplicate name"""
        tool1 = mock_tools[0]
        tool2 = MockTool("tool1")  # Same name
        
        executor.register_tool(tool1)
        
        with pytest.raises(ValueError, match="Tool tool1 is already registered"):
            executor.register_tool(tool2)

    def test_unregister_tool(self, executor, mock_tools):
        """Test tool unregistration"""
        tool = mock_tools[0]
        
        executor.register_tool(tool)
        assert tool.name in executor.tools
        
        executor.unregister_tool(tool.name)
        assert tool.name not in executor.tools

    def test_unregister_nonexistent_tool(self, executor):
        """Test unregistering non-existent tool"""
        with pytest.raises(ValueError, match="Tool nonexistent_tool is not registered"):
            executor.unregister_tool("nonexistent_tool")

    def test_list_tools(self, executor, mock_tools):
        """Test listing registered tools"""
        # Initially empty
        assert executor.list_tools() == []
        
        # Register tools
        for tool in mock_tools:
            executor.register_tool(tool)
        
        tools = executor.list_tools()
        assert len(tools) == 3
        assert "tool1" in tools
        assert "tool2" in tools
        assert "tool3" in tools

    def test_get_tool(self, executor, mock_tools):
        """Test getting a specific tool"""
        tool = mock_tools[0]
        executor.register_tool(tool)
        
        retrieved_tool = executor.get_tool("tool1")
        assert retrieved_tool == tool

    def test_get_nonexistent_tool(self, executor):
        """Test getting non-existent tool"""
        assert executor.get_tool("nonexistent_tool") is None

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, executor, mock_tools):
        """Test successful tool execution"""
        tool = mock_tools[0]
        executor.register_tool(tool)
        
        result = await executor.execute_tool(
            tool_name="tool1",
            query="test query"
        )
        
        assert result.success is True
        assert "Mock result for tool1" in result.data["result"]
        assert result.data["query"] == "test query"

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self, executor):
        """Test executing non-existent tool"""
        with pytest.raises(ValueError, match="Tool nonexistent_tool is not registered"):
            await executor.execute_tool(
                tool_name="nonexistent_tool",
                query="test query"
            )

    @pytest.mark.asyncio
    async def test_execute_tool_failure(self, executor, mock_tools):
        """Test tool execution failure"""
        tool = mock_tools[2]  # This tool is configured to fail
        executor.register_tool(tool)
        
        result = await executor.execute_tool(
            tool_name="tool3",
            query="test query"
        )
        
        assert result.success is False
        assert "Tool tool3 failed as requested" in result.error

    @pytest.mark.asyncio
    async def test_execute_tool_with_timeout(self, executor, mock_tools):
        """Test tool execution with timeout"""
        tool = MockTool("slow_tool", execution_time=2.0)
        executor.register_tool(tool)
        
        # Execute with timeout shorter than execution time
        with pytest.raises(asyncio.TimeoutError):
            await executor.execute_tool(
                tool_name="slow_tool",
                query="test query",
                timeout=1.0
            )

    @pytest.mark.asyncio
    async def test_execute_tool_with_validation(self, executor, mock_tools):
        """Test tool execution with parameter validation"""
        tool = mock_tools[0]
        executor.register_tool(tool)
        
        # Execute without required parameter
        result = await executor.execute_tool(tool_name="tool1")
        
        assert result.success is False
        assert "validation" in result.error.lower()

    @pytest.mark.asyncio
    async def test_find_and_execute_tool(self, executor, mock_tools):
        """Test finding and executing tool based on text"""
        for tool in mock_tools:
            executor.register_tool(tool)
        
        result = await executor.find_and_execute(
            text="Please use tool1 to process this query",
            query="test query"
        )
        
        assert result.success is True
        assert result.tool_name == "tool1"

    @pytest.mark.asyncio
    async def test_find_and_execute_no_match(self, executor, mock_tools):
        """Test finding and executing tool with no match"""
        for tool in mock_tools:
            executor.register_tool(tool)
        
        result = await executor.find_and_execute(
            text="Please use nonexistent_tool to process this query",
            query="test query"
        )
        
        assert result.success is False
        assert "No suitable tool found" in result.error

    @pytest.mark.asyncio
    async def test_execute_multiple_tools(self, executor, mock_tools):
        """Test executing multiple tools"""
        for tool in mock_tools[:2]:  # Register only the first two (successful) tools
            executor.register_tool(tool)
        
        results = await executor.execute_multiple_tools(
            tool_names=["tool1", "tool2"],
            common_params={"query": "test query"}
        )
        
        assert len(results) == 2
        assert all(result.success for result in results)
        assert results[0].tool_name == "tool1"
        assert results[1].tool_name == "tool2"

    @pytest.mark.asyncio
    async def test_execute_multiple_tools_with_failure(self, executor, mock_tools):
        """Test executing multiple tools with one failure"""
        for tool in mock_tools:
            executor.register_tool(tool)
        
        results = await executor.execute_multiple_tools(
            tool_names=["tool1", "tool3", "tool2"],
            common_params={"query": "test query"}
        )
        
        assert len(results) == 3
        assert results[0].success is True  # tool1
        assert results[1].success is False  # tool3 (fails)
        assert results[2].success is True  # tool2

    @pytest.mark.asyncio
    async def test_execute_tools_in_parallel(self, executor, mock_tools):
        """Test executing tools in parallel"""
        slow_tool1 = MockTool("slow_tool1", execution_time=0.2)
        slow_tool2 = MockTool("slow_tool2", execution_time=0.2)
        
        executor.register_tool(slow_tool1)
        executor.register_tool(slow_tool2)
        
        start_time = time.time()
        
        results = await executor.execute_tools_in_parallel(
            tool_configs=[
                {"tool_name": "slow_tool1", "params": {"query": "test1"}},
                {"tool_name": "slow_tool2", "params": {"query": "test2"}}
            ]
        )
        
        execution_time = time.time() - start_time
        
        # Should execute in parallel, so time should be less than sum of individual times
        assert execution_time < 0.35  # Less than 0.2 + 0.2
        assert len(results) == 2
        assert all(result.success for result in results)

    @pytest.mark.asyncio
    async def test_execute_tools_with_dependencies(self, executor, mock_tools):
        """Test executing tools with dependencies"""
        for tool in mock_tools[:2]:
            executor.register_tool(tool)
        
        # Define dependencies: tool2 depends on tool1
        dependencies = {
            "tool2": ["tool1"]
        }
        
        results = await executor.execute_tools_with_dependencies(
            tool_configs=[
                {"tool_name": "tool1", "params": {"query": "test1"}},
                {"tool_name": "tool2", "params": {"query": "test2"}}
            ],
            dependencies=dependencies
        )
        
        assert len(results) == 2
        assert results[0].success is True  # tool1
        assert results[1].success is True  # tool2
        
        # Check execution order
        assert results[0].tool_name == "tool1"
        assert results[1].tool_name == "tool2"

    @pytest.mark.asyncio
    async def test_execute_tools_with_circular_dependencies(self, executor, mock_tools):
        """Test executing tools with circular dependencies"""
        for tool in mock_tools[:2]:
            executor.register_tool(tool)
        
        # Define circular dependencies
        dependencies = {
            "tool1": ["tool2"],
            "tool2": ["tool1"]
        }
        
        with pytest.raises(ValueError, match="Circular dependency detected"):
            await executor.execute_tools_with_dependencies(
                tool_configs=[
                    {"tool_name": "tool1", "params": {"query": "test1"}},
                    {"tool_name": "tool2", "params": {"query": "test2"}}
                ],
                dependencies=dependencies
            )

    @pytest.mark.asyncio
    async def test_get_execution_stats(self, executor, mock_tools):
        """Test getting execution statistics"""
        # Initially should have no stats
        stats = executor.get_execution_stats()
        assert stats["total_executions"] == 0
        assert stats["successful_executions"] == 0
        assert stats["failed_executions"] == 0
        
        # Register and execute tools
        for tool in mock_tools:
            executor.register_tool(tool)
        
        await executor.execute_tool(tool_name="tool1", query="test1")
        await executor.execute_tool(tool_name="tool3", query="test3")  # This will fail
        
        stats = executor.get_execution_stats()
        assert stats["total_executions"] == 2
        assert stats["successful_executions"] == 1
        assert stats["failed_executions"] == 1
        assert "tool1" in stats["tool_stats"]
        assert "tool3" in stats["tool_stats"]

    @pytest.mark.asyncio
    async def test_reset_stats(self, executor, mock_tools):
        """Test resetting execution statistics"""
        # Register and execute tools
        for tool in mock_tools:
            executor.register_tool(tool)
        
        await executor.execute_tool(tool_name="tool1", query="test1")
        
        # Check stats exist
        stats = executor.get_execution_stats()
        assert stats["total_executions"] == 1
        
        # Reset stats
        executor.reset_stats()
        
        # Check stats are reset
        stats = executor.get_execution_stats()
        assert stats["total_executions"] == 0
        assert stats["successful_executions"] == 0
        assert stats["failed_executions"] == 0

    def test_should_use_with_executor_keywords(self, executor):
        """Test should_use method with executor keywords"""
        assert executor.should_use("execute tool1") is True
        assert executor.should_use("run tool2") is True
        assert executor.should_use("use tool3") is True

    def test_should_use_without_executor_keywords(self, executor):
        """Test should_use method without executor keywords"""
        assert executor.should_use("create a file") is False
        assert executor.should_use("calculate something") is False

    def test_validate_parameters_valid(self, executor):
        """Test validate_parameters with valid parameters"""
        assert executor.validate_parameters(tool_name="test_tool") is True
        assert executor.validate_parameters(
            tool_name="test_tool",
            query="test query"
        ) is True

    def test_validate_parameters_missing_required(self, executor):
        """Test validate_parameters with missing required parameter"""
        assert executor.validate_parameters() is False

    @pytest.mark.asyncio
    async def test_execute_with_timeout_wrapper(self, executor, mock_tools):
        """Test execute_with_timeout method"""
        tool = mock_tools[0]
        executor.register_tool(tool)
        
        result = await executor.execute_with_timeout(
            tool_name="tool1",
            query="test query",
            timeout=5.0
        )
        
        assert result.success is True
        assert result.data["success"] is True

    @pytest.mark.asyncio
    async def test_get_tool_usage_stats(self, executor, mock_tools):
        """Test getting tool usage statistics"""
        for tool in mock_tools:
            executor.register_tool(tool)
        
        # Execute tools multiple times
        await executor.execute_tool(tool_name="tool1", query="test1")
        await executor.execute_tool(tool_name="tool1", query="test2")
        await executor.execute_tool(tool_name="tool2", query="test3")
        
        stats = executor.get_tool_usage_stats()
        
        assert stats["tool1"]["execution_count"] == 2
        assert stats["tool2"]["execution_count"] == 1
        assert stats["tool1"]["total_execution_time"] > 0
        assert stats["tool2"]["total_execution_time"] > 0

    def test_tool_category_and_version(self, executor):
        """Test executor category and version"""
        assert executor.category == "execution"
        assert executor.version == "1.0.0"

    def test_tool_auth_requirements(self, executor):
        """Test executor authentication requirements"""
        assert executor.requires_api_key is False
        assert executor.requires_auth is False

    @pytest.mark.asyncio
    async def test_execute_tool_with_context(self, executor, mock_tools):
        """Test tool execution with context"""
        tool = mock_tools[0]
        executor.register_tool(tool)
        
        context = {
            "user_id": "test_user",
            "session_id": "test_session",
            "available_tools": ["tool1", "tool2"]
        }
        
        result = await executor.execute_tool(
            tool_name="tool1",
            query="test query",
            context=context
        )
        
        assert result.success is True
        assert result.context == context

    @pytest.mark.asyncio
    async def test_execute_tool_with_retry(self, executor, mock_tools):
        """Test tool execution with retry logic"""
        # Create a tool that fails once then succeeds
        class RetryTool(MockTool):
            def __init__(self):
                super().__init__("retry_tool")
                self._call_count = 0
            
            async def execute(self, **kwargs):
                self._call_count += 1
                if self._call_count == 1:
                    raise Exception("Temporary failure")
                return await super().execute(**kwargs)
        
        retry_tool = RetryTool()
        executor.register_tool(retry_tool)
        
        result = await executor.execute_tool(
            tool_name="retry_tool",
            query="test query",
            max_retries=1
        )
        
        assert result.success is True
        assert retry_tool._call_count == 2  # Should have been called twice

    @pytest.mark.asyncio
    async def test_execute_tool_with_retry_exhausted(self, executor, mock_tools):
        """Test tool execution with exhausted retries"""
        tool = mock_tools[2]  # This tool always fails
        executor.register_tool(tool)
        
        result = await executor.execute_tool(
            tool_name="tool3",
            query="test query",
            max_retries=2
        )
        
        assert result.success is False
        assert "Tool tool3 failed as requested" in result.error

    @pytest.mark.asyncio
    async def test_execute_tool_chain(self, executor, mock_tools):
        """Test executing a chain of tools"""
        for tool in mock_tools[:2]:
            executor.register_tool(tool)
        
        # Define a tool chain where output of tool1 is input to tool2
        tool_chain = [
            {"tool_name": "tool1", "params": {"query": "initial query"}},
            {"tool_name": "tool2", "params": {"query": "result_from_tool1"}}
        ]
        
        # Mock the chain execution
        with patch.object(executor, '_execute_chain_step') as mock_chain_step:
            mock_chain_step.side_effect = [
                ToolResult(success=True, data={"result": "tool1 result"}),
                ToolResult(success=True, data={"result": "tool2 result"})
            ]
            
            results = await executor.execute_tool_chain(tool_chain)
            
            assert len(results) == 2
            assert all(result.success for result in results)
            assert mock_chain_step.call_count == 2