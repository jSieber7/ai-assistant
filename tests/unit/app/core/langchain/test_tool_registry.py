"""
Unit tests for LangChain Tool Registry.

This module tests the LangChain-based tool registry functionality,
including tool registration, discovery, and management.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import uuid
from typing import List, Dict, Any, Optional

from app.core.langchain.tool_registry import (
    LangChainToolRegistry,
    ToolType,
    ToolCategory,
    ToolConfig,
    ToolStats,
    tool_registry
)


class TestToolType:
    """Test ToolType enum"""

    def test_tool_type_values(self):
        """Test that ToolType has expected values"""
        expected_types = [
            "native",
            "custom",
            "langchain",
            "api",
            "function"
        ]
        
        actual_types = [tool_type.value for tool_type in ToolType]
        assert actual_types == expected_types


class TestToolCategory:
    """Test ToolCategory enum"""

    def test_tool_category_values(self):
        """Test that ToolCategory has expected values"""
        expected_categories = [
            "search",
            "calculation",
            "code",
            "data",
            "communication",
            "utility",
            "general"
        ]
        
        actual_categories = [category.value for category in ToolCategory]
        assert actual_categories == expected_categories


class TestToolConfig:
    """Test ToolConfig dataclass"""

    def test_tool_config_defaults(self):
        """Test ToolConfig default values"""
        config = ToolConfig(
            name="test_tool",
            description="A test tool",
            tool_type=ToolType.NATIVE
        )
        
        assert config.name == "test_tool"
        assert config.description == "A test tool"
        assert config.tool_type == ToolType.NATIVE
        assert config.categories == []
        assert config.enabled is True
        assert config.parameters == {}
        assert config.returns == {}
        assert config.examples == []
        assert config.dependencies == []
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.metadata == {}

    def test_tool_config_with_values(self):
        """Test ToolConfig with provided values"""
        categories = [ToolCategory.SEARCH, ToolCategory.DATA]
        parameters = {
            "query": {"type": "string", "required": True},
            "limit": {"type": "integer", "default": 10}
        }
        returns = {"results": {"type": "array"}}
        examples = [
            {"input": {"query": "test"}, "output": {"results": []}}
        ]
        dependencies = ["requests", "beautifulsoup4"]
        metadata = {"version": "1.0", "author": "test"}
        
        config = ToolConfig(
            name="advanced_tool",
            description="An advanced tool",
            tool_type=ToolType.CUSTOM,
            categories=categories,
            enabled=False,
            parameters=parameters,
            returns=returns,
            examples=examples,
            dependencies=dependencies,
            timeout=60,
            max_retries=5,
            metadata=metadata
        )
        
        assert config.name == "advanced_tool"
        assert config.description == "An advanced tool"
        assert config.tool_type == ToolType.CUSTOM
        assert config.categories == categories
        assert config.enabled is False
        assert config.parameters == parameters
        assert config.returns == returns
        assert config.examples == examples
        assert config.dependencies == dependencies
        assert config.timeout == 60
        assert config.max_retries == 5
        assert config.metadata == metadata


class TestToolStats:
    """Test ToolStats dataclass"""

    def test_tool_stats_defaults(self):
        """Test ToolStats default values"""
        stats = ToolStats(
            tool_name="test_tool",
            tool_type="native"
        )
        
        assert stats.tool_name == "test_tool"
        assert stats.tool_type == "native"
        assert stats.total_executions == 0
        assert stats.successful_executions == 0
        assert stats.failed_executions == 0
        assert stats.total_execution_time == 0.0
        assert stats.average_execution_time == 0.0
        assert stats.last_used is None
        assert stats.created_at is not None

    def test_tool_stats_with_values(self):
        """Test ToolStats with provided values"""
        created_at = datetime.now()
        last_used = datetime.now()
        
        stats = ToolStats(
            tool_name="advanced_tool",
            tool_type="custom",
            total_executions=10,
            successful_executions=9,
            failed_executions=1,
            total_execution_time=45.5,
            average_execution_time=4.55,
            last_used=last_used,
            created_at=created_at
        )
        
        assert stats.tool_name == "advanced_tool"
        assert stats.tool_type == "custom"
        assert stats.total_executions == 10
        assert stats.successful_executions == 9
        assert stats.failed_executions == 1
        assert stats.total_execution_time == 45.5
        assert stats.average_execution_time == 4.55
        assert stats.last_used == last_used
        assert stats.created_at == created_at


class TestLangChainToolRegistry:
    """Test LangChainToolRegistry class"""

    @pytest.fixture
    def tool_registry_instance(self):
        """Create a fresh tool registry instance for testing"""
        return LangChainToolRegistry()

    @pytest.fixture
    def mock_monitoring(self):
        """Mock monitoring system"""
        with patch('app.core.langchain.tool_registry.LangChainMonitoring') as mock:
            mock_instance = Mock()
            mock_instance.initialize = AsyncMock()
            mock_instance.track_tool_registration = AsyncMock()
            mock_instance.track_tool_execution = AsyncMock()
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def mock_langchain_tool(self):
        """Mock LangChain tool"""
        mock_tool = Mock()
        mock_tool.name = "mock_tool"
        mock_tool.description = "A mock tool"
        mock_tool.args = {"query": {"type": "string"}}
        mock_tool.ainvoke = AsyncMock(return_value={"result": "mock result"})
        return mock_tool

    @pytest.mark.asyncio
    async def test_initialize(self, tool_registry_instance, mock_monitoring):
        """Test tool registry initialization"""
        await tool_registry_instance.initialize()
        
        assert tool_registry_instance._initialized is True
        mock_monitoring.return_value.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, tool_registry_instance, mock_monitoring):
        """Test that initialize is idempotent"""
        await tool_registry_instance.initialize()
        await tool_registry_instance.initialize()
        
        assert tool_registry_instance._initialized is True
        # Should only initialize once
        mock_monitoring.return_value.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_tool(
        self, 
        tool_registry_instance, 
        mock_langchain_tool, 
        mock_monitoring
    ):
        """Test registering a tool"""
        await tool_registry_instance.initialize()
        
        config = ToolConfig(
            name="search_tool",
            description="Search the web",
            tool_type=ToolType.NATIVE,
            categories=[ToolCategory.SEARCH],
            parameters={
                "query": {"type": "string", "required": True},
                "limit": {"type": "integer", "default": 10}
            }
        )
        
        result = await tool_registry_instance.register_tool(
            tool=mock_langchain_tool,
            config=config
        )
        
        assert result is True
        assert "search_tool" in tool_registry_instance._tools
        assert "search_tool" in tool_registry_instance._tool_configs
        assert "search_tool" in tool_registry_instance._tool_stats
        
        # Verify monitoring was called
        mock_monitoring.return_value.track_tool_registration.assert_called_once()
        call_args = mock_monitoring.return_value.track_tool_registration.call_args[1]
        assert call_args["tool_name"] == "search_tool"
        assert call_args["tool_type"] == "native"
        assert call_args["success"] is True

    @pytest.mark.asyncio
    async def test_register_tool_duplicate(
        self, 
        tool_registry_instance, 
        mock_langchain_tool, 
        mock_monitoring
    ):
        """Test registering duplicate tool"""
        await tool_registry_instance.initialize()
        
        config = ToolConfig(
            name="duplicate_tool",
            description="A duplicate tool",
            tool_type=ToolType.NATIVE
        )
        
        # Register first time
        result1 = await tool_registry_instance.register_tool(
            tool=mock_langchain_tool,
            config=config
        )
        assert result1 is True
        
        # Register second time
        result2 = await tool_registry_instance.register_tool(
            tool=mock_langchain_tool,
            config=config
        )
        assert result2 is False
        
        # Verify monitoring was called for failure
        assert mock_monitoring.return_value.track_tool_registration.call_count == 2
        failure_call = mock_monitoring.return_value.track_tool_registration.call_args_list[1][1]
        assert failure_call["success"] is False
        assert "already registered" in failure_call["error"]

    @pytest.mark.asyncio
    async def test_register_langchain_tools(
        self, 
        tool_registry_instance, 
        mock_monitoring
    ):
        """Test registering LangChain tools"""
        await tool_registry_instance.initialize()
        
        # Mock LangChain tools
        mock_tools = [
            Mock(name="search_tool", description="Search tool"),
            Mock(name="calculator_tool", description="Calculator tool")
        ]
        
        # Mock the import function
        with patch('app.core.langchain.tool_registry.load_langchain_tools') as mock_load:
            mock_load.return_value = mock_tools
            
            result = await tool_registry_instance.register_langchain_tools()
            
            assert result is True
            assert len(tool_registry_instance._tools) == 2
            assert "search_tool" in tool_registry_instance._tools
            assert "calculator_tool" in tool_registry_instance._tools

    @pytest.mark.asyncio
    async def test_get_tool(self, tool_registry_instance):
        """Test getting a tool"""
        await tool_registry_instance.initialize()
        
        # Create mock tool
        mock_tool = Mock()
        tool_registry_instance._tools["test_tool"] = mock_tool
        
        # Get existing tool
        result = tool_registry_instance.get_tool("test_tool")
        assert result == mock_tool
        
        # Get non-existent tool
        result = tool_registry_instance.get_tool("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_tools(self, tool_registry_instance):
        """Test listing tools"""
        await tool_registry_instance.initialize()
        
        # Create mock tools and configs
        tools = {
            "search_tool": Mock(),
            "calculator_tool": Mock(),
            "disabled_tool": Mock()
        }
        
        configs = {
            "search_tool": ToolConfig(
                name="search_tool",
                description="Search tool",
                tool_type=ToolType.NATIVE,
                categories=[ToolCategory.SEARCH],
                enabled=True
            ),
            "calculator_tool": ToolConfig(
                name="calculator_tool",
                description="Calculator tool",
                tool_type=ToolType.NATIVE,
                categories=[ToolCategory.CALCULATION],
                enabled=True
            ),
            "disabled_tool": ToolConfig(
                name="disabled_tool",
                description="Disabled tool",
                tool_type=ToolType.NATIVE,
                enabled=False
            )
        }
        
        tool_registry_instance._tools = tools
        tool_registry_instance._tool_configs = configs
        
        # List all tools
        result = await tool_registry_instance.list_tools()
        assert len(result) == 3
        
        # List enabled tools only
        result = await tool_registry_instance.list_tools(enabled_only=True)
        assert len(result) == 2
        tool_names = [tool["name"] for tool in result]
        assert "search_tool" in tool_names
        assert "calculator_tool" in tool_names
        assert "disabled_tool" not in tool_names

    @pytest.mark.asyncio
    async def test_list_tools_by_category(self, tool_registry_instance):
        """Test listing tools by category"""
        await tool_registry_instance.initialize()
        
        # Create mock tools and configs
        tools = {
            "search_tool": Mock(),
            "web_search_tool": Mock(),
            "calculator_tool": Mock(),
            "math_tool": Mock()
        }
        
        configs = {
            "search_tool": ToolConfig(
                name="search_tool",
                description="Search tool",
                tool_type=ToolType.NATIVE,
                categories=[ToolCategory.SEARCH],
                enabled=True
            ),
            "web_search_tool": ToolConfig(
                name="web_search_tool",
                description="Web search tool",
                tool_type=ToolType.NATIVE,
                categories=[ToolCategory.SEARCH],
                enabled=True
            ),
            "calculator_tool": ToolConfig(
                name="calculator_tool",
                description="Calculator tool",
                tool_type=ToolType.NATIVE,
                categories=[ToolCategory.CALCULATION],
                enabled=True
            ),
            "math_tool": ToolConfig(
                name="math_tool",
                description="Math tool",
                tool_type=ToolType.NATIVE,
                categories=[ToolCategory.CALCULATION, ToolCategory.DATA],
                enabled=True
            )
        }
        
        tool_registry_instance._tools = tools
        tool_registry_instance._tool_configs = configs
        
        # List search tools
        result = await tool_registry_instance.list_tools(category="search")
        assert len(result) == 2
        tool_names = [tool["name"] for tool in result]
        assert "search_tool" in tool_names
        assert "web_search_tool" in tool_names
        
        # List calculation tools
        result = await tool_registry_instance.list_tools(category="calculation")
        assert len(result) == 2
        tool_names = [tool["name"] for tool in result]
        assert "calculator_tool" in tool_names
        assert "math_tool" in tool_names

    @pytest.mark.asyncio
    async def test_find_relevant_tools(
        self, 
        tool_registry_instance, 
        mock_monitoring
    ):
        """Test finding relevant tools for a query"""
        await tool_registry_instance.initialize()
        
        # Create mock tools and configs
        tools = {
            "search_tool": Mock(),
            "calculator_tool": Mock(),
            "code_tool": Mock()
        }
        
        configs = {
            "search_tool": ToolConfig(
                name="search_tool",
                description="Search the web for information",
                tool_type=ToolType.NATIVE,
                categories=[ToolCategory.SEARCH],
                enabled=True
            ),
            "calculator_tool": ToolConfig(
                name="calculator_tool",
                description="Perform mathematical calculations",
                tool_type=ToolType.NATIVE,
                categories=[ToolCategory.CALCULATION],
                enabled=True
            ),
            "code_tool": ToolConfig(
                name="code_tool",
                description="Execute code and programming tasks",
                tool_type=ToolType.NATIVE,
                categories=[ToolCategory.CODE],
                enabled=True
            )
        }
        
        tool_registry_instance._tools = tools
        tool_registry_instance._tool_configs = configs
        
        # Find relevant tools for search query
        result = await tool_registry_instance.find_relevant_tools(
            query="search for information about machine learning",
            max_results=2
        )
        
        assert len(result) == 2
        assert result[0].name == "search_tool"  # Most relevant
        
        # Find relevant tools for math query
        result = await tool_registry_instance.find_relevant_tools(
            query="calculate 2 + 2",
            max_results=2
        )
        
        assert len(result) == 2
        assert result[0].name == "calculator_tool"  # Most relevant

    @pytest.mark.asyncio
    async def test_execute_tool(
        self, 
        tool_registry_instance, 
        mock_langchain_tool, 
        mock_monitoring
    ):
        """Test executing a tool"""
        await tool_registry_instance.initialize()
        
        # Register tool
        config = ToolConfig(
            name="search_tool",
            description="Search tool",
            tool_type=ToolType.NATIVE,
            categories=[ToolCategory.SEARCH]
        )
        
        await tool_registry_instance.register_tool(
            tool=mock_langchain_tool,
            config=config
        )
        
        # Execute tool
        result = await tool_registry_instance.execute_tool(
            tool_name="search_tool",
            input_data={"query": "test query"}
        )
        
        assert result["success"] is True
        assert "result" in result
        assert result["tool_name"] == "search_tool"
        assert result["execution_time"] > 0
        
        # Verify monitoring was called
        mock_monitoring.return_value.track_tool_execution.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(
        self, 
        tool_registry_instance, 
        mock_monitoring
    ):
        """Test executing non-existent tool"""
        await tool_registry_instance.initialize()
        
        result = await tool_registry_instance.execute_tool(
            tool_name="nonexistent_tool",
            input_data={"query": "test query"}
        )
        
        assert result["success"] is False
        assert "not found" in result["error"]
        assert result["tool_name"] == "nonexistent_tool"
        
        # Verify monitoring was called for failure
        mock_monitoring.return_value.track_tool_execution.assert_called_once()
        call_args = mock_monitoring.return_value.track_tool_execution.call_args[1]
        assert call_args["success"] is False
        assert "not found" in call_args["error"]

    @pytest.mark.asyncio
    async def test_execute_tool_failure(
        self, 
        tool_registry_instance, 
        mock_monitoring
    ):
        """Test executing tool that fails"""
        await tool_registry_instance.initialize()
        
        # Create mock tool that fails
        mock_tool = Mock()
        mock_tool.name = "failing_tool"
        mock_tool.ainvoke = AsyncMock(side_effect=Exception("Tool execution failed"))
        
        config = ToolConfig(
            name="failing_tool",
            description="Failing tool",
            tool_type=ToolType.NATIVE
        )
        
        await tool_registry_instance.register_tool(
            tool=mock_tool,
            config=config
        )
        
        result = await tool_registry_instance.execute_tool(
            tool_name="failing_tool",
            input_data={"query": "test query"}
        )
        
        assert result["success"] is False
        assert "Tool execution failed" in result["error"]
        assert result["tool_name"] == "failing_tool"

    @pytest.mark.asyncio
    async def test_enable_disable_tool(self, tool_registry_instance):
        """Test enabling and disabling tools"""
        await tool_registry_instance.initialize()
        
        # Register tool
        mock_tool = Mock()
        config = ToolConfig(
            name="test_tool",
            description="Test tool",
            tool_type=ToolType.NATIVE,
            enabled=True
        )
        
        await tool_registry_instance.register_tool(tool=mock_tool, config=config)
        
        # Disable tool
        result = await tool_registry_instance.disable_tool("test_tool")
        assert result is True
        assert tool_registry_instance._tool_configs["test_tool"].enabled is False
        
        # Enable tool
        result = await tool_registry_instance.enable_tool("test_tool")
        assert result is True
        assert tool_registry_instance._tool_configs["test_tool"].enabled is True

    @pytest.mark.asyncio
    async def test_unregister_tool(
        self, 
        tool_registry_instance, 
        mock_monitoring
    ):
        """Test unregistering a tool"""
        await tool_registry_instance.initialize()
        
        # Register tool
        mock_tool = Mock()
        config = ToolConfig(
            name="test_tool",
            description="Test tool",
            tool_type=ToolType.NATIVE
        )
        
        await tool_registry_instance.register_tool(tool=mock_tool, config=config)
        
        # Unregister tool
        result = await tool_registry_instance.unregister_tool("test_tool")
        
        assert result is True
        assert "test_tool" not in tool_registry_instance._tools
        assert "test_tool" not in tool_registry_instance._tool_configs
        assert "test_tool" not in tool_registry_instance._tool_stats

    @pytest.mark.asyncio
    async def test_get_tool_config(self, tool_registry_instance):
        """Test getting tool configuration"""
        await tool_registry_instance.initialize()
        
        # Create mock config
        config = ToolConfig(
            name="test_tool",
            description="Test tool",
            tool_type=ToolType.NATIVE,
            categories=[ToolCategory.SEARCH]
        )
        
        tool_registry_instance._tool_configs["test_tool"] = config
        
        # Get existing config
        result = tool_registry_instance.get_tool_config("test_tool")
        assert result == config
        
        # Get non-existent config
        result = tool_registry_instance.get_tool_config("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_tool_stats(self, tool_registry_instance):
        """Test getting tool statistics"""
        await tool_registry_instance.initialize()
        
        # Create mock stats
        stats = ToolStats(
            tool_name="test_tool",
            tool_type="native",
            total_executions=10,
            successful_executions=9,
            failed_executions=1,
            total_execution_time=45.5,
            average_execution_time=4.55
        )
        
        tool_registry_instance._tool_stats["test_tool"] = stats
        
        result = tool_registry_instance.get_tool_stats("test_tool")
        
        assert result is not None
        assert result["tool_name"] == "test_tool"
        assert result["tool_type"] == "native"
        assert result["total_executions"] == 10
        assert result["successful_executions"] == 9
        assert result["failed_executions"] == 1
        assert result["total_execution_time"] == 45.5
        assert result["average_execution_time"] == 4.55

    @pytest.mark.asyncio
    async def test_get_registry_stats(self, tool_registry_instance):
        """Test getting registry statistics"""
        await tool_registry_instance.initialize()
        
        # Create mock tools, configs, and stats
        tools = {
            "search_tool": Mock(),
            "calculator_tool": Mock(),
            "code_tool": Mock()
        }
        
        configs = {
            "search_tool": ToolConfig(
                name="search_tool",
                description="Search tool",
                tool_type=ToolType.NATIVE,
                categories=[ToolCategory.SEARCH],
                enabled=True
            ),
            "calculator_tool": ToolConfig(
                name="calculator_tool",
                description="Calculator tool",
                tool_type=ToolType.CUSTOM,
                categories=[ToolCategory.CALCULATION],
                enabled=True
            ),
            "code_tool": ToolConfig(
                name="code_tool",
                description="Code tool",
                tool_type=ToolType.LANGCHAIN,
                categories=[ToolCategory.CODE],
                enabled=False  # Disabled tool
            )
        }
        
        stats = {
            "search_tool": ToolStats(
                tool_name="search_tool",
                tool_type="native",
                total_executions=10,
                successful_executions=9,
                failed_executions=1
            ),
            "calculator_tool": ToolStats(
                tool_name="calculator_tool",
                tool_type="custom",
                total_executions=5,
                successful_executions=5,
                failed_executions=0
            ),
            "code_tool": ToolStats(
                tool_name="code_tool",
                tool_type="langchain",
                total_executions=0,
                successful_executions=0,
                failed_executions=0
            )
        }
        
        tool_registry_instance._tools = tools
        tool_registry_instance._tool_configs = configs
        tool_registry_instance._tool_stats = stats
        
        result = await tool_registry_instance.get_registry_stats()
        
        assert result["total_tools"] == 3
        assert result["enabled_tools"] == 2
        assert result["disabled_tools"] == 1
        assert len(result["tool_types"]) == 3
        assert result["tool_types"]["native"] == 1
        assert result["tool_types"]["custom"] == 1
        assert result["tool_types"]["langchain"] == 1
        assert len(result["categories"]) == 3
        assert result["categories"]["search"] == 1
        assert result["categories"]["calculation"] == 1
        assert result["categories"]["code"] == 1
        assert result["total_executions"] == 15
        assert result["successful_executions"] == 14
        assert result["failed_executions"] == 1

    @pytest.mark.asyncio
    async def test_reset_stats(self, tool_registry_instance):
        """Test resetting tool statistics"""
        await tool_registry_instance.initialize()
        
        # Create mock stats
        stats = ToolStats(
            tool_name="test_tool",
            tool_type="native",
            total_executions=10,
            successful_executions=9,
            failed_executions=1,
            total_execution_time=45.5,
            average_execution_time=4.55
        )
        
        tool_registry_instance._tool_stats["test_tool"] = stats
        
        # Reset stats
        result = await tool_registry_instance.reset_stats("test_tool")
        
        assert result is True
        reset_stats = tool_registry_instance._tool_stats["test_tool"]
        assert reset_stats.total_executions == 0
        assert reset_stats.successful_executions == 0
        assert reset_stats.failed_executions == 0
        assert reset_stats.total_execution_time == 0.0
        assert reset_stats.average_execution_time == 0.0

    @pytest.mark.asyncio
    async def test_reset_all_stats(self, tool_registry_instance):
        """Test resetting all statistics"""
        await tool_registry_instance.initialize()
        
        # Create mock stats
        stats = {
            "tool1": ToolStats(
                tool_name="tool1",
                tool_type="native",
                total_executions=10,
                successful_executions=9,
                failed_executions=1
            ),
            "tool2": ToolStats(
                tool_name="tool2",
                tool_type="custom",
                total_executions=5,
                successful_executions=5,
                failed_executions=0
            )
        }
        
        tool_registry_instance._tool_stats = stats
        
        # Reset all stats
        await tool_registry_instance.reset_all_stats()
        
        # Check all stats are reset
        for key, stats in tool_registry_instance._tool_stats.items():
            assert stats.total_executions == 0
            assert stats.successful_executions == 0
            assert stats.failed_executions == 0
            assert stats.total_execution_time == 0.0
            assert stats.average_execution_time == 0.0

    def test_calculate_relevance_score(self, tool_registry_instance):
        """Test calculating relevance score for tools"""
        config = ToolConfig(
            name="search_tool",
            description="Search the web for information",
            tool_type=ToolType.NATIVE,
            categories=[ToolCategory.SEARCH]
        )
        
        # High relevance query
        score = tool_registry_instance._calculate_relevance_score(
            config, "search for information"
        )
        assert score > 0.5  # Should be high relevance
        
        # Low relevance query
        score = tool_registry_instance._calculate_relevance_score(
            config, "calculate 2 + 2"
        )
        assert score < 0.3  # Should be low relevance

    @pytest.mark.asyncio
    async def test_track_execution(
        self, 
        tool_registry_instance, 
        mock_monitoring
    ):
        """Test tracking tool execution"""
        await tool_registry_instance.initialize()
        
        # Create mock stats
        stats = ToolStats(
            tool_name="test_tool",
            tool_type="native"
        )
        
        tool_registry_instance._tool_stats["test_tool"] = stats
        
        # Track successful execution
        await tool_registry_instance._track_execution(
            tool_name="test_tool",
            success=True,
            execution_time=2.5,
            input_data={"query": "test"},
            output_data={"result": "success"}
        )
        
        updated_stats = tool_registry_instance._tool_stats["test_tool"]
        assert updated_stats.total_executions == 1
        assert updated_stats.successful_executions == 1
        assert updated_stats.failed_executions == 0
        assert updated_stats.total_execution_time == 2.5
        assert updated_stats.average_execution_time == 2.5
        assert updated_stats.last_used is not None
        
        # Verify monitoring was called
        mock_monitoring.return_value.track_tool_execution.assert_called_once()

    def test_validate_tool_config(self, tool_registry_instance):
        """Test validating tool configuration"""
        # Valid config
        valid_config = ToolConfig(
            name="valid_tool",
            description="A valid tool",
            tool_type=ToolType.NATIVE
        )
        
        result = tool_registry_instance._validate_tool_config(valid_config)
        assert result is True
        
        # Invalid config (missing name)
        invalid_config = ToolConfig(
            name="",
            description="Invalid tool",
            tool_type=ToolType.NATIVE
        )
        
        result = tool_registry_instance._validate_tool_config(invalid_config)
        assert result is False


class TestGlobalToolRegistry:
    """Test global tool registry instance"""

    def test_global_instance(self):
        """Test that global tool registry instance exists"""
        from app.core.langchain.tool_registry import tool_registry
        assert tool_registry is not None
        assert isinstance(tool_registry, LangChainToolRegistry)