"""
Unit tests for Tool Registry.

This module tests the tool registry system for managing tool registration and discovery.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import uuid
from typing import List, Dict, Any, Optional, Tuple
import json
import time

from app.core.tools.execution.registry import ToolRegistry
from app.core.tools.base.base import BaseTool, ToolResult


class MockTool(BaseTool):
    """Mock tool for testing"""
    
    def __init__(self, name="mock_tool", category="test", tags=None, version="1.0.0"):
        super().__init__()
        self.name = name
        self.description = f"Mock tool {name}"
        self.keywords = ["mock", "test", name]
        self.category = category
        self.version = version
        self.tags = tags or []
        self.requires_api_key = False
        self.requires_auth = False
    
    def get_parameters(self):
        return {
            "query": {
                "type": "string",
                "required": True,
                "description": "Test query parameter"
            }
        }
    
    async def execute(self, **kwargs):
        return ToolResult(
            success=True,
            data={"result": f"Mock result for {self.name}", "query": kwargs.get("query")},
            message=f"Successfully executed {self.name}",
            execution_time=0.1
        )
    
    def validate_parameters(self, **kwargs):
        return "query" in kwargs and kwargs["query"] is not None
    
    def should_use(self, text, context=None):
        return self.name in text.lower()


class TestToolRegistry:
    """Test ToolRegistry class"""

    @pytest.fixture
    def registry(self):
        """Create a ToolRegistry instance"""
        return ToolRegistry()

    @pytest.fixture
    def mock_tools(self):
        """Create mock tools for testing"""
        return [
            MockTool("tool1", "web", ["search", "scraping"]),
            MockTool("tool2", "data", ["analysis", "processing"]),
            MockTool("tool3", "ai", ["generation", "nlp"]),
            MockTool("tool4", "web", ["scraping"], "2.0.0"),
            MockTool("tool5", "utility", ["helper"])
        ]

    def test_registry_init(self, registry):
        """Test ToolRegistry initialization"""
        assert registry.name == "tool_registry"
        assert registry.description == "Registry for managing tool registration and discovery"
        assert isinstance(registry.tools, dict)
        assert isinstance(registry.categories, dict)
        assert isinstance(registry.tags, dict)
        assert len(registry.tools) == 0
        assert len(registry.categories) == 0
        assert len(registry.tags) == 0

    def test_register_tool(self, registry, mock_tools):
        """Test tool registration"""
        tool = mock_tools[0]
        
        registry.register_tool(tool)
        
        assert tool.name in registry.tools
        assert registry.tools[tool.name] == tool
        assert tool.category in registry.categories
        assert tool.name in registry.categories[tool.category]
        
        # Check tags
        for tag in tool.tags:
            assert tag in registry.tags
            assert tool.name in registry.tags[tag]

    def test_register_tool_with_duplicate(self, registry, mock_tools):
        """Test tool registration with duplicate name"""
        tool1 = mock_tools[0]
        tool2 = MockTool("tool1")  # Same name
        
        registry.register_tool(tool1)
        
        with pytest.raises(ValueError, match="Tool tool1 is already registered"):
            registry.register_tool(tool2)

    def test_unregister_tool(self, registry, mock_tools):
        """Test tool unregistration"""
        tool = mock_tools[0]
        
        registry.register_tool(tool)
        assert tool.name in registry.tools
        
        registry.unregister_tool(tool.name)
        assert tool.name not in registry.tools
        assert tool.name not in registry.categories[tool.category]
        
        # Check tags are removed
        for tag in tool.tags:
            if tag in registry.tags and tool.name in registry.tags[tag]:
                assert False, f"Tool {tool.name} still in tag {tag}"

    def test_unregister_nonexistent_tool(self, registry):
        """Test unregistering non-existent tool"""
        with pytest.raises(ValueError, match="Tool nonexistent_tool is not registered"):
            registry.unregister_tool("nonexistent_tool")

    def test_list_tools(self, registry, mock_tools):
        """Test listing all registered tools"""
        # Initially empty
        assert registry.list_tools() == []
        
        # Register tools
        for tool in mock_tools:
            registry.register_tool(tool)
        
        tools = registry.list_tools()
        assert len(tools) == 5
        assert all(tool.name in tools for tool in mock_tools)

    def test_list_tools_by_category(self, registry, mock_tools):
        """Test listing tools by category"""
        # Register tools
        for tool in mock_tools:
            registry.register_tool(tool)
        
        web_tools = registry.list_tools_by_category("web")
        assert len(web_tools) == 2
        assert "tool1" in web_tools
        assert "tool4" in web_tools
        
        data_tools = registry.list_tools_by_category("data")
        assert len(data_tools) == 1
        assert "tool2" in data_tools

    def test_list_tools_by_tag(self, registry, mock_tools):
        """Test listing tools by tag"""
        # Register tools
        for tool in mock_tools:
            registry.register_tool(tool)
        
        scraping_tools = registry.list_tools_by_tag("scraping")
        assert len(scraping_tools) == 2
        assert "tool1" in scraping_tools
        assert "tool4" in scraping_tools
        
        search_tools = registry.list_tools_by_tag("search")
        assert len(search_tools) == 1
        assert "tool1" in search_tools

    def test_get_tool(self, registry, mock_tools):
        """Test getting a specific tool"""
        tool = mock_tools[0]
        registry.register_tool(tool)
        
        retrieved_tool = registry.get_tool("tool1")
        assert retrieved_tool == tool

    def test_get_nonexistent_tool(self, registry):
        """Test getting non-existent tool"""
        assert registry.get_tool("nonexistent_tool") is None

    def test_find_tools_by_keyword(self, registry, mock_tools):
        """Test finding tools by keyword"""
        # Register tools
        for tool in mock_tools:
            registry.register_tool(tool)
        
        search_tools = registry.find_tools_by_keyword("search")
        assert len(search_tools) == 1
        assert search_tools[0].name == "tool1"
        
        scraping_tools = registry.find_tools_by_keyword("scraping")
        assert len(scraping_tools) == 2
        assert all(tool.name in ["tool1", "tool4"] for tool in scraping_tools)

    def test_find_tools_by_text(self, registry, mock_tools):
        """Test finding tools by text match"""
        # Register tools
        for tool in mock_tools:
            registry.register_tool(tool)
        
        matching_tools = registry.find_tools_by_text("use tool1 for searching")
        assert len(matching_tools) == 1
        assert matching_tools[0].name == "tool1"
        
        matching_tools = registry.find_tools_by_text("scrape web content")
        assert len(matching_tools) == 2
        assert all(tool.name in ["tool1", "tool4"] for tool in matching_tools)

    def test_get_categories(self, registry, mock_tools):
        """Test getting all categories"""
        # Initially empty
        assert registry.get_categories() == []
        
        # Register tools
        for tool in mock_tools:
            registry.register_tool(tool)
        
        categories = registry.get_categories()
        assert len(categories) == 4
        assert "web" in categories
        assert "data" in categories
        assert "ai" in categories
        assert "utility" in categories

    def test_get_tags(self, registry, mock_tools):
        """Test getting all tags"""
        # Initially empty
        assert registry.get_tags() == []
        
        # Register tools
        for tool in mock_tools:
            registry.register_tool(tool)
        
        tags = registry.get_tags()
        assert len(tags) == 6
        assert "search" in tags
        assert "scraping" in tags
        assert "analysis" in tags
        assert "processing" in tags
        assert "generation" in tags
        assert "nlp" in tags

    def test_update_tool_metadata(self, registry, mock_tools):
        """Test updating tool metadata"""
        tool = mock_tools[0]
        registry.register_tool(tool)
        
        # Update tags
        new_tags = ["search", "scraping", "web", "updated"]
        registry.update_tool_metadata("tool1", tags=new_tags)
        
        # Check tags were updated
        assert "updated" in registry.tags
        assert "tool1" in registry.tags["updated"]
        
        # Check old tags are still there
        assert "tool1" in registry.tags["search"]
        assert "tool1" in registry.tags["scraping"]

    def test_update_tool_metadata_nonexistent(self, registry):
        """Test updating metadata for non-existent tool"""
        with pytest.raises(ValueError, match="Tool nonexistent_tool is not registered"):
            registry.update_tool_metadata("nonexistent_tool", tags=["new"])

    def test_get_tool_info(self, registry, mock_tools):
        """Test getting tool information"""
        tool = mock_tools[0]
        registry.register_tool(tool)
        
        info = registry.get_tool_info("tool1")
        
        assert info["name"] == "tool1"
        assert info["description"] == "Mock tool tool1"
        assert info["category"] == "web"
        assert info["version"] == "1.0.0"
        assert info["tags"] == ["search", "scraping"]
        assert info["keywords"] == ["mock", "test", "tool1"]
        assert info["requires_api_key"] is False
        assert info["requires_auth"] is False

    def test_get_tool_info_nonexistent(self, registry):
        """Test getting info for non-existent tool"""
        assert registry.get_tool_info("nonexistent_tool") is None

    def test_search_tools(self, registry, mock_tools):
        """Test searching tools with multiple criteria"""
        # Register tools
        for tool in mock_tools:
            registry.register_tool(tool)
        
        # Search by category
        results = registry.search_tools(category="web")
        assert len(results) == 2
        assert all(tool.category == "web" for tool in results)
        
        # Search by tag
        results = registry.search_tools(tags=["scraping"])
        assert len(results) == 2
        assert all("scraping" in tool.tags for tool in results)
        
        # Search by keyword
        results = registry.search_tools(keyword="search")
        assert len(results) == 1
        assert results[0].name == "tool1"
        
        # Search by multiple criteria
        results = registry.search_tools(category="web", tags=["scraping"])
        assert len(results) == 2
        assert all(tool.category == "web" and "scraping" in tool.tags for tool in results)

    def test_get_tool_dependencies(self, registry, mock_tools):
        """Test getting tool dependencies"""
        tool = mock_tools[0]
        registry.register_tool(tool)
        
        dependencies = registry.get_tool_dependencies("tool1")
        assert isinstance(dependencies, list)
        # MockTool doesn't have dependencies by default

    def test_register_tool_with_dependencies(self, registry):
        """Test registering tool with dependencies"""
        # Create tools with dependencies
        base_tool = MockTool("base_tool")
        dependent_tool = MockTool("dependent_tool")
        
        # Manually add dependencies
        dependent_tool.dependencies = ["base_tool"]
        
        registry.register_tool(base_tool)
        registry.register_tool(dependent_tool)
        
        dependencies = registry.get_tool_dependencies("dependent_tool")
        assert "base_tool" in dependencies

    def test_validate_tool_dependencies(self, registry):
        """Test validating tool dependencies"""
        # Create tools with dependencies
        base_tool = MockTool("base_tool")
        dependent_tool = MockTool("dependent_tool")
        
        # Manually add dependencies
        dependent_tool.dependencies = ["base_tool"]
        
        # Register only dependent tool (missing dependency)
        registry.register_tool(dependent_tool)
        
        # Should detect missing dependency
        with pytest.raises(ValueError, match="Dependency base_tool not found"):
            registry.validate_tool_dependencies("dependent_tool")

    def test_get_execution_order(self, registry):
        """Test getting execution order based on dependencies"""
        # Create tools with dependencies
        base_tool = MockTool("base_tool")
        middle_tool = MockTool("middle_tool")
        top_tool = MockTool("top_tool")
        
        # Set up dependencies: top -> middle -> base
        middle_tool.dependencies = ["base_tool"]
        top_tool.dependencies = ["middle_tool"]
        
        registry.register_tool(base_tool)
        registry.register_tool(middle_tool)
        registry.register_tool(top_tool)
        
        order = registry.get_execution_order(["top_tool", "middle_tool", "base_tool"])
        
        # Check order: base should be first, top should be last
        assert order[0] == "base_tool"
        assert order[1] == "middle_tool"
        assert order[2] == "top_tool"

    def test_get_execution_order_with_cycle(self, registry):
        """Test getting execution order with circular dependencies"""
        # Create tools with circular dependencies
        tool1 = MockTool("tool1")
        tool2 = MockTool("tool2")
        
        # Set up circular dependencies
        tool1.dependencies = ["tool2"]
        tool2.dependencies = ["tool1"]
        
        registry.register_tool(tool1)
        registry.register_tool(tool2)
        
        with pytest.raises(ValueError, match="Circular dependency detected"):
            registry.get_execution_order(["tool1", "tool2"])

    def test_enable_disable_tool(self, registry, mock_tools):
        """Test enabling and disabling tools"""
        tool = mock_tools[0]
        registry.register_tool(tool)
        
        # Tool should be enabled by default
        assert registry.is_tool_enabled("tool1") is True
        
        # Disable tool
        registry.disable_tool("tool1")
        assert registry.is_tool_enabled("tool1") is False
        
        # Enable tool
        registry.enable_tool("tool1")
        assert registry.is_tool_enabled("tool1") is True

    def test_enable_disable_nonexistent_tool(self, registry):
        """Test enabling/disabling non-existent tool"""
        with pytest.raises(ValueError, match="Tool nonexistent_tool is not registered"):
            registry.disable_tool("nonexistent_tool")
        
        with pytest.raises(ValueError, match="Tool nonexistent_tool is not registered"):
            registry.enable_tool("nonexistent_tool")

    def test_list_enabled_tools(self, registry, mock_tools):
        """Test listing enabled tools"""
        # Register tools
        for tool in mock_tools:
            registry.register_tool(tool)
        
        # All tools should be enabled by default
        enabled_tools = registry.list_enabled_tools()
        assert len(enabled_tools) == 5
        
        # Disable one tool
        registry.disable_tool("tool1")
        
        enabled_tools = registry.list_enabled_tools()
        assert len(enabled_tools) == 4
        assert "tool1" not in enabled_tools

    def test_get_registry_stats(self, registry, mock_tools):
        """Test getting registry statistics"""
        # Initially empty
        stats = registry.get_registry_stats()
        assert stats["total_tools"] == 0
        assert stats["total_categories"] == 0
        assert stats["total_tags"] == 0
        assert stats["enabled_tools"] == 0
        assert stats["disabled_tools"] == 0
        
        # Register tools
        for tool in mock_tools:
            registry.register_tool(tool)
        
        stats = registry.get_registry_stats()
        assert stats["total_tools"] == 5
        assert stats["total_categories"] == 4
        assert stats["total_tags"] == 6
        assert stats["enabled_tools"] == 5
        assert stats["disabled_tools"] == 0
        
        # Disable one tool
        registry.disable_tool("tool1")
        
        stats = registry.get_registry_stats()
        assert stats["enabled_tools"] == 4
        assert stats["disabled_tools"] == 1

    def test_export_registry(self, registry, mock_tools):
        """Test exporting registry configuration"""
        # Register tools
        for tool in mock_tools:
            registry.register_tool(tool)
        
        exported = registry.export_registry()
        
        assert "tools" in exported
        assert "categories" in exported
        assert "tags" in exported
        assert "metadata" in exported
        
        # Check tools are exported
        assert len(exported["tools"]) == 5
        assert all(tool.name in exported["tools"] for tool in mock_tools)
        
        # Check tool info
        tool1_info = exported["tools"]["tool1"]
        assert tool1_info["name"] == "tool1"
        assert tool1_info["category"] == "web"
        assert tool1_info["version"] == "1.0.0"

    def test_import_registry(self, registry, mock_tools):
        """Test importing registry configuration"""
        # Register tools
        for tool in mock_tools:
            registry.register_tool(tool)
        
        # Export registry
        exported = registry.export_registry()
        
        # Create new registry and import
        new_registry = ToolRegistry()
        new_registry.import_registry(exported)
        
        # Check tools were imported
        assert len(new_registry.tools) == 5
        assert all(tool.name in new_registry.tools for tool in mock_tools)
        
        # Check categories and tags
        assert len(new_registry.categories) == 4
        assert len(new_registry.tags) == 6

    def test_should_use_with_registry_keywords(self, registry):
        """Test should_use method with registry keywords"""
        assert registry.should_use("register tool1") is True
        assert registry.should_use("find tool for searching") is True
        assert registry.should_use("list all web tools") is True

    def test_should_use_without_registry_keywords(self, registry):
        """Test should_use method without registry keywords"""
        assert registry.should_use("create a file") is False
        assert registry.should_use("calculate something") is False

    def test_validate_parameters_valid(self, registry):
        """Test validate_parameters with valid parameters"""
        assert registry.validate_parameters(tool_name="test_tool") is True
        assert registry.validate_parameters(
            tool_name="test_tool",
            category="web"
        ) is True

    def test_validate_parameters_missing_required(self, registry):
        """Test validate_parameters with missing required parameter"""
        assert registry.validate_parameters() is False

    @pytest.mark.asyncio
    async def test_execute_with_timeout_wrapper(self, registry, mock_tools):
        """Test execute_with_timeout method"""
        tool = mock_tools[0]
        registry.register_tool(tool)
        
        result = await registry.execute_with_timeout(
            tool_name="tool1",
            query="test query",
            timeout=5.0
        )
        
        assert result.success is True
        assert result.data["success"] is True

    def test_tool_category_and_version(self, registry):
        """Test registry category and version"""
        assert registry.category == "registry"
        assert registry.version == "1.0.0"

    def test_tool_auth_requirements(self, registry):
        """Test registry authentication requirements"""
        assert registry.requires_api_key is False
        assert registry.requires_auth is False

    @pytest.mark.asyncio
    async def test_batch_register_tools(self, registry, mock_tools):
        """Test batch registration of tools"""
        await registry.batch_register_tools(mock_tools)
        
        assert len(registry.tools) == 5
        assert all(tool.name in registry.tools for tool in mock_tools)

    @pytest.mark.asyncio
    async def test_batch_unregister_tools(self, registry, mock_tools):
        """Test batch unregistration of tools"""
        # Register tools
        for tool in mock_tools:
            registry.register_tool(tool)
        
        # Batch unregister
        await registry.batch_unregister_tools(["tool1", "tool2", "tool3"])
        
        assert len(registry.tools) == 2
        assert "tool1" not in registry.tools
        assert "tool2" not in registry.tools
        assert "tool3" not in registry.tools
        assert "tool4" in registry.tools
        assert "tool5" in registry.tools

    def test_get_tools_by_version(self, registry, mock_tools):
        """Test getting tools by version"""
        # Register tools
        for tool in mock_tools:
            registry.register_tool(tool)
        
        # Get tools with version 1.0.0
        v1_tools = registry.get_tools_by_version("1.0.0")
        assert len(v1_tools) == 4
        assert "tool4" not in v1_tools  # tool4 is version 2.0.0
        
        # Get tools with version 2.0.0
        v2_tools = registry.get_tools_by_version("2.0.0")
        assert len(v2_tools) == 1
        assert "tool4" in v2_tools

    def test_get_tools_newer_than(self, registry, mock_tools):
        """Test getting tools newer than a specific version"""
        # Register tools
        for tool in mock_tools:
            registry.register_tool(tool)
        
        # Get tools newer than 1.0.0
        newer_tools = registry.get_tools_newer_than("1.0.0")
        assert len(newer_tools) == 1
        assert "tool4" in newer_tools

    def test_update_tool_version(self, registry, mock_tools):
        """Test updating tool version"""
        tool = mock_tools[0]
        registry.register_tool(tool)
        
        # Update version
        registry.update_tool_version("tool1", "1.1.0")
        
        # Check version was updated
        info = registry.get_tool_info("tool1")
        assert info["version"] == "1.1.0"