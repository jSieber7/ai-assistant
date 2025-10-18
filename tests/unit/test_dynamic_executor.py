"""
Unit tests for the Dynamic Tool Execution Manager.
"""

import pytest
import asyncio
from typing import Dict, Any
from unittest.mock import AsyncMock, patch, MagicMock
from app.core.tools.dynamic_executor import (
    DynamicToolExecutor,
    TaskRequest,
    TaskResult,
    TaskType,
)
from app.core.tools.base import BaseTool, ToolResult
from app.core.tools.registry import ToolRegistry


class MockTool(BaseTool):
    """Mock tool for testing"""

    def __init__(self, name: str, keywords: list = None, categories: list = None):
        super().__init__()
        self._name = name
        self._keywords = keywords or []
        self._categories = categories or ["general"]
        self._execute_result = {"mock": "data", "tool": name}

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Mock tool {self._name}"

    @property
    def keywords(self) -> list:
        return self._keywords

    @property
    def categories(self) -> list:
        return self._categories

    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        """Define mock parameters based on tool type"""
        if "search" in self._name:
            return {
                "query": {
                    "type": str,
                    "description": "Search query",
                    "required": True,
                },
                "results_count": {
                    "type": int,
                    "description": "Number of results",
                    "required": False,
                    "default": 10,
                },
            }
        elif "scrape" in self._name:
            return {
                "url": {
                    "type": str,
                    "description": "URL to scrape",
                    "required": True,
                }
            }
        elif "rerank" in self._name:
            return {
                "query": {
                    "type": str,
                    "description": "Query for reranking",
                    "required": True,
                },
                "documents": {
                    "type": list,
                    "description": "Documents to rerank",
                    "required": True,
                },
            }
        else:
            return {
                "query": {
                    "type": str,
                    "description": "Query parameter",
                    "required": False,
                    "default": "default",
                }
            }

    async def execute(self, **kwargs):
        return self._execute_result


@pytest.mark.unit
class TestDynamicToolExecutor:
    """Test cases for DynamicToolExecutor"""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock tool registry"""
        registry = ToolRegistry()

        # Register mock tools
        search_tool = MockTool(
            "search_tool", keywords=["search", "web", "find"], categories=["search"]
        )
        scrape_tool = MockTool(
            "scrape_tool",
            keywords=["scrape", "extract", "content"],
            categories=["scraping"],
        )
        rerank_tool = MockTool(
            "rerank_tool", keywords=["rerank", "sort", "rank"], categories=["ranking"]
        )
        general_tool = MockTool(
            "general_tool", keywords=["general", "helper"], categories=["general"]
        )

        registry.register(search_tool, "search")
        registry.register(scrape_tool, "scraping")
        registry.register(rerank_tool, "ranking")
        registry.register(general_tool, "general")

        return registry

    @pytest.fixture
    def executor(self, mock_registry):
        """Create a DynamicToolExecutor instance"""
        return DynamicToolExecutor(mock_registry)

    @pytest.mark.asyncio
    async def test_execute_search_task(self, executor):
        """Test executing a search task"""
        request = TaskRequest(
            task_type=TaskType.SEARCH,
            query="test search query",
            context={"results_count": 5},
        )

        result = await executor.execute_task(request)

        assert result.success is True
        assert result.task_type == TaskType.SEARCH
        assert len(result.tool_results) > 0
        assert "search_tool" in [r.tool_name for r in result.tool_results]

    @pytest.mark.asyncio
    async def test_execute_scrape_task(self, executor):
        """Test executing a scrape task"""
        request = TaskRequest(
            task_type=TaskType.SCRAPE,
            query="scrape https://example.com",
            context={"url": "https://example.com"},
        )

        result = await executor.execute_task(request)

        assert result.success is True
        assert result.task_type == TaskType.SCRAPE
        assert len(result.tool_results) > 0
        assert "scrape_tool" in [r.tool_name for r in result.tool_results]

    @pytest.mark.asyncio
    async def test_execute_rerank_task(self, executor):
        """Test executing a rerank task"""
        request = TaskRequest(
            task_type=TaskType.RERANK,
            query="rerank documents",
            context={"documents": ["doc1", "doc2", "doc3"]},
        )

        result = await executor.execute_task(request)

        assert result.success is True
        assert result.task_type == TaskType.RERANK
        assert len(result.tool_results) > 0
        assert "rerank_tool" in [r.tool_name for r in result.tool_results]

    @pytest.mark.asyncio
    async def test_execute_task_with_required_tools(self, executor):
        """Test executing a task with specific required tools"""
        request = TaskRequest(
            task_type=TaskType.SEARCH,
            query="test query",
            context={},
            required_tools=["search_tool"],
        )

        result = await executor.execute_task(request)

        assert result.success is True
        assert len(result.tool_results) == 1
        assert result.tool_results[0].tool_name == "search_tool"

    @pytest.mark.asyncio
    async def test_execute_task_with_excluded_tools(self, executor):
        """Test executing a task with excluded tools"""
        request = TaskRequest(
            task_type=TaskType.SEARCH,
            query="test query",
            context={},
            excluded_tools=["general_tool"],
        )

        result = await executor.execute_task(request)

        assert result.success is True
        tool_names = [r.tool_name for r in result.tool_results]
        assert "general_tool" not in tool_names

    @pytest.mark.asyncio
    async def test_execute_task_no_suitable_tools(self, executor):
        """Test executing a task with no suitable tools"""
        request = TaskRequest(
            task_type=TaskType.SEARCH,
            query="test query",
            context={},
            required_tools=["nonexistent_tool"],
        )

        result = await executor.execute_task(request)

        assert result.success is False
        assert "No suitable tools found" in result.error
        assert len(result.tool_results) == 0

    @pytest.mark.asyncio
    async def test_tool_selection_by_keywords(self, executor):
        """Test that tools are selected based on keywords"""
        request = TaskRequest(
            task_type=TaskType.SEARCH, query="find information about search", context={}
        )

        result = await executor.execute_task(request)

        assert result.success is True
        # Should select search tool due to keyword matching
        tool_names = [r.tool_name for r in result.tool_results]
        assert "search_tool" in tool_names

    @pytest.mark.asyncio
    async def test_tool_selection_by_categories(self, executor):
        """Test that tools are selected based on categories"""
        request = TaskRequest(task_type=TaskType.SEARCH, query="test query", context={})

        result = await executor.execute_task(request)

        assert result.success is True
        # Should select tools from search category
        tool_names = [r.tool_name for r in result.tool_results]
        assert "search_tool" in tool_names

    @pytest.mark.asyncio
    async def test_parameter_mapping(self, executor):
        """Test that parameters are correctly mapped to tools"""
        request = TaskRequest(
            task_type=TaskType.SEARCH,
            query="test query",
            context={},
            parameters={"results_count": 15},
        )

        result = await executor.execute_task(request)

        assert result.success is True
        # Verify parameters were passed to tool
        for tool_result in result.tool_results:
            assert tool_result.success is True

    @pytest.mark.asyncio
    async def test_url_extraction_from_query(self, executor):
        """Test URL extraction from query"""
        request = TaskRequest(
            task_type=TaskType.SCRAPE,
            query="scrape https://example.com/page",
            context={},
        )

        result = await executor.execute_task(request)

        assert result.success is True
        # Should extract URL from query
        tool_names = [r.tool_name for r in result.tool_results]
        assert "scrape_tool" in tool_names

    @pytest.mark.asyncio
    async def test_combine_search_results(self, executor):
        """Test combining multiple search results"""
        # Create a mock search result
        search_result = ToolResult(
            success=True,
            data={
                "results": [
                    {"title": "Result 1", "url": "https://example.com/1"},
                    {"title": "Result 2", "url": "https://example.com/2"},
                ],
                "engines": ["google"],
                "search_time": 0.5,
            },
            tool_name="search_tool",
            execution_time=0.5,
        )

        combined = executor._combine_search_results([search_result])

        assert combined["total_results"] == 2
        assert len(combined["results"]) == 2
        assert combined["engines"] == ["google"]
        assert combined["search_time"] == 0.5

    @pytest.mark.asyncio
    async def test_combine_scrape_results(self, executor):
        """Test combining multiple scrape results"""
        # Create a mock scrape result
        scrape_result = ToolResult(
            success=True,
            data={
                "url": "https://example.com",
                "title": "Example Page",
                "content": "This is the content",
                "content_length": 19,
            },
            tool_name="scrape_tool",
            execution_time=1.0,
        )

        combined = executor._combine_scrape_results([scrape_result])

        assert combined["total_content_length"] == 19
        assert len(combined["scraped_content"]) == 1
        assert combined["urls_scraped"] == ["https://example.com"]

    @pytest.mark.asyncio
    async def test_execution_history(self, executor):
        """Test execution history tracking"""
        request = TaskRequest(task_type=TaskType.SEARCH, query="test query", context={})

        # Execute a task
        await executor.execute_task(request)

        # Check history
        history = executor.get_execution_history()
        assert len(history) == 1
        assert history[0].task_type == TaskType.SEARCH

        # Clear history
        executor.clear_execution_history()
        history = executor.get_execution_history()
        assert len(history) == 0

    @pytest.mark.asyncio
    async def test_executor_statistics(self, executor):
        """Test executor statistics"""
        request = TaskRequest(task_type=TaskType.SEARCH, query="test query", context={})

        # Execute multiple tasks
        await executor.execute_task(request)
        await executor.execute_task(request)

        stats = executor.get_stats()

        assert stats["total_executions"] == 2
        assert stats["successful_executions"] == 2
        assert stats["success_rate"] == 1.0
        assert "average_execution_time" in stats
        assert "task_type_distribution" in stats
        assert "most_used_tools" in stats

    @pytest.mark.asyncio
    async def test_max_tools_limit(self, executor):
        """Test that max_tools parameter is respected"""
        request = TaskRequest(
            task_type=TaskType.GENERAL, query="test query", context={}, max_tools=2
        )

        result = await executor.execute_task(request)

        assert result.success is True
        assert len(result.tool_results) <= 2

    @pytest.mark.asyncio
    async def test_task_request_validation(self, executor):
        """Test TaskRequest validation"""
        # Test with missing required fields
        with pytest.raises(TypeError):
            TaskRequest()  # Missing required fields

        # Test with valid request
        request = TaskRequest(task_type=TaskType.SEARCH, query="test query", context={})

        assert request.task_type == TaskType.SEARCH
        assert request.query == "test query"
        assert request.context == {}

    @pytest.mark.asyncio
    async def test_task_result_structure(self, executor):
        """Test TaskResult structure"""
        request = TaskRequest(task_type=TaskType.SEARCH, query="test query", context={})

        result = await executor.execute_task(request)

        assert isinstance(result, TaskResult)
        assert hasattr(result, "success")
        assert hasattr(result, "data")
        assert hasattr(result, "tool_results")
        assert hasattr(result, "execution_time")
        assert hasattr(result, "task_type")
        assert hasattr(result, "error")
        assert hasattr(result, "metadata")

    @pytest.mark.asyncio
    async def test_error_handling_in_tool_execution(self, executor, mock_registry):
        """Test error handling when tool execution fails"""
        # Create a tool that raises an exception
        failing_tool = MockTool("failing_tool")
        failing_tool.execute = AsyncMock(side_effect=Exception("Tool failed"))
        mock_registry.register(failing_tool, "failing")

        request = TaskRequest(
            task_type=TaskType.GENERAL,
            query="test query",
            context={},
            required_tools=["failing_tool"],
        )

        result = await executor.execute_task(request)

        # Should handle the error gracefully
        assert result.success is False
        assert "Tool failed" in result.error

    @pytest.mark.asyncio
    async def test_parameter_validation(self, executor, mock_registry):
        """Test parameter validation for tools"""

        # Create a tool with specific parameter requirements
        class ValidatedTool(MockTool):
            @property
            def parameters(self):
                return {
                    "required_param": {
                        "type": str,
                        "description": "Required parameter",
                        "required": True,
                    }
                }

        validated_tool = ValidatedTool("validated_tool")
        mock_registry.register(validated_tool, "validated")

        request = TaskRequest(
            task_type=TaskType.GENERAL,
            query="test query",
            context={},
            required_tools=["validated_tool"],
            parameters={},  # Missing required parameter
        )

        result = await executor.execute_task(request)

        # Should fail due to missing required parameter
        assert result.success is False
        assert "Invalid parameters" in result.error


@pytest.mark.unit
class TestTaskRequest:
    """Test cases for TaskRequest"""

    def test_task_request_creation(self):
        """Test creating a TaskRequest"""
        request = TaskRequest(
            task_type=TaskType.SEARCH,
            query="test query",
            context={"key": "value"},
            parameters={"param": "value"},
            required_tools=["search_tool"],
            excluded_tools=["excluded_tool"],
            max_tools=3,
        )

        assert request.task_type == TaskType.SEARCH
        assert request.query == "test query"
        assert request.context == {"key": "value"}
        assert request.parameters == {"param": "value"}
        assert request.required_tools == ["search_tool"]
        assert request.excluded_tools == ["excluded_tool"]
        assert request.max_tools == 3

    def test_task_request_defaults(self):
        """Test TaskRequest with default values"""
        request = TaskRequest(task_type=TaskType.SEARCH, query="test query", context={})

        assert request.parameters is None
        assert request.required_tools is None
        assert request.excluded_tools is None
        assert request.max_tools == 5


@pytest.mark.unit
class TestTaskResult:
    """Test cases for TaskResult"""

    def test_task_result_creation(self):
        """Test creating a TaskResult"""
        tool_results = [
            ToolResult(
                success=True,
                data={"result": "data"},
                tool_name="test_tool",
                execution_time=1.0,
            )
        ]

        result = TaskResult(
            success=True,
            data={"combined": "result"},
            tool_results=tool_results,
            execution_time=2.0,
            task_type=TaskType.SEARCH,
            metadata={"test": "metadata"},
        )

        assert result.success is True
        assert result.data == {"combined": "result"}
        assert len(result.tool_results) == 1
        assert result.execution_time == 2.0
        assert result.task_type == TaskType.SEARCH
        assert result.metadata == {"test": "metadata"}
        assert result.error is None

    def test_task_result_with_error(self):
        """Test creating a TaskResult with an error"""
        result = TaskResult(
            success=False,
            data=None,
            tool_results=[],
            execution_time=0.5,
            task_type=TaskType.SEARCH,
            error="Something went wrong",
        )

        assert result.success is False
        assert result.data is None
        assert result.error == "Something went wrong"
