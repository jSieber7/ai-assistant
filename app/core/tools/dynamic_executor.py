"""
Dynamic Tool Execution Manager for the AI Assistant system.

This module provides a dynamic tool execution manager that can select and run
tools based on context, requirements, and available tools in the registry.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from .base import BaseTool, ToolResult
from .registry import ToolRegistry
from ..agents.base import AgentResult

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of tasks that can be executed"""

    SEARCH = "search"
    SCRAPE = "scrape"
    RERANK = "rerank"
    CALCULATE = "calculate"
    GENERAL = "general"


@dataclass
class TaskRequest:
    """Request for tool execution"""

    task_type: TaskType
    query: str
    context: Dict[str, Any]
    parameters: Optional[Dict[str, Any]] = None
    required_tools: Optional[List[str]] = None
    excluded_tools: Optional[List[str]] = None
    max_tools: int = 5


@dataclass
class TaskResult:
    """Result from task execution"""

    success: bool
    data: Any
    tool_results: List[ToolResult]
    execution_time: float
    task_type: TaskType
    error: Optional[str] = None
    metadata: Dict[str, Any] = None


class DynamicToolExecutor:
    """
    Dynamic tool execution manager that selects and runs tools based on context.

    This executor analyzes task requirements, selects appropriate tools from the
    registry, and executes them with proper parameter mapping and error handling.
    """

    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
        self._execution_history: List[TaskResult] = []

        # Task type to tool keywords mapping
        self._task_tool_mapping = {
            TaskType.SEARCH: ["search", "web", "internet", "find", "lookup", "query"],
            TaskType.SCRAPE: [
                "scrape",
                "crawl",
                "extract",
                "content",
                "web",
                "firecrawl",
            ],
            TaskType.RERANK: ["rerank", "rank", "sort", "jina", "relevance"],
            TaskType.CALCULATE: ["calculate", "math", "compute", "calculator"],
            TaskType.GENERAL: ["general", "utility", "helper", "assistant"],
        }

    async def execute_task(self, request: TaskRequest) -> TaskResult:
        """
        Execute a task by dynamically selecting and running appropriate tools.

        Args:
            request: Task request with type, query, and context

        Returns:
            TaskResult with execution results and metadata
        """
        start_time = time.time()

        try:
            # Step 1: Select appropriate tools
            selected_tools = await self._select_tools(request)

            if not selected_tools:
                task_result = TaskResult(
                    success=False,
                    data=None,
                    tool_results=[],
                    execution_time=time.time() - start_time,
                    task_type=request.task_type,
                    error="No suitable tools found for the task",
                    metadata={"request": request},
                )
                self._execution_history.append(task_result)
                return task_result

            # Step 2: Execute tools in sequence or parallel as appropriate
            tool_results = await self._execute_tools(request, selected_tools)

            # Step 3: Process and combine results
            combined_result = await self._process_results(request, tool_results)

            execution_time = time.time() - start_time

            # Determine if task was successful (at least one tool succeeded)
            successful_tools = [r for r in tool_results if r.success]
            task_success = len(successful_tools) > 0

            # Store execution history
            task_result = TaskResult(
                success=task_success,
                data=combined_result if task_success else None,
                tool_results=tool_results,
                execution_time=execution_time,
                task_type=request.task_type,
                error=None if task_success else "No tools executed successfully",
                metadata={
                    "request": request,
                    "tools_used": [tool.name for tool in selected_tools],
                    "tool_count": len(selected_tools),
                },
            )

            self._execution_history.append(task_result)
            return task_result

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Task execution failed: {str(e)}"
            logger.error(error_msg)

            task_result = TaskResult(
                success=False,
                data=None,
                tool_results=[],
                execution_time=execution_time,
                task_type=request.task_type,
                error=error_msg,
                metadata={"request": request, "error_type": type(e).__name__},
            )

            self._execution_history.append(task_result)
            return task_result

    async def _select_tools(self, request: TaskRequest) -> List[BaseTool]:
        """
        Select appropriate tools for the given task.

        Args:
            request: Task request

        Returns:
            List of selected tools
        """
        # Start with explicit required tools
        selected_tools = []

        if request.required_tools:
            for tool_name in request.required_tools:
                tool = self.tool_registry.get_tool(tool_name)
                if tool and tool.enabled:
                    selected_tools.append(tool)
                else:
                    logger.warning(f"Required tool '{tool_name}' not found or disabled")

        # Find additional relevant tools based on task type and keywords
        if len(selected_tools) < request.max_tools:
            # Get keywords for task type
            task_keywords = self._task_tool_mapping.get(request.task_type, [])

            # Find relevant tools
            relevant_tools = self.tool_registry.find_relevant_tools(
                query=request.query,
                context=request.context,
                max_results=request.max_tools - len(selected_tools),
            )

            # Filter by task relevance and exclusions
            for tool in relevant_tools:
                if tool.name in [t.name for t in selected_tools]:
                    continue  # Already selected

                if request.excluded_tools and tool.name in request.excluded_tools:
                    continue  # Excluded tool

                # Check if tool is relevant to task type
                if self._is_tool_relevant_to_task(
                    tool, request.task_type, task_keywords
                ):
                    selected_tools.append(tool)

                if len(selected_tools) >= request.max_tools:
                    break

        return selected_tools

    def _is_tool_relevant_to_task(
        self, tool: BaseTool, task_type: TaskType, task_keywords: List[str]
    ) -> bool:
        """
        Check if a tool is relevant to a specific task type.

        Args:
            tool: Tool to check
            task_type: Type of task
            task_keywords: Keywords associated with the task type

        Returns:
            True if tool is relevant to the task
        """
        # Check if tool keywords match task keywords
        tool_keywords = [kw.lower() for kw in tool.keywords]
        task_keywords_lower = [kw.lower() for kw in task_keywords]

        # Check for keyword matches (both directions)
        for tool_kw in tool_keywords:
            for task_kw in task_keywords_lower:
                if tool_kw in task_kw or task_kw in tool_kw:
                    return True

        # Check tool categories
        task_type_str = task_type.value.lower()
        for category in tool.categories:
            if task_type_str in category.lower() or category.lower() in task_type_str:
                return True

        # Check tool name
        if task_type_str in tool.name.lower() or tool.name.lower() in task_type_str:
            return True

        return False

    async def _execute_tools(
        self, request: TaskRequest, tools: List[BaseTool]
    ) -> List[ToolResult]:
        """
        Execute the selected tools with appropriate parameters.

        Args:
            request: Task request
            tools: List of tools to execute

        Returns:
            List of tool execution results
        """
        tool_results = []

        # Prepare parameters for each tool
        for tool in tools:
            try:
                # Map request parameters to tool parameters
                tool_params = self._map_parameters(request, tool)

                # Validate parameters
                if not tool.validate_parameters(**tool_params):
                    logger.warning(
                        f"Invalid parameters for tool '{tool.name}': {tool_params}"
                    )
                    tool_results.append(
                        ToolResult(
                            success=False,
                            data=None,
                            error=f"Invalid parameters: {tool_params}",
                            tool_name=tool.name,
                            execution_time=0.0,
                        )
                    )
                    continue

                # Execute tool
                result = await tool.execute_with_timeout(**tool_params)
                tool_results.append(result)

                logger.info(
                    f"Executed tool '{tool.name}' with success: {result.success}"
                )

            except Exception as e:
                logger.error(f"Error executing tool '{tool.name}': {str(e)}")
                tool_results.append(
                    ToolResult(
                        success=False,
                        data=None,
                        error=str(e),
                        tool_name=tool.name,
                        execution_time=0.0,
                    )
                )

        return tool_results

    def _map_parameters(self, request: TaskRequest, tool: BaseTool) -> Dict[str, Any]:
        """
        Map request parameters to tool-specific parameters.

        Args:
            request: Task request
            tool: Tool to map parameters for

        Returns:
            Mapped parameters for the tool
        """
        # Start with tool-specific parameters from request
        tool_params = request.parameters.copy() if request.parameters else {}

        # Add common parameters based on task type
        if request.task_type == TaskType.SEARCH:
            if "query" not in tool_params:
                tool_params["query"] = request.query

            # Set default search parameters if not provided
            if (
                "results_count" not in tool_params
                and "results_count" in tool.parameters
            ):
                tool_params["results_count"] = tool.parameters["results_count"].get(
                    "default", 10
                )

        elif request.task_type == TaskType.SCRAPE:
            if "url" not in tool_params:
                # Try to extract URL from query or context
                url = self._extract_url_from_query(request.query, request.context)
                if url:
                    tool_params["url"] = url

        elif request.task_type == TaskType.RERANK:
            if "query" not in tool_params:
                tool_params["query"] = request.query

            # Documents might be in context
            if "documents" not in tool_params and "documents" in request.context:
                tool_params["documents"] = request.context["documents"]

        # Add context parameters
        for key, value in request.context.items():
            if key in tool.parameters and key not in tool_params:
                tool_params[key] = value

        return tool_params

    def _extract_url_from_query(
        self, query: str, context: Dict[str, Any]
    ) -> Optional[str]:
        """
        Extract URL from query or context.

        Args:
            query: Query string
            context: Context dictionary

        Returns:
            Extracted URL or None
        """
        import re

        # Try to extract URL from query
        url_pattern = r"https?://[^\s]+|www\.[^\s]+"
        urls = re.findall(url_pattern, query)

        if urls:
            return urls[0]

        # Try to get URL from context
        if "url" in context:
            return context["url"]

        if "urls" in context and isinstance(context["urls"], list) and context["urls"]:
            return context["urls"][0]

        return None

    async def _process_results(
        self, request: TaskRequest, tool_results: List[ToolResult]
    ) -> Any:
        """
        Process and combine tool execution results.

        Args:
            request: Original task request
            tool_results: Results from tool executions

        Returns:
            Processed and combined result
        """
        # Filter successful results
        successful_results = [r for r in tool_results if r.success]

        if not successful_results:
            return {
                "error": "No tools executed successfully",
                "tool_results": [r.error for r in tool_results if r.error],
            }

        # For single tool results, return directly
        if len(successful_results) == 1:
            return successful_results[0].data

        # For multiple results, combine based on task type
        if request.task_type == TaskType.SEARCH:
            return self._combine_search_results(successful_results)
        elif request.task_type == TaskType.SCRAPE:
            return self._combine_scrape_results(successful_results)
        elif request.task_type == TaskType.RERANK:
            return self._combine_rerank_results(successful_results)
        else:
            # Default combination
            return {
                "combined_results": [r.data for r in successful_results],
                "tool_names": [r.tool_name for r in successful_results],
                "result_count": len(successful_results),
            }

    def _combine_search_results(self, results: List[ToolResult]) -> Dict[str, Any]:
        """Combine multiple search results"""
        combined = {
            "results": [],
            "total_results": 0,
            "engines": [],
            "search_time": 0,
            "sources": [],
        }

        for result in results:
            data = result.data
            if isinstance(data, dict):
                # Add results
                if "results" in data:
                    combined["results"].extend(data["results"])

                # Add metadata
                if "engines" in data:
                    combined["engines"].extend(data["engines"])

                if "search_time" in data:
                    combined["search_time"] += data["search_time"]

                combined["sources"].append(result.tool_name)

        combined["total_results"] = len(combined["results"])
        return combined

    def _combine_scrape_results(self, results: List[ToolResult]) -> Dict[str, Any]:
        """Combine multiple scrape results"""
        combined = {
            "scraped_content": [],
            "total_content_length": 0,
            "urls_scraped": [],
            "sources": [],
        }

        for result in results:
            data = result.data
            if isinstance(data, dict):
                # Add content
                if "content" in data:
                    combined["scraped_content"].append(
                        {
                            "url": data.get("url", ""),
                            "title": data.get("title", ""),
                            "content": data["content"],
                            "content_length": len(data["content"]),
                            "source": result.tool_name,
                        }
                    )
                    combined["total_content_length"] += len(data["content"])

                if "url" in data:
                    combined["urls_scraped"].append(data["url"])

                combined["sources"].append(result.tool_name)

        return combined

    def _combine_rerank_results(self, results: List[ToolResult]) -> Dict[str, Any]:
        """Combine multiple rerank results"""
        # For reranking, typically only one tool is used
        if results:
            return results[0].data

        return {"error": "No reranking results available"}

    def get_execution_history(self) -> List[TaskResult]:
        """Get the execution history"""
        return self._execution_history.copy()

    def clear_execution_history(self):
        """Clear the execution history"""
        self._execution_history.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics"""
        if not self._execution_history:
            return {"total_executions": 0}

        total_executions = len(self._execution_history)
        successful_executions = sum(1 for r in self._execution_history if r.success)

        task_type_counts = {}
        for result in self._execution_history:
            task_type = result.task_type.value
            task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1

        avg_execution_time = (
            sum(r.execution_time for r in self._execution_history) / total_executions
        )

        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "success_rate": successful_executions / total_executions,
            "average_execution_time": avg_execution_time,
            "task_type_distribution": task_type_counts,
            "most_used_tools": self._get_most_used_tools(),
        }

    def _get_most_used_tools(self) -> List[Dict[str, Any]]:
        """Get the most used tools from execution history"""
        tool_counts = {}

        for result in self._execution_history:
            for tool_result in result.tool_results:
                tool_name = tool_result.tool_name
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

        # Sort by usage count
        sorted_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)

        return [
            {"tool_name": name, "usage_count": count}
            for name, count in sorted_tools[:10]
        ]
